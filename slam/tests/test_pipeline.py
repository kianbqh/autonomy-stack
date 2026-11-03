"""Synthetic CLI-chain contracts; all subprocesses mocked, no KITTI/native run."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import run_pipeline as pipeline


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("M3_PIPELINE_TEST_TMP", Path(__file__).resolve().parents[1] / "results/.pipeline-test-tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-wrapper-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.calls = []
        self.mode = "normal"
        self.inputs = {}
        for name in ("archives-manifest", "build-receipt", "binary", "eval-python", "official-oracle"):
            self.inputs[name] = self.root / name
            self.inputs[name].write_text("synthetic mocked input; never executed\n")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"kind": "kitti_full", "sequence": "07", "expected_frames": 1101}))
        self.output = self.root / "outputs"

    def arguments(self, official=True):
        args = []
        for name in ("archives-manifest", "build-receipt", "binary", "eval-python"):
            args += ["--" + name, str(self.inputs[name])]
        args += ["--output-parent", str(self.output), "--config", str(self.config)]
        if official:
            args += ["--official-oracle", str(self.inputs["official-oracle"])]
        return args

    def child(self, argv, cwd, stdout, stderr):
        self.calls.append(argv)
        name = Path(argv[1]).name
        value = lambda option: Path(argv[argv.index(option) + 1])
        if name == "import_kitti.py":
            data = value("--output")
            data.mkdir(parents=True)
            path = data / "input-manifest.json"
            path.write_text(json.dumps({"root": str(data)}))
            self.truth = data / "dataset/poses/07.txt"
            self.truth.parent.mkdir(parents=True)
            self.truth.write_text("synthetic truth placeholder; not parsed\n")
        else:
            parent = value("--output-parent")
            parent.mkdir(parents=True)
            path = parent / ("this-invocation-" + name.removesuffix(".py"))
            path.mkdir()
            if name == "run_experiment.py":
                self.native_run = path
                (path / "run.json").write_text(json.dumps({"status": "complete"}))
                decoy = parent / "zz-decoy-never-selected"
                decoy.mkdir()
                (decoy / "run.json").write_text(json.dumps({"status": "complete"}))
                if self.mode == "native-failure":
                    (path / "run.json").write_text(json.dumps({"status": "failed", "scope": "synthetic mocked child"}))
                    stderr.write(b"synthetic retained child failure\n")
                    return mock.Mock(returncode=7)
            elif name == "eval_traj.py":
                self.assertEqual(value("--native-run"), self.native_run / "native")
                self.assertEqual(value("--truth"), self.truth)
                official = "--official-oracle" in argv
                metrics = {"own_metrics_status": "complete", "evo_status": "passed",
                           "official_devkit_status": "passed" if official else "not_run"}
                if self.mode == "evo-failure":
                    metrics["evo_status"] = "failed"
                (path / "metrics.json").write_text(json.dumps(metrics))
                if self.mode == "escaped-output":
                    path = self.root / "outside-assigned-evaluation"
                    path.mkdir()
                    (path / "metrics.json").write_text(json.dumps(metrics))
            elif name == "calib_check.py":
                (path / "calibration.json").write_text(json.dumps({"status": "complete"}))
            elif name == "export_replay.py":
                self.assertEqual(value("--run"), self.native_run)
                self.assertEqual(value("--truth"), self.truth)
                (path / "export.json").write_text(json.dumps({"status": "complete"}))
            else:
                self.fail("unexpected child: " + name)
        stdout.write((str(path) + "\n").encode("utf-8"))
        if self.mode == "ambiguous-output" and name == "import_kitti.py":
            stdout.write((str(path) + "\n").encode("utf-8"))
        return mock.Mock(returncode=0)

    def invoke(self, argv):
        with mock.patch.object(pipeline, "git_state", return_value={"git_sha": "a" * 40}), \
             mock.patch.object(pipeline, "source_manifest", return_value={"synthetic": "fixture"}), \
             mock.patch.object(pipeline.subprocess, "run", side_effect=self.child), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pipeline.main(argv)
        attempts = list(self.output.iterdir())
        self.assertEqual(len(attempts), 1)
        return code, attempts[0], json.loads((attempts[0] / "pipeline.json").read_text())

    def test_exact_returned_paths_and_order_with_requested_oracle(self):
        original = self.config.read_bytes()
        code, output, record = self.invoke(self.arguments())
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "complete")
        self.assertEqual([Path(c[1]).name for c in self.calls],
                         ["import_kitti.py", "run_experiment.py", "eval_traj.py", "calib_check.py", "export_replay.py"])
        self.assertIn("--evo", self.calls[2])
        self.assertIn("--official-oracle", self.calls[2])
        self.assertEqual(Path(record["paths"]["run"]), self.native_run)
        self.assertNotIn("zz-decoy", record["paths"]["run"])
        self.assertEqual(self.config.read_bytes(), original)
        self.assertTrue(all(s["logs"] for s in record["stages"]))

    def test_omitted_oracle_remains_explicitly_partial(self):
        code, output, record = self.invoke(self.arguments(False))
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["official_devkit_status"], "not_run")
        self.assertNotIn("--official-oracle", self.calls[2])

    def test_child_failure_stops_and_preserves_attempt_and_raw_logs(self):
        self.mode = "native-failure"
        code, output, record = self.invoke(self.arguments())
        self.assertEqual(code, 1)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(record["stages"][-1]["returncode"], 7)
        self.assertEqual((output / "run.stderr.txt").read_bytes(), b"synthetic retained child failure\n")
        self.assertEqual(json.loads((self.native_run / "run.json").read_text())["status"], "failed")

    def test_path_escape_is_rejected_without_searching_alternatives(self):
        self.mode = "escaped-output"
        code, output, record = self.invoke(self.arguments())
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("outside its assigned", record["failure"]["message"])

    def test_ambiguous_stdout_cannot_select_a_latest_path(self):
        self.mode = "ambiguous-output"
        code, output, record = self.invoke(self.arguments())
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("exactly one", record["failure"]["message"])

    def test_zero_exit_evo_failure_is_still_failure(self):
        self.mode = "evo-failure"
        code, output, record = self.invoke(self.arguments(False))
        self.assertEqual(code, 1)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("actual evo", record["failure"]["message"])

    def test_required_paths_have_environment_fallbacks(self):
        variables = {"M3_SLAM_" + key.replace("-", "_").upper(): str(path)
                     for key, path in self.inputs.items() if key != "official-oracle"}
        variables["M3_SLAM_OUTPUT_PARENT"] = str(self.output)
        with mock.patch.dict(os.environ, variables, clear=True):
            code, output, record = self.invoke(["--config", str(self.config)])
        self.assertEqual(code, 0)
        self.assertEqual(record["sequence"], "07")
        self.assertEqual(record["status"], "partial")

    def test_stretch00_requires_both_explicit_controls(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            pipeline.main(["00", *self.arguments(False)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())

    def test_help_requires_no_paths_and_launches_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
            pipeline.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--archives-manifest", output.getvalue())
        self.assertIn("--official-oracle", output.getvalue())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
