"""One explicit offline entrypoint over the existing import/run/eval/export CLIs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from common import git_state, new_run, sha256, source_manifest, utc, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequence", nargs="?", default="07", choices=("07", "00"))
    for flag, variable in (
        ("archives-manifest", "M3_SLAM_ARCHIVES_MANIFEST"),
        ("build-receipt", "M3_SLAM_BUILD_RECEIPT"),
        ("binary", "M3_SLAM_BINARY"),
        ("eval-python", "M3_SLAM_EVAL_PYTHON"),
        ("output-parent", "M3_SLAM_OUTPUT_PARENT"),
    ):
        p.add_argument("--" + flag, type=Path, default=os.environ.get(variable),
                       help="Explicit path; environment fallback: " + variable)
    p.add_argument("--config", type=Path, help="Default: frozen seq07 config; stretch00 requires an explicit config")
    p.add_argument("--data-root", type=Path, help="Optional existing verified extraction to revalidate/reuse")
    p.add_argument("--official-oracle", type=Path, default=os.environ.get("M3_SLAM_OFFICIAL_ORACLE"))
    p.add_argument("--official-timeout-s", type=float, default=60.0)
    p.add_argument("--git", default=os.environ.get("M3_SLAM_GIT", "git"))
    p.add_argument("--allow-sequence-00", action="store_true", help="Explicit nonprimary stretch sequence; never implied by the default")
    return p


def returned_path(stdout, parent, marker=None, exact=None):
    """Use only this invocation's sole returned path, never directory timestamps."""
    lines = stdout.decode("utf-8").strip().splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("stage must return exactly one output path; inspect its stdout/stderr")
    path = Path(lines[0]).resolve(strict=True)
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("stage returned a path outside its assigned output parent")
    if exact is not None and path != exact.resolve():
        raise ValueError("import returned an unexpected manifest path")
    if marker is not None and not (path / marker).is_file():
        raise ValueError("stage returned no expected receipt: " + marker)
    return path


def run(args, repo):
    output = new_run(args.output_parent, "kitti-pipeline-" + args.sequence)
    record = {"status": "started", "started_utc": utc(), "command": sys.argv,
              "sequence": args.sequence, "output": str(output.resolve()),
              "official_requested": args.official_oracle is not None, "stages": []}
    start = time.monotonic()
    receipt = output / "pipeline.json"

    def stage(label, argv, parent, marker=None, exact=None):
        item = {"stage": label, "argv": argv, "output_parent": str(parent), "started_utc": utc()}
        record["stages"].append(item)
        write_json(receipt, record)
        stdout, stderr = output / (label + ".stdout.txt"), output / (label + ".stderr.txt")
        print(label + ": " + str(parent), file=sys.stderr, flush=True)
        then = time.monotonic()
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                completed = subprocess.run(argv, cwd=repo, stdout=out, stderr=err)
            item["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(label + " exited " + str(completed.returncode) + "; retain " + str(stderr))
            path = returned_path(stdout.read_bytes(), parent, marker, exact)
            item.update(status="complete", returned_path=str(path))
            return path
        except BaseException as exc:
            item.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            item.update(finished_utc=utc(), wall_s=time.monotonic() - then)
            item["logs"] = {str(p.name): sha256(p) for p in (stdout, stderr) if p.exists()}
            write_json(receipt, record)

    try:
        record.update(git_state(repo, args.git))
        record["source_files"] = source_manifest(repo / "slam")
        record["inputs"] = {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
            ("archives_manifest", args.archives_manifest), ("build_receipt", args.build_receipt),
            ("binary", args.binary), ("config", args.config))}
        scripts = repo / "slam/python"
        python = str(args.eval_python)
        data = args.data_root or output / ("data-seq" + args.sequence)
        imported = [python, str(scripts / "import_kitti.py"), "--archives", str(args.archives_manifest),
                    "--config", str(args.config), "--output", str(data)]
        if args.sequence == "00":
            imported.append("--allow-sequence-00")
        manifest = stage("import", imported, data, exact=data / "input-manifest.json")
        imported_data = json.loads(manifest.read_text())
        if Path(imported_data["root"]).resolve() != data.resolve():
            raise ValueError("returned manifest root differs from assigned extraction")
        truth = data / ("dataset/poses/" + args.sequence + ".txt")
        common = ["--repo-root", str(repo), "--git", args.git]
        run_dir = stage("run", [python, str(scripts / "run_experiment.py"), "--config", str(args.config),
                        "--input-manifest", str(manifest), "--binary", str(args.binary),
                        "--build-receipt", str(args.build_receipt), *common,
                        "--output-parent", str(output / "runs")], output / "runs", "run.json")
        evaluation = [python, str(scripts / "eval_traj.py"), "--config", str(args.config),
                      "--native-run", str(run_dir / "native"), "--truth", str(truth), *common,
                      "--output-parent", str(output / "evaluations"), "--evo"]
        if args.official_oracle:
            evaluation += ["--official-oracle", str(args.official_oracle),
                           "--official-timeout-s", str(args.official_timeout_s)]
        evaluated = stage("evaluate", evaluation, output / "evaluations", "metrics.json")
        metrics = json.loads((evaluated / "metrics.json").read_text())
        if metrics.get("own_metrics_status") != "complete" or metrics.get("evo_status") != "passed":
            raise ValueError("evaluation did not pass own metrics and actual evo checks")
        if args.official_oracle and metrics.get("official_devkit_status") != "passed":
            raise ValueError("requested official comparison did not pass")
        calibration = stage("calibrate", [python, str(scripts / "calib_check.py"), "--config", str(args.config),
                            "--input-manifest", str(manifest), *common, "--output-parent", str(output / "calibration")],
                            output / "calibration", "calibration.json")
        replay = stage("export", [python, str(scripts / "export_replay.py"), "--run", str(run_dir),
                       "--truth", str(truth), *common, "--output-parent", str(output / "replay")],
                       output / "replay", "export.json")
        record.update(status="complete" if args.official_oracle else "partial",
                      official_devkit_status=metrics.get("official_devkit_status", "not_run"),
                      paths={"input_manifest": str(manifest), "run": str(run_dir), "evaluation": str(evaluated),
                             "calibration": str(calibration), "replay": str(replay)},
                      scope="existing offline CLI chain only; browser acceptance and resource-lease validation remain separate")
        if not args.official_oracle:
            record["partial_reason"] = "official oracle omitted; recorded-run official parity remains unverified"
        return 0
    except BaseException as exc:
        record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        print("pipeline failed: " + str(exc) + "; receipt: " + str(receipt), file=sys.stderr)
        return 1
    finally:
        record.update(finished_utc=utc(), wall_s=time.monotonic() - start)
        write_json(receipt, record)
        print(output)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ("archives_manifest", "build_receipt", "binary", "eval_python", "output_parent"):
        if getattr(args, name) is None:
            p.error("--" + name.replace("_", "-") + " or its named environment variable is required")
    repo = Path(__file__).resolve().parents[2]
    if args.sequence == "00" and (not args.allow_sequence_00 or args.config is None):
        p.error("stretch00 requires --allow-sequence-00 and an explicit frozen --config")
    if args.sequence == "00" and args.official_oracle:
        p.error("the accepted official-oracle adapter is explicitly restricted to full07")
    args.config = (args.config or repo / "slam/configs/seq07.json").resolve()
    for name in ("archives_manifest", "build_receipt", "binary", "eval_python", "config", "official_oracle"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            if not path.is_file():
                p.error("missing input file: " + str(path))
            setattr(args, name, path)
    if not math.isfinite(args.official_timeout_s) or args.official_timeout_s <= 0:
        p.error("official timeout must be positive")
    config = json.loads(args.config.read_text())
    if config.get("kind") != "kitti_full" or config.get("sequence") != args.sequence:
        p.error("config must be the declared full KITTI sequence")
    args.output_parent = args.output_parent.resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.resolve()
    return run(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
