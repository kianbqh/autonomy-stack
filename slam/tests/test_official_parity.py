"""Stdlib-only synthetic adapter tests. Every subprocess is mocked; no oracle is run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import official_parity as adapter


class OracleAdapterTests(unittest.TestCase):
    def setUp(self):
        # Always use the task's G-backed directory; tests create no system temp cache.
        scratch = Path(os.environ.get("M3_ORACLE_TEST_TMP", Path(__file__).resolve().parents[1] / "results/.oracle-test-tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-mocked-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.oracle = self.root / "fake-oracle-never-executed"
        self.oracle.write_bytes(b"offline test, not an executable")
        self.truth = self.root / "synthetic-truth.txt"
        self.prediction = self.root / "synthetic-prediction.txt"
        self.truth.write_text("synthetic fixture; not a KITTI trajectory\n")
        self.prediction.write_text("synthetic fixture; not a KITTI trajectory\n")
        self.own = {"frame_count": 4, "distance_bits": [0, 1120403456, 1128792064, 1128857600],
                    "endpoints": [{"start": 0, "length_m": 100, "end": 2},
                                  {"start": 0, "length_m": 200, "end": 3},
                                  {"start": 0, "length_m": 300, "end": -1}],
                    "segments": {"count": 2,
                                 "samples": [{"i": 0, "j": 2, "length_m": 100, "translation_ratio": 1.0, "rotation_rad_per_m": 0.5},
                                             {"i": 0, "j": 3, "length_m": 200, "translation_ratio": 1.0, "rotation_rad_per_m": 0.5}],
                                 "translation_ratio_mean": 1.0, "rotation_rad_per_m_mean": 0.5}}

    def outputs(self, destination):
        destination.mkdir(exist_ok=True)
        (destination / "distances.tsv").write_text("frame\tdistance_m\tbits\n0\t0\t0\n1\t100\t1120403456\n2\t200\t1128792064\n3\t201\t1128857600\n")
        (destination / "endpoints.tsv").write_text("start\tlength_m\tend\n0\t100\t2\n0\t200\t3\n0\t300\t-1\n")
        (destination / "errors.tsv").write_text("start\tend\tlength_m\ttranslation_ratio\trotation_rad_per_m\tspeed_mps\ttranslation_bits\trotation_bits\n0\t2\t100\t1\t0.5\t10\t1065353216\t1056964608\n0\t3\t200\t1\t0.5\t20\t1065353216\t1056964608\n")
        (destination / "stats-exact.tsv").write_text("count\ttranslation_ratio_mean\trotation_rad_per_m_mean\ttranslation_bits\trotation_bits\n2\t1\t0.5\t1065353216\t1056964608\n")
        (destination / "stats.txt").write_text("1.000000 0.500000\n")

    def compare(self, own=None):
        output = self.root / "synthetic-comparison"
        self.outputs(output)
        return output, self.own if own is None else own

    def replace(self, path, old, new):
        content = path.read_text()
        self.assertIn(old, content)
        path.write_text(content.replace(old, new))

    def invoke(self, effect):
        expected = adapter.identity(self.oracle)
        output = self.root / "mock-subprocess-receipt"
        with mock.patch.object(adapter, "ACCEPTED_ORACLE_SHA256", expected["sha256"]), \
             mock.patch.object(adapter, "ACCEPTED_ORACLE_BYTES", expected["bytes"]), \
             mock.patch.object(adapter.subprocess, "run", side_effect=effect) as run:
            result = adapter.run_official_crosscheck(self.oracle, self.truth, self.prediction,
                output, self.own, {"data_kind": "synthetic unit fixture; subprocess mocked; NOT real07"})
        self.assertEqual(run.call_count, 1)
        return output, result

    def test_exact_order_bits_scalars_and_original_aggregate(self):
        output, own = self.compare()
        result = adapter.compare_outputs(output, own)
        self.assertTrue(result["passed"])
        self.assertEqual((result["frames"], result["endpoint_rows"], result["segments"]), (4, 3, 2))
        self.assertEqual(result["exact_segment_scalars"], {"translation_ratio": 2, "rotation_rad_per_m": 2})

    def test_default_or_missing_gate_never_reports_complete(self):
        self.assertEqual(adapter.completion_status(True, False, [])["official_devkit_status"], "not_run")
        self.assertNotEqual(adapter.completion_status(True, False, [])["status"], "complete")
        self.assertEqual(adapter.completion_status(True, True, [])["official_devkit_status"], "failed")
        passed = {"status": "passed", "passed": True}
        self.assertEqual(adapter.completion_status(True, True, [passed])["official_devkit_status"], "failed")

    def test_both_actual_gate_receipts_required_for_complete(self):
        passed = {"status": "passed", "passed": True}
        self.assertEqual(adapter.completion_status(True, True, [passed, passed])["status"], "complete")
        self.assertNotEqual(adapter.completion_status(False, True, [passed, passed])["status"], "complete")
        failed = {"status": "failed", "passed": False}
        self.assertEqual(adapter.completion_status(True, True, [passed, failed])["official_devkit_status"], "failed")

    def test_small_numeric_difference_uses_frozen_tolerance(self):
        output, own = self.compare(copy.deepcopy(self.own))
        own["segments"]["samples"][0]["translation_ratio"] += 1e-7
        result = adapter.compare_outputs(output, own)
        self.assertEqual(result["exact_segment_scalars"]["translation_ratio"], 1)
        self.assertGreater(result["maximum_abs_scalar_difference"]["translation_ratio"], 0)

    def test_distance_bits_fail_even_if_decimal_text_is_unchanged(self):
        output, own = self.compare()
        self.replace(output / "distances.tsv", "1120403456", "1120403457")
        with self.assertRaisesRegex(ValueError, "bits mismatch"):
            adapter.compare_outputs(output, own)

    def test_invalid_endpoint_sentinel_or_order_fails(self):
        output, own = self.compare()
        self.replace(output / "endpoints.tsv", "0\t300\t-1", "0\t300\t3")
        with self.assertRaisesRegex(ValueError, "endpoint identity/order"):
            adapter.compare_outputs(output, own)

    def test_valid_segment_order_is_exact(self):
        output, own = self.compare()
        path = output / "errors.tsv"
        lines = path.read_text().splitlines(True)
        path.write_text(lines[0] + lines[2] + lines[1])
        with self.assertRaisesRegex(ValueError, "segment identity/order"):
            adapter.compare_outputs(output, own)

    def test_missing_or_extra_valid_segment_fails(self):
        output, own = self.compare()
        path = output / "errors.tsv"
        path.write_text("".join(path.read_text().splitlines(True)[:-1]))
        with self.assertRaisesRegex(ValueError, "segment count"):
            adapter.compare_outputs(output, own)

    def test_scalar_failure_is_distinct_from_count_and_pair_parity(self):
        output, own = self.compare(copy.deepcopy(self.own))
        own["segments"]["samples"][0]["translation_ratio"] = 1.1
        with self.assertRaisesRegex(ValueError, "per-segment scalar"):
            adapter.compare_outputs(output, own)

    def test_aggregate_cannot_be_replaced_by_recomputed_own_mean(self):
        output, own = self.compare(copy.deepcopy(self.own))
        own["segments"]["translation_ratio_mean"] = 0.75
        with self.assertRaisesRegex(ValueError, "original saveStats aggregate"):
            adapter.compare_outputs(output, own)

    def test_nonfinite_capture_fails(self):
        output, own = self.compare()
        self.replace(output / "errors.tsv", "1065353216", "2139095040")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            adapter.compare_outputs(output, own)

    def test_text_columns_cannot_disagree_with_raw_bits(self):
        output, own = self.compare()
        self.replace(output / "errors.tsv", "\t1\t0.5\t10", "\t9\t0.5\t10")
        with self.assertRaisesRegex(ValueError, "text differs"):
            adapter.compare_outputs(output, own)

    def test_malformed_schema_and_original_formatted_stats_fail(self):
        output, own = self.compare()
        self.replace(output / "stats.txt", "1.000000", "0.000000")
        with self.assertRaisesRegex(ValueError, "formatted saveStats"):
            adapter.compare_outputs(output, own)
        self.replace(output / "distances.tsv", "distance_m", "wrong")
        with self.assertRaisesRegex(ValueError, "table schema"):
            adapter.compare_outputs(output, own)

    def test_unaccepted_binary_is_never_launched(self):
        output = self.root / "unaccepted"
        with mock.patch.object(adapter.subprocess, "run") as run:
            record = adapter.run_official_crosscheck(self.oracle, self.truth, self.prediction, output,
                self.own, {"data_kind": "synthetic test; no invocation"})
        run.assert_not_called()
        self.assertFalse(record["passed"])
        self.assertIn("accepted binary", record["failure"]["message"])

    def test_nonzero_retains_returncode_raw_streams_and_partial_table(self):
        def nonzero(command, **options):
            Path(command[-1], "distances.tsv").write_text("partial synthetic output\n")
            return subprocess.CompletedProcess(command, 4, b"partial stdout\x00", b"no valid synthetic segments")
        output, record = self.invoke(nonzero)
        self.assertEqual(record["returncode"], 4)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["passed"])
        self.assertEqual((output / "stdout.bin").read_bytes(), b"partial stdout\x00")
        self.assertIn("distances.tsv", record["artifacts"])
        self.assertNotIn("comparison", record)

    def test_timeout_is_failed_not_exit_zero_and_preserves_partial_bytes(self):
        def timeout(command, **options):
            raise subprocess.TimeoutExpired(command, options["timeout"], output=b"before timeout", stderr=b"diagnostic")
        output, record = self.invoke(timeout)
        self.assertTrue(record["timed_out"])
        self.assertIsNone(record["returncode"])
        self.assertEqual(record["failure"]["kind"], "timeout")
        self.assertEqual((output / "stderr.bin").read_bytes(), b"diagnostic")

    def test_loader_error_is_preserved_without_a_fake_exitcode(self):
        def missing_loader(command, **options):
            raise OSError("synthetic missing ELF loader")
        _, record = self.invoke(missing_loader)
        self.assertEqual(record["failure"]["type"], "OSError")
        self.assertIsNone(record["returncode"])
        self.assertFalse(record["passed"])

    def test_mock_success_receipt_only_no_oracle_invoked(self):
        def completed(command, **options):
            self.outputs(Path(command[-1]))
            return subprocess.CompletedProcess(command, 0, b"mocked synthetic tables", b"")
        output, record = self.invoke(completed)
        self.assertTrue(record["passed"])
        self.assertEqual(record["comparison"]["segments"], 2)
        self.assertTrue(record["inputs_unchanged_after_invocation"])
        self.assertIn("NOT real07", record["context"]["data_kind"])
        self.assertEqual(json.loads((output / "receipt.json").read_text())["status"], "passed")
        self.assertEqual(record["receipt"]["sha256"], adapter.identity(output / "receipt.json")["sha256"])

    def test_exit_zero_without_official_outputs_still_fails(self):
        def incomplete(command, **options):
            return subprocess.CompletedProcess(command, 0, b"mock process exited zero without tables", b"")
        _, record = self.invoke(incomplete)
        self.assertEqual(record["returncode"], 0)
        self.assertFalse(record["passed"])
        self.assertEqual(record["failure"]["type"], "FileNotFoundError")

    def test_input_changed_during_mock_invocation_fails(self):
        def modified(command, **options):
            self.outputs(Path(command[-1]))
            self.prediction.write_text("changed synthetic fixture")
            return subprocess.CompletedProcess(command, 0, b"", b"")
        _, record = self.invoke(modified)
        self.assertFalse(record["passed"])
        self.assertIn("identity changed", record["failure"]["message"])

    def test_previous_attempt_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "receipt.json"
        marker.write_text("preserve existing test receipt")
        with self.assertRaises(FileExistsError):
            adapter.run_official_crosscheck(self.oracle, self.truth, self.prediction, output, self.own, {})
        self.assertEqual(marker.read_text(), "preserve existing test receipt")


if __name__ == "__main__":
    unittest.main()
