"""Own adapter for an explicitly supplied, pinned external KITTI oracle.

No official implementation is copied/imported here. Invocation is opt-in; the
caller must supply a completed real run and its computed comparison payload.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

ACCEPTED_ORACLE_SHA256 = "8ed947db73d62909a4bbda75e2475f9e147928390ab9f621b7f7a1127e41bbd7"
ACCEPTED_ORACLE_BYTES = 68856
ATOL = RTOL = 1e-6
OUTPUTS = ("distances.tsv", "endpoints.tsv", "errors.tsv", "stats-exact.tsv", "stats.txt")


def completion_status(evo_passed, official_requested, reports):
    official_passed = bool(official_requested and len(reports) == 2 and
                           all(row.get("passed") is True and row.get("status") == "passed" for row in reports))
    official = "passed" if official_passed else ("failed" if official_requested else "not_run")
    if evo_passed and official_passed:
        status = "complete"
    elif official_requested and not official_passed:
        status = "partial_official_devkit_crosscheck_failed"
    elif not evo_passed:
        status = "partial_independent_crosscheck_not_passed"
    else:
        status = "partial_official_devkit_parity_not_run"
    return {"official_devkit_status": official, "status": status}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite oracle/comparison value")
    return result


def uint32(value):
    result = int(value)
    if str(result) != str(value) or not 0 <= result < 2 ** 32:
        raise ValueError("invalid uint32 bit pattern")
    return result


def from_bits(value):
    return number(struct.unpack("<f", struct.pack("<I", uint32(value)))[0])


def close(actual, expected):
    actual, expected = number(actual), number(expected)
    return abs(actual - expected) <= ATOL + RTOL * abs(expected)


def rows(path, header):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != header:
            raise ValueError("unexpected oracle table schema: " + Path(path).name)
        result = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in result):
        raise ValueError("malformed oracle table row: " + Path(path).name)
    return result


def compare_outputs(output, own):
    """Compare ordered tables and original saveStats floats; never recompute a replacement mean."""
    output = Path(output)
    distances = rows(output / "distances.tsv", ["frame", "distance_m", "bits"])
    expected_bits = own["distance_bits"]
    if len(distances) != own["frame_count"] or len(expected_bits) != len(distances):
        raise ValueError("official distance/frame count mismatch")
    for index, (actual, expected) in enumerate(zip(distances, expected_bits)):
        bits = uint32(actual["bits"])
        if int(actual["frame"]) != index or bits != uint32(expected):
            raise ValueError("official distance frame/order/float32 bits mismatch")
        if not close(actual["distance_m"], from_bits(bits)):
            raise ValueError("official distance text differs from captured bits")

    endpoints = rows(output / "endpoints.tsv", ["start", "length_m", "end"])
    if len(endpoints) != len(own["endpoints"]):
        raise ValueError("official endpoint count mismatch")
    for actual, expected in zip(endpoints, own["endpoints"]):
        entry = (int(actual["start"]), number(actual["length_m"]), int(actual["end"]))
        if entry != (expected["start"], number(expected["length_m"]), expected["end"]):
            raise ValueError("official endpoint identity/order mismatch")

    errors = rows(output / "errors.tsv", ["start", "end", "length_m", "translation_ratio", "rotation_rad_per_m", "speed_mps", "translation_bits", "rotation_bits"])
    segments = own["segments"]
    if not errors or len(errors) != segments["count"] or len(errors) != len(segments["samples"]):
        raise ValueError("official valid segment count mismatch or empty result")
    max_difference = {"translation_ratio": 0.0, "rotation_rad_per_m": 0.0}
    exact = {key: 0 for key in max_difference}
    for actual, expected in zip(errors, segments["samples"]):
        entry = (int(actual["start"]), int(actual["end"]), number(actual["length_m"]))
        if entry != (expected["i"], expected["j"], number(expected["length_m"])):
            raise ValueError("official valid segment identity/order mismatch")
        number(actual["speed_mps"])
        for key, bits_key in (("translation_ratio", "translation_bits"), ("rotation_rad_per_m", "rotation_bits")):
            official = from_bits(actual[bits_key])
            if not close(actual[key], official):
                raise ValueError("official scalar text differs from captured bits")
            value = number(expected[key])
            if not close(value, official):
                raise ValueError("official per-segment scalar mismatch: " + key)
            max_difference[key] = max(max_difference[key], abs(value - official))
            exact[key] += int(value == official)

    stats = rows(output / "stats-exact.tsv", ["count", "translation_ratio_mean", "rotation_rad_per_m_mean", "translation_bits", "rotation_bits"])
    if len(stats) != 1 or int(stats[0]["count"]) != len(errors):
        raise ValueError("official saveStats count mismatch")
    official_means = {}
    mean_differences = {}
    for key, bits_key in (("translation_ratio_mean", "translation_bits"), ("rotation_rad_per_m_mean", "rotation_bits")):
        official = from_bits(stats[0][bits_key])
        if number(stats[0][key]) != official:
            raise ValueError("official captured mean text/bits mismatch")
        value = number(segments[key])
        if not close(value, official):
            raise ValueError("official original saveStats aggregate mismatch: " + key)
        official_means[key] = official
        mean_differences[key] = abs(value - official)
    formatted = (output / "stats.txt").read_text(encoding="utf-8").split()
    if len(formatted) != 2 or any(not close(value, expected) for value, expected in zip(formatted, official_means.values())):
        raise ValueError("original formatted saveStats output mismatch")
    return {"passed": True, "frames": len(distances), "endpoint_rows": len(endpoints),
            "segments": len(errors), "distance_bits_exact": True, "endpoint_and_segment_order_exact": True,
            "atol": ATOL, "rtol": RTOL, "tolerance_rule": "abs(actual-expected) <= atol + rtol*abs(expected)",
            "exact_segment_scalars": exact, "maximum_abs_scalar_difference": max_difference,
            "original_official_saveStats_means": official_means, "aggregate_abs_difference": mean_differences}


def run_official_crosscheck(binary, truth, prediction, output, own, context, timeout_s=60.0):
    """Preserve subprocess/partial files and return failed receipt on any execution/parity error.

    A pre-existing output directory raises rather than overwriting an attempt.
    The caller decides the final evaluator exit status from this receipt.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "started", "passed": False, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "python_executable": sys.executable, "context": context, "returncode": None,
              "timed_out": False, "timeout_s": timeout_s, "expected_binary_sha256": ACCEPTED_ORACLE_SHA256,
              "expected_binary_bytes": ACCEPTED_ORACLE_BYTES,
              "official_source_location": "external private GPL oracle; no vendor source included in this adapter"}
    stdout = stderr = b""
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("oracle timeout must be finite and positive")
        record["inputs"] = {"truth": identity(truth), "prediction": identity(prediction)}
        record["binary"] = identity(binary)
        if record["binary"]["sha256"] != ACCEPTED_ORACLE_SHA256 or record["binary"]["bytes"] != ACCEPTED_ORACLE_BYTES:
            raise ValueError("external oracle differs from the accepted binary")
        dump(output / "own-comparison.json", own)
        record["own_comparison"] = identity(output / "own-comparison.json")
        record["command"] = [str(Path(binary).resolve()), str(Path(truth).resolve()), str(Path(prediction).resolve()), str(output.resolve())]
        process = subprocess.run(record["command"], capture_output=True, timeout=timeout_s, check=False)
        stdout, stderr = process.stdout, process.stderr
        record["returncode"] = process.returncode
        record["inputs_after"] = {"truth": identity(truth), "prediction": identity(prediction)}
        record["binary_after"] = identity(binary)
        record["inputs_unchanged_after_invocation"] = record["inputs_after"] == record["inputs"] and record["binary_after"] == record["binary"]
        if process.returncode != 0:
            raise RuntimeError("official oracle exited nonzero")
        if not record["inputs_unchanged_after_invocation"]:
            raise ValueError("oracle input/binary identity changed during invocation")
        record["comparison"] = compare_outputs(output, own)
        record.update(status="passed", passed=True)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        record.update(status="failed", timed_out=True, failure={"kind": "timeout", "type": type(error).__name__, "message": str(error)})
    except Exception as error:
        record.update(status="failed", failure={"kind": "execution_or_parity", "type": type(error).__name__, "message": str(error)})
    finally:
        (output / "stdout.bin").write_bytes(stdout)
        (output / "stderr.bin").write_bytes(stderr)
        record["artifacts"] = {name: identity(output / name) for name in (*OUTPUTS, "stdout.bin", "stderr.bin", "own-comparison.json") if (output / name).is_file()}
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["wall_s"] = time.monotonic() - started
        dump(output / "receipt.json", record)
    return dict(record, receipt=identity(output / "receipt.json"))
