"""Synthetic CLI contract checks; no KITTI data or measured SLAM result."""
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from eval_traj import evo_crosscheck, metrics, rpe

EVO_VERSION = "1.31.0"
ATOL = RTOL = 1e-6
CFG = {"segment_lengths_m": [100, 200, 300, 400, 500, 600, 700, 800],
       "segment_start_stride_frames": 10, "rpe_distance_m": 100}


def synthetic_trajectories():
    """Non-collinear 3D positions, changing yaw, translation/rotation drift."""
    count = 36
    truth = np.repeat(np.eye(4)[None], count, axis=0)
    pred = truth.copy()
    for i in range(count):
        for poses, yaw in ((truth, .02 * i),
                           (pred, .024 * i + .02 * np.sin(i / 5))):
            c, s = np.cos(yaw), np.sin(yaw)
            poses[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        truth[i, :3, 3] = [10 * i + 4, 8 * np.sin(i / 4) - 7,
                           3 * np.cos(i / 3) + 1]
        pred[i, :3, 3] = truth[i, :3, 3] + [.25 * i + 14, .1 * i - 9,
                                                          -.03 * i + 5]
    return pred, truth


def test_synthetic_evo_cli_exact_pairs_and_boundary_artifacts(tmp_path, monkeypatch):
    assert importlib.metadata.version("evo") == EVO_VERSION
    artifact = tmp_path / "synthetic-evo-contract"
    artifact.mkdir()
    out = artifact / "cli"
    out.mkdir()
    print(f"SYNTHETIC_EVO_CONTRACT_ARTIFACT_DIR={artifact}", flush=True)
    (artifact / "fixture.json").write_text(json.dumps({
        "synthetic": True, "description": synthetic_trajectories.__doc__,
        "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
        "evo_version": EVO_VERSION, "path_cli_lookup_disabled": True,
    }, indent=2) + "\n", encoding="utf-8")
    pred, truth = synthetic_trajectories()
    ref, est = artifact / "synthetic-reference.txt", artifact / "synthetic-estimate.txt"
    np.savetxt(ref, truth[:, :3, :].reshape(-1, 12), fmt="%.17g")
    np.savetxt(est, pred[:, :3, :].reshape(-1, 12), fmt="%.17g")
    own = metrics(pred, truth, CFG)
    (artifact / "own-metrics.json").write_text(json.dumps(own, indent=2) + "\n", encoding="utf-8")
    # An interpreter-bound CLI must still run when PATH cannot resolve evo_*.
    empty_path = artifact / "empty-path"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    result = evo_crosscheck(ref, est, out, EVO_VERSION, CFG, own)
    (artifact / "crosscheck.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert result["passed"]
    assert result["identity"]["python_executable"] == sys.executable
    assert result["identity"]["distribution_version"] == EVO_VERSION
    assert result["identity"]["distribution_root"] == str(importlib.metadata.distribution("evo").locate_file(""))
    pairs = [(row["i"], row["j"]) for row in own["rpe_100m"]["samples"]]
    assert len(pairs) > 1  # This fixture must exercise actual pair boundaries.
    assert result["distance_pair_count"] == len(pairs)
    ids = [index for pair in pairs for index in pair]
    adjacent = list(zip(ids, ids[1:]))
    assert adjacent[::2] == pairs
    selection = json.loads((out / "100m-pair-selection.json").read_text(encoding="utf-8"))
    assert selection["original_pairs"] == [list(pair) for pair in pairs]
    assert selection["selected_evo_error_indices"] == list(range(0, len(adjacent), 2))
    assert selection["boundary_error_indices_ignored"] == list(range(1, len(adjacent), 2))
    expanded_own = rpe(pred, truth, adjacent)

    def values(samples, relation):
        if relation == "trans_part":
            return np.array([row["translation_m"] for row in samples])
        return np.rad2deg([row["rotation_rad"] for row in samples])

    expected = {"ape": np.array(own["ate"]["per_frame_m"])}
    for relation in ("trans_part", "angle_deg"):
        expected["rpe_1frame_" + relation] = values(own["rpe_1frame"]["samples"], relation)
        expected["rpe_100m_exact_" + relation] = values(expanded_own["samples"], relation)
    assert {row["label"] for row in result["commands"]} == set(expected)
    for record in result["commands"]:
        label, command = record["label"], record["command"]
        assert record["exit_code"] == 0 and record["passed"]
        assert command[:2] == [sys.executable, "-c"]
        assert command[3] == "kitti"
        entrypoint = "ape" if label == "ape" else "rpe"
        assert f"from evo.entry_points import {entrypoint}" in command[2]
        assert repr(EVO_VERSION) in command[2]
        with zipfile.ZipFile(out / (label + ".zip")) as archive:
            errors = np.load(io.BytesIO(archive.read("error_array.npy")), allow_pickle=False)
            stats = json.loads(archive.read("stats.json"))
        assert errors.ndim == 1 and np.isfinite(errors).all()
        assert len(errors) == len(expected[label]) == record["raw_pair_count"] == record["expected_raw_pair_count"]
        # Compare all raw errors too, including the discarded boundary samples.
        np.testing.assert_allclose(errors, expected[label], atol=ATOL, rtol=RTOL)
        is_expanded = label.startswith("rpe_100m_exact_")
        assert record["raw_evo_stats_include_pair_boundaries"] is is_expanded
        chosen = errors[selection["selected_evo_error_indices"]] if is_expanded else errors
        assert len(chosen) == record["selected_pair_count"] == record["expected_pair_count"]
        if is_expanded:
            assert len(errors) == 2 * len(pairs) - 1 and len(chosen) == len(pairs)
            relation = "trans_part" if label.endswith("trans_part") else "angle_deg"
            np.testing.assert_allclose(chosen, values(own["rpe_100m"]["samples"], relation), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(stats["mean"], np.mean(errors), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(stats["rmse"], np.sqrt(np.mean(errors ** 2)), atol=ATOL, rtol=RTOL)
        assert (out / (label + ".stdout.txt")).is_file()
        assert (out / (label + ".stderr.txt")).is_file()


def test_evo_version_mismatch_rejected_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="evo version mismatch"):
        evo_crosscheck(tmp_path / "unused-ref", tmp_path / "unused-est",
                       tmp_path, "0.0.0", CFG, {})
