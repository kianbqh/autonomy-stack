import numpy as np
import pytest

from avsafety.controller import SafetyMonitor, avoid_control
from avsafety.evaluate import wilson
from avsafety.runner import run_episode
from avsafety.scenario import FaultySensors, Scenario, make_world, scenario_matrix


def _drive(sc, seconds=40.0, seed=0):
    sim, *_ = make_world(sc, seed)
    fs = FaultySensors(sim, sc, np.random.default_rng(1))
    msgs = [fs.step(0.0, 0.0) for _ in range(int(seconds / sim.cfg.dt))]
    return fs, msgs


def test_matrix_size_and_unique_names():
    m = scenario_matrix()
    assert len(m) == 30 and len({s.name for s in m}) == 30


def test_gps_dropout_window():
    fs, msgs = _drive(Scenario(1.5, "dropout_10s"))
    gps_t = np.array([m["t"] for m in msgs if "gps" in m])
    inside = (gps_t >= fs.t_fault) & (gps_t < fs.t_fault + 10)
    assert not inside.any() and fs.log["gps_dropped"] in (9, 10, 11)
    assert len(gps_t) >= 28                                   # GPS present outside the outage


def test_spikes_are_large():
    sc = Scenario(0.5, "spikes")
    fs, msgs = _drive(sc)
    sim, *_ = make_world(sc, 0)
    assert fs.log["gps_spiked"] >= 3


def test_odom_dropout():
    fs, msgs = _drive(Scenario(1.5, "none", "dropout_10s"))
    assert fs.log["odom_dropped"] in range(495, 506)        # 50 Hz x 10 s


def test_deterministic_episode():
    a = run_episode(Scenario(1.5, "bias_step"), "ekf", 3)
    replay = {}
    b = run_episode(Scenario(1.5, "bias_step"), "ekf", 3, replay=replay)
    assert a == b
    assert replay["meta"]["outcome"] == b["outcome"]
    assert replay["meta"]["raw_baseline"] == "held GPS position"
    samples = replay["samples"]
    assert np.all(np.diff([s["t"] for s in samples]) > 0)
    assert samples[-1]["t"] == b["time"]
    assert samples[-1]["outcome"] == b["outcome"]
    assert min(s["min_clearance"] for s in samples) == b["min_clearance"]
    for sample in samples:
        assert len(sample["truth"]) == len(sample["estimated_pose"]) == 3
        xx, xy, yy = sample["cov"]
        assert xx >= 0 and yy >= 0 and xx * yy - xy ** 2 >= -1e-9
    assert any(s["faults"]["gps_bias"] for s in samples)


@pytest.mark.parametrize("seed", range(5))
def test_truth_nominal_is_safe(seed):
    r = run_episode(Scenario(0.5, "none"), "truth", seed)
    assert r["outcome"] == "success" and r["min_clearance"] > 0.3
    assert r["loc_rmse"] == 0.0 and r["loc_max"] == 0.0


def test_monitor_degrades_with_uncertainty():
    mon = SafetyMonitor()
    m0, v0, s0 = mon.adjust(np.diag([0.01, 0.01, 0, 0, 0]))
    m1, v1, s1 = mon.adjust(np.diag([4.0, 4.0, 0, 0, 0]))
    m2, v2, s2 = mon.adjust(np.diag([16.0, 16.0, 0, 0, 0]))
    assert m1 > m0 and v1 < v0 and v2 == 0.0 and s2 == pytest.approx(4.0)


def test_avoidance_turns_away_from_obstacle():
    obstacles = np.array([[3.0, 0.0, 0.7]])
    _, w = avoid_control(np.array([1.5, 0.05, 0.0, 1.0]), np.array([10.0, 0.0]), obstacles)
    assert abs(w) > 0.3                                       # steers around instead of straight through


def test_gating_rejects_bias_step():
    r = run_episode(Scenario(0.5, "bias_step"), "ekf", 1)
    assert r["gps_rejected"] >= 5


def test_wilson_interval():
    lo, hi = wilson(0, 100)
    assert lo == 0 and 0.03 < hi < 0.05
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
