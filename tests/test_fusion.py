import numpy as np
import pytest

from fusion_core.ekf import EKF, EKFConfig, wrap_angle
from fusion_core.evaluate_fusion import run_episode


def test_wrap_angle():
    assert wrap_angle(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-0.5) == pytest.approx(-0.5)
    assert abs(wrap_angle(2 * np.pi + 0.1) - 0.1) < 1e-12


def test_fused_beats_raw_gps_and_dead_reckoning():
    r = run_episode(seed=123, duration=60.0)
    assert r["fused"] < 0.6 * r["gps"]
    assert r["fused"] < r["dead_reckoning"]


def test_covariance_stays_symmetric_psd():
    ekf = EKF()
    rng = np.random.default_rng(0)
    for k in range(2000):
        ekf.predict(rng.normal(0, 0.3), rng.normal(0, 0.5), 0.01)
        if k % 2 == 0:
            ekf.update_odom(1.0 + rng.normal(0, 0.05))
        if k % 100 == 0:
            ekf.update_gps(ekf.x[:2] + rng.normal(0, 1.5, 2))
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


def test_mahalanobis_gate_rejects_outlier():
    ekf = EKF(x0=[0, 0, 0, 1, 0], P0=np.diag([0.1, 0.1, 0.01, 0.01, 1e-4]))
    assert ekf.update_gps([0.5, -0.3]) is True
    before = ekf.x.copy()
    assert ekf.update_gps([25.0, 25.0]) is False      # 25 m jump: rejected
    np.testing.assert_array_equal(ekf.x, before)
    nog = EKF(EKFConfig(use_gating=False), x0=[0, 0, 0, 1, 0], P0=np.diag([0.1, 0.1, 0.01, 0.01, 1e-4]))
    assert nog.update_gps([25.0, 25.0]) is True


def test_gyro_bias_is_learned():
    """Constant gyro bias with zero true rotation should be estimated from GPS/odometry."""
    ekf = EKF(x0=[0, 0, 0, 1, 0])
    true_bias, rng = 0.05, np.random.default_rng(1)
    x = 0.0
    for k in range(6000):  # 60 s straight line at 1 m/s
        ekf.predict(true_bias + rng.normal(0, 0.02), 0.0, 0.01)
        x += 0.01
        if k % 2 == 0:
            ekf.update_odom(1.0)
        if k % 100 == 0:
            ekf.update_gps([x + rng.normal(0, 1.5), rng.normal(0, 1.5)])
    assert ekf.x[4] == pytest.approx(true_bias, abs=0.02)
