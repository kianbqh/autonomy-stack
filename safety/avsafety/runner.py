"""Run one episode (scenario x estimator x seed) and return safety metrics."""
from __future__ import annotations

import numpy as np

from fusion_core.ekf import EKF, EKFConfig, wrap_angle

from .controller import ROBOT_R, SafetyMonitor, avoid_control
from .scenario import FaultySensors, Scenario, make_world

ESTIMATORS = ("truth", "raw", "ekf_nogate", "ekf", "ekf_monitor")
DT_CTRL_SUB = 5            # controller at 20 Hz, sim at 100 Hz
MAX_T = 90.0               # long enough to recover after a 30 s GPS outage


def run_episode(sc: Scenario, estimator: str, seed: int, *, replay: dict | None = None) -> dict:
    sim, goal, obstacles, th0, rng = make_world(sc, seed)
    sens = FaultySensors(sim, sc, np.random.default_rng(seed + 7_000_003))
    ekf = EKF(EKFConfig(gps_noise=sc.gps_sigma, use_gating=estimator != "ekf_nogate"),
              x0=[0, 0, th0, 0, 0], P0=np.diag([0.01, 0.01, 0.01, 0.01, 0.05 ** 2]))
    mon = SafetyMonitor() if estimator == "ekf_monitor" else None
    gps, dr_th, dr_v = np.zeros(2), th0, 0.0
    dt = sim.cfg.dt
    min_clear, loc_err, outcome, t = np.inf, [], "timeout", 0.0
    max_sigma, t_closest, nees = 0.0, 0.0, []
    if replay is not None:
        if not estimator.startswith("ekf"):
            raise ValueError("Covariance replays require an EKF estimator")
        replay.update({"meta": {"schema": "autonomy-replay/1", "source": "avsafety.run_episode + shared fusion_core",
                               "seed": seed, "estimator": estimator, "scenario": sc.name,
                               "gps_sigma": sc.gps_sigma, "gps_fault": sc.gps_fault, "odom_fault": sc.odom_fault,
                               "fault_onset_s": sens.t_fault, "sample_hz": 20, "frame": "ENU metres, z up",
                               "data_kind": "synthetic 2-D simulation", "raw_baseline": "held GPS position"},
                       "world": {"goal": goal.tolist(), "obstacles": obstacles.tolist(), "robot_radius": ROBOT_R},
                       "samples": [], "gps": []})
    while t < MAX_T:
        counts_before = sens.log.copy() if replay is not None else None
        if estimator == "truth":
            pose = sim.state.copy()
        elif estimator == "raw":
            pose = np.array([*gps, dr_th, dr_v])
        else:
            pose = ekf.x[:4].copy()
        margin, v_cap = 0.6, 1.2
        if mon is not None:
            margin, v_cap, sigma = mon.adjust(ekf.P)
            max_sigma = max(max_sigma, sigma)
        v_cmd, w = avoid_control(pose, goal, obstacles, margin, v_cap)
        for _ in range(DT_CTRL_SUB):
            a = np.clip(2.0 * (v_cmd - sim.state[3]), -2.0, 2.0)
            m = sens.step(a, w)
            g, acc = m["imu"]
            ekf.predict(g, acc, dt)
            dr_th = wrap_angle(dr_th + g * dt)
            if "odom" in m:
                ekf.update_odom(m["odom"]); dr_v = m["odom"]
            if "gps" in m:
                accepted = ekf.update_gps(m["gps"]); gps = m["gps"]
                if replay is not None:
                    replay["gps"].append({"t": float(sim.t), "xy": gps.tolist(), "accepted": bool(accepted)})
            p = sim.state[:2]
            clear = float(np.min(np.hypot(obstacles[:, 0] - p[0], obstacles[:, 1] - p[1]) - obstacles[:, 2]) - ROBOT_R)
            if clear < min_clear:
                min_clear, t_closest = clear, sim.t
            if clear < 0:
                outcome = "collision"; break
        t = sim.t
        # Compare the updated estimate and truth at the same simulator time.
        if estimator == "truth":
            estimated_position = sim.state[:2]
        elif estimator == "raw":
            estimated_position = gps
        else:
            estimated_position = ekf.x[:2]
        loc_err.append(float(np.hypot(*(estimated_position - sim.state[:2]))))
        if replay is not None:
            flags = sens.flags(t)
            flags.update({f"{name}_event": sens.log[name] > counts_before[name] for name in sens.log})
            replay["samples"].append({"t": float(t), "truth": sim.state[:3].tolist(),
                                      "estimated_pose": ekf.x[:3].tolist(),
                                      "cov": [float(ekf.P[0, 0]), float(ekf.P[0, 1]), float(ekf.P[1, 1])],
                                      "dr": gps.tolist(), "speed": float(sim.state[3]),
                                      "clearance": clear, "min_clearance": min_clear,
                                      "faults": flags, "outcome": "running"})
        if estimator.startswith("ekf"):                       # filter consistency: 2-DoF position NEES
            e = ekf.x[:2] - sim.state[:2]
            nees.append(float(e @ np.linalg.solve(ekf.P[:2, :2], e)))
        if outcome == "collision":
            break
        if np.hypot(*(goal - sim.state[:2])) < 0.75 and sim.state[3] < 0.3:
            outcome = "success"; break
    le = np.array(loc_err)
    summary = {"scenario": sc.name, "gps_sigma": sc.gps_sigma, "gps_fault": sc.gps_fault, "odom_fault": sc.odom_fault,
            "estimator": estimator, "seed": seed, "outcome": outcome, "min_clearance": min_clear, "time": t,
            "loc_rmse": float(np.sqrt(np.mean(le ** 2))), "loc_max": float(le.max()),
            "gps_rejected": ekf.n_rejected, "max_sigma": max_sigma,
            "nees_mean": float(np.mean(nees)) if nees else np.nan,
            "nees_gt_chi2_95": float(np.mean(np.array(nees) > 5.99)) if nees else np.nan,
            "fault_at_closest": sens.active(t_closest), "t_closest": t_closest,
            "degraded_frac": (mon.degraded_steps / max(1, len(le))) if mon else 0.0, **sens.log}
    if replay is not None:
        replay["meta"].update({"outcome": outcome, "duration_s": t, "metrics": summary})
        replay["samples"][-1]["outcome"] = outcome
    return summary
