"""Monte-Carlo evaluation: fused EKF vs raw GPS vs IMU+odometry dead reckoning."""
import argparse
import json
import time

import numpy as np

from .ekf import EKF, EKFConfig, wrap_angle
from .sim import RobotSim, SensorConfig, random_drive_controls


def run_episode(seed: int, duration: float = 120.0, gating: bool = True):
    cfg = SensorConfig()
    rng = np.random.default_rng(seed + 10_000)
    sim = RobotSim(cfg, seed=seed, x0=[0, 0, rng.uniform(-np.pi, np.pi), 1.0])
    n = int(duration / cfg.dt)
    omega, vt = random_drive_controls(rng, n, cfg.dt)
    ekf = EKF(EKFConfig(use_gating=gating), x0=[0, 0, sim.state[2], 1.0, 0.0],
              P0=np.diag([0.01, 0.01, 0.01, 0.01, 0.05 ** 2]))
    dr = np.array([0.0, 0.0, sim.state[2], 1.0])  # dead reckoning: gyro heading + wheel speed
    last_gps = None
    err_f, err_g, err_d, err_th, err_fix, err_f_fix = [], [], [], [], [], []
    for k in range(n):
        a = np.clip(1.5 * (vt[k] - sim.state[3]), -1.0, 1.0)
        meas = sim.step(a, omega[k])
        g, acc = meas["imu"]
        ekf.predict(g, acc, cfg.dt)
        dr[2] = wrap_angle(dr[2] + g * cfg.dt)
        dr[0] += dr[3] * np.cos(dr[2]) * cfg.dt
        dr[1] += dr[3] * np.sin(dr[2]) * cfg.dt
        if "odom" in meas:
            ekf.update_odom(meas["odom"])
            dr[3] = meas["odom"]
        if "gps" in meas:
            ekf.update_gps(meas["gps"])
            last_gps = meas["gps"]
            err_fix.append(np.sum((last_gps - sim.state[:2]) ** 2))   # GPS scored only at fix instants
            err_f_fix.append(np.sum((ekf.x[:2] - sim.state[:2]) ** 2))
        if k % 10 == 0 and last_gps is not None:  # score at 10 Hz once a fix exists
            p = sim.state[:2]
            err_f.append(np.sum((ekf.x[:2] - p) ** 2))
            err_g.append(np.sum((last_gps - p) ** 2))  # raw GPS held at last fix (what a consumer sees)
            err_d.append(np.sum((dr[:2] - p) ** 2))
            err_th.append(wrap_angle(ekf.x[2] - sim.state[2]) ** 2)
    rm = lambda e: float(np.sqrt(np.mean(e)))
    return {"fused": rm(err_f), "gps": rm(err_g), "dead_reckoning": rm(err_d),
            "heading_deg": float(np.degrees(rm(err_th))),
            "gps_at_fix": rm(err_fix), "fused_at_fix": rm(err_f_fix),
            "gps_rejected": ekf.n_rejected, "gps_total": ekf.n_gps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default="results/fusion_metrics.json")
    args = ap.parse_args()
    t0 = time.time()
    res = {}
    for gating in (True, False):
        eps = [run_episode(s, args.duration, gating) for s in range(args.episodes)]
        key = "ekf_gated" if gating else "ekf_no_gating"
        res[key] = {m: float(np.mean([e[m] for e in eps])) for m in ("fused", "gps", "dead_reckoning", "heading_deg", "gps_at_fix", "fused_at_fix")}
        res[key]["fused_std"] = float(np.std([e["fused"] for e in eps]))
        res[key]["gps_rejected_frac"] = float(sum(e["gps_rejected"] for e in eps) / sum(e["gps_total"] for e in eps))
    g = res["ekf_gated"]
    res["summary"] = {
        "episodes": args.episodes, "duration_s": args.duration,
        "position_rmse_m": {"raw_gps_at_fix_instants": g["gps_at_fix"], "ekf_at_fix_instants": g["fused_at_fix"],
                            "raw_gps_held_10hz": g["gps"], "dead_reckoning": g["dead_reckoning"],
                            "ekf_no_gating": res["ekf_no_gating"]["fused"], "ekf_gated": g["fused"]},
        "rmse_reduction_vs_gps_pct": 100 * (1 - g["fused"] / g["gps"]),
        "rmse_reduction_vs_gps_at_fix_pct": 100 * (1 - g["fused_at_fix"] / g["gps_at_fix"]),
        "rmse_reduction_vs_dead_reckoning_pct": 100 * (1 - g["fused"] / g["dead_reckoning"]),
        "heading_rmse_deg": g["heading_deg"],
        "wall_time_s": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
