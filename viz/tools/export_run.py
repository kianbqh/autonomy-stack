"""Export one simulated drive from ros2-sensor-fusion-sim (ground truth, EKF estimate + 2x2 position covariance,
raw GPS fixes with the chi-square gate decision, IMU+odom dead reckoning) to JSON for the viewer.
Usage: python tools/export_run.py --fusion-repo ../ros2-sensor-fusion-sim --seed 3 --duration 120"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fusion-repo", default=os.environ.get("FUSION_REPO", "../ros2-sensor-fusion-sim"))
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--duration", type=float, default=120.0)
ap.add_argument("--rate", type=float, default=10.0, help="output sample rate (Hz)")
ap.add_argument("--out", default="public/data/run.json")
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.fusion_repo))
from fusion_core.ekf import EKF, EKFConfig, wrap_angle  # noqa: E402
from fusion_core.sim import RobotSim, SensorConfig, random_drive_controls  # noqa: E402

cfg = SensorConfig(); rng = np.random.default_rng(a.seed + 10_000)
sim = RobotSim(cfg, seed=a.seed, x0=[0, 0, 0.3, 1.0]); n = int(a.duration / cfg.dt)
omega, vt = random_drive_controls(rng, n, cfg.dt)
ekf = EKF(EKFConfig(), x0=[0, 0, 0.3, 1.0, 0], P0=np.diag([0.01] * 4 + [0.05 ** 2]))
dr = np.array([0.0, 0.0, 0.3, 1.0]); every = max(1, int(round(1 / (a.rate * cfg.dt))))
samples, gps = [], []
r4 = lambda v: [round(float(x), 4) for x in v]
for k in range(n):
    m = sim.step(np.clip(1.5 * (vt[k] - sim.state[3]), -1, 1), omega[k])
    g, acc = m["imu"]; ekf.predict(g, acc, cfg.dt)
    dr[2] = wrap_angle(dr[2] + g * cfg.dt); dr[:2] += dr[3] * np.array([np.cos(dr[2]), np.sin(dr[2])]) * cfg.dt
    if "odom" in m:
        ekf.update_odom(m["odom"]); dr[3] = m["odom"]
    if "gps" in m:
        accepted = ekf.update_gps(m["gps"])
        gps.append({"t": round((k + 1) * cfg.dt, 3), "xy": r4(m["gps"]), "accepted": bool(accepted)})
    if k % every == 0:
        P = ekf.P[:2, :2]
        samples.append({"t": round((k + 1) * cfg.dt, 3), "truth": r4([*sim.state[:3]]), "ekf": r4([*ekf.x[:3]]),
                        "cov": r4([P[0, 0], P[0, 1], P[1, 1]]), "dr": r4(dr[:2]), "speed": round(float(sim.state[3]), 3)})
out = {"meta": {"source": "ros2-sensor-fusion-sim fusion_core (EKF: IMU + wheel odometry + GPS)", "seed": a.seed,
                "duration_s": a.duration, "rate_hz": a.rate, "frame": "ENU metres, z up"},
       "samples": samples, "gps": gps}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), separators=(",", ":"))
print(f"wrote {a.out}: {len(samples)} samples, {len(gps)} GPS fixes ({sum(not g['accepted'] for g in gps)} rejected)")
