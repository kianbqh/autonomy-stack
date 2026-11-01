"""Plot one 120 s trajectory: ground truth vs GPS fixes vs dead reckoning vs EKF."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ekf import EKF, EKFConfig, wrap_angle
from .sim import RobotSim, SensorConfig, random_drive_controls


def main(seed=3, out="results/trajectory.png"):
    cfg = SensorConfig()
    rng = np.random.default_rng(seed + 10_000)
    sim = RobotSim(cfg, seed=seed, x0=[0, 0, 0.3, 1.0])
    n = int(120 / cfg.dt)
    omega, vt = random_drive_controls(rng, n, cfg.dt)
    ekf = EKF(EKFConfig(), x0=[0, 0, 0.3, 1.0, 0], P0=np.diag([0.01] * 4 + [0.05 ** 2]))
    dr = np.array([0.0, 0.0, 0.3, 1.0])
    T, E, D, G = [], [], [], []
    for k in range(n):
        m = sim.step(np.clip(1.5 * (vt[k] - sim.state[3]), -1, 1), omega[k])
        g, a = m["imu"]
        ekf.predict(g, a, cfg.dt)
        dr[2] = wrap_angle(dr[2] + g * cfg.dt)
        dr[:2] += dr[3] * np.array([np.cos(dr[2]), np.sin(dr[2])]) * cfg.dt
        if "odom" in m:
            ekf.update_odom(m["odom"]); dr[3] = m["odom"]
        if "gps" in m:
            ekf.update_gps(m["gps"]); G.append(m["gps"])
        T.append(sim.state[:2].copy()); E.append(ekf.x[:2].copy()); D.append(dr[:2].copy())
    T, E, D, G = map(np.array, (T, E, D, G))
    plt.figure(figsize=(7, 6))
    plt.plot(*T.T, "k-", lw=2, label="ground truth")
    plt.scatter(*G.T, s=12, c="tab:orange", label="raw GPS (1 Hz)")
    plt.plot(*D.T, "--", c="tab:gray", label="IMU+odom dead reckoning")
    plt.plot(*E.T, c="tab:blue", lw=1.5, label="EKF fused")
    plt.xlim(T[:, 0].min() - 10, T[:, 0].max() + 10); plt.ylim(T[:, 1].min() - 10, T[:, 1].max() + 10)
    plt.axis("equal"); plt.legend(); plt.title("120 s drive: EKF vs raw sensors"); plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
