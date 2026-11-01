"""2-D unicycle robot with realistic, imperfect sensors.

State (truth): x, y [m], theta [rad], v [m/s].  Controls: longitudinal accel a, yaw rate omega.
Sensors
  * IMU   @100 Hz: gyro = omega + bias + noise (bias does a random walk), accel = a + noise
  * Wheel odometry @50 Hz: speed with 3 % scale error + noise
  * GPS   @1 Hz: x, y with sigma = 1.5 m and occasional multipath outliers (~8 m jumps)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorConfig:
    dt: float = 0.01               # simulation / IMU period [s]
    odom_every: int = 2            # 50 Hz
    gps_every: int = 100           # 1 Hz
    gyro_noise: float = 0.02       # rad/s
    gyro_bias0: float = 0.03       # rad/s initial bias std
    gyro_bias_rw: float = 0.0005   # rad/s/sqrt(s)
    accel_noise: float = 0.15      # m/s^2
    odom_noise: float = 0.05       # m/s
    odom_scale_err: float = 0.03   # fractional
    gps_noise: float = 1.5         # m
    gps_outlier_prob: float = 0.03
    gps_outlier_mag: float = 8.0   # m


class RobotSim:
    def __init__(self, cfg: SensorConfig | None = None, seed: int = 0, x0=None):
        self.cfg = cfg or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4) if x0 is None else np.asarray(x0, float).copy()
        self.bias = self.rng.normal(0.0, self.cfg.gyro_bias0)
        self.odom_scale = 1.0 + self.rng.uniform(-self.cfg.odom_scale_err, self.cfg.odom_scale_err)
        self.k = 0
        self.t = 0.0

    def step(self, a: float, omega: float) -> dict:
        """Advance one dt with controls; return dict of sensor readings available this tick."""
        c, dt = self.cfg, self.cfg.dt
        x, y, th, v = self.state
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = (th + omega * dt + np.pi) % (2 * np.pi) - np.pi
        v = max(0.0, v + a * dt)
        self.state[:] = (x, y, th, v)
        self.bias += self.rng.normal(0.0, c.gyro_bias_rw * np.sqrt(dt))
        self.k += 1
        self.t += dt
        out = {"t": self.t,
               "imu": (omega + self.bias + self.rng.normal(0, c.gyro_noise),
                       a + self.rng.normal(0, c.accel_noise))}
        if self.k % c.odom_every == 0:
            out["odom"] = v * self.odom_scale + self.rng.normal(0, c.odom_noise)
        if self.k % c.gps_every == 0:
            z = self.state[:2] + self.rng.normal(0, c.gps_noise, 2)
            if self.rng.random() < c.gps_outlier_prob:
                ang = self.rng.uniform(0, 2 * np.pi)
                z += c.gps_outlier_mag * np.array([np.cos(ang), np.sin(ang)])
            out["gps"] = z
        return out


def random_drive_controls(rng: np.random.Generator, n: int, dt: float, v_target=(0.5, 2.0)):
    """Smooth random excitation: sinusoidal yaw-rate mix and speed set-points changing every few s."""
    t = np.arange(n) * dt
    f = rng.uniform(0.02, 0.15, 3)
    ph = rng.uniform(0, 2 * np.pi, 3)
    amp = rng.uniform(0.1, 0.4, 3)
    omega = sum(amp[i] * np.sin(2 * np.pi * f[i] * t + ph[i]) for i in range(3))
    seg = int(5.0 / dt)
    vt = np.repeat(rng.uniform(*v_target, n // seg + 1), seg)[:n]
    return omega, vt
