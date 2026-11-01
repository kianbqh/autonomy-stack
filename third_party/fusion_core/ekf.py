"""Extended Kalman Filter for IMU + wheel-odometry + GPS fusion.

State s = [x, y, theta, v, b_g]   (b_g = gyro bias, estimated online)
Predict  (100 Hz): IMU gyro/accel as control input, Jacobian-linearised unicycle model.
Update   (50 Hz):  wheel speed  z = v
Update   (1 Hz):   GPS          z = [x, y], with chi-square (Mahalanobis) outlier gating.
"""
from dataclasses import dataclass

import numpy as np

CHI2_2DOF_99 = 9.21  # 99 % gate for 2-D measurement


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EKFConfig:
    gyro_noise: float = 0.02
    accel_noise: float = 0.15
    bias_rw: float = 0.0005
    odom_noise: float = 0.08        # inflated vs. truth to absorb unmodelled scale error
    gps_noise: float = 1.5
    gate: float = CHI2_2DOF_99
    use_gating: bool = True


class EKF:
    def __init__(self, cfg: EKFConfig | None = None, x0=None, P0=None):
        self.cfg = cfg or EKFConfig()
        self.x = np.zeros(5) if x0 is None else np.asarray(x0, float).copy()
        self.P = np.diag([4.0, 4.0, 0.05, 0.25, 0.05 ** 2]) if P0 is None else np.asarray(P0, float).copy()
        self.n_gps = 0
        self.n_rejected = 0

    def predict(self, gyro: float, accel: float, dt: float):
        x, y, th, v, bg = self.x
        c, s = np.cos(th), np.sin(th)
        self.x = np.array([x + v * c * dt, y + v * s * dt, wrap_angle(th + (gyro - bg) * dt),
                           max(0.0, v + accel * dt), bg])
        F = np.eye(5)
        F[0, 2], F[0, 3] = -v * s * dt, c * dt
        F[1, 2], F[1, 3] = v * c * dt, s * dt
        F[2, 4] = -dt
        G = np.zeros((5, 3))  # noise inputs: gyro, accel, bias random walk
        G[2, 0], G[3, 1], G[4, 2] = dt, dt, np.sqrt(dt)
        Qn = np.diag([self.cfg.gyro_noise ** 2, self.cfg.accel_noise ** 2, self.cfg.bias_rw ** 2])
        self.P = F @ self.P @ F.T + G @ Qn @ G.T

    def _update(self, z, h, H, R, angle_idx=None):
        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form: stays symmetric PSD

    def update_odom(self, speed: float):
        H = np.array([[0, 0, 0, 1.0, 0]])
        self._update(np.array([speed]), self.x[3:4], H, np.array([[self.cfg.odom_noise ** 2]]))

    def update_gps(self, z) -> bool:
        """Returns True if accepted, False if rejected by the Mahalanobis gate."""
        self.n_gps += 1
        H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * self.cfg.gps_noise ** 2
        innov = np.asarray(z) - self.x[:2]
        S = H @ self.P @ H.T + R
        d2 = float(innov @ np.linalg.solve(S, innov))
        if self.cfg.use_gating and d2 > self.cfg.gate:
            self.n_rejected += 1
            return False
        self._update(np.asarray(z, float), self.x[:2], H, R)
        return True
