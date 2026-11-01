"""Gymnasium environment: drive a noisy-sensor robot to a goal using an estimated state.

Observation modes
  truth : privileged ground-truth pose (used only by the expert / upper bound)
  ekf   : EKF-fused pose from IMU + odometry + GPS
  raw   : no fusion - last GPS fix for position, gyro-integrated heading, wheel speed
Observation: [dist/10, cos(bearing), sin(bearing), v/2], bearing = goal angle in robot frame.
Success: true distance < 0.75 m with speed < 0.3 m/s (parked) within 40 s.
Action: [-1, 1]^2 -> speed set-point 0..1.5 m/s, yaw rate -1.5..1.5 rad/s.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ekf import EKF, EKFConfig, wrap_angle
from .sim import RobotSim, SensorConfig

V_MAX, W_MAX = 1.5, 1.5


def make_obs(pose, goal):
    x, y, th, v = pose
    dx, dy = goal[0] - x, goal[1] - y
    d = np.hypot(dx, dy)
    b = wrap_angle(np.arctan2(dy, dx) - th)
    return np.array([d / 10.0, np.cos(b), np.sin(b), v / 2.0], dtype=np.float32)


def expert_action(obs):
    """Proportional go-to-goal controller (the demonstrator)."""
    d = obs[0] * 10.0
    b = np.arctan2(obs[2], obs[1])
    w = np.clip(2.0 * b, -W_MAX, W_MAX)
    v = np.clip(0.6 * d, 0.0, V_MAX) * max(0.0, np.cos(b))
    return np.array([2 * v / V_MAX - 1, w / W_MAX], dtype=np.float32)


class GoalReach2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_mode: str = "ekf", success_radius: float = 0.75, max_steps: int = 800,
                 substeps: int = 5):
        assert obs_mode in ("truth", "ekf", "raw")
        self.obs_mode, self.r_succ, self.max_steps, self.substeps = obs_mode, success_radius, max_steps, substeps
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    # -- helpers --
    def truth_obs(self):
        return make_obs(self.sim.state, self.goal)

    def _est_pose(self):
        if self.obs_mode == "truth":
            return self.sim.state
        if self.obs_mode == "ekf":
            return self.ekf.x[:4]
        return np.array([*self.gps, self.dr_th, self.dr_v])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(1 << 31))
        th0 = self.np_random.uniform(-np.pi, np.pi)
        self.sim = RobotSim(SensorConfig(), seed=s, x0=[0, 0, th0, 0])
        ang, dist = self.np_random.uniform(-np.pi, np.pi), self.np_random.uniform(8, 20)
        self.goal = np.array([dist * np.cos(ang), dist * np.sin(ang)])
        self.ekf = EKF(EKFConfig(), x0=[0, 0, th0, 0, 0], P0=np.diag([0.01, 0.01, 0.01, 0.01, 0.05 ** 2]))
        self.gps, self.dr_th, self.dr_v = np.zeros(2), th0, 0.0
        self.steps = 0
        self.prev_d = np.hypot(*self.goal)
        return make_obs(self._est_pose(), self.goal), {}

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1, 1)
        v_cmd, w = (action[0] + 1) / 2 * V_MAX, action[1] * W_MAX
        dt = self.sim.cfg.dt
        for _ in range(self.substeps):
            a = np.clip(2.0 * (v_cmd - self.sim.state[3]), -2.0, 2.0)
            m = self.sim.step(a, w)
            g, acc = m["imu"]
            self.ekf.predict(g, acc, dt)
            self.dr_th = wrap_angle(self.dr_th + g * dt)
            if "odom" in m:
                self.ekf.update_odom(m["odom"])
                self.dr_v = m["odom"]
            if "gps" in m:
                self.ekf.update_gps(m["gps"])
                self.gps = m["gps"]
        self.steps += 1
        d = float(np.hypot(*(self.goal - self.sim.state[:2])))
        success = d < self.r_succ and self.sim.state[3] < 0.3  # must arrive AND stop (park)
        reward = (self.prev_d - d) - 0.01 + (10.0 if success else 0.0)
        self.prev_d = d
        truncated = self.steps >= self.max_steps
        return make_obs(self._est_pose(), self.goal), reward, success, truncated, {"success": success,
                                                                                  "dist": d}


gym.register(id="GoalReach2D-v0", entry_point=GoalReach2DEnv)
