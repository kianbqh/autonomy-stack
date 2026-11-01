"""Go-to-goal + obstacle-avoidance controller on an ESTIMATED pose, with an optional uncertainty-aware
runtime safety monitor (uses the EKF position covariance)."""
from __future__ import annotations

import numpy as np

from fusion_core.ekf import wrap_angle

V_MAX, W_MAX, ROBOT_R = 1.2, 1.5, 0.3


def avoid_control(pose, goal, obstacles, margin=0.6, v_cap=V_MAX):
    """Vector field: attraction to goal + repulsion (with tangential swirl) from obstacles within reach."""
    x, y, th, _ = pose
    p = np.array([x, y])
    to_goal = goal - p
    d_goal = np.linalg.norm(to_goal)
    vec = to_goal / (d_goal + 1e-9)
    slow = 1.0
    for ox, oy, r in obstacles:
        diff = p - np.array([ox, oy])
        dist = np.linalg.norm(diff)
        clear = dist - r - ROBOT_R
        reach = margin + 1.5
        if clear < reach:
            away = diff / (dist + 1e-9)
            tang = np.array([-away[1], away[0]])
            if tang @ vec < 0:
                tang = -tang
            k = (reach - clear) / reach
            vec = vec + 3.0 * k ** 2 * away + 1.5 * k * tang
            slow = min(slow, max(0.25, (clear - margin * 0.5) / reach))
    b = wrap_angle(np.arctan2(vec[1], vec[0]) - th)
    w = np.clip(2.0 * b, -W_MAX, W_MAX)
    v = np.clip(0.6 * d_goal, 0.0, v_cap) * max(0.0, np.cos(b)) * slow
    return v, w


class SafetyMonitor:
    """Degraded mode when localization is uncertain: inflate the obstacle margin by k*sigma and cap speed;
    stop if sigma exceeds the hard limit."""

    def __init__(self, k_sigma=3.0, sigma_soft=0.5, sigma_hard=3.0):
        self.k, self.soft, self.hard = k_sigma, sigma_soft, sigma_hard
        self.degraded_steps = self.stop_steps = 0

    def adjust(self, P):
        sigma = float(np.sqrt(np.max(np.linalg.eigvalsh(P[:2, :2]))))
        margin = 0.6 + self.k * sigma
        v_cap = V_MAX
        if sigma > self.soft:
            self.degraded_steps += 1
            v_cap = V_MAX * max(0.3, self.soft / sigma)
        if sigma > self.hard:
            self.stop_steps += 1
            v_cap = 0.0
        return margin, v_cap, sigma
