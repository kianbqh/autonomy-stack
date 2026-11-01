"""Scenario definition + fault injection on top of fusion_core.RobotSim.

A scenario = sensor noise level x GPS fault x odometry fault, on a goal-reaching drive with static obstacles
placed across the straight-line path. Faults are injected between the simulator and the estimator."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from fusion_core.sim import RobotSim, SensorConfig

GPS_FAULTS = ("none", "dropout_10s", "dropout_30s", "spikes", "bias_step")
ODOM_FAULTS = ("ok", "dropout_10s")
NOISE_LEVELS = (0.5, 1.5, 3.0)          # GPS sigma [m]


@dataclass(frozen=True)
class Scenario:
    gps_sigma: float = 1.5
    gps_fault: str = "none"
    odom_fault: str = "ok"

    @property
    def name(self):
        return f"gps{self.gps_sigma:g}_{self.gps_fault}_odom-{self.odom_fault}"


def scenario_matrix(levels=NOISE_LEVELS, gps=GPS_FAULTS, odom=ODOM_FAULTS):
    return [Scenario(s, g, o) for s, g, o in itertools.product(levels, gps, odom)]


class FaultySensors:
    """Wraps RobotSim.step(): drops / corrupts measurements according to the scenario."""

    def __init__(self, sim: RobotSim, sc: Scenario, rng: np.random.Generator):
        self.sim, self.sc, self.rng = sim, sc, rng
        self.t_fault = rng.uniform(3.0, 10.0)                  # fault onset [s]
        self.bias_vec = 4.0 * np.array([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a)])
        self.log = {"gps_dropped": 0, "gps_spiked": 0, "gps_biased": 0, "odom_dropped": 0}

    def _in(self, t, dur):
        return self.t_fault <= t < self.t_fault + dur

    def active(self, t) -> bool:
        """Was any injected fault active at time t?"""
        g = {"dropout_10s": 10, "dropout_30s": 30, "bias_step": 15}.get(self.sc.gps_fault)
        return bool((g and self._in(t, g)) or self.sc.gps_fault == "spikes"
                    or (self.sc.odom_fault == "dropout_10s" and self._in(t + 2.0, 10)))

    def flags(self, t) -> dict[str, bool]:
        duration = {"dropout_10s": 10, "dropout_30s": 30}.get(self.sc.gps_fault)
        return {"gps_dropout": bool(duration and self._in(t, duration)),
                "gps_bias": bool(self.sc.gps_fault == "bias_step" and self._in(t, 15)),
                "gps_spike_regime": self.sc.gps_fault == "spikes",
                "odom_dropout": bool(self.sc.odom_fault == "dropout_10s" and self._in(t + 2.0, 10))}

    def step(self, a, w):
        m = self.sim.step(a, w)
        t, f = m["t"], self.sc.gps_fault
        if "gps" in m:
            if (f == "dropout_10s" and self._in(t, 10)) or (f == "dropout_30s" and self._in(t, 30)):
                del m["gps"]; self.log["gps_dropped"] += 1
            elif f == "spikes" and self.rng.random() < 0.2:
                ang = self.rng.uniform(0, 2 * np.pi)
                m["gps"] = m["gps"] + 15.0 * np.array([np.cos(ang), np.sin(ang)]); self.log["gps_spiked"] += 1
            elif f == "bias_step" and self._in(t, 15):        # multipath / spoofing-like persistent offset
                m["gps"] = m["gps"] + self.bias_vec; self.log["gps_biased"] += 1
        if "odom" in m and self.sc.odom_fault == "dropout_10s" and self._in(t + 2.0, 10):
            del m["odom"]; self.log["odom_dropped"] += 1
        return m


def make_world(sc: Scenario, seed: int):
    """Start pose, goal and 2 obstacles that block the straight path (deterministic in seed)."""
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(-np.pi, np.pi)
    ang, dist = rng.uniform(-np.pi, np.pi), rng.uniform(12, 20)
    goal = dist * np.array([np.cos(ang), np.sin(ang)])
    u = goal / dist; n = np.array([-u[1], u[0]])
    obs = []
    for frac in (rng.uniform(0.3, 0.45), rng.uniform(0.6, 0.75)):
        r = rng.uniform(0.5, 1.0)
        obs.append((*(frac * goal + rng.uniform(-0.6, 0.6) * n), r))
    cfg = SensorConfig(gps_noise=sc.gps_sigma)
    sim = RobotSim(cfg, seed=int(rng.integers(1 << 31)), x0=[0, 0, th0, 0])
    return sim, goal, np.array(obs), th0, rng
