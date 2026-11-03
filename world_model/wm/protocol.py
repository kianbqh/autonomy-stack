"""Dependency-free physics, split and aggregation contracts."""
from __future__ import annotations
import hashlib
import math

STATE_DIM = 18
ACTION_DIM = 4
MODEL_SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))
HORIZONS = (1, 5, 10, 20, 30)
SOURCES = ("expert", "bc", "noisy_bc")
ANGLES = slice(6, 9)
STATE_LAYOUT = {"grip_position": [0, 3], "object_position": [3, 6],
                "object_euler": [6, 9], "object_world_velocity": [9, 12],
                "object_angular_velocity": [12, 15], "grip_world_velocity": [15, 18]}
FETCH_LAYOUT = {"grip_position": [0, 3], "object_position": [3, 6],
                "object_relative_position": [6, 9], "finger_position": [9, 11],
                "object_euler": [11, 14], "object_relative_velocity_times_dt": [14, 17],
                "object_angular_velocity_times_dt": [17, 20],
                "grip_velocity_times_dt": [20, 23], "finger_velocity_times_dt": [23, 25]}

def wrap_angle(x):
    return (float(x) + math.pi) % (2 * math.pi) - math.pi

def finite_vector(values, size):
    out = [float(x) for x in values]
    if len(out) != size or not all(math.isfinite(x) for x in out):
        raise ValueError(f"expected {size} finite values")
    return out

def observation_to_state(observation, control_dt):
    o = finite_vector(observation, 25)
    dt = float(control_dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("control_dt must be physical seconds >0")
    return (o[:6] + [wrap_angle(x) for x in o[11:14]]
            + [(o[14+i] + o[20+i]) / dt for i in range(3)]
            + [x / dt for x in o[17:20]] + [x / dt for x in o[20:23]])

def state_to_observation(state, template, control_dt):
    s, o = finite_vector(state, 18), finite_vector(template, 25)
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("invalid dt")
    o[:6] = s[:6]
    o[6:9] = [s[3+i] - s[i] for i in range(3)]
    o[11:14] = [wrap_angle(x) for x in s[6:9]]
    o[14:17] = [(s[9+i] - s[15+i]) * control_dt for i in range(3)]
    o[17:20] = [x * control_dt for x in s[12:15]]
    o[20:23] = [x * control_dt for x in s[15:18]]
    return {"observation": o, "achieved_goal": o[3:6]}

def state_delta(state, next_state):
    x, y = finite_vector(state, 18), finite_vector(next_state, 18)
    d = [b-a for a, b in zip(x, y)]
    d[6:9] = [wrap_angle(v) for v in d[6:9]]
    return d

def add_delta(state, delta):
    s = [a+b for a, b in zip(finite_vector(state, 18), finite_vector(delta, 18))]
    s[6:9] = [wrap_angle(v) for v in s[6:9]]
    return s

def constant_velocity(state, steps, dt):
    s = finite_vector(state, 18)
    if steps < 0 or dt <= 0:
        raise ValueError("invalid CV duration")
    for i in range(3):
        s[i] += s[15+i] * steps * dt
        s[3+i] += s[9+i] * steps * dt
        s[6+i] = wrap_angle(s[6+i] + s[12+i] * steps * dt)
    return s

def split_for_reset(seed):
    if type(seed) is not int or not 0 <= seed < 900_000:
        raise ValueError("held-out reset forbidden in dynamics data")
    bucket = int(hashlib.sha256(f"fetch-wm-v1:{seed}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"

def child_seed(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**32)

def position_error_mm(predicted, target, offset):
    return 1000 * math.sqrt(sum((float(predicted[offset+i])-float(target[offset+i]))**2 for i in range(3)))

def population_summary(values):
    xs = [float(x) for x in values]
    if not xs or not all(math.isfinite(x) for x in xs):
        return {"count": 0, "mean": None, "population_std": None, "std_definition": "ddof=0"}
    mean = sum(xs) / len(xs)
    return {"count": len(xs), "mean": mean,
            "population_std": math.sqrt(sum((x-mean)**2 for x in xs)/len(xs)),
            "std_definition": "ddof=0"}




def formal_planner(config):
    required={'horizon':10,'candidates':128,'iterations':3,'elites':16}
    if any(config.get(k)!=v for k,v in required.items()) or config.get('scheme')!='deterministic_latent':
        raise ValueError('root-accepted main CEM10/128/3/16 differs')
    if config.get('initial_std')!=.5 or config.get('minimum_std')!=.01 or config.get('action_penalty')!=.001:
        raise ValueError('frozen CEM cost/sampling settings differ')
