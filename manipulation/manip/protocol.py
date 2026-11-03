"""Pure protocol logic: testable without numpy, Torch or MuJoCo."""
from __future__ import annotations
import hashlib
import json
import math

SEEDS = (0, 1, 2)
EVAL_SEEDS = tuple(range(1_000_000, 1_000_100))
VALIDATION_SEEDS = tuple(range(900_000, 900_100))
HORIZON = 50
FIELD_MAP = {"grip_position": [0, 3], "object_position": [3, 6],
             "object_relative_position": [6, 9], "finger_position": [9, 11],
             "object_euler": [11, 14], "object_relative_velocity_times_dt": [14, 17],
             "object_angular_velocity_times_dt": [17, 20],
             "grip_velocity_times_dt": [20, 23], "finger_velocity_times_dt": [23, 25]}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def split_for_reset(seed):
    require_training_reset(seed)
    bucket = int(hashlib.sha256(f"fetch-wm-v1:{seed}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"

def require_training_reset(seed):
    if type(seed) is not int or not 0 <= seed < 900_000:
        raise ValueError(f"held-out/nontraining reset seed forbidden: {seed}")


def child_seed(*values):
    return int(digest(list(values))[:16], 16) % (2**32)







def wilson(successes, count, z=1.959963984540054):
    if count == 0:
        return [None, None]
    p = successes / count
    d = 1 + z*z/count
    center = (p + z*z/(2*count))/d
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/d
    return [max(0.0, center-radius), min(1.0, center+radius)]

def summarize(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("no real episodes")
    successes = [r for r in rows if r["terminal_success"]]
    return {"episodes": n, "successes": len(successes), "success_rate": len(successes)/n,
            "wilson_95": wilson(len(successes), n),
            "ever_successes": sum(r["ever_success"] for r in rows),
            "mean_length_on_success": sum(r["length"] for r in successes)/len(successes) if successes else None,
            "physical_violation_episodes": (sum(bool(r['physical_violations']) for r in rows)
                if all(r.get('physical_audit_active',False) for r in rows) else None),
            "physical_audit_status": 'active for all episodes' if all(r.get('physical_audit_active',False) for r in rows) else 'not measured'}


def aggregate_policies(summaries):
    values = [x["success_rate"] for x in summaries]
    if len(values) < 3:
        raise ValueError("training result requires >=3 independent policies")
    mean = sum(values)/len(values)
    return {"mean": mean, "population_std": math.sqrt(sum((x-mean)**2 for x in values)/len(values)),
            "std_definition": "population ddof=0", "policies": summaries}

def assert_formal(config):
    if config.get('mode')!='formal' or config.get('protocol')!='m3-reduced-v2':
        raise ValueError('reduced formal protocol required')
    if config['env_id']!='FetchPush-v4' or config['seeds']!=list(SEEDS) or config['horizon']!=50 or config['demo_budget']!=100:
        raise ValueError('main BC task/seeds/horizon/demo budget changed')
    if config.get('method') not in ('bc','bc_dr'):
        raise ValueError('only nominal/DR BC is in the reduced main experiment')

def assert_panel(spec):
    expected={'mass':[.5,2.],'friction':[.5,1.5],'noise_sigma':[.005,.010],'latency':[1,2],'gain':[.8,1.2]}
    nominal={'mass':1.,'friction':1.,'noise_sigma':0.,'latency':0,'gain':1.}
    cells=[{'name':'nominal',**nominal}]
    for key,values in expected.items():
        for i,value in enumerate(values): cells.append({'name':key+('_low' if i==0 else '_high'),**nominal,key:value})
    if spec.get('protocol')!='fixed11-v2' or spec.get('cells')!=cells or spec.get('episodes_per_policy')!=100 or spec.get('policy_seeds')!=[0,1,2]:
        raise ValueError('fixed11-cell predeclared panel/count/seed protocol changed')
