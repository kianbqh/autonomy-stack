"""Actual MuJoCo episode traces for a viewer; absent outcomes stay absent."""
import json
from pathlib import Path
from .envs import make_env
from .evaluate import evaluate_policy
from .matrix import load_adapter
from .protocol import EVAL_SEEDS

def export(config,attempt,spec,device):
    env=make_env(config)
    policy=load_adapter(spec,device)
    saved=set()
    def trace(arrays,metadata):
        outcome="success" if metadata["terminal_success"] else "failure"
        if outcome in saved:
            return
        steps=[]
        for t in range(50):
            steps.append({"control_step":t,"time_seconds":t*metadata["control_dt"],
                          "gripper_position":arrays["clean_obs"][t,:3].tolist(),
                          "object_position":arrays["clean_obs"][t,3:6].tolist(),
                          "goal":arrays["desired_goal"][t].tolist(),"visible_observation":arrays["obs"][t].tolist(),
                          "command_action":arrays["action"][t].tolist(),"applied_action":arrays["applied_action"][t].tolist(),
                          "reward":float(arrays["reward"][t]),"is_success":bool(arrays["is_success"][t]),
                          "terminated":bool(arrays["terminated"][t]),"truncated":bool(arrays["truncated"][t])})
        attempt.write(f"trace-{outcome}.json",{"kind":"actual Fetch simulation state trace","outcome":outcome,
                      "policy_spec":spec,"episode":metadata,"steps":steps,
                      "terminal_gripper_position":arrays["next_clean_obs"][-1,:3].tolist(),
                      "terminal_object_position":arrays["next_clean_obs"][-1,3:6].tolist()})
        saved.add(outcome)
    try:
        result=evaluate_policy(env,policy,EVAL_SEEDS,spec["model_seed"],spec["method"],trace)
        attempt.write("trace-index.json",{"actual_outcomes_saved":sorted(saved),"evaluation":result,
                      "missing_outcomes":sorted({"success","failure"}-saved),"reason_if_missing":"No such actual outcome among the100 fixed episodes"})
    finally:
        env.close()
