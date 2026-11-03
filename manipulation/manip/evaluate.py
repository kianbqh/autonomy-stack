"""Fixed-reset evaluation. Policies receive only visible dict observations."""
from __future__ import annotations
import copy
import numpy as np
from .protocol import summarize,digest

def rollout(env,policy,reset_seed,model_seed=None,source_policy="unknown",dataset_eligible=False):
    obs,_=env.reset(seed=int(reset_seed))
    required=getattr(policy,"required_env_contract_sha256",None)
    if required is not None and required!=digest(env.contract):
        raise ValueError("external policy installed environment contract mismatch")
    required_dt=getattr(policy,"control_dt",None)
    if required_dt is not None and float(required_dt)!=float(env.contract["control_dt"]):
        raise ValueError("external policy native control_dt mismatch")
    policy.reset(seed=int(reset_seed))
    fields={k:[] for k in ("obs","next_obs","clean_obs","next_clean_obs","action","applied_action",
            "desired_goal","next_desired_goal","achieved_goal","next_achieved_goal",
            "clean_achieved_goal","next_clean_achieved_goal","reward","terminated","truncated","is_success")}
    states=[env.integration_state()]
    first_success=None
    for t in range(50):
        clean=copy.deepcopy(env.clean_observation)
        action=np.asarray(policy.act(copy.deepcopy(obs),deterministic=True),dtype=np.float32)
        nxt,reward,terminated,truncated,info=env.step(action)
        clean_next=env.clean_observation
        for label,item in (("",obs),("next_",nxt)):
            fields[label+"obs"].append(item["observation"].copy())
            fields[label+"desired_goal"].append(item["desired_goal"].copy())
            fields[label+"achieved_goal"].append(item["achieved_goal"].copy())
        for label,item in (("",clean),("next_",clean_next)):
            fields[label+"clean_obs"].append(item["observation"].copy())
            fields[label+"clean_achieved_goal"].append(item["achieved_goal"].copy())
        fields["action"].append(action)
        fields["applied_action"].append(env.last_applied_action.copy())
        for key,value in (("reward",reward),("terminated",terminated),("truncated",truncated),("is_success",bool(info["is_success"]))):
            fields[key].append(value)
        states.append(env.integration_state())
        if info["is_success"] and first_success is None:
            first_success=t+1
        obs=nxt
        if terminated or truncated:
            if t!=49 or terminated or not truncated:
                raise ValueError("native 50-step continuing-task boundary changed")
            break
    if len(fields["reward"])!=50 or not fields["truncated"][-1]:
        raise ValueError("missing native time-limit truncation")
    arrays={k:np.asarray(v) for k,v in fields.items()}
    arrays["simulator_state"]=np.asarray(states)
    audit_active=env.audit_shield is not None
    metadata={"schema":"fetch-episode-v1","source_policy":source_policy,"reset_seed":int(reset_seed),
              "model_seed":model_seed,"wrapper_seed":env.wrapper_seed,"randomization_seed":env.randomization_seed,
              "control_dt":env.contract["control_dt"],"env_contract":env.contract,
              "perturbation":copy.deepcopy(env.current),"dataset_eligible":dataset_eligible,
              "terminal_success":bool(fields["is_success"][-1]),"ever_success":any(fields["is_success"]),
              "first_success_step":first_success,"length":50,"physical_audit_active":audit_active,
              "physical_violations":len(env.physical_events) if audit_active else None,
              "physical_events":copy.deepcopy(env.physical_events)}
    return arrays,metadata

def evaluate_policy(env,policy,reset_seeds,model_seed=None,source_policy="unknown",trace_callback=None):
    rows=[]
    for seed in reset_seeds:
        arrays,row=rollout(env,policy,int(seed),model_seed,source_policy)
        if hasattr(policy,"samples"):
            row["planner_samples"]=copy.deepcopy(policy.samples)
            row["planner_timing_scope"]="external adapter's raw per-step planner samples; full-control scope measured by TrackB runner"
        rows.append(row)
        if trace_callback:
            trace_callback(arrays,row)
    return {**summarize(rows),"model_seed":model_seed,"source_policy":source_policy,"episodes_detail":rows,
            "outcome":"native is_success at final step50; ever_success separate"}
