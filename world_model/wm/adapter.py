"""Track A external policy seam; ordinary MPC has no environment/clean-state reference."""
from __future__ import annotations
import time
import numpy as np
from .mpc import Planner
from .protocol import child_seed, observation_to_state
from .train import load_model
from .provenance import digest

class MPCPolicy:
    def __init__(self,spec,device="cpu"):
        checkpoint=spec["checkpoint"] if isinstance(spec["checkpoint"],dict) else {
            "path":spec["checkpoint"],"sha256":spec["checkpoint_sha256"],"model_seed":spec["model_seed"]}
        self.model,self.checkpoint=load_model(checkpoint,device)
        if spec["model_seed"]!=self.checkpoint["model_seed"]: raise ValueError("adapter model seed mismatch")
        self.planner=Planner(spec["planner"],device,formal=spec.get("mode","formal")=="formal")
        self.dt=float(self.checkpoint["control_dt"])
        self.control_dt=self.dt
        self.required_env_contract_sha256=self.checkpoint["env_contract_sha256"]
        self.model_seed=spec["model_seed"]
        self.spec=spec; self.reset(0)
    def reset(self,seed=None):
        self.seed=int(seed); self.step=0; self.samples=[]; self.planner.reset(); self.control_start=None
    def act(self,obs,deterministic=True):
        self.control_start=time.perf_counter()
        # Deliberately read only these two public fields; achieved_goal could be redundant.
        visible=np.asarray(obs["observation"],dtype=np.float64)
        state=observation_to_state(visible,self.dt)
        goal=np.asarray(obs["desired_goal"],dtype=np.float32)
        seed=child_seed("cem",self.model_seed,self.seed,self.step,self.spec["planner"]["sampling_seed"])
        action,stats=self.planner.plan(state,goal,self.model,seed,remaining_steps=50-self.step)
        stats.update({"episode_reset_seed":self.seed,"model_seed":self.model_seed,"control_step":self.step,
                      "policy_act_ms":(time.perf_counter()-self.control_start)*1000})
        self.samples.append(stats); self.step+=1
        return action

def load_policy(spec,device="cpu"):
    if spec.get("kind") not in ("external","learned_mpc"):
        raise ValueError("expected external learned-MPC policy spec")
    return MPCPolicy(spec,device)
