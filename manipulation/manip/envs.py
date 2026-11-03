"""Official Fetch factory with noncompounding DR, paired faults and physical substep audit.

Heavy libraries are imported only when this runtime module is explicitly used.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import types
import xml.etree.ElementTree as ET
import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
from .protocol import FIELD_MAP, child_seed, digest
from .provenance import sha256

NOMINAL={"mass":1.0,"friction":1.0,"noise_sigma":0.0,"latency":0,"gain":1.0,
         "dropout_hold":0,"dropout_probability":0.1,"gripper_failure_probability":0.0}

def xml_inputs(path):
    """Hash XML includes recursively; runtime binary/model identity is recorded separately."""
    found={}
    def visit(current):
        current=current.resolve()
        if str(current) in found:
            return
        found[str(current)]=sha256(current)
        for include in ET.parse(current).getroot().iter("include"):
            visit(current.parent/include.attrib["file"])
    visit(Path(path))
    return found

class FetchExperiment(gym.Wrapper):
    def __init__(self, env, config, perturbation=None, shield=None):
        super().__init__(env)
        self.config=copy.deepcopy(config)
        self.perturbation={**NOMINAL,**(perturbation or {})}
        self.randomization=config.get("training_randomization")
        self.shield=shield
        self.audit_shield=shield  # May be present while projection is disabled.
        self.shield_enabled=config.get("shield_enabled",shield is not None)
        self.base=self.unwrapped
        self.nominal={k:getattr(self.base.model,k).copy() for k in ("body_mass","body_inertia","geom_friction")}
        self._native_step=self.base._mujoco_step
        self._monitor=False
        # Test-only stepping mode, excluded from physical snapshot identity. Exactness
        # is a deferred runtime gate; this switch is never set by experiment drivers.
        self._force_substep_reference=False
        outer=self
        def substeps(base, action):
            auditing=outer._monitor and outer.audit_shield is not None
            if not auditing and not outer._force_substep_reference:
                return outer._native_step(action)
            for _ in range(base.n_substeps):
                if auditing:
                    before=base._utils.get_site_xpos(base.model,base.data,"robot0:grip").copy()
                base._mujoco.mj_step(base.model,base.data,nstep=1)
                if auditing:
                    after=base._utils.get_site_xpos(base.model,base.data,"robot0:grip").copy()
                    failures=outer.audit_shield.violations(before,after,0.0,base.model.opt.timestep)
                    if failures:
                        outer.physical_events.append({"control_step":outer.steps,"time":float(base.data.time),
                                                      "position":after.tolist(),"violations":failures})
        self.base._mujoco_step=types.MethodType(substeps,self.base)
        self.contract=None

    def _model_id(self, kind, name):
        value=mujoco.mj_name2id(self.base.model,kind,name)
        if value<0:
            raise ValueError(f"required installed XML name absent: {name}")
        return value

    def _restore_nominal(self):
        for key,value in self.nominal.items():
            getattr(self.base.model,key)[:]=value

    def _parameters(self):
        self._restore_nominal()
        b=self.base
        object_id=self._model_id(mujoco.mjtObj.mjOBJ_BODY,"object0")
        table_id=self._model_id(mujoco.mjtObj.mjOBJ_BODY,"table0")
        b.model.body_mass[object_id]*=self.current["mass"]
        b.model.body_inertia[object_id]*=self.current["mass"]
        geoms=np.isin(b.model.geom_bodyid,[object_id,table_id])
        if not geoms.any():
            raise ValueError("no object/table friction geoms")
        b.model.geom_friction[geoms]*=self.current["friction"]
        mujoco.mj_setConst(b.model,b.data)
        self.parameter_receipt={"object_body_id":int(object_id),"table_body_id":int(table_id),
                                "friction_geom_ids":np.flatnonzero(geoms).tolist(),
                                "body_mass":float(b.model.body_mass[object_id]),
                                "body_inertia":b.model.body_inertia[object_id].tolist(),
                                "geom_friction":b.model.geom_friction[geoms].tolist()}

    def reset(self, *, seed=None, options=None):
        if seed is None:
            raise ValueError("explicit reset seed required")
        self.steps=0
        self.reset_seed=int(seed)
        self.wrapper_seed=child_seed("wrapper",seed,self.config.get("fault_stream_seed",0))
        self.randomization_seed=child_seed("randomization",seed,self.config.get("dr_stream_seed",0))
        self.rng=np.random.default_rng(self.wrapper_seed)
        self.dr_rng=np.random.default_rng(self.randomization_seed)
        self.current=copy.deepcopy(self.perturbation)
        if self.randomization:
            for key,values in self.randomization.items():
                self.current[key]=values[int(self.dr_rng.integers(len(values)))]
        self._restore_nominal()
        self._parameters()
        self._monitor=False
        obs,info=self.env.reset(seed=seed,options=options)
        self._monitor=True
        self.physical_events=[]
        self.queue=[np.zeros(4,dtype=np.float32) for _ in range(int(self.current["latency"]))]
        self.noise=self.rng.standard_normal((51,3))*float(self.current["noise_sigma"])
        self.dropout_draws=self.rng.random(51)
        self.failure_draws=self.rng.random(51)
        self.held_position=None
        self.drop_remaining=0
        self.clean_observation=copy.deepcopy(obs)
        self.action_space.seed(child_seed("action",self.reset_seed))
        self.last_applied_action=np.zeros(4,dtype=np.float32)
        self.last_shield={}
        visible=self._visible(obs)
        if self.contract is None:
            self.contract=self._contract(obs)
        info={**info,"physical_events":[],"perturbation":copy.deepcopy(self.current)}
        return visible,info

    def _visible(self, obs):
        out=copy.deepcopy(obs)
        p=out["observation"][3:6]+self.noise[self.steps]
        hold=int(self.current["dropout_hold"])
        if hold and self.drop_remaining==0 and self.dropout_draws[self.steps]<self.current["dropout_probability"]:
            self.drop_remaining=hold
        if self.drop_remaining and self.held_position is not None:
            p=self.held_position.copy()
            self.drop_remaining-=1
        else:
            self.held_position=p.copy()
            if self.drop_remaining:
                self.drop_remaining-=1
        out["observation"][3:6]=p
        out["observation"][6:9]=p-out["observation"][:3]
        out["achieved_goal"]=p.copy()
        return out

    def step(self, action):
        if self.steps>=50:
            raise RuntimeError("reset after native time limit")
        act=np.asarray(action,dtype=np.float32).copy()
        if act.shape!=(4,) or not np.isfinite(act).all():
            raise ValueError("invalid action")
        self.queue.append(np.clip(act,-1,1))
        applied=self.queue.pop(0)
        applied[:3]*=float(self.current["gain"])
        applied=np.clip(applied,-1,1)
        if self.failure_draws[self.steps]<self.current["gripper_failure_probability"]:
            applied[3]=0.0  # Inert for blocked FetchPush; labelled N/A in fault report.
        self.last_shield={}
        if self.shield is not None and self.shield_enabled:
            applied,diag=self.shield.project(applied,self.clean_observation["observation"][:3],
                                              float(self.current["noise_sigma"]),[x.tolist() for x in self.queue])
            applied=np.asarray(applied,dtype=np.float32)
            self.last_shield=diag
        self.last_applied_action=applied.copy()
        obs,reward,terminated,truncated,info=self.env.step(applied)
        self.steps+=1
        self.clean_observation=copy.deepcopy(obs)
        info={**info,"physical_events":copy.deepcopy(self.physical_events),"shield":self.last_shield,
              "applied_action":applied.tolist(),"perturbation":copy.deepcopy(self.current)}
        return self._visible(obs),reward,terminated,truncated,info

    def _contract(self,obs):
        b=self.base
        if obs["observation"].shape!=(25,) or obs["desired_goal"].shape!=(3,) or self.action_space.shape!=(4,):
            raise ValueError("installed Fetch observation/action schema changed")
        if not np.allclose(obs["observation"][3:6],obs["achieved_goal"],rtol=0,atol=1e-12):
            raise ValueError("object position mapping changed")
        if not np.allclose(obs["observation"][6:9],obs["observation"][3:6]-obs["observation"][:3],atol=1e-12):
            raise ValueError("relative position mapping changed")
        if self.spec.id!="FetchPush-v4" or self.spec.max_episode_steps!=50 or b.reward_type!="sparse" or not b.block_gripper:
            raise ValueError("primary task/native reward/horizon changed")
        self.action_space.seed(child_seed("action",self.reset_seed))
        return {"env_id":self.spec.id,"max_episode_steps":self.spec.max_episode_steps,"reward_type":b.reward_type,
                "distance_threshold":float(b.distance_threshold),"control_dt":float(b.dt),
                "physics_dt":float(b.model.opt.timestep),"n_substeps":int(b.n_substeps),
                "action_scale_m":0.05,"field_map":FIELD_MAP,"xml_sha256":sha256(b.fullpath),
                "xml_inputs":xml_inputs(b.fullpath),"integration_state_spec":int(mujoco.mjtState.mjSTATE_INTEGRATION),
                "integration_state_size":int(mujoco.mj_stateSize(b.model,mujoco.mjtState.mjSTATE_INTEGRATION)),
                "gain_semantics":"Cartesian mocap delta multiplier before native clipping; not joint actuator gain",
                "gripper_failure_applicability":"N/A: native FetchPush block_gripper=True",
                "source_sha256":sha256(__file__)}

    def integration_state(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        result=np.empty(mujoco.mj_stateSize(self.base.model,spec))
        mujoco.mj_getState(self.base.model,self.base.data,result,spec)
        return result

    def snapshot(self):
        local=("steps","reset_seed","wrapper_seed","randomization_seed","current","queue","noise","dropout_draws",
               "failure_draws","held_position","drop_remaining","clean_observation","last_applied_action",
               "last_shield","physical_events","parameter_receipt","_monitor")
        wrappers=[]
        item=self.env
        while isinstance(item,gym.Wrapper):
            fields={k:copy.deepcopy(item.__dict__[k]) for k in ("_elapsed_steps","_has_reset","_disable_render_order_enforcing") if k in item.__dict__}
            # Reject unknown stateful wrappers rather than claiming a complete oracle snapshot.
            if type(item).__name__ not in ("TimeLimit","OrderEnforcing","PassiveEnvChecker"):
                raise TypeError(f"unreviewed oracle wrapper {type(item).__name__}")
            for key in ("checked_reset","checked_step","checked_render","close_called"):
                if key in item.__dict__:
                    fields[key]=copy.deepcopy(item.__dict__[key])
            wrappers.append((type(item).__name__,fields))
            item=item.env
        return {"integration":self.integration_state(),"goal":self.base.goal.copy(),
                "contract_sha256":digest(self.contract),"factory_config_sha256":digest(self.config),
                "perturbation_spec":copy.deepcopy(self.perturbation),
                "shield_config":copy.deepcopy(self.shield.config) if self.shield else None,
                "model":{k:getattr(self.base.model,k).copy() for k in self.nominal},
                "local":{k:copy.deepcopy(getattr(self,k)) for k in local},"wrappers":wrappers,
                "rng":copy.deepcopy(self.rng.bit_generator.state),"dr_rng":copy.deepcopy(self.dr_rng.bit_generator.state),
                "env_rng":copy.deepcopy(self.base.np_random.bit_generator.state),
                "env_rng_seed":copy.deepcopy(getattr(self.base,"_np_random_seed",None)),
                "action_rng":copy.deepcopy(self.action_space.np_random.bit_generator.state)}

    def restore(self,snapshot):
        b=self.base
        if digest(self.contract)!=snapshot["contract_sha256"] or digest(self.config)!=snapshot["factory_config_sha256"]:
            raise ValueError("snapshot requires identical installed environment/factory configuration")
        if self.perturbation!=snapshot["perturbation_spec"]:
            raise ValueError("snapshot perturbation specification mismatch")
        if (self.shield.config if self.shield else None)!=snapshot["shield_config"]:
            raise ValueError("snapshot shield configuration mismatch")
        for k,v in snapshot["model"].items():
            getattr(b.model,k)[:]=v
        mujoco.mj_setConst(b.model,b.data)
        b.goal=snapshot["goal"].copy()
        mujoco.mj_setState(b.model,b.data,snapshot["integration"],mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(b.model,b.data)
        # mj_forward may update solver workspaces. Restore integration state again for exact warmstart.
        mujoco.mj_setState(b.model,b.data,snapshot["integration"],mujoco.mjtState.mjSTATE_INTEGRATION)
        for k,v in snapshot["local"].items():
            setattr(self,k,copy.deepcopy(v))
        item=self.env
        for name,fields in snapshot["wrappers"]:
            if type(item).__name__!=name:
                raise ValueError("snapshot wrapper stack mismatch")
            for key,value in fields.items():
                setattr(item,key,copy.deepcopy(value))
            item=item.env
        self.rng.bit_generator.state=copy.deepcopy(snapshot["rng"])
        self.dr_rng.bit_generator.state=copy.deepcopy(snapshot["dr_rng"])
        b.np_random.bit_generator.state=copy.deepcopy(snapshot["env_rng"])
        b._np_random_seed=copy.deepcopy(snapshot["env_rng_seed"])
        self.action_space.np_random.bit_generator.state=copy.deepcopy(snapshot["action_rng"])

    def compute_reward(self,achieved_goal,desired_goal,info):
        return self.base.compute_reward(achieved_goal,desired_goal,info)

def make_env(config,perturbation=None,shield=None):
    gym.register_envs(gymnasium_robotics)
    env=gym.make(config.get("env_id","FetchPush-v4"),reward_type="sparse",render_mode=None)
    return FetchExperiment(env,config,perturbation,shield)
