"""Deferred actual physics/BC engineering gates. Not research measurements."""
import copy
import numpy as np
import torch
import gymnasium as gym
from .envs import make_env
from .bc import fit_bc
from .evaluate import rollout,evaluate_policy
from .expert import PushExpert
from .seed import seed_all
from .shield import Shield
from .sensor_checks import command_fifo_checks,observation_schedule_checks
def exact(a,b,where='root'):
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or not np.array_equal(a,b): raise AssertionError('exact array mismatch: '+where)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError('keys mismatch: '+where)
        for k in a: exact(a[k],b[k],where+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError('length mismatch: '+where)
        for i,(x,y) in enumerate(zip(a,b)): exact(x,y,where+'.'+str(i))
    elif a!=b: raise AssertionError('scalar mismatch: '+where)
def stepping_regression(config):
    # Actual full native episodes, including queued commands and changed parameters.
    conditions=[{}, {'mass':2.}, {'friction':.5}, {'gain':1.2}, {'latency':2},
                {'mass':.5,'friction':1.5,'gain':.8,'latency':1,'noise_sigma':.01}]
    receipts=[]
    for index,perturb in enumerate(conditions):
        env=make_env({**config,'training_randomization':None},perturb); clone=make_env({**config,'training_randomization':None},perturb)
        try:
            env.reset(seed=200+index); clone.reset(seed=200+index); actions=np.random.default_rng(91).uniform(-.5,.5,(50,4)).astype(np.float32)
            for step,action in enumerate(actions):
                before=env.snapshot(); env._force_substep_reference=False
                native=env.step(action); native_state=env.snapshot()
                env.restore(before); env._force_substep_reference=True
                reference=env.step(action); reference_state=env.snapshot()
                exact(native[:4],reference[:4],'observation/reward/terminated/truncated')
                exact(native_state,reference_state,'complete_state')
                if step in (0,7,25,49):
                    real=env.snapshot(); clone.restore(before); clone._force_substep_reference=False; cloned=clone.step(action)
                    exact(native[:4],cloned[:4],'independent_clone_outcome'); exact(native_state,clone.snapshot(),'independent_clone_state')
                    exact(real,env.snapshot(),'clone_preserved_real_env')
                env._force_substep_reference=False
            receipts.append({'perturbation':perturb,'paired_control_steps':50,'physics_substeps_per_control':env.contract['n_substeps'],
                             'bit_exact_integration_and_wrapper_state':True,'bit_exact_observation_reward_termination':True,'independent_clone_checks':4})
        finally: clone.close(); env.close()
    return receipts
def run_checks(config,attempt):
    if config['mode']!='smoke': raise ValueError('engineering smoke must be explicitly labelled')
    seed_all(0); checks={}
    # Registration occurs in make_env, before using native FetchReach.
    env=make_env(config)
    try:
        obs,_=env.reset(seed=0); base=env.unwrapped
        if not np.allclose(obs['observation'][20:23],base._utils.get_site_xvelp(base.model,base.data,'robot0:grip')*base.dt,rtol=0,atol=1e-12):
            raise AssertionError('installed gripper dt mapping changed')
        if not np.allclose(obs['observation'][14:17]+obs['observation'][20:23],base._utils.get_site_xvelp(base.model,base.data,'object0')*base.dt,atol=1e-12):
            raise AssertionError('installed object relative velocity changed')
        checks['contract']=env.contract
        expert=evaluate_policy(env,PushExpert(),range(10),None,'expert')
        attempt.write('expert-training-domain-gate.json',expert)
        if expert['successes']<config['expert_min_successes_out_of_10']: raise AssertionError('expert floor failed; formal data collection blocked')
        a,_=rollout(env,PushExpert(),17,None,'expert',True); b,_=rollout(env,PushExpert(),18,None,'expert',True)
        x=np.concatenate((a['obs'],a['desired_goal']),1); vx=np.concatenate((b['obs'],b['desired_goal']),1)
        selected=[]; _,history=fit_bc(x,a['action'],vx,b['action'],config['bc'],'cpu',lambda e,actor,opt,loss:selected.append((e,loss)))
        checks['bc_smoke']={'epochs':len(history),'history':history,'validation_selected_epochs':selected,'independent_episode_validation':True}
    finally: env.close()
    reach=gym.make('FetchReach-v4',reward_type='sparse',render_mode=None)
    try: reach.reset(seed=123); reach.step(np.zeros(4,dtype=np.float32)); checks['reach_sanity']={'env_id':reach.spec.id}
    finally: reach.close()
    checks['native_vs_substeps_and_clone']=stepping_regression(config)
    checks['full_command_fifo']=command_fifo_checks(config)
    checks['noise_dropout_schedule']=observation_schedule_checks(config)
    random=make_env(config,{'mass':2.,'friction':1.5,'noise_sigma':.01,'latency':2,'gain':1.2,'dropout_hold':3})
    try:
        first,_=random.reset(seed=73); parameters=copy.deepcopy(random.parameter_receipt)
        random.step([.2,0,0,0]); random.reset(seed=74); again,_=random.reset(seed=73)
        exact(parameters,random.parameter_receipt); exact(first,again)
        if not np.array_equal(again['observation'][3:6],again['achieved_goal']): raise AssertionError('pose truth leak')
        if not np.allclose(again['observation'][6:9],again['observation'][3:6]-again['observation'][:3]): raise AssertionError('relative pose leak')
        random.step([.2,0,0,0]);
        if np.any(random.last_applied_action): raise AssertionError('latency queue did not begin with zeros')
        checks['sensor_dr']={'paired_reset':True,'no_parameter_compounding':True,'redundant_pose_consistency':True,'initial_latency_zero':True}
    finally: random.close()
    impulses=[]
    for gain in (.8,1.,1.2):
        e=make_env(config,{'gain':gain})
        try:
            e.reset(seed=79); before=e.clean_observation['observation'][:3].copy(); e.step([.2,0,0,0])
            impulses.append({'gain':gain,'applied':e.last_applied_action.tolist(),'displacement':(e.clean_observation['observation'][:3]-before).tolist()})
        finally: e.close()
    if np.array_equal(impulses[0]['displacement'],impulses[2]['displacement']): raise AssertionError('command gain has no physical effect')
    checks['cartesian_gain_impulse']=impulses
    # Small engineering check of the actual shield/audit path, never a fault matrix.
    from pathlib import Path
    import json
    shield_config=json.loads((Path(__file__).resolve().parents[1]/'configs/shield.yaml').read_text())
    monitored=make_env(config)
    try:
        monitored.reset(seed=81)
        monitored.shield=Shield(shield_config,monitored.unwrapped.dt)
        monitored.audit_shield=monitored.shield; monitored.shield_enabled=True
        for _ in range(5):
            monitored.step([.2,0,0,0])
            if not monitored.last_shield: raise AssertionError('actual actuation shield path was bypassed')
        checks['physical_shield_path']={'control_steps':5,'projection_diagnostics_present':True,
            'physical_events':monitored.physical_events,'claim':'engineering path check only; no success or unconditional safety result'}
    finally: monitored.close()
    attempt.write('runtime-gates.json',{'status':'engineering_gates_passed','checks':checks,'research_result':False})
