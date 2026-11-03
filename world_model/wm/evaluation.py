"""Matched nominal100held-out ×3trained small models, with raw control timings."""
import copy
import time
import numpy as np
from .protocol import EVAL_SEEDS,MODEL_SEEDS,formal_planner,population_summary
from .provenance import digest,dump_new
class TimedEnvironment:
    def __init__(self,env,policy): self.env=env; self.policy=policy
    def __getattr__(self,key): return getattr(self.env,key)
    def reset(self,*a,**kw): return self.env.reset(*a,**kw)
    def step(self,action):
        result=self.env.step(action); self.policy.samples[-1]['full_control_step_ms']=(time.perf_counter()-self.policy.control_start)*1000
        return result
def timing_summary(samples,config):
    result={'raw_sample_count':len(samples),'warmup_control_steps':config['warmup_control_steps'],'warmup_reset_seed':config['warmup_reset_seed'],
        'repeats':'50control steps ×100fixed held-out episodes/model seed','concurrency':1,'percentile_method':'numpy linear',
        'exclusions':'env reset,pre-act deepcopy,episode summarization,disk I/O',
        'full_scope':'policy.act entry through env.step return; includes preprocessing,synchronized CEM,host copies,native physics and sensor wrapper'}
    for key in ('planner_ms','host_policy_ms','policy_act_ms','full_control_step_ms'):
        xs=[r[key] for r in samples]
        result[key]={**population_summary(xs),'p50':float(np.percentile(xs,50)),'p95':float(np.percentile(xs,95)),'p99':float(np.percentile(xs,99))} if xs else None
    return result
def evaluate_nominal(config,run):
    from manip.envs import make_env
    from manip.evaluate import evaluate_policy
    from .adapter import MPCPolicy
    import torch
    checkpoints=config.get('checkpoints',[])
    if config['mode']=='formal':
        formal_planner(config['planner'])
        if config['model_seeds']!=list(MODEL_SEEDS) or config['eval_reset_seeds']!=list(EVAL_SEEDS): raise ValueError('formal MPC requires3seeds ×100fixed held-out resets')
        if sorted(c['model_seed'] for c in checkpoints)!=list(MODEL_SEEDS) or len({c['sha256'] for c in checkpoints})!=3:
            raise ValueError('three independently trained actual small-model checkpoints required')
    results=[]; samples=[]
    for seed in config['model_seeds']:
        env=make_env({**config['env_config'],'training_randomization':None})
        try:
            checkpoint=next(c for c in checkpoints if c['model_seed']==seed)
            policy=MPCPolicy({'kind':'learned_mpc','mode':config['mode'],'checkpoint':checkpoint,'model_seed':seed,'planner':config['planner']},config['device'])
            timed=TimedEnvironment(env,policy); obs,_=timed.reset(seed=config['warmup_reset_seed']); policy.reset(config['warmup_reset_seed'])
            if policy.required_env_contract_sha256!=digest(env.contract) or policy.control_dt!=env.contract['control_dt']: raise ValueError('model/env contract or dt mismatch')
            for _ in range(config['warmup_control_steps']): obs,*_=timed.step(policy.act(obs))
            run.event('warmup_complete',model_seed=seed,steps=config['warmup_control_steps'])
            def trace(arrays,row):
                raw=copy.deepcopy(policy.samples); samples.extend(raw)
                run.write(f"trace-seed-{seed}-reset-{row['reset_seed']}.json",{'reset_seed':row['reset_seed'],'model_seed':seed,
                    'planner_samples':raw,'terminal_success':row['terminal_success'],'ever_success':row['ever_success']})
            result=evaluate_policy(timed,policy,config['eval_reset_seeds'],seed,'learned_mpc',trace)
            result.update({'mode':config['mode'],'env_contract':env.contract,'env_contract_sha256':digest(env.contract),
                'eval_reset_seeds':config['eval_reset_seeds'],'planner_config':config['planner'],'policy_kind':'visible-state learned encoder/latent dynamics MPC'})
            dump_new(run.path/f'nominal-seed-{seed}.json',{**run.base,**result}); results.append(result)
        finally: env.close()
    return {'method':'learned_mpc','model_seeds':config['model_seeds'],'policies':results,
        'success':population_summary([r['success_rate'] for r in results]),'timing':timing_summary(samples,config),'raw_samples':samples,
        'device':config['device'],'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name() if config['device']=='cuda' else None,
        'formal_evaluation':'native step50 success; same100held-out resets as BC; min(10,50-policy_step); no oracle/grid experiment'}
