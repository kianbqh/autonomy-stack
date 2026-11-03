"""Released BC drivers; all failed expert attempts and split counts are retained."""
import json
from pathlib import Path
import numpy as np
import torch
from .bc import ActorPolicy,fit_bc,load_actor
from .envs import make_env
from .episodes import load_episode,write_episode
from .evaluate import evaluate_policy,rollout
from .expert import PushExpert
from .protocol import EVAL_SEEDS,assert_formal,digest,split_for_reset
from .provenance import dump_new,sha256
from .seed import seed_all,rng_state
def save_new(path,payload):
    with Path(path).open('xb') as stream: torch.save(payload,stream)
def collect(config,attempt):
    assert_formal(config)
    env=make_env(config); expert=PushExpert(config['expert']); records=[]; successful=[]
    producer=attempt.producer_receipt(config['expert'])
    try:
        for seed in range(config['reset_start'],config['reset_start']+config['max_attempts']):
            arrays,meta=rollout(env,expert,seed,None,'expert',True)
            meta.update(episode_id=f"{config['condition']}-expert-r{seed}",source_sha256=digest(attempt.record['source_hashes']),
                policy_sha256=producer['policy']['sha256'],checkpoint_sha256=None,config_sha256=digest(config),
                provenance=attempt.record,producer_receipt=producer)
            record=write_episode(attempt.path/'episodes',arrays,meta); records.append(record)
            if meta['terminal_success']: successful.append(record['episode_id'])
            attempt.event('collection_attempt',reset_seed=seed,terminal_success=meta['terminal_success'],successful=len(successful))
            if len(successful)==100: break
        manifest={'status':'complete' if len(successful)==100 else 'incomplete','condition':config['condition'],
            'episode_directory':'episodes','episodes':records,'attempts':len(records),'successful':len(successful),
            'success_rate':len(successful)/len(records),'successful_ids':successful,'demo_budget':100,'provenance':attempt.record}
        dump_new(attempt.path/'manifest.json',manifest)
        if manifest['status']!='complete': raise RuntimeError('expert gate: could not collect100 successful native50-step episodes')
        # Expert benchmark is nominal; randomization is disabled explicitly.
        if config['condition']=='nominal':
            evaluation=evaluate_policy(env,expert,EVAL_SEEDS,None,'expert')
            attempt.write('expert-heldout.json',evaluation)
    finally: env.close()
def demo_records(path,budget=100):
    path=Path(path); m=json.loads(path.read_text(encoding='utf-8'))
    if budget!=100 or m['status']!='complete' or m['successful']!=100 or m['demo_budget']!=100:
        raise ValueError('one complete100-success demo pool required')
    ids=m['successful_ids']; byid={r['episode_id']:r for r in m['episodes']}
    if len(ids)!=100 or len(set(ids))!=100: raise ValueError('invalid successful pool')
    records=[byid[i] for i in ids]
    if any(not r['terminal_success'] for r in records): raise ValueError('failed episode in success pool')
    return path.parent/m['episode_directory'],records,m
def train_bc(config,attempt,manifest_path,seed,budget,device):
    assert_formal(config); seed_all(seed)
    root,records,manifest=demo_records(manifest_path,budget)
    if manifest['condition']!=config['condition']: raise ValueError('nominal/DR demo condition mismatch')
    groups={k:[] for k in ('train','val','test')}; reset_ids={k:[] for k in groups}
    for record in records:
        split=split_for_reset(record['reset_seed']); reset_ids[split].append(record['reset_seed'])
        if split!='test': groups[split].append(load_episode(root,record))
    def arrays(split):
        if not groups[split]: raise ValueError('empty grouped split; no outcome-guided repartition')
        return (np.concatenate([np.concatenate((e['obs'],e['desired_goal']),axis=1) for e in groups[split]]),
                np.concatenate([e['action'] for e in groups[split]]))
    x,y=arrays('train'); vx,vy=arrays('val'); selected=[]
    common={'schema':'fetch-bc-v2','method':config['method'],'model_seed':seed,'demo_budget':100,
        'split_reset_ids':reset_ids,'episode_counts':{k:len(v) for k,v in reset_ids.items()},
        'input_manifest_sha256':sha256(manifest_path),'normalizer':'identity','provenance':attempt.record}
    def save(epoch,actor,opt,loss):
        path=attempt.path/f'bc-epoch-{epoch:04d}.pt'
        save_new(path,{**common,'actor':actor.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
                      'epoch':epoch,'validation_mse':loss}); selected.append((loss,path,epoch))
    _,history=fit_bc(x,y,vx,vy,config['bc'],device,save)
    _,path,epoch=min(selected,key=lambda t:t[0]); actor,payload=load_actor(path,device)
    final=attempt.path/'final-policy.pt'; save_new(final,{**common,'actor':actor.state_dict(),'epoch':epoch,'validation_mse':payload['validation_mse']})
    # DR collection must never silently leak into nominal held-out evaluation.
    env=make_env({**config,'training_randomization':None})
    try: evaluation=evaluate_policy(env,ActorPolicy(actor,device,payload),EVAL_SEEDS,seed,config['method'])
    finally: env.close()
    attempt.write('bc-result.json',{'status':'complete','method':config['method'],'model_seed':seed,'demo_budget_pool':100,
        'collection_summary':{k:manifest[k] for k in ('condition','attempts','successful','success_rate')},
        'actual_fit_episodes':len(reset_ids['train']),'validation_episodes':len(reset_ids['val']),'unused_test_episodes':len(reset_ids['test']),
        'split_reset_ids':reset_ids,'actual_fit_transitions':len(x),'validation_transitions':len(vx),'history':history,
        'selection':'minimum grouped training-domain validation action MSE; heldout unused',
        'selected_epoch':epoch,'checkpoint':str(final),'checkpoint_sha256':sha256(final),
        'input_manifest_sha256':sha256(manifest_path),'evaluation':evaluation})
