"""Actual held-out recorded-action open-loop predictions versus constant velocity."""
import numpy as np
import torch
from .protocol import HORIZONS,constant_velocity,population_summary
@torch.no_grad()
def evaluate_prediction(model,checkpoint,dataset,config):
    if checkpoint['identity']['dataset']!=dataset.identity or config['horizons']!=list(HORIZONS): raise ValueError('prediction dataset/horizons differ from frozen protocol')
    device=next(model.parameters()).device; rows=[]
    for ep in dataset.by_split['test']:
        n=ep.length; latent=model.encode(torch.as_tensor(ep.visible,device=device)); actions=torch.as_tensor(ep.arrays['action'],device=device,dtype=torch.float32)
        for step in range(1,max(HORIZONS)+1):
            action=actions[(torch.arange(n,device=device)+step-1).clamp_max(n-1)]; latent=model.latent_step(latent,action)
            if step not in HORIZONS: continue
            valid=n-step+1; predicted=model.decode(latent[:valid]).cpu().numpy(); truth=ep.clean_next[step-1:]
            cv=np.asarray([constant_velocity(x,step,ep.metadata['control_dt']) for x in ep.visible[:valid]])
            errors={'object_mm':np.linalg.norm(predicted[:,3:6]-truth[:,3:6],axis=1)*1000,
                    'gripper_mm':np.linalg.norm(predicted[:,:3]-truth[:,:3],axis=1)*1000,
                    'cv_object_mm':np.linalg.norm(cv[:,3:6]-truth[:,3:6],axis=1)*1000,
                    'cv_gripper_mm':np.linalg.norm(cv[:,:3]-truth[:,:3],axis=1)*1000}
            if any(not np.isfinite(v).all() for v in errors.values()): raise FloatingPointError('nonfinite held-out prediction')
            rows.append({'episode_id':ep.metadata['episode_id'],'reset_seed':ep.metadata['reset_seed'],'source':ep.metadata['source_policy'],
                'model_seed':checkpoint['model_seed'],'horizon':step,'starts':valid,**{k:v.tolist() for k,v in errors.items()},
                **{'episode_mean_'+k:float(v.mean()) for k,v in errors.items()}})
    summaries=[]
    for h in HORIZONS:
        cells=[r for r in rows if r['horizon']==h]
        summaries.append({'horizon':h,'episodes':len(cells),'starts':sum(r['starts'] for r in cells),
            **{k:population_summary([r['episode_mean_'+k] for r in cells]) for k in ('object_mm','gripper_mm','cv_object_mm','cv_gripper_mm')}})
    return {'model_seed':checkpoint['model_seed'],'summary':summaries,'episodes_detail':rows,
        'aggregation':'mean valid recorded-action starts within episode; population mean/std over episodes; overlapping starts are nested',
        'rollout':'encode once then action-conditioned latent rollout without teacher forcing; decode only for scoring',
        'split':'frozen test; never used for checkpoint selection','uncertainty':'single model per seed; no ensemble uncertainty claim'}
