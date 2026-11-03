"""One small model per independent seed; validation-only checkpoint selection."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from .dynamics import SmallLatentDynamics,fit_normalizers
from .protocol import child_seed
from .provenance import artifact_root,digest,seed_all,sha256,verify_file
def _save(path,payload):
    with Path(path).open('xb') as f: torch.save(payload,f)
def train_seed(dataset,config,seed,run,checkpoint_root,resume=None):
    if config['hidden']!=64 or config['latent']!=8 or config['model_seeds']!=[0,1,2]: raise ValueError('main small model protocol changed')
    seed_all(seed); device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested; no relabelled CPU fallback')
    normalizers=fit_normalizers(dataset.flat['train']); model=SmallLatentDynamics(normalizers,64,8).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    splits={s:{k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in dataset.flat[s].items() if k!='episode_index'} for s in ('train','val')}
    identity={'dataset':dataset.identity,'train_config':digest(config),'source_sha256':run.identity['source_sha256'],'seed':seed,
              'fit_episode_ids':[e.metadata['episode_id'] for e in dataset.by_split['train']]}
    best=float('inf'); best_state=None; best_epoch=None; stale=0; start=0; history=[]
    rng=np.random.default_rng(child_seed(seed,'minibatches'))
    if resume:
        verify_file(resume['path'],resume['sha256']); saved=torch.load(resume['path'],map_location=device,weights_only=False)
        if saved['identity']!=identity or saved['normalizers']!=normalizers: raise ValueError('resume data/config/source/fit identity mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); best=saved['best']; best_state=saved['best_state']
        best_epoch=saved['best_epoch']; stale=saved['stale']; history=saved['history']; start=saved['epoch']+1
        rng.bit_generator.state=saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        run.event('resume',checkpoint_sha256=resume['sha256'],epoch=start)
    directory=artifact_root(checkpoint_root)/run.path.name/f'seed-{seed}'; directory.mkdir(parents=True,exist_ok=False)
    for epoch in range(start,config['max_epochs']):
        begin=time.monotonic(); model.train(); totals={}; samples=0; gradient_probe=None
        for ids in np.array_split(rng.permutation(len(splits['train']['state'])),max(1,int(np.ceil(len(splits['train']['state'])/config['batch_size'])))):
            batch={k:v[torch.as_tensor(ids,device=device)] for k,v in splits['train'].items()}
            loss,parts=model.loss(**batch,weights=config['loss_weights'])
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite latent training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if gradient_probe is None:
                gradient_probe={name:float(sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt().detach().cpu()) for name,module in [('encoder',model.encoder),('dynamics',model.dynamics),('decoder',model.decoder)]}
                if any(not np.isfinite(v) or v<=0 for v in gradient_probe.values()): raise FloatingPointError('a learned component received no finite nonzero gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']); optimizer.step()
            for k,v in {'total':loss,**parts}.items(): totals[k]=totals.get(k,0)+float(v.detach().cpu())*len(ids)
            samples+=len(ids)
        model.eval(); validation={}; count=0
        with torch.no_grad():
            for offset in range(0,len(splits['val']['state']),config['batch_size']):
                batch={k:v[offset:offset+config['batch_size']] for k,v in splits['val'].items()}; n=len(batch['state'])
                loss,parts=model.loss(**batch,weights=config['loss_weights'])
                for k,v in {'total':loss,**parts}.items(): validation[k]=validation.get(k,0)+float(v.cpu())*n
                count+=n
            latent_std=model.encode(splits['val']['state']).std(0,unbiased=False).cpu().tolist()
        validation={k:v/count for k,v in validation.items()}
        if not all(np.isfinite(v) for v in validation.values()) or max(latent_std)<1e-8: raise FloatingPointError('nonfinite validation or numerically constant latent; retain failure, do not retune on test')
        if validation['total']<best-config['early_stop']['min_delta']:
            best=validation['total']; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        row={'epoch':epoch+1,'train':{k:v/samples for k,v in totals.items()},'validation':validation,'latent_std':latent_std,
             'first_batch_component_gradient_norms':gradient_probe,'wall_seconds':time.monotonic()-begin}
        history.append(row); run.event('training_epoch',model_seed=seed,**row)
        saved={'identity':identity,'normalizers':normalizers,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'best':best,'best_state':best_state,'best_epoch':best_epoch,'stale':stale,'history':history,'epoch':epoch,
               'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        checkpoint=directory/f'resume-epoch-{epoch:04d}.pt'; _save(checkpoint,saved)
        run.event('checkpoint',model_seed=seed,epoch=epoch+1,path=str(checkpoint),sha256=sha256(checkpoint))
        if epoch+1>=config['early_stop']['minimum_epochs'] and stale>=config['early_stop']['patience']: break
    if best_state is None: raise RuntimeError('no validation-selected checkpoint')
    model.load_state_dict(best_state); model.eval(); checkpoint=directory/'small-latent-model.pt'
    _save(checkpoint,{'schema':'fetch-small-latent-v2','identity':identity,'normalizers':normalizers,'model':model.state_dict(),
          'hidden':64,'latent':8,'model_seed':seed,'control_dt':dataset.audit['control_dt'],'env_contract_sha256':dataset.audit['env_contract_sha256'],
          'dataset_audit':dataset.audit,'best_epoch':best_epoch+1,'best_validation_total':best})
    return {'model_seed':seed,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'dataset_identity':dataset.identity,
            'normalizers_sha256':digest(normalizers),'best_epoch':best_epoch+1,'best_validation_total':best,'epochs_completed':len(history),
            'history':history,'selection':'validation total next-state/reconstruction/latent loss only; test unused','device':str(device),
            'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
def load_model(spec,device='cpu'):
    verify_file(spec['path'],spec['sha256']); p=torch.load(spec['path'],map_location=device,weights_only=False)
    if p.get('schema')!='fetch-small-latent-v2': raise ValueError('small trained latent checkpoint required')
    for name in ('dynamics.py','protocol.py'):
        if p['identity']['source_sha256']['wm/'+name]!=sha256(Path(__file__).with_name(name)): raise ValueError('trained model/state source changed')
    if 'model_seed' in spec and spec['model_seed']!=p['model_seed']: raise ValueError('actual checkpoint seed mismatch')
    model=SmallLatentDynamics(p['normalizers'],p['hidden'],p['latent']).to(device); model.load_state_dict(p['model'],strict=True); model.eval()
    return model,p
