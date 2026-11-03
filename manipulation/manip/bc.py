"""Matched nominal/DR behavior-cloning policy; no RL actor or stochastic head."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .protocol import split_for_reset
def features(obs):
    return np.concatenate((obs['observation'],obs['desired_goal']),axis=-1).astype(np.float32)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(28,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,4),nn.Tanh())
    def deterministic(self,x): return self.net(x)
class ActorPolicy:
    def __init__(self,actor,device='cpu',metadata=None):
        self.actor,self.device=actor.to(device),device; self.actor.eval()
        self.model_seed=(metadata or {}).get('model_seed')
    def reset(self,seed=None): pass
    def act(self,obs,deterministic=True):
        with torch.no_grad():
            x=torch.as_tensor(features(obs),dtype=torch.float32,device=self.device).unsqueeze(0)
            return self.actor.deterministic(x)[0].cpu().numpy()
def load_actor(path,device='cpu'):
    p=torch.load(path,map_location=device,weights_only=False)
    if p.get('schema')!='fetch-bc-v2' or p.get('method') not in ('bc','bc_dr'):
        raise ValueError('a reduced-protocol trained BC checkpoint is required')
    ids=p['split_reset_ids']
    if any(not ids[k] for k in ('train','val')) or any(split_for_reset(s)!=k for k,ss in ids.items() for s in ss):
        raise ValueError('checkpoint fit/validation reset identity is invalid')
    if p['demo_budget']!=100 or p['normalizer']!='identity': raise ValueError('wrong BC main protocol')
    actor=Actor().to(device); actor.load_state_dict(p['actor'],strict=True)
    return actor,p
def fit_bc(train_x,train_y,val_x,val_y,config,device,checkpoint_callback):
    if not len(train_x) or not len(val_x): raise ValueError('nonempty grouped fit/validation required')
    actor=Actor().to(device); optimizer=torch.optim.Adam(actor.parameters(),lr=config['learning_rate'])
    x,y,vx,vy=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in (train_x,train_y,val_x,val_y)]
    history=[]; best=float('inf')
    for epoch in range(config['epochs']):
        actor.train(); total=0.; count=0
        for batch in torch.randperm(len(x),device=device).split(config['batch_size']):
            loss=(actor.deterministic(x[batch])-y[batch]).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite BC loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total+=float(loss.detach().cpu())*len(batch); count+=len(batch)
        actor.eval()
        with torch.no_grad(): validation=float((actor.deterministic(vx)-vy).square().mean().cpu())
        history.append({'epoch':epoch+1,'train_mse':total/count,'validation_mse':validation})
        if validation<best:
            best=validation; checkpoint_callback(epoch+1,actor,optimizer,validation)
    return actor,history
