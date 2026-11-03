"""One root-accepted GPU-batched latent CEM configuration; no ensemble sampling."""
import time
import numpy as np
import torch
from .protocol import child_seed,formal_planner
def synchronize(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)
def stage_cost(states,actions,goal,penalty):
    return (states[...,3:6]-goal).norm(dim=-1)+penalty*actions.square().sum(-1)
class Planner:
    def __init__(self,config,device='cpu',formal=True):
        if formal: formal_planner(config)
        self.config=dict(config); self.device=torch.device(device); self.warm=None
    def reset(self): self.warm=None
    @torch.no_grad()
    def plan(self,state,goal,model,seed,remaining_steps=50):
        c=self.config; h=min(c['horizon'],int(remaining_steps))
        if h<=0: raise ValueError('episode must reset after50 actions')
        synchronize(self.device); whole=time.perf_counter()
        state=torch.as_tensor(state,device=self.device,dtype=torch.float32).reshape(1,18)
        goal=torch.as_tensor(goal,device=self.device,dtype=torch.float32).reshape(3)
        synchronize(self.device); start=time.perf_counter(); n=c['candidates']
        rng=np.random.default_rng(child_seed(seed,'candidate')); initial=model.encode(state)
        mean=torch.zeros((h,4),device=self.device)
        if self.warm is not None: mean[:min(h,len(self.warm))]=self.warm[:h]
        std=torch.full_like(mean,c['initial_std']); best_cost=float('inf'); sequence=None; iterations=[]
        for _ in range(c['iterations']):
            noise=torch.as_tensor(rng.standard_normal((n,h,4)),device=self.device,dtype=torch.float32)
            candidates=(mean+std*noise).clamp(-1,1); z=initial.expand(n,-1).clone(); cost=torch.zeros(n,device=self.device)
            for step in range(h):
                z=model.latent_step(z,candidates[:,step]); predicted=model.decode(z)
                cost+=stage_cost(predicted,candidates[:,step],goal,c['action_penalty'])
            if not torch.isfinite(cost).all(): raise FloatingPointError('nonfinite learned latent rollout cost')
            elite_cost,ids=torch.topk(cost,c['elites'],largest=False,sorted=True); elites=candidates[ids]
            mean=elites.mean(0); std=elites.std(0,unbiased=False).clamp_min(c['minimum_std'])
            value=float(elite_cost[0].cpu()); iterations.append(value)
            if value<best_cost: best_cost=value; sequence=candidates[ids[0]].clone()
        synchronize(self.device); planner_ms=1000*(time.perf_counter()-start)
        self.warm=torch.cat((sequence[1:],torch.zeros_like(sequence[:1])),0); action=sequence[0].cpu().numpy().copy()
        synchronize(self.device)
        return action,{'planner_ms':planner_ms,'host_policy_ms':1000*(time.perf_counter()-whole),'effective_horizon':h,
            'candidates':n,'iterations':c['iterations'],'elites':c['elites'],'predicted_latent_transitions':n*h*c['iterations'],
            'best_cost':best_cost,'iteration_best_costs':iterations,'seed':int(seed),'scheme':'deterministic latent rollout; one model per seed',
            'timer':'CUDA synchronized before/after; planner includes encoder,NumPy candidate generation/H2D,latent rollout,decode,cost,selection; excludes observation H2D/action D2H;host_policy includes both'}
