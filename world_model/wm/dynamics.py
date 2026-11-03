"""Small actually trainable bottleneck encoder, action-conditioned dynamics and decoder."""
import numpy as np
import torch
from torch import nn
class Normalizer(nn.Module):
    def __init__(self,mean,std):
        super().__init__(); self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32)); self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(1e-6))
    def forward(self,x): return (x-self.mean)/self.std
    def inverse(self,x): return x*self.std+self.mean
class SmallLatentDynamics(nn.Module):
    def __init__(self,normalizers,hidden=64,latent=8):
        super().__init__()
        if hidden!=64 or latent!=8: raise ValueError('main architecture is fixed64hidden/8latent')
        self.state_norm=Normalizer(**normalizers['state']); self.action_norm=Normalizer(**normalizers['action'])
        self.encoder=nn.Sequential(nn.Linear(18,hidden),nn.SiLU(),nn.Linear(hidden,latent))
        self.dynamics=nn.Sequential(nn.Linear(latent+4,hidden),nn.SiLU(),nn.Linear(hidden,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.SiLU(),nn.Linear(hidden,18))
    def encode(self,state): return self.encoder(self.state_norm(state))
    def latent_step(self,latent,action):
        return latent+self.dynamics(torch.cat((latent,self.action_norm(action)),dim=-1))
    def decode(self,latent):
        state=self.state_norm.inverse(self.decoder(latent)); angles=torch.remainder(state[...,6:9]+torch.pi,2*torch.pi)-torch.pi
        return torch.cat((state[...,:6],angles,state[...,9:]),dim=-1)
    def forward(self,state,action): return self.decode(self.latent_step(self.encode(state),action))
    def normalized_error(self,predicted,target):
        d=predicted-target; angular=torch.remainder(d[...,6:9]+torch.pi,2*torch.pi)-torch.pi
        return torch.cat((d[...,:6],angular,d[...,9:]),dim=-1)/self.state_norm.std
    def loss(self,state,action,clean_state,next_state,weights):
        latent=self.encode(state); target_latent=self.encode(next_state); predicted_latent=self.latent_step(latent,action)
        next_loss=self.normalized_error(self.decode(predicted_latent),next_state).square().mean()
        current_reconstruction=self.normalized_error(self.decode(latent),clean_state).square().mean()
        next_reconstruction=self.normalized_error(self.decode(target_latent),next_state).square().mean()
        reconstruction=(current_reconstruction+next_reconstruction)/2
        latent_consistency=(predicted_latent-target_latent.detach()).square().mean()
        total=next_loss+weights['reconstruction']*reconstruction+weights['latent_consistency']*latent_consistency
        return total,{'next_state_mse':next_loss,'reconstruction_mse':reconstruction,'latent_consistency_mse':latent_consistency}
def fit_normalizers(train):
    # Only fit-split visible states and supervised clean labels enter these statistics.
    state=np.concatenate((train['state'],train['clean_state'],train['next_state']),axis=0)
    return {k:{'mean':v.mean(0).tolist(),'std':v.std(0,ddof=0).clip(1e-6).tolist()} for k,v in (('state',state),('action',train['action']))}
