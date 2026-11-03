"""Reduced BC policy loader; MPC uses the explicit external adapter seam."""
def load_policy(kind,checkpoint=None,device='cpu'):
    if kind in ('bc','bc_dr'):
        from .bc import ActorPolicy,load_actor
        actor,payload=load_actor(checkpoint,device)
        if payload['method']!=kind: raise ValueError('checkpoint condition differs from requested BC kind')
        return ActorPolicy(actor,device,payload)
    raise ValueError('only actual reduced BC checkpoints are supported')
