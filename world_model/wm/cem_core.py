"""Pure-list CEM used only for meaningful algorithmic CPU tests."""
import math
import random

def cem(objective,dimension,seed=0,candidates=500,elites=50,iterations=5,lower=-1,upper=1):
    if not 0<elites<candidates or dimension<=0: raise ValueError("invalid CEM dimensions")
    rng=random.Random(seed); mean=[0.0]*dimension; std=[0.5]*dimension
    best=None; best_cost=float("inf")
    for _ in range(iterations):
        xs=[[min(upper,max(lower,rng.gauss(m,s))) for m,s in zip(mean,std)] for _ in range(candidates)]
        scores=[float(objective(x)) for x in xs]
        if not all(math.isfinite(v) for v in scores): raise ValueError("nonfinite objective")
        ids=sorted(range(candidates),key=lambda i:scores[i])[:elites]
        if scores[ids[0]]<best_cost: best,best_cost=xs[ids[0]],scores[ids[0]]
        mean=[sum(xs[i][j] for i in ids)/elites for j in range(dimension)]
        std=[max(0.01,math.sqrt(sum((xs[i][j]-mean[j])**2 for i in ids)/elites)) for j in range(dimension)]
    return best,best_cost
