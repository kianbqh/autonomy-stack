"""Command-space projection and a separate physical-path checker, all SI units."""
from __future__ import annotations
import math

def norm(v):
    return math.sqrt(sum(float(x)**2 for x in v))

def distance_segment(point, start, end):
    delta = [e-s for s, e in zip(start, end)]
    den = sum(x*x for x in delta)
    alpha = max(0.0, min(1.0, sum((p-s)*d for p,s,d in zip(point,start,delta))/den)) if den else 0.0
    return norm([s+alpha*d-p for s,d,p in zip(start,delta,point)])

class Shield:
    def __init__(self, config, dt):
        self.config, self.dt = config, float(dt)

    def violations(self, previous, current, sigma=0.0, seconds=None):
        c = self.config
        margin = c["end_effector_radius"] + c["k_sigma"]*sigma
        failures = []
        if any(x < lo+margin or x > hi-margin for x,lo,hi in zip(current,c["workspace_low"],c["workspace_high"])):
            failures.append("workspace")
        if norm([b-a for a,b in zip(previous,current)]) > c["max_speed_m_s"]*(seconds or self.dt)+1e-10:
            failures.append("speed")
        for sphere in c["keepout_spheres"]:
            if distance_segment(sphere["center"],previous,current) < sphere["radius"]+margin:
                failures.append("keepout")
        return failures

    def project(self, action, measured_state, uncertainty=0.0, pending_actuation=None):
        """Input is final gained/delayed native action; no queue is permitted after this call."""
        p = [float(x) for x in measured_state[:3]]
        a = [max(-1.0,min(1.0,float(x))) for x in action]
        scale = self.config["native_cartesian_scale_m"]
        delta = [scale*x for x in a[:3]]
        limit = self.config["max_speed_m_s"]*self.dt
        length = norm(delta)
        if length > limit:
            delta = [x*limit/length for x in delta]
        margin = self.config["end_effector_radius"] + self.config["k_sigma"]*uncertainty
        already_unsafe = bool(self.violations(p,p,uncertainty))
        if already_unsafe:
            return [0.0,0.0,0.0,a[3]], {"fallback": "already_unsafe_hold", "infeasible": True,
                                        "already_unsafe": True, "pending_actuation": pending_actuation}
        end = [max(lo+margin,min(hi-margin,x+d)) for x,d,lo,hi in zip(p,delta,self.config["workspace_low"],self.config["workspace_high"])]
        # Workspace is convex; for nonconvex sphere exclusions shorten along the original ray.
        if self.violations(p,end,uncertainty):
            lo,hi = 0.0,1.0
            for _ in range(50):
                mid=(lo+hi)/2
                trial=[x+mid*(y-x) for x,y in zip(p,end)]
                if self.violations(p,trial,uncertainty):
                    hi=mid
                else:
                    lo=mid
            end=[x+lo*(y-x) for x,y in zip(p,end)]
        projected=[(y-x)/scale for x,y in zip(p,end)]+[a[3]]
        return projected,{"fallback": None, "infeasible": False, "already_unsafe": False,
                          "changed": norm([x-y for x,y in zip(projected,a)])>1e-10,
                          "pending_actuation": pending_actuation}
