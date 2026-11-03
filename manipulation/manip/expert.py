"""Deterministic waypoint push controller; its success is an empirical runtime gate."""
import math

class PushExpert:
    def reset(self, seed=None):
        self.phase = "approach"

    def __init__(self, config=None):
        self.config = config or {"behind_m": 0.055, "hover_m": 0.08, "position_gain": 20.0,
                                 "alignment_m": 0.018, "push_ahead_m": 0.05}
        self.reset()

    def act(self, obs, deterministic=True):
        v,goal=obs["observation"],obs["desired_goal"]
        grip,obj=list(v[:3]),list(v[3:6])
        line=[goal[i]-obj[i] for i in range(2)]
        d=math.hypot(*line)
        direction=[x/max(d,1e-9) for x in line]
        behind=[obj[i]-self.config["behind_m"]*direction[i] for i in range(2)]+[obj[2]+0.006]
        lateral=math.hypot(*(grip[i]-behind[i] for i in range(2)))
        if d < 0.018:
            target=grip
        elif self.phase == "approach":
            target=behind[:2]+[obj[2]+self.config["hover_m"]]
            if lateral < self.config["alignment_m"]:
                self.phase="descend"
        elif self.phase == "descend":
            target=behind
            if abs(grip[2]-behind[2])<0.012 and lateral<0.025:
                self.phase="push"
        else:
            # Re-align after lateral slippage; never descend through the object.
            to_grip=[grip[i]-obj[i] for i in range(2)]
            cross=abs(to_grip[0]*direction[1]-to_grip[1]*direction[0])
            if cross>0.045 or sum(to_grip[i]*direction[i] for i in range(2))>0.02:
                self.phase="approach"
                target=grip[:2]+[obj[2]+self.config["hover_m"]]
            else:
                target=[obj[i]+self.config["push_ahead_m"]*direction[i] for i in range(2)]+[obj[2]+0.006]
        return [max(-1.,min(1.,self.config["position_gain"]*(t-p))) for t,p in zip(target,grip)]+[0.0]
