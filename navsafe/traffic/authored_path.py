"""Shared, deterministic motion for the BEV editor and actual traffic runtime."""
from __future__ import annotations
import math
import numpy as np
from navsafe.traffic.navsafe import register_driver
from navsafe.traffic.stage_free import Driver
from navsafe.traffic.timed_braking import BrakeProfile


def smooth_path(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1] not in (2,3) or not 2<=len(p)<=200:
        raise ValueError("A path needs 2–200 points")
    if not np.isfinite(p).all() or np.abs(p).max()>100000:
        raise ValueError("Invalid path coordinates")
    if p.shape[1]==2: p=np.column_stack([p,np.zeros(len(p))])
    if np.any(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)<.05):
        raise ValueError("Consecutive path points must be at least 5 cm apart")
    result=[]
    for i in range(len(p)-1):
        a=p[max(0,i-1)]; b=p[i]; c=p[i+1]; d=p[min(len(p)-1,i+2)]
        # Hermite endpoints pass through every point placed on the BEV.
        m0=(c-a)/2 if i else c-b
        m1=(d-b)/2 if i+2<len(p) else c-b
        n=max(3,min(500,int(np.linalg.norm(c-b)/.25)+1))
        for u in np.linspace(0,1,n,endpoint=False):
            result.append((2*u**3-3*u**2+1)*b+(u**3-2*u**2+u)*m0
                          +(-2*u**3+3*u**2)*c+(u**3-u**2)*m1)
    result.append(p[-1])
    return np.asarray(result)


class AuthoredMotion:
    def __init__(self,policy):
        self.policy=dict(policy)
        self.path=np.asarray(policy["path_polyline"],dtype=float)
        if self.path.ndim!=2 or self.path.shape[1]!=3 or len(self.path)<2:
            raise ValueError("authored_path requires an XYZ path")
        if not np.isfinite(self.path).all(): raise ValueError("nonfinite authored path")
        self.arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.path[:,:2],axis=0),axis=1))]
        if self.arc[-1]<.05 or np.any(np.diff(self.arc)<=0):
            raise ValueError("path contains repeated points")
        self.kind=policy["event_kind"]
        if self.kind not in ("cut_in","lead_brake","vru_cross"): raise ValueError("unknown event")
        self.onset=float(policy["onset_s"]); self.speed=float(policy["speed"])
        if not all(math.isfinite(x) for x in (self.onset,self.speed)) or self.onset<0 or not 0<self.speed<=35:
            raise ValueError("invalid onset/speed")
        self.enabled=bool(policy.get("enabled",True))
        self.brake=BrakeProfile(speed=self.speed,onset_s=0,
            deceleration=float(policy.get("deceleration",3)),
            speed_reduction=float(policy.get("speed_reduction",self.speed)),
            hold_s=float(policy.get("hold_s",1)),
            recovery_acceleration=float(policy.get("recovery_acceleration",2)),
            recovery_speed=(float(policy["recovery_speed"])
                            if policy.get("recovery_speed") is not None else None),
            enabled=self.enabled)

    def sample(self,t):
        if not math.isfinite(t) or t<0: raise ValueError("invalid motion time")
        elapsed=t-self.onset
        if elapsed<=0:
            distance=0 if self.kind=="vru_cross" else elapsed*self.speed
            speed=0 if self.kind=="vru_cross" else self.speed
        elif self.kind=="lead_brake":
            distance,speed,_=self.brake.sample(elapsed)
        elif self.kind=="vru_cross" and not self.enabled:
            distance,speed=0,0
        else: distance,speed=elapsed*self.speed,self.speed
        first=self.path[1]-self.path[0]
        first=first/np.linalg.norm(first[:2])
        exhausted=distance>self.arc[-1]+1e-6
        if self.kind=="cut_in" and not self.enabled:
            xyz=self.path[0]+distance*first; direction=first; exhausted=False
        elif distance<0:
            xyz=self.path[0]+distance*first; direction=first
        else:
            s=min(distance,self.arc[-1])
            i=min(int(np.searchsorted(self.arc,s,side="right"))-1,len(self.path)-2)
            i=max(0,i)
            ratio=(s-self.arc[i])/(self.arc[i+1]-self.arc[i])
            xyz=self.path[i]+ratio*(self.path[i+1]-self.path[i])
            direction=self.path[i+1]-self.path[i]
        heading=math.atan2(direction[1],direction[0])
        if exhausted: speed=0
        return dict(position=xyz,heading=heading,
                    velocity=speed*np.array([math.cos(heading),math.sin(heading)]),
                    speed=float(speed),exhausted=bool(exhausted))


@register_driver("authored_path")
class AuthoredPathDriver(Driver):
    strict=True
    def __init__(self,agent_id,spawn,*,policy=None,**_):
        self.agent_id=agent_id; self.spawn=dict(spawn)
        self.motion=AuthoredMotion(policy or {})
        self.reset(spawn=self.spawn)
    def reset(self,*,spawn):
        self.time_s=0.; self.length=float(spawn.get("length",4.5)); self.width=float(spawn.get("width",1.8))
        expected=self.motion.sample(0)["position"]
        if not np.allclose(np.asarray(spawn["position"])[:2],expected[:2],atol=1e-4):
            raise ValueError("spawn does not match authored trajectory at t=0")
    def step(self,world,dt):
        self.time_s=float(world.t)*float(dt)
        state=self.motion.sample(self.time_s)
        if state["exhausted"] and self.motion.kind!="vru_cross":
            raise ValueError("The vehicle's event path is exhausted; extend the path in the BEV editor")
    def pose(self):
        state=self.motion.sample(self.time_s)
        return {**state,"length":self.length,"width":self.width}
