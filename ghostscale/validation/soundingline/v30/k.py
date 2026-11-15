"""K finite gap scheduling; exact matched policies and independent retained-seed reconstruction."""
from pathlib import Path
import numpy as np
from numba import njit
from .common import rng,mean_blocks,censored_blocks,audit_controls,write

@njit
def uniform(seed,counter):
    z=np.uint64(seed)+np.uint64(counter)*np.uint64(0x9e3779b97f4a7c15)
    z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    z=z^(z>>np.uint64(31))
    return float(z>>np.uint64(11))*(1./9007199254740992.)

def reference_uniform(seed,counters):
    with np.errstate(over="ignore"):
        z=np.uint64(seed)+np.asarray(counters,dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)
        z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        z^=z>>np.uint64(31)
    return (z>>np.uint64(11)).astype(float)/9007199254740992.

def settings():
    return [dict(arrival=a,opening=w,q=q,cue=r,deadline=deadline,steps=3000,worlds=200,blocks=8,
                 policy=policy,work_sizes=[3]) for a in (.02,.05,.15) for w in (.3,.1)
            for deadline in (False,True) for policy,q,r in
            [("NOPLAN",0.,0.)]+[("PLAN",q,r) for q in (1.,.75,.5) for r in (.9,.6,.3,0.)]]

@njit
def simulate(seeds,T,arrival,opening,q,cue,planned,deadline,sizes):
    W=len(seeds);history=np.zeros((W,T,8));births=np.full((W,T),-1,np.int32)
    finished=np.full((W,T),-1,np.int32);needed=np.zeros((W,T),np.int16)
    for world in range(W):
        remaining=np.zeros(T,np.int16);quiet=np.zeros(T,np.bool_);expired=np.zeros(T,np.bool_);count=0
        for t in range(T):
            new=-1
            if uniform(seeds[world],4*t)<arrival:
                new=count;births[world,count]=t;needed[world,count]=sizes[count%len(sizes)]
                remaining[count]=needed[world,count];count+=1
            stress=0.;open_count=0;penalty=0.;attended=-1
            for j in range(count):
                if remaining[j]<=0:continue
                open_count+=1
                if deadline and quiet[j] and not expired[j] and t-births[world,j]>=60:
                    expired[j]=True;penalty+=3.
                s=1-q if quiet[j] and not expired[j] else 1.
                stress+=s
            # Planning is immediate on arrival, consumes the one available step.
            if planned and new>=0:
                quiet[new]=True;history[world,t,0]=new+1;history[world,t,2]=new+1
                history[world,t,3]=open_count;history[world,t,4]=stress
                history[world,t,7]=.5*open_count+penalty
                continue
            if uniform(seeds[world],4*t+1)<1-np.exp(-.3*stress):
                for j in range(count):
                    if remaining[j]>0 and (not quiet[j] or expired[j] or q<1):
                        attended=j;break
            work=-1
            if attended>=0:
                base=4*T+2*(t*T+attended)
                if uniform(seeds[world],base)<opening:work=attended
            if work<0 and planned:
                for j in range(count):
                    if remaining[j]<=0 or not quiet[j]:continue
                    base=4*T+2*(t*T+j)
                    if uniform(seeds[world],base)<opening and uniform(seeds[world],base+1)<cue:
                        work=j;break
            closed=0
            if work>=0:
                remaining[work]-=1
                if remaining[work]==0:finished[world,work]=t;closed=1
            history[world,t,0]=attended+1;history[world,t,1]=work+1
            history[world,t,3]=open_count;history[world,t,4]=stress
            history[world,t,5]=1. if attended<0 and work<0 else 0.
            history[world,t,6]=closed;history[world,t,7]=.5*open_count+penalty
    return history,births,finished,needed

def reference_simulate(seeds,s):
    T=s["steps"];hist=np.zeros((len(seeds),T,8));births=np.full((len(seeds),T),-1,np.int32)
    finished=np.full_like(births,-1);needed=np.zeros_like(births,dtype=np.int16)
    for world,seed in enumerate(seeds):
        rem=[];quiet=[];expired=[];born=[];plan=s["policy"]=="PLAN"
        for t in range(T):
            arrival,attend=reference_uniform(seed,[4*t,4*t+1]);new=-1
            if arrival<s["arrival"]:
                new=len(rem);born.append(t);rem.append(s["work_sizes"][new%len(s["work_sizes"])])
                quiet.append(False);expired.append(False);births[world,new]=t;needed[world,new]=rem[-1]
            remarray=np.asarray(rem);live=remarray>0;penalty=0.
            for j in np.flatnonzero(live):
                if s["deadline"] and quiet[j] and not expired[j] and t-born[j]>=60:
                    expired[j]=True;penalty+=3
            stresses=np.array([1-s["q"] if quiet[j] and not expired[j] else 1 for j in range(len(rem))])*live
            stress=float(stresses.sum());count=int(live.sum());row=hist[world,t]
            row[3]=count;row[4]=stress;row[7]=.5*count+penalty
            if plan and new>=0:quiet[new]=True;row[0]=new+1;row[2]=new+1;continue
            at=-1
            if attend<1-np.exp(-.3*stress):
                candidates=np.flatnonzero(stresses>0)
                if len(candidates):at=int(candidates[0])
            ids=np.arange(len(rem));window=reference_uniform(seed,4*T+2*(t*T+ids))<s["opening"]
            work=at if at>=0 and window[at] else -1
            if work<0 and plan:
                caught=reference_uniform(seed,4*T+2*(t*T+ids)+1)<s["cue"]
                candidates=np.flatnonzero(live&np.array(quiet,bool)&window&caught)
                if len(candidates):work=int(candidates[0])
            if work>=0:
                rem[work]-=1
                if rem[work]==0:finished[world,work]=t;row[6]=1
            row[0]=at+1;row[1]=work+1;row[5]=int(at<0 and work<0)
    return hist,births,finished,needed

def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("K",lane,{k:v for k,v in s.items() if k not in ("q","cue","policy","deadline")})
    seeds=r.integers(1,2**63,size=s["worlds"],dtype=np.int64)
    h,b,f,n=simulate(seeds,s["steps"],s["arrival"],s["opening"],s["q"],s["cue"],
                     s["policy"]=="PLAN",s["deadline"],np.array(s["work_sizes"]))
    pulse(phase="generated",worlds=s["worlds"])
    return dict(seeds=seeds,history=h,births=b,finished=f,needed=n)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    if reference:
        h,b,f,n=reference_simulate(raw["seeds"],s)
        if not all(np.array_equal(x,y) for x,y in zip((h,b,f,n),(raw["history"],raw["births"],raw["finished"],raw["needed"]))):
            raise ArithmeticError("independent gap trajectory differs")
    else:h,b,f,n=raw["history"],raw["births"],raw["finished"],raw["needed"]
    active=b>=0;closed=f>=0;delays=f-b+1;count=active.sum(1)
    return dict(closed_per_1000=h[:,:,6].sum(1)*1000/s["steps"],
        mean_stress=h[:,:,4].mean(1),other_work=h[:,:,5].sum(1),attention_lost=(h[:,:,0]>0).mean(1),
        cue_or_attention_work_share=(h[:,:,1]>0).mean(1),gap_cost=h[:,:,7].sum(1),
        net_other_work_minus_gap_cost=h[:,:,5].sum(1)-h[:,:,7].sum(1),
        open_at_end=(active&~closed).sum(1),arrivals=count,
        _closure_delay=np.where(closed,delays,s["steps"]+1),_active_gap=active,_closed_gap=closed,
        _followup=np.where(closed,delays,s["steps"]-b),
        _attention=(h[:,:,0]>0).astype(np.int8))

def summarize(s,m):
    rows=[dict(measure=k,**mean_blocks(v,s["blocks"])) for k,v in m.items() if not k.startswith("_")]
    counts=m["_active_gap"].sum(1);closed=m["_closed_gap"].sum(1)
    # Whole-world blocks, including variable numbers of nested gaps.
    block_count=counts.reshape(s["blocks"],-1).sum(1);block_closed=closed.reshape(s["blocks"],-1).sum(1)
    draws=rng("K","closure-bootstrap").integers(s["blocks"],size=(2000,s["blocks"]))
    rates=block_closed[draws].sum(1)/np.maximum(1,block_count[draws].sum(1))
    rows.append(dict(measure="gap_closed_fraction",estimate=float(closed.sum()/max(1,counts.sum())),
                     interval95=np.quantile(rates,[.025,.975]).tolist(),gaps=int(counts.sum()),
                     block_gap_counts=block_count.tolist(),unclosed_gaps=int((counts-closed).sum())))
    # Kaplan-Meier with arrival-specific censoring; resample whole maker blocks.
    T=s["steps"];event_hist=np.zeros((s["blocks"],T+1));end_hist=np.zeros_like(event_hist)
    perblock=s["worlds"]//s["blocks"]
    for world in range(s["worlds"]):
        active=m["_active_gap"][world];dur=m["_followup"][world,active].astype(int)
        event=m["_closed_gap"][world,active];block=world//perblock
        event_hist[block]+=np.bincount(dur[event],minlength=T+1)
        end_hist[block]+=np.bincount(dur,minlength=T+1)
    def km_median(events,ends):
        risk=np.cumsum(ends[::-1])[::-1]
        hazard=np.divide(events,risk,out=np.zeros_like(events),where=risk>0)
        survival=np.cumprod(1-hazard);cross=np.flatnonzero(survival<=.5)
        return int(cross[0]) if len(cross) else T+1
    central=km_median(event_hist.sum(0),end_hist.sum(0))
    medians=sorted(km_median(event_hist[d].sum(0),end_hist[d].sum(0)) for d in draws)
    encode=lambda x:"unresolved" if x>T else int(x)
    rows.append(dict(measure="gap_closing_median_Kaplan_Meier",estimate=encode(central),
        interval95=[encode(medians[50]),encode(medians[1949])],blocks=s["blocks"],
        censoring="each unfinished gap censored at its own observed age"))
    # Conditional completion times explicitly exclude unfinished gaps.
    mask=m["_closed_gap"];values=m["_closure_delay"]
    sums=np.where(mask,values,0).reshape(s["blocks"],-1).sum(1)
    den=block_closed[draws].sum(1);num=sums[draws].sum(1);sample=num[den>0]/den[den>0]
    rows.append(dict(measure="conditional_mean_completed_gap_steps",
        estimate=float(sums.sum()/closed.sum()) if closed.sum() else None,
        interval95=np.quantile(sample,[.025,.975]).tolist() if len(sample) else [None,None],
        completed_gaps=int(closed.sum()),unclosed_gaps=int((counts-closed).sum())))
    return rows

def pilot_exclusion(s,m):return None

def controls(dest=None,pulse=lambda **kw:None):
    checks=[];seed=12345;u=np.array([uniform(seed,j) for j in range(100)])
    checks.append(dict(name="independent counter random arithmetic",passed=bool(np.array_equal(u,reference_uniform(seed,np.arange(100))))))
    for q,r,deadline in ((1.,0.,False),(1.,.9,False),(.5,.3,True),(1.,0.,True)):
        s=dict(arrival=.15,opening=.3,q=q,cue=r,deadline=deadline,steps=128,worlds=8,blocks=8,policy="PLAN",work_sizes=[3])
        raw=generate(s,"known-answer");a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=f"independent full trajectory q={q},r={r},deadline={deadline}",
                           passed=all(np.array_equal(a[k],b[k]) for k in a)))
        if q==1 and r==0 and not deadline:checks.append(dict(name="fully quiet uncued plans never close",
                                                             passed=bool(np.all(a["closed_per_1000"]==0))))
    checks.append(dict(name="singleton attention law",passed=abs(1-np.exp(-.3)-.2591817793182821)<1e-14))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
