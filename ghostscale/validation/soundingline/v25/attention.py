"""W7: exact marginal likelihoods for final units, attention and replaced proposals."""
import itertools,math
import numpy as np
from .common import rng,posterior,scores,record,close

GOALS=list(itertools.combinations(range(1,7),3))
HABITS=np.array([[1]*8,[1,2,3,4,1,2,3,4],[4,1,4,1,4,1,4,1]],float);HABITS/=HABITS.sum(1)[:,None]
HYPOTHESES=list(itertools.product(range(20),range(3)))

def likelihood(final,attention,proposal,start,epsilon,reader,independent=False,fixture_habit=None):
    values=[]
    for g,h in HYPOTHESES:
        habit=HABITS[h] if fixture_habit is None else fixture_habit
        veto=GOALS[g];v=math.fsum(float(habit[j]) for j in veto) if independent else float(habit[list(veto)].sum())
        if final in veto:values.append(0.);continue
        pf=float(habit[final]/(1-v));other=1-(1-epsilon)*(1-(.5 if start else 0))
        if reader=='final':value=pf
        elif reader in ('attention','misread'):
            pa=v+(1-v)*(0 if reader=='misread' else other);value=pf*(pa if attention else 1-pa)
        else:
            if proposal is not None:value=float(habit[proposal])*pf if proposal in veto and attention else 0.
            else:value=float(habit[final])*(other if attention else 1-other)
        values.append(value)
    return np.array(values)

def block(setting,identity,independent=False):
    r=rng(*identity);out=[];raw=[];epsilon=setting['epsilon'];h=setting['habit']
    for maker in range(8):
        g=int(r.integers(20));habit=HABITS[h];veto=GOALS[g];allowed=habit.copy();allowed[list(veto)]=0;allowed/=allowed.sum()
        logs={k:np.zeros(60) for k in ('final','attention','revision','misread')};history=[];violations=0;attended=0
        for t in range(64):
            proposal=int(r.choice(8,p=habit));violation=proposal in veto;start=t%4==0
            final=int(r.choice(8,p=allowed)) if violation else proposal
            other=bool(r.random()<epsilon or (start and r.random()<.5));attention=violation or other
            violations+=int(violation);attended+=int(attention)
            row=dict(final=final,attention=bool(attention),proposal=proposal if violation else None,start=start);history.append(row)
            for reader in logs:
                law=likelihood(**row,epsilon=epsilon,reader=reader,independent=independent)
                with np.errstate(divide='ignore'):logs[reader]+=np.log(law)
            if t+1 in (4,8,16,32,64):
                ps={k:posterior(v,independent=independent).reshape(20,3).sum(1) for k,v in logs.items()}
                for reader,p in ps.items():out.append(record(scores(p,g),maker=maker,reader=reader,history=t+1))
                delta=math.log(ps['attention'][g])-math.log(ps['final'][g]);mis=math.log(ps['misread'][g])-math.log(ps['final'][g]);revision=math.log(ps['revision'][g])-math.log(ps['attention'][g])
                out.append(record(dict(primary=delta,misread_gain=mis,revision_gain=revision,violation_events=violations,attention_events=attended),maker=maker,reader='contrasts',history=t+1))
        raw.append(dict(goal=g,habit=h,observations=history))
    # Ratios are total evidence / total events within each whole-world block.
    for n in (4,8,16,32,64):
        rows=[x['metrics'] for x in out if x['labels']['reader']=='contrasts' and x['labels']['history']==n]
        gain=math.fsum(x['primary'] for x in rows);events=sum(x['violation_events'] for x in rows);all_events=sum(x['attention_events'] for x in rows)
        out.append(record(dict(nats_per_violation=None if events==0 else gain/events,nats_per_attention=None if all_events==0 else gain/all_events),reader='pooled_event_rate',history=n))
    return dict(public=dict(epsilon=epsilon,planning_probability=.5,goal_vetoes=GOALS,habit_grid=HABITS.tolist()),truth_and_observations=raw,readouts=out)

def controls():
    habit=np.array([.5,0,0,0,0,0,0,.5]);a=np.ones(60)/60;b=a.copy()
    for final,att,start in [(0,True,True),(7,False,False),(0,False,True)]:
        a*=likelihood(final,att,None,start,.1,'final',fixture_habit=habit);a/=a.sum()
        b*=likelihood(final,att,None,start,.1,'attention',fixture_habit=habit);b/=b.sum()
    close(a.tolist(),b.tolist(),tol=1e-12)
    for reader in ('final','attention','revision','misread'):
        close(likelihood(0,True,1,True,.1,reader).tolist(),likelihood(0,True,1,True,.1,reader,True).tolist(),tol=1e-12)
    return dict(passed=True,no_veto_fixed_habit_alias=True,independent_likelihood=True)
