"""S.b R2: identical learning model with independent rescaled-log reconstruction."""
import math
import numpy as np
from scipy.special import logsumexp,expit
from numba import njit
from .s_stay import policy,matched
from .common import rng,write,rows_for

GRID=np.linspace(.001,.999,201)

def initial(kind):
    mean=float(expit(matched('SHIFT'))) if kind=='SHIFT' else .5
    a,b=4*mean,4*(1-mean)
    w=(a-1)*np.log(GRID)+(b-1)*np.log1p(-GRID)
    return w-logsumexp(w)

def terminal(stay,p,reference=False):
    h=len(stay);out=np.zeros((h+1,2))
    if reference:
        survival=np.r_[1.,np.cumprod(stay)]
        noarrival=1-p+p*.85**np.arange(h+1)
        out[:h,0]=survival[:h]*noarrival[:h]*(1-stay)
        out[1:,1]=survival[1:]*p*.15*.85**np.arange(h)
        out[h,0]=survival[-1]*noarrival[-1]
    else:
        absent=1-p;coming=p
        for t,q in enumerate(stay):
            out[t,0]=(absent+coming)*(1-q);absent*=q;coming*=q
            out[t+1,1]=.15*coming;coming*=.85
        out[h,0]=absent+coming
    return out.ravel()

def learn(weights,event,reference=False):
    wait=event//2;reward=event%2
    # Only observed terminal information is used. Quitting censors arrival;
    # an unreceived reward is never labelled an absent reward by the learner.
    likelihood=GRID*.15*.85**(wait-1) if reward else 1-GRID+GRID*.85**wait
    if reference:
        # Normalize log masses without converting the posterior to probabilities.
        # Scalar logs and compensated summation independently check scipy's route.
        terms=np.array([float(w)+math.log(float(q)) for w,q in zip(weights,likelihood)])
        peak=max(terms)
        normalizer=peak+math.log(math.fsum(math.exp(float(x-peak)) for x in terms))
        return terms-normalizer
    new=weights+np.log(likelihood);return new-logsumexp(new)

@njit(cache=True)
def fast_policy(mean,cost,temperature,kind,parameter):
    if kind==2:mean=1/(1+np.exp(-parameter*np.log(mean/(1-mean))))
    weight=parameter if kind==1 else 1.;power=parameter if kind==3 else 1.
    stay=np.zeros(40);future=0.
    for t in range(39,-1,-1):
        remaining=.85**(t*power);belief=mean*remaining/(1-mean+mean*remaining)
        q=belief*.15*20-weight*cost+(1-belief*.15)*future
        stay[t]=1/(1+np.exp(-q/temperature));future=temperature*np.logaddexp(0.,q/temperature)
    return stay

def choice_policy(mean,kind,cost,temperature,parameter,reference=False):
    name=kind if kind in ('COST','SHARPEN','SLOW') else 'BASE'
    if reference:return policy(mean,cost=cost,temperature=temperature,kind=name,parameter=parameter)
    return fast_policy(mean,cost,temperature,{'BASE':0,'COST':1,'SHARPEN':2,'SLOW':3}[name],parameter)

def update(weights,event,kind,parameter,reference=False):
    updated=learn(weights,event,reference)
    if kind=='SLOW':
        updated=weights+parameter*(updated-weights);updated-=logsumexp(updated)
    return updated

def generate(s,lane,pulse=lambda **kw:None):
    result=[];parameter=matched(s['truth'],temperature=s['temperature']) if s['truth'] in ('COST','SHARPEN','SLOW') else 0.
    for w in range(s['worlds']):
        g=rng('S-learning',s,lane,w);weights=initial(s['truth']);events=[]
        for t in range(s['steps']):
            mean=float(np.exp(weights)@GRID);stay=choice_policy(mean,s['truth'],s['cost'],s['temperature'],parameter)
            distribution=terminal(stay,s['odds']);event=g.choice(len(distribution),p=distribution/distribution.sum())
            events.append(event);weights=update(weights,event,s['truth'],parameter)
        result.append(events);pulse(world=w+1,phase='generate')
    return dict(events=np.array(result,dtype=np.int16))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    kinds=s['kinds'];truth=kinds.index(s['truth']);parameters={k:matched(k,temperature=s['temperature']) if k in ('COST','SHARPEN','SLOW') else 0. for k in kinds};out={k:[] for k in ['separation','wrongly_settled','shift_wait_gap_at50','shift_wait_gap_last50']}
    for w,events in enumerate(raw['events']):
        weights=[initial(k) for k in kinds];ll=np.zeros((len(kinds),len(events)));gaps=[];running=np.zeros(len(kinds))
        # Counterfactual priors are updated on the same observed history, so
        # this is a conditional learning diagnostic, not a policy-value estimate.
        diagnostic=[initial('BASE'),initial('SHIFT')]
        for t,event in enumerate(events):
            waits=[]
            for k in range(2):
                pmean=float(np.exp(diagnostic[k])@GRID);waits.append(float(np.cumprod(choice_policy(pmean,'BASE',s['cost'],s['temperature'],0.,reference)).sum()))
                diagnostic[k]=learn(diagnostic[k],int(event),reference)
            gaps.append(waits[1]-waits[0])
            for k,kind in enumerate(kinds):
                mean=float(np.exp(weights[k])@GRID);stay=choice_policy(mean,kind,s['cost'],s['temperature'],parameters[kind],reference)
                dist=terminal(stay,s['odds'],reference);running[k]+=np.log(max(dist[event],1e-300));ll[k,t]=running[k]
                weights[k]=update(weights[k],int(event),kind,parameters[kind],reference)
        gap=ll[truth]-np.max(np.delete(ll,truth,0),axis=0);hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        out['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1)
        final=ll[:,-1];order=np.sort(final);out['wrongly_settled'].append(float(np.argmax(final)!=truth and order[-1]-order[-2]>=3))
        out['shift_wait_gap_at50'].append(float(np.mean(gaps[40:50])))
        out['shift_wait_gap_last50'].append(float(np.mean(gaps[-50:])))
        pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in out.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten trials'
    return None

def controls(dest,pulse=lambda **kw:None):
    checks={};a=initial('BASE');checks['zero_wait_is_uninformative']=bool(np.allclose(learn(a,0),a,atol=1e-14,rtol=0))
    checks['received_reward_increases_odds']=float(np.exp(learn(a,3))@GRID)>.5
    checks['censored_wait_decreases_odds']=float(np.exp(learn(a,40))@GRID)<.5
    checks['independent_update']=bool(np.allclose(learn(a,40),learn(a,40,True),atol=1e-13,rtol=0))
    stay=policy(.75);checks['independent_terminal']=bool(np.allclose(terminal(stay,.75),terminal(stay,.75,True),atol=1e-14,rtol=0))
    checks['independent_dynamic_programme']=all(np.allclose(choice_policy(p,k,1.,.5,matched(k) if k!='BASE' else 0.),choice_policy(p,k,1.,.5,matched(k) if k!='BASE' else 0.,True),rtol=1e-12,atol=1e-12) for p in (.1,.5,.9) for k in ('BASE','COST','SHARPEN','SLOW'))
    extreme=-np.linspace(0.,2000.,len(GRID))
    extreme-=logsumexp(extreme)
    for event in (0,3,40,80):
        ref=learn(extreme,event,True)
        checks[f'finite_extreme_support_{event}']=bool(np.isfinite(ref).all())
        checks[f'independent_extreme_update_{event}']=bool(np.allclose(ref,learn(extreme,event),atol=1e-12,rtol=0))
        checks[f'normalized_extreme_update_{event}']=bool(abs(logsumexp(ref))<1e-13)
        checks[f'log_shift_invariance_{event}']=bool(np.allclose(ref,learn(extreme+4096.,event,True),atol=1e-12,rtol=0))
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
