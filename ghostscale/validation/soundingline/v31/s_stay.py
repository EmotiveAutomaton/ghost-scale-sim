"""Text-first exact finite waiting task; independent forward likelihood check."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq,minimize_scalar
from .common import rng,write,rows_for,sustained

KINDS=('COST','SHIFT','SHARPEN','SLOW')
BOUNDS={'COST':(.02,1.),'SHIFT':(0.,8.),'SHARPEN':(1.,12.),'SLOW':(.02,1.)}
FIT_BOUNDS={'COST':(.02,3.),'SHIFT':(-6.,8.),'SHARPEN':(.2,8.),'SLOW':(0.,2.)}

def policy(p,cost=1.,reward=20.,temperature=.5,kind='BASE',parameter=0.,law='geometric',reference=False):
    h=40 if law=='geometric' else 20
    initial=logit(p)
    if kind=='SHIFT': initial+=parameter
    if kind=='SHARPEN': initial*=parameter
    prior=expit(initial); weight=parameter if kind=='COST' else 1.
    power=parameter if kind=='SLOW' else 1.
    remain=.85**np.arange(h+1) if law=='geometric' else np.maximum(0,1-np.arange(h+1)/h)
    belief=prior*remain**power/(1-prior+prior*remain**power)
    hazard=np.full(h,.15) if law=='geometric' else 1/(h-np.arange(h))
    stay=np.zeros(h);future=0.
    for t in range(h-1,-1,-1):
        arrival=belief[t]*hazard[t]
        q=arrival*reward-weight*cost+(1-arrival)*future
        stay[t]=expit(q/temperature)
        future=temperature*np.logaddexp(0.,q/temperature)
    return stay

def distribution(p,cost=1.,reward=20.,temperature=.5,kind='BASE',parameter=0.,law='geometric',reference=False):
    stay=policy(p,cost,reward,temperature,kind,parameter,law,reference);h=len(stay)
    out=np.zeros((h+1,2))
    if reference:
        # Enumerate all possible arrival times plus absent reward. No recursive
        # mixture update, and no evaluator knowledge enters the maker's policy.
        arrivals=p*.15*.85**np.arange(h) if law=='geometric' else np.full(h,p/h)
        times=np.r_[arrivals,1-arrivals.sum()]
        for end,mass in enumerate(times):
            alive=mass
            for t in range(h):
                out[t,0]+=alive*(1-stay[t]);alive*=stay[t]
                if t==end:out[t+1,1]+=alive;alive=0;break
            out[h,0]+=alive
    else:
        absent=1-p;coming=p
        for t in range(h):
            out[t,0]+=(absent+coming)*(1-stay[t])
            absent*=stay[t];coming*=stay[t]
            hazard=.15 if law=='geometric' else 1/(h-t)
            out[t+1,1]+=coming*hazard;coming*=1-hazard
        out[h,0]+=absent+coming
    return out.ravel()

def noarrival_wait(parameter,kind,temperature=.5,law='geometric',reward=20.):
    stay=policy(.75,reward=reward,temperature=temperature,kind=kind,parameter=parameter,law=law)
    return float(np.cumprod(stay).sum())

def matched(kind,lift=4.,temperature=.5,law='geometric',reward=20.):
    base=noarrival_wait(0,'BASE',temperature,law,reward)
    return float(brentq(lambda p:noarrival_wait(p,kind,temperature,law,reward)-base-lift,*BOUNDS[kind]))

def fit_rival(truth,kind,conditions,temp,law='geometric',reference=False):
    def loss(par):
        return np.mean([np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(distribution(p,c,r,temp,kind,par,law,reference),1e-300))))
            for a,(p,c,r) in zip(truth,conditions)])
    grid=np.linspace(*FIT_BOUNDS[kind],65);v=np.array([loss(x) for x in grid]);j=int(np.argmin(v))
    opt=minimize_scalar(loss,bounds=(grid[max(0,j-1)],grid[min(64,j+1)]),method='bounded',options={'xatol':1e-10})
    candidates=[(float(opt.fun),float(opt.x)),(float(v[j]),float(grid[j]))]
    return min(candidates)

def generate(s,lane,pulse=lambda **kw:None):
    conditions=s['conditions'];kind=s['truth'];temp=s['temperature'];law=s.get('law','geometric')
    par=matched(kind,s['lift'],temp,law,s.get('match_reward',20.))
    truth=np.array([distribution(p,c,r,temp,kind,par,law) for p,c,r in conditions])
    fitted=[par,fit_rival(truth,s['rival'],conditions,temp,law)[1]]
    events=[];which=[]
    for w in range(s['worlds']):
        g=rng('S',s,lane,w);cond=g.integers(len(conditions),size=s['steps'])
        events.append(np.array([g.choice(truth.shape[1],p=truth[j]/truth[j].sum()) for j in cond]));which.append(cond)
        pulse(world=w+1,phase='generate')
    return dict(events=np.asarray(events,dtype=np.int16),condition=np.asarray(which,dtype=np.int16),parameters=np.array(fitted))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    probs=np.array([[distribution(p,c,r,s['temperature'],k,raw['parameters'][i],s.get('law','geometric'),reference)
        for p,c,r in s['conditions']] for i,k in enumerate((s['truth'],s['rival']))])
    result={'separation':[],'wrongly_settled':[],'final_gap':[],'alias':[]}
    idx=0
    for w,(events,conditions) in enumerate(zip(raw['events'],raw['condition'])):
        ll=np.cumsum(np.log(np.maximum(probs[:,conditions,events],1e-300)),axis=1)
        gap=ll[idx]-np.max(np.delete(ll,idx,axis=0),axis=0)
        hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        result['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1)
        order=np.sort(ll[:,-1]);winner=np.argmax(ll[:,-1])
        result['wrongly_settled'].append(float(winner!=idx and order[-1]-order[-2]>=3))
        result['final_gap'].append(float(gap[-1]));result['alias'].append(float(np.max(abs(probs[0]-probs[1]))<1e-10))
        for n in s.get('reported_prefixes',[s['steps']]):
            if n>s['steps']:continue
            result.setdefault('settled_'+str(n),[]).append(float(gap[n-1]>=3))
            result.setdefault('wrongly_settled_'+str(n),[]).append(float(gap[n-1]<=-3))
        pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in result.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten trials'
    if np.all(m['alias']==1):return 'exact trial-distribution alias after rival refit'
    return None

def controls(dest,pulse=lambda **kw:None):
    errors=[];normal=[]
    for p in (.25,.5,.75,.9):
        for k in KINDS:
            par=matched(k);a=distribution(p,kind=k,parameter=par);b=distribution(p,kind=k,parameter=par,reference=True)
            errors.append(float(np.max(abs(a-b))));normal.append(abs(a.sum()-1)<1e-12)
    checks={'independent_probability':max(errors)<1e-12,'normalization':all(normal),
        'matched_lift':all(abs(noarrival_wait(matched(k),k)-noarrival_wait(0,'BASE')-4)<1e-8 for k in KINDS),
        'sharp_even_alias':np.allclose(distribution(.5,kind='SHARPEN',parameter=2),distribution(.5),rtol=0,atol=1e-14)}
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
