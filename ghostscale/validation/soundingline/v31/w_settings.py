"""Text-first four-seat maker and exact full-view reader.

Partial views use a separately gated bootstrap particle filter. They are never
admitted by a full-view control alone. Calibration panels are excluded.
"""
import numpy as np
from numba import njit
from .common import rng,write,rows_for

BASE=np.array([.5,.35,2.,1.,0.])

@njit(cache=True)
def softmax(x):
    y=np.exp(x-x.max());return y/y.sum()

@njit(cache=True)
def transition(belief,cue,v,r,a):
    prior=belief*(1-v)+(1-belief)*v/3
    logp=2*(1-a)*np.log(prior)
    for j in range(4):logp[j]+=2*a*np.log(r if j==cue else (1-r)/3)
    return softmax(logp)

@njit(cache=True)
def simulate(draws,v,r,f,theta):
    a,ts,d,s,d2=theta;belief=np.ones(4)/4;schema=0;context=0;cbar=.5;u=0.;plan=0
    rows=np.zeros((len(draws),11))
    for t in range(len(draws)):
        z=draws[t]
        if z[0]<v:context=(context+1+int(z[1]*3))%4
        cue=context if z[2]<r else (context+1+int(z[3]*3))%4
        drop=z[4]<.1;actionable=drop and z[5]<f
        u+=drop
        if plan==0 and actionable:plan=4
        available=plan>0
        belief=transition(belief,cue,v,r,a)
        logits=np.log(belief);logits[schema]+=1
        probs=softmax(logits/ts);cs=np.cumsum(probs);schema=min(np.searchsorted(cs,z[6]),3)
        conf=belief[schema];newbar=.8*cbar+.2*conf;rise=max(0.,newbar-cbar);cbar=newbar
        work=z[7]<1/(1+np.exp(-(d*(cbar-.5)+1)))
        logrt=-.5*d*(cbar-.5)-d2*rise+.3*z[10]
        abandon=work and available and z[8]<1/(1+np.exp(-(s*u-3)))
        completed=False
        if abandon:plan=0
        elif work and plan>0:
            plan-=1
            if plan==0:completed=True;u=max(0.,u-1)
        u*=.95
        rows[t]=np.array([context,cue,drop,actionable,schema,work,logrt if work else 0.,available,abandon,completed,u])
    return rows

@njit(cache=True)
def full_paths(path,v,r,models):
    result=np.zeros((len(models),len(path)))
    for h in range(len(models)):
        a,ts,d,s,d2=models[h];belief=np.ones(4)/4;schema=0;cbar=.5;u=0.;ll=0.
        for t in range(len(path)):
            row=path[t];u+=row[2]
            belief=transition(belief,int(row[1]),v,r,a)
            logits=np.log(belief);logits[schema]+=1;prob=softmax(logits/ts);schema=int(row[4]);ll+=np.log(prob[schema])
            nextbar=.8*cbar+.2*belief[schema];rise=max(0.,nextbar-cbar);cbar=nextbar
            pw=1/(1+np.exp(-(d*(cbar-.5)+1)))
            ll+=np.log(pw if row[5] else 1-pw)
            mean=-.5*d*(cbar-.5)-d2*rise
            if row[5]:ll-=.5*((row[6]-mean)/.3)**2
            if row[7] and row[5]:
                pa=1/(1+np.exp(-(s*u-3)));ll+=np.log(pa if row[8] else 1-pa)
            if row[9]:u=max(0.,u-1)
            u*=.95;result[h,t]=ll
    return result

def full_reference(path,v,r,models):
    result=[]
    for a,ts,d,s,d2 in models:
        b=np.full(4,.25);old=0;cbar=.5;u=0.;increments=[]
        for row in path:
            u+=row[2]
            transition_matrix=np.full((4,4),v/3);np.fill_diagonal(transition_matrix,1-v)
            prior=b@transition_matrix;lik=np.full(4,(1-r)/3);lik[int(row[1])]=r
            b=prior**(2*(1-a))*lik**(2*a);b/=b.sum()
            prob=b**(1/ts);prob[old]*=np.exp(1/ts);prob/=prob.sum();old=int(row[4])
            nextbar=.8*cbar+.2*b[old];rise=max(0,nextbar-cbar);cbar=nextbar
            # Stable scalar log logistic probabilities, no production kernel.
            z=d*(cbar-.5)+1;term=-np.logaddexp(0,-z if row[5] else z)+np.log(prob[old])
            if row[5]:term-=.5*((row[6]+.5*d*(cbar-.5)+d2*rise)/.3)**2
            if row[7] and row[5]:term-=np.logaddexp(0,-(s*u-3) if row[8] else s*u-3)
            if row[9]:u=max(0,u-1)
            u*=.95;increments.append(term)
        result.append(np.cumsum(increments))
    return np.array(result)

def generate(s,lane,pulse=lambda **kw:None):
    paths=[]
    for w in range(s['worlds']):
        g=rng('W',s,lane,w);z=g.random((s['steps'],11));z[:,10]=g.normal(size=s['steps'])
        paths.append(simulate(z,s['volatility'],s['reliability'],s['actionable'],np.array(s['models'][s['truth']])))
        pulse(world=w+1,phase='generate')
    return dict(records=np.array(paths))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    if s['view']!='V4':raise ValueError('Partial-view accuracy admission required; no privileged full-view fallback')
    results={k:[] for k in ('separation','wrongly_settled','switches_per100','plans_per1000','right_work','alias')}
    for w,path in enumerate(raw['records']):
        ll=(full_reference if reference else full_paths)(path,s['volatility'],s['reliability'],np.array(s['models']))
        truth=s['truth'];gap=ll[truth]-np.max(np.delete(ll,truth,0),axis=0);hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        last=ll[:,-1];order=np.sort(last)
        results['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1)
        results['wrongly_settled'].append(float(np.argmax(last)!=truth and order[-1]-order[-2]>=3))
        results['switches_per100'].append(float(np.mean(path[1:,4]!=path[:-1,4])*100))
        results['plans_per1000'].append(float(path[:,9].mean()*1000))
        results['right_work'].append(float(np.mean(path[:,5]*(path[:,4]==path[:,0]))))
        results['alias'].append(float(np.max(abs(gap))<1e-10));pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in results.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=10):return 'all forty paths separate within ten steps'
    if np.all(m['alias']==1):return 'exact observed-probability alias'
    return None

def controls(dest,pulse=lambda **kw:None):
    b=np.array([.1,.2,.3,.4]);cue=2;v=.05;r=.75
    prior=b*(1-v)+(1-b)*v/3;like=np.full(4,(1-r)/3);like[cue]=r;expected=prior*like;expected/=expected.sum()
    g=np.random.default_rng(3104);draw=g.random((512,11));draw[:,10]=g.normal(size=512)
    path=simulate(draw,v,r,.5,BASE);models=np.array([BASE,[.6,.35,2,1,0],[.5,.4,2,1,0],[.5,.35,3,1,0],[.5,.35,2,1.2,0]])
    first=full_paths(path,v,r,models);second=full_reference(path,v,r,models)
    checks={'exact_half_weight_update':bool(np.allclose(transition(b,cue,v,r,.5),expected,atol=1e-14,rtol=0)),
        'independent_full_view':float(np.max(abs(first-second)))<1e-8,
        'identical_law_alias':bool(np.array_equal(full_paths(path,v,r,np.array([BASE,BASE]))[0],full_paths(path,v,r,np.array([BASE,BASE]))[1]))}
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
