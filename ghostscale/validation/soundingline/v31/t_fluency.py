"""Exact marginal-attention fluency reader; text-first finite four-kind world."""
import numpy as np
from numba import njit
from scipy.special import expit
from .common import rng,write,rows_for

KINDS=('HABIT','HEURISTIC','VALUE','IDENTITY')
PRICES=np.array([0.,1.,8.,64.]);BASES=np.array([1.,0.,0.,2.])

@njit(cache=True)
def choice(k,c,m,n,q,price,b0):
    ready=1/(1+np.exp(-(b0+3*k+.5*np.log1p(n))))
    attended=1. if (c==0 or c<price) and (m<0 or m<price) else 0.
    return (1-q)*ready+q*attended

def probability(k,c,m,n,q,price,b0,reference=False):
    if reference:
        logodds=b0+3*k+.5*np.log(1+n)
        ready=float(expit(logodds));attend=float((c==0 or c<price) and (m<0 or m<price))
        return np.dot([1-q,q],[ready,attend])
    return choice(k,c,m,n,q,price,b0)

def effect_mean(kind,act):
    if kind=='HEURISTIC':return .25*act
    if kind=='VALUE':return 1.*act
    if kind=='IDENTITY':return -1.*(1-act)
    return 0.

@njit(cache=True)
def make_path(k,c,m,u,q,price,b0):
    out=np.zeros(len(k),dtype=np.int8);uses=0
    for t in range(len(k)):
        out[t]=u[t]<choice(k[t],c[t],m[t],uses,q,price,b0);uses+=out[t]
    return out

def generate(s,lane,pulse=lambda **kw:None):
    rows=[];truth=KINDS.index(s['truth'])
    for w in range(s['worlds']):
        g=rng('T',s,lane,w);t=s['steps'];k=g.random(t)
        c=g.choice([1.,2.,4.,8.,16.],t)*(g.random(t)<s['cost_rate'])
        m=np.where(g.random(t)<s['offer_rate'],g.choice([1.,2.,4.,8.,16.,64.,128.],t),-1.)
        act=make_path(k,c,m,g.random(t),s['attention'],PRICES[truth],BASES[truth])
        effect=np.array([effect_mean(s['truth'],int(x)) for x in act])+g.normal(0,1,t)
        rows.append(np.column_stack((k,c,m,act,effect)));pulse(world=w+1,phase='generate')
    return dict(records=np.array(rows))

@njit(cache=True)
def logpaths(raw,q,view,prices,bases,effects):
    models=len(prices);out=np.zeros((models,len(raw)));uses=0
    for t in range(len(raw)):
        k,c,m,act,eff=raw[t]
        for j in range(models):
            p=choice(k,c,m,uses,q,prices[j],bases[j]);p=min(max(p,1e-300),1-1e-15)
            ll=np.log(p if act else 1-p)
            if view:
                mean=effects[j,0]*act+effects[j,1]*(1-act)
                ll-=.5*(eff-mean)**2
            out[j,t]=(out[j,t-1] if t else 0)+ll
        uses+=int(act)
    return out

def reference_paths(raw,q,view,prices,bases,effects):
    uses=np.r_[0,np.cumsum(raw[:-1,3])];values=[]
    for price,b0,ef in zip(prices,bases,effects):
        ready=expit(b0+3*raw[:,0]+.5*np.log1p(uses))
        attend=((raw[:,1]==0)|(raw[:,1]<price))&((raw[:,2]<0)|(raw[:,2]<price))
        p=(1-q)*ready+q*attend;p=np.clip(p,1e-300,1-1e-15)
        ll=np.where(raw[:,3]>0,np.log(p),np.log1p(-p))
        if view:ll-=.5*(raw[:,4]-(ef[0]*raw[:,3]+ef[1]*(1-raw[:,3])))**2
        values.append(np.cumsum(ll))
    return np.array(values)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    effects=np.array([[0,0],[.25,0],[1,0],[0,-1]])
    f=reference_paths if reference else logpaths;truth=KINDS.index(s['truth']);result={k:[] for k in ('separation','wrongly_settled','price_error','readiness_error','alias','acts')}
    # A separate rectangular price/readiness library measures confounding from
    # actions only; kinds/effects are never used as privileged parameter labels.
    prices,bases=np.meshgrid([0,1,2,4,8,16,32,64,128],[-1,0,1,2,3],indexing='ij');prices=prices.ravel();bases=bases.ravel()
    for w,path in enumerate(raw['records']):
        ll=f(path,s['attention'],s['effects'],PRICES,BASES,effects)
        gap=ll[truth]-np.max(np.delete(ll,truth,0),axis=0);hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        last=ll[:,-1];order=np.sort(last)
        result['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1)
        result['wrongly_settled'].append(float(np.argmax(last)!=truth and order[-1]-order[-2]>=3))
        fit=f(path,s['attention'],False,prices,bases,np.zeros((len(prices),2)))[:,-1]
        post=np.exp(fit-fit.max());post/=post.sum()
        result['price_error'].append(float(abs(post@prices-PRICES[truth])))
        result['readiness_error'].append(float(abs(post@bases-BASES[truth])))
        result['alias'].append(float(np.any([np.max(abs(ll[j]-ll[truth]))<1e-10 for j in range(4) if j!=truth])))
        result['acts'].append(float(path[:,3].mean()));pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in result.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten occasions'
    if np.all(m['alias']==1):return 'observed act-probability alias in every pilot world'
    return None

def controls(dest,pulse=lambda **kw:None):
    checks={'attended_habit_cost':choice(.2,1,-1,4,1,0,1)==0,
        'attended_habit_offer':choice(.2,0,1,4,1,0,1)==0,
        'value_refuses_small_offer':choice(.2,0,4,4,1,8,0)==1,
        'price_tie_not_hidden':choice(.2,8,-1,4,1,8,0)==0,
        'identity_price':choice(.2,0,16,4,1,64,2)==1,
        'off_attention_price_alias':choice(.2,1,128,4,0,0,1)==choice(.2,1,128,4,0,64,1)}
    s=dict(truth='VALUE',steps=40,worlds=2,cost_rate=.25,offer_rate=.1,attention=.5,effects=True,blocks=2)
    raw=generate(s,'control');a=infer(raw,s);b=infer(raw,s,True)
    checks['independent_full_reader']=all(np.allclose(a[k],b[k],rtol=1e-10,atol=1e-8) for k in a)
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
