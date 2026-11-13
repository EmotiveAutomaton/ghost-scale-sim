"""Queue reading from exact stationary states; built before supplied pre-flight."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng, logistic, class_evidence, sustained, mean_blocks, censored_blocks, write, digest, audit_controls

PROFILES={'even':([.15,.15,.15],[1.,1.,1.]),
          'skewed':([.2,.15,.1],[1.,1.5,3.]),
          'one-big':([.05,.05,.3],[1.,1.,3.])}
STATES=np.array(list(product(range(6),repeat=3)),dtype=int)
READINGS=('COUNT','TOTAL','LOUDEST')


def index(d):return d[...,0]*36+d[...,1]*6+d[...,2]


@lru_cache(None)
def stationary(profile):
    arrivals,a=map(np.asarray,PROFILES[profile]);p=np.zeros((216,216))
    for i,d in enumerate(STATES):
        winner=int(np.argmax(a*d))
        served=d.copy();served[winner]=max(0,served[winner]-1)
        for bits in product((0,1),repeat=3):
            inc=np.array(bits);chance=float(np.prod(np.where(inc,arrivals,1-arrivals)))
            p[i,index(np.minimum(5,d+inc))]+=.1*chance
            p[i,index(np.minimum(5,served+inc))]+=.9*chance
    mass=np.full(216,1/216.)
    for iteration in range(20000):
        new=mass@p
        if abs(new-mass).max()<1e-15:break
        mass=new
    else:raise ArithmeticError('stationary distribution not converged')
    values=np.column_stack([(STATES>0).sum(1),STATES@a,(STATES*a).max(1)]).astype(float)
    center=mass@values;sd=np.sqrt(mass@((values-center)**2))
    scaled=values/sd
    quantiles=np.empty((3,5));median=np.empty(3)
    for j in range(3):
        order=np.argsort(scaled[:,j],kind='stable');cdf=np.cumsum(mass[order])
        quantiles[j]=[scaled[order[min(215,np.searchsorted(cdf,q))],j] for q in np.linspace(.3,.7,5)]
        median[j]=scaled[order[min(215,np.searchsorted(cdf,.5))],j]
    return mass,scaled,quantiles,median,float(abs(mass@p-mass).max())


def settings_main():
    return [dict(part='Q3',truth=truth,profile=profile,rate=rate,steps=2048,worlds=200,blocks=8)
            for truth in range(3) for profile in PROFILES for rate in (.1,.3)]


def candidates(profile):
    _,scaled,q,median,_=stationary(profile)
    # Quantile duplicates remain members, as the declared five-point prior requires.
    params=np.array([(j,r,theta) for j in range(3) for r in (.05,.1,.2,.3) for theta in q[j]])
    return params,scaled,median


def generate_main(s,lane='study',worlds=None):
    n=worlds or s['worlds'];steps=s['steps'];r=rng('Q3',lane,s)
    mass,scaled,_,median,_=stationary(s['profile']);arrivals,a=map(np.asarray,PROFILES[s['profile']])
    d=STATES[r.choice(216,n,p=mass)].copy();y=np.zeros(n,int)
    states=np.empty((n,steps),np.int16);rides=np.zeros((n,steps),np.int8);eligible=np.zeros_like(rides)
    actions=np.zeros((n,steps),np.int8)
    for t in range(steps):
        i=index(d);states[:,t]=i;y=np.maximum(0,y-1);eligible[:,t]=y==0
        p=s['rate']*logistic(s.get('slope',1.5)*(median[s['truth']]-scaled[i,s['truth']]),s.get('tremble',0.))
        ride=(y==0)&(r.random(n)<p);rides[:,t]=ride;y+=2*ride
        pick=(a*d).argmax(1);serve=(r.random(n)<.9)&(d.sum(1)>0);actions[:,t]=np.where(serve,pick+1,0)
        d[np.arange(n),pick]-=serve
        d=np.minimum(5,d+(r.random((n,3))<arrivals))
    return dict(states=states,rides=rides,eligible=eligible,actions=actions)


def infer_main(raw,s):
    params,scaled,_=candidates(s['profile']);classes=params[:,0].astype(int);n,steps=raw['states'].shape
    table=params[None,:,1]*logistic(s.get('slope',1.5)*(params[None,:,2]-scaled[:,classes]),s.get('tremble',0.))
    same=np.ones((n,3),bool)
    ll=np.zeros((n,len(params)));last=np.full((n,3),-1);visit=np.zeros((n,216),int)
    for t in range(steps):
        p=table[raw['states'][:,t]];act=raw['rides'][:,t,None]
        ll+=raw['eligible'][:,t,None]*np.where(act,np.log(p),np.log1p(-p))
        evidence=class_evidence(ll,classes);odds=evidence[:,s['truth'],None]-evidence
        last=np.where(odds<3,t,last);same &= abs(odds)<1e-12
        visit[np.arange(n),raw['states'][:,t]]+=raw['eligible'][:,t]
    # Truth-vs-rival probabilities at each truth's own median, matched rate.
    _,_,median=candidates(s['profile'])
    probabilities=s['rate']*logistic(s.get('slope',1.5)*(median[None,:]-scaled),s.get('tremble',0.))
    p=probabilities[:,s['truth'],None];q=probabilities
    kl=p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    same[:,s['truth']]=False
    return dict(same_probability=same.astype(float),pair_separation=last+2,final_class_odds=odds,final_likelihood=ll,visits=visit,
                divergence=kl,informative_share=((visit@(kl.max(1)>.01))/np.maximum(1,visit.sum(1))))


def base_controls(destination,pulse=lambda **kw:None):
    checks=[]
    for profile in PROFILES:
        mass,scaled,q,median,residual=stationary(profile)
        checks.append(dict(name=profile+' stationary balance',passed=residual<1e-12,maximum_residual=residual))
        sd=np.sqrt(mass@((scaled-mass@scaled)**2))
        checks.append(dict(name=profile+' unit standard deviations',passed=bool(np.allclose(sd,1,atol=1e-12))))
    checks.append(dict(name='state index bijection',passed=bool(np.array_equal(index(STATES),np.arange(216)))))
    p=np.array([.01,.1,.3]);kl=p*np.log(p/p)+(1-p)*np.log((1-p)/(1-p))
    checks.append(dict(name='equal readings zero divergence',passed=bool(np.array_equal(kl,np.zeros(3)))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result


def selection_probabilities(d,a,costs,rule,lam=1.):
    """Q3.b: the action probability is identical; only the selected need changes."""
    score=a*d if rule=='loudest' else a*(d-.5)-lam*costs
    score=np.where(d>0,score,-np.inf)
    selected=score.argmax(axis=-1)
    return selected,(np.sum(d,axis=-1)>0)*.9


def settings():
    return settings_main()+[dict(part='Q3b',truth=truth,costs=list(costs),steps=2048,worlds=200,blocks=8) for truth in (0,1) for costs in ((1,1,1),(1,2,3),(3,2,1))]

def selection_laws(d,s,reference=False):
    n=len(d);out=np.zeros((n,2,4));a=np.array([1.,1.5,3.]);cost=np.array(s['costs']);lapse=s.get('tremble',0.)
    if reference:
        for w in range(n):
            available=[j for j in range(3) if d[w,j]>0]
            for model in range(2):
                out[w,model,0]=.1 if available else 1.
                if available:
                    score=[a[j]*d[w,j] if model==0 else a[j]*(d[w,j]-.5)-cost[j] for j in available]
                    chosen=available[int(np.argmax(score))]
                    for j in available:out[w,model,j+1]=.9*(lapse/len(available)+(1-lapse)*(j==chosen))
    else:
        for model in range(2):
            chosen,act=selection_probabilities(d,a,cost,'loudest' if model==0 else 'cost',1.)
            out[:,model,0]=1-act
            for j in range(3):out[:,model,j+1]=act*(lapse*(d[:,j]>0)/np.maximum(1,(d>0).sum(1))+(1-lapse)*(chosen==j))
    return out

def generate_selection(s,lane='study'):
    n=s['worlds'];r=rng('Q3b',lane,s);d=np.zeros((n,3),int);states=np.empty((n,s['steps'],3),np.int8);actions=np.empty((n,s['steps']),np.int8)
    for t in range(s['steps']):
        states[:,t]=d;p=selection_laws(d,s)[:,s['truth']];act=(r.random((n,1))>np.cumsum(p,axis=1)).sum(1);actions[:,t]=act
        for j in range(3):d[:,j]-=act==j+1
        d=np.minimum(5,d+(r.random((n,3))<np.array([.2,.15,.1])))
    return dict(deficits=states,actions=actions)

def infer_selection(raw,s,reference=False):
    n,steps=raw['actions'].shape;ll=np.zeros((n,2));last=np.full(n,-1);conflict=np.zeros(n);same=np.ones(n,bool)
    for t in range(steps):
        p=selection_laws(raw['deficits'][:,t],s,reference);chosen=p[np.arange(n)[:,None],np.arange(2)[None,:],raw['actions'][:,t,None]]
        with np.errstate(divide='ignore'):ll+=np.log(chosen)
        odds=ll[:,s['truth']]-ll[:,1-s['truth']];last=np.where(odds<3,t,last)
        differ=np.any(abs(p[:,0]-p[:,1])>1e-12,axis=1);conflict+=differ;same &= ~differ
    return dict(separation=last+2,final_likelihood=ll,conflict_share=conflict/steps,same_probability=same[:,None].astype(float))

def generate(s,lane='study',worlds=None):
    if worlds is not None:s=s|dict(worlds=worlds)
    return generate_selection(s,lane) if s['part']=='Q3b' else generate_main(s,lane)

def infer(raw,s,reference=False):
    if s['part']=='Q3b':return infer_selection(raw,s,reference)
    from .q3_reference import infer as reconstruct
    out=(reconstruct if reference else infer_main)(raw,s)
    out['separation']=np.max(np.delete(out['pair_separation'],s['truth'],axis=1),axis=1)
    return out

def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);rows=[];files={};roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        if s['part']=='Q3b':rows.append(dict(setting=s,measure='selection_conflict_share',**mean_blocks(out['conflict_share'],s['blocks'])))
        else:
            for j in range(3):
                if j!=s['truth']:rows.append(dict(setting=s,measure='pair_separation',rival=READINGS[j],**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
            rows.append(dict(setting=s,measure='informative_opportunity_share',**mean_blocks(out['informative_share'],s['blocks'])))
            for j in range(3):
                if j==s['truth']:continue
                top=np.argsort(-out['divergence'][:,j],kind='stable')[:5]
                for state in top:rows.append(dict(setting=s,measure='telling_state',rival=READINGS[j],state=STATES[state].tolist(),divergence_nats=float(out['divergence'][state,j]),visits=mean_blocks(out['visits'][:,state],s['blocks'])))
        pulse(family='Q3',cell=i+1,cells=len(roster))
    result=dict(family='Q3',settings=roster,rows=rows,no_registered_claim=True,selection_note='Primary selection rules have no within-action tremble. A conflicting selected act has infinite likelihood ratio; conflict frequency and separation time remain finite. No claim of finite near-zero KL is made.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result

def controls(destination,pulse=lambda **kw:None):
    dest=Path(destination);base=base_controls(dest/'base',pulse);checks=base['checks']
    for setting in [settings_main()[0],settings_main()[9],settings_main()[-1],settings()[-1],settings()[-3]]:
        s=setting|dict(worlds=8,blocks=8,steps=64);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(s)+' independent entire filter',passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
    for costs in ((1,1,1),(1,2,3),(3,2,1)):
        p=selection_laws(STATES,dict(costs=costs));checks.append(dict(name=str(costs)+' identical act propensity',passed=bool(np.allclose(p[:,0,0],p[:,1,0])) and bool(np.allclose(p.sum(2),1))))
    from .q3_reference import filter_table
    states=np.zeros((8,32),np.int16);rides=np.zeros_like(states);eligible=np.ones_like(states);table=np.full((216,3),.1)
    pair,odds,*_=filter_table(states,rides,eligible,table,np.arange(3),0)
    checks.append(dict(name='identical class laws zero evidence',passed=bool(np.all(odds==0)) and bool(np.all(pair==33))))
    result=audit_controls(checks);write(dest/'CONTROLS.json',result);return result
