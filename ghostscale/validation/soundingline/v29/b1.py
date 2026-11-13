"""Visible band re-entry, centre latching and a lower-noise move option."""
from pathlib import Path
import math
import numpy as np
from scipy.special import logsumexp
from numba import njit
from .common import rng,class_evidence,sustained,mean_blocks,censored_blocks,conditional_blocks,write,digest,audit_controls
from .ledger_reference import lse
NAMES=('PROPORTIONAL','EDGE','CENTRE','UNCERTAIN')


def candidates(simple=False):
    if simple:return np.array([(0,theta,pi,0.) for theta in range(4,23) for pi in (.25,.5,1.,2.,4.)])
    return np.array([(typ,theta,pi,delta) for typ in range(4) for theta in (8,10,12,14,16,18,20) for pi in (.5,1.,2.) for delta in ((2.,4.) if typ in (1,2) else (0.,))])


def settings():
    truths=[(0,12,pi,0) for pi in (1.,.5)]+[(typ,12,1,delta) for typ in (1,2) for delta in (2.,4.)]+[(3,12,pi,0) for pi in (1.,.5)]
    return [dict(truth=list(truth),regime=regime,steps=512,worlds=200,blocks=8,move_offer=.1) for truth in truths for regime in ('N','D','W')]


def probabilities(stock,offered,params,active,s):
    typ=params[:,0].astype(int);theta=params[:,1];pi=params[:,2];delta=params[:,3];x=stock[:,None]
    active=np.where(typ[None,:]==2,np.where(abs(x-theta)<1e-12,False,active|(abs(x-theta)>delta)),False)
    lower=np.where(typ==1,theta-delta,theta);upper=np.where(typ==1,theta+delta,theta)
    up=pi*np.maximum(0,lower-x)-3;down=pi*np.maximum(0,x-upper)-3
    up=np.where((typ[None,:]==2)&~active,-3.,up);down=np.where((typ[None,:]==2)&~active,-3.,down)
    utility=s.get('slope',1.)*np.stack([np.zeros_like(up),up,down],axis=2);utility-=utility.max(2,keepdims=True)
    ordinary=np.exp(utility);ordinary/=ordinary.sum(2,keepdims=True);lapse=s.get('tremble',0.);ordinary=(1-lapse)*ordinary+lapse/3
    move=.8*offered[:,None]*(typ[None,:]==3);p=np.concatenate([ordinary*(1-move[:,:,None]),move[:,:,None]],axis=2)
    return p,active,ordinary


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];r=rng('B1',lane,s);params=np.array([s['truth']],float);active=np.zeros((n,1),bool);stock=np.full(n,12.);place=np.zeros(n,bool);steps=s['steps']
    stocks=np.empty((n,steps));actions=np.empty((n,steps),np.int8);offers=np.empty_like(actions);places=np.empty_like(actions);disturb=np.empty((n,steps))
    for t in range(steps):
        offer=(~place)&(r.random(n)<s.get('move_offer',.1));stocks[:,t]=stock;offers[:,t]=offer;places[:,t]=place
        p,active,_=probabilities(stock,offer,params,active,s);act=(r.random((n,1))>p[:,0].cumsum(1)).sum(1);actions[:,t]=act
        moved=act==3;place |= moved;stock=np.where(moved,s['truth'][1]+3,stock)
        change=np.where(act==1,3.,np.where(act==2,-3.,0.))
        if s['regime']=='D' and 64<=t<128:change=np.where(act==1,0.,change)
        noise=(r.random((n,1))>np.array([.5,.85])).sum(1)*np.where(place,.2,1.);disturb[:,t]=noise
        stock=stock+change-noise
        if s['regime']=='W' and 64<=t<128:stock+=2
        stock=np.clip(stock,0,24)
    return dict(stocks=stocks,actions=actions,offers=offers,places=places,disturbances=disturb)


@njit(cache=True)
def reference_paths(stocks,actions,offers,params,nfull,truth,slope,lapse):
    n,steps=stocks.shape;ll=np.zeros((n,nfull));simple=np.zeros((n,len(params)-nfull));trace=np.empty((n,steps,4));truth_cond=np.zeros(n)
    for w in range(n):
        active=np.zeros(len(params),np.bool_)
        for t in range(steps):
            stock=stocks[w,t];act=actions[w,t];offered=offers[w,t]
            for j in range(len(params)):
                typ,theta,pi,delta=params[j]
                if typ==2:
                    if abs(stock-theta)<1e-12:active[j]=False
                    elif abs(stock-theta)>delta:active[j]=True
                low=theta-delta if typ==1 else theta;high=theta+delta if typ==1 else theta
                up=pi*max(0.,low-stock)-3;down=pi*max(0.,stock-high)-3
                if typ==2 and not active[j]:up=-3.;down=-3.
                utilities=np.array([0.,slope*up,slope*down]);peak=max(utilities);norm=0.
                for k in range(3):norm+=math.exp(utilities[k]-peak)
                ordinary=0.
                if act<3:ordinary=(1-lapse)*math.exp(utilities[act]-peak)/norm+lapse/3
                move=.8 if typ==3 and offered else 0.
                chance=move if act==3 else ordinary*(1-move)
                if j<nfull:ll[w,j]+=math.log(chance) if chance>0 else -math.inf
                elif act<3:simple[w,j-nfull]+=math.log(ordinary)
                if j==truth and act<3:truth_cond[w]+=math.log(ordinary)
            for k in range(4):
                take=params[:nfull,0]==k;trace[w,t,k]=lse(ll[w,take])-math.log(np.sum(take))
    return ll,simple,trace,truth_cond


def paths(raw,s,reference=False):
    full=candidates();simple=candidates(True);params=np.vstack([full,simple]);n=len(raw['stocks']);steps=s['steps'];truth=int(np.flatnonzero(np.all(full==s['truth'],axis=1))[0])
    if reference:return reference_paths(raw['stocks'],raw['actions'],raw['offers'],params,len(full),truth,s.get('slope',1.),s.get('tremble',0.))
    active=np.zeros((n,len(params)),bool);ll=np.zeros((n,len(full)));simple_ll=np.zeros((n,len(simple)));trace=np.empty((n,steps,4));truth_cond=np.zeros(n)
    for t in range(steps):
        p,active,ordinary=probabilities(raw['stocks'][:,t],raw['offers'][:,t],params,active,s);act=raw['actions'][:,t]
        chosen=np.take_along_axis(p[:,:len(full)],act[:,None,None],axis=2)[:,:,0]
        with np.errstate(divide='ignore'):ll+=np.log(chosen)
        conditional=np.take_along_axis(ordinary,np.minimum(act,2)[:,None,None],axis=2)[:,:,0]
        simple_ll+=np.where(act[:,None]<3,np.log(conditional[:,len(full):]),0.);truth_cond+=np.where(act<3,np.log(conditional[:,truth]),0.)
        trace[:,t]=class_evidence(ll,full[:,0].astype(int))
    return ll,simple_ll,trace,truth_cond


def simple_choice(ll,raw,s):
    """Equal utility laws on the observed stock support get the same exact tie class."""
    params=candidates(True);chosen=np.empty(len(ll),int)
    for w in range(len(ll)):
        values=np.unique(raw['stocks'][w,raw['actions'][w]<3]);groups={}
        for j,(_,theta,pi,_) in enumerate(params):
            signature=tuple(np.concatenate([pi*np.maximum(0,theta-values)-3,pi*np.maximum(0,values-theta)-3]))
            groups.setdefault(signature,[]).append(j)
        indices=list(groups.values());best=int(np.argmax([ll[w,g].mean() for g in indices]));chosen[w]=indices[best][0]
    return chosen


def infer(raw,s,reference=False):
    ll,simple,trace,truth_cond=paths(raw,s,reference);n,steps=raw['stocks'].shape;truth=int(s['truth'][0]);odds=trace[:,:,truth,None]-trace
    pair=np.stack([sustained(odds[:,:,j]) for j in range(4)],axis=1);same=np.all(abs(odds)<1e-12,axis=1);same[:,truth]=False
    chosen=simple_choice(simple,raw,s);params=candidates(True);ordinary_count=(raw['actions']<3).sum(1)
    delta=s['truth'][3] if truth in (1,2) else 2.;outside=abs(raw['stocks']-s['truth'][1])>delta
    entering=outside&~np.concatenate([np.zeros((n,1),bool),outside[:,:-1]],axis=1);counts=entering.cumsum(1)
    target=pair[:,2 if truth==1 else 1];observed=target<=steps;at=np.minimum(target-1,steps-1)
    return dict(separation=np.max(np.delete(pair,truth,axis=1),axis=1),pair_separation=pair,final_likelihood=ll,simple_likelihood=simple,same_probability=same.astype(float),fitted_theta=params[chosen,1],fitted_precision=params[chosen,2],conditional_in_sample_loss=(truth_cond-simple[np.arange(n),chosen])/np.maximum(1,ordinary_count),move_share=np.mean(raw['actions']==3,axis=1),excursions=counts[:,-1],excursions_at_pair_separation=counts[np.arange(n),at],pair_resolved=observed.astype(float))


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);roster=settings() if selected is None else selected;rows=[];files={}
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        rows.append(dict(setting=s,measure='separation',**censored_blocks(out['separation'],s['steps'],s['blocks'])))
        for j in range(4):
            if j!=s['truth'][0]:rows.append(dict(setting=s,measure='pair_separation',rival=NAMES[j],**censored_blocks(out['pair_separation'][:,j],s['steps'],s['blocks'])))
        for key in ('fitted_theta','fitted_precision','conditional_in_sample_loss','move_share','excursions'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        if s['truth'][0] in (1,2):rows.append(dict(setting=s,measure='excursions_until_other_band_rule_separates_given_resolution',**conditional_blocks(out['excursions_at_pair_separation'],out['pair_resolved']>0,s['blocks'])))
        pulse(family='B1',cell=i+1,cells=len(roster))
    result=dict(family='B1',settings=roster,rows=rows,no_registered_claim=True,simple_reader_limit='POINT fitted conditionally on ordinary actions; move events have zero support and are reported separately. In-sample fit loss is not held-out predictive loss.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for base in (settings()[0],settings()[6],settings()[12],settings()[18]):
        s=base|dict(steps=64,worlds=8,blocks=8);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(s['truth'])+' independent full path and fit',passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
    params=np.array([[1,12,1,2],[2,12,1,2]],float);p,_,_=probabilities(np.array([12.,13.]),np.zeros(2),params,np.zeros((2,2),bool),{})
    checks.append(dict(name='untriggered EDGE and CENTRE equal inside band',passed=bool(np.array_equal(p[:,0],p[:,1]))))
    p,active,_=probabilities(np.array([8.]),np.array([False]),params,np.zeros((1,2),bool),{});q,_,_=probabilities(np.array([11.]),np.array([False]),params,active,{})
    checks.append(dict(name='after excursion CENTRE stays active and differs on reentry',passed=bool(active[0,1]) and bool(np.max(abs(q[:,0]-q[:,1]))>.01)))
    params=candidates();p,_,_=probabilities(np.array([0.,12.,24.]),np.ones(3),params,np.zeros((3,len(params)),bool),{})
    checks.append(dict(name='choice laws normalized and move opportunity exact',passed=bool(np.allclose(p.sum(2),1)) and bool(np.all(p[:,:,3][:,params[:,0]==3]==.8)) and bool(np.all(p[:,:,3][:,params[:,0]!=3]==0))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result