"""Finite visible resource classes: surplus exposure, separation and weights."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng,softmax,mean,median,proportion,row,separation,read,write,digest

CLASSES=('TOOL2','TOOL4','SAVER10','SAVER15','SAVER20','CAPTURED3','CAPTURED5')
RESERVES=np.array([2,4,10,15,20,-1,-1])
WEIGHTS=np.array([(2.,2.),(3.,1.),(1.,3.)])
RATES=((.3,.1),(.3,.15),(.3,.2),(.5,.2),(.5,.25),(.5,.3),(.8,.15),(.3,.3))


def settings():
    return [dict(q_in=a,q_need=b,truth=truth,steps=512,initial='uniform_integer_0_30')
            for a,b in RATES for truth in (5,6,2,4,0)]


def keep(stock):
    costs=np.where(stock[:,None]<=RESERVES,3.,0.)
    costs[:,5]=3.;costs[:,6]=5.
    return costs


def probabilities(stock,needs):
    cost=keep(stock)
    gains=needs[:,None,None,:]*WEIGHTS[None,None,:,:]-cost[:,:,None,None]
    gains=np.where((stock[:,None,None,None]>0)&(needs[:,None,None,:]>0),gains,-np.inf)
    holds=np.zeros((*gains.shape[:-1],1))
    return softmax(np.concatenate([holds,gains],axis=-1))


def generate(setting,lane='study',worlds=400):
    r=rng('G3',lane,setting);steps=setting['steps'];stock=r.integers(0,31,worlds)
    if setting.get('initial')=='fixed_5':stock[:]=5
    need=np.zeros((worlds,2),int);weights=r.integers(3,size=worlds)
    initial=stock.copy();stocks=np.empty((worlds,steps),dtype=np.int8)
    needs=np.empty((worlds,steps,2),dtype=np.int8);actions=np.empty((worlds,steps),dtype=np.int8)
    inflows=np.empty_like(actions);increments=np.empty_like(needs)
    for t in range(steps):
        inflow=r.random(worlds)<setting['q_in'];inc=r.random((worlds,2))<setting['q_need']
        stock=np.minimum(30,stock+inflow);need=np.minimum(3,need+inc)
        stocks[:,t]=stock;needs[:,t]=need;inflows[:,t]=inflow;increments[:,t]=inc
        p=probabilities(stock,need)[np.arange(worlds),setting['truth'],weights]
        action=(r.random(worlds)[:,None]>np.cumsum(p,axis=1)).sum(axis=1)
        actions[:,t]=action
        for i in (1,2):
            selected=action==i;need[selected,i-1]-=1;stock[selected]-=1
    return dict(stocks=stocks,needs=needs,actions=actions,weights=weights,
        initial_stock=initial,inflows=inflows,increments=increments)


def infer(raw,setting):
    worlds,steps=raw['stocks'].shape;ll=np.zeros((worlds,7,3))
    last=np.full((worlds,7),-1,int);weight_last=np.full(worlds,-1,int)
    exposures=np.zeros((worlds,3),int);before=np.zeros((worlds,3),int);seen=np.zeros((worlds,3),bool)
    history=np.zeros((worlds,steps,3),dtype=bool)
    for t in range(steps):
        stock=raw['stocks'][:,t];need=raw['needs'][:,t]
        p=probabilities(stock,need)
        selected=np.take_along_axis(p,raw['actions'][:,t,None,None,None],axis=3)[:,:,:,0]
        with np.errstate(divide='ignore'):ll+=np.log(selected)
        class_ll=logsumexp(ll,axis=2)-np.log(3)
        odds=class_ll[:,setting['truth'],None]-class_ll
        last=np.where(odds<1,t,last)
        weight_ll=logsumexp(ll,axis=1)-np.log(7)
        weight_odds=weight_ll[:,1]-weight_ll[:,2]
        weight_last=np.where(weight_odds<1,t,weight_last)
        history[:,t]=(stock[:,None]>np.array([10,15,20]))&np.any(need>0,axis=1)[:,None]
    counts=separation(last,odds)
    for r in range(3):
        for w in range(worlds):
            # Moments up to the last below-bar step, literally before the next
            # sustained crossing. Censored cases retain a censored count too.
            before[w,r]=history[w,:last[w,r+2]+1,r].sum()
    return dict(separation=counts,last_below=last,final_odds=odds,
        exposure_world=history.any(axis=1),exposure_step=history.mean(axis=1),
        exposure_before=before.astype(float),weight_separation=separation(weight_last,weight_odds),
        weight_posterior=softmax(weight_ll)[np.arange(worlds),raw['weights']])


def run(destination,mode='original',original=None,pulse=lambda **kw:None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);rows=[];files={}
    from .g3_reference import infer as reconstruct
    for index,setting in enumerate(settings()):
        name=f'cell-{index:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:
            raw=generate(setting);np.savez_compressed(destination/name,**raw);files[name]=digest(destination/name)
        result=(reconstruct(raw,setting) if mode=='reconstruct' else infer(raw,setting))
        np.savez_compressed(destination/f'cell-{index:03d}_measures.npz',**result)
        for r,reserve in enumerate((10,15,20)):
            if setting['truth']==5:
                rows.append(row(setting,'last_below_plus_one',median(result['separation'][:,r+2]),rival='SAVER'+str(reserve)))
                moments=np.where(np.isinf(result['separation'][:,r+2]),np.inf,result['exposure_before'][:,r])
                rows.append(row(setting,'telling_moments_before_separation',median(moments),rival='SAVER'+str(reserve)))
            rows.append(row(setting,'worlds_with_telling_moment',proportion(result['exposure_world'][:,r]),reserve=reserve))
            rows.append(row(setting,'share_of_steps_with_telling_moment',mean(result['exposure_step'][:,r]),reserve=reserve))
        selected=raw['weights']==1
        rows.append(row(setting,'weight_3_1_against_1_3',median(result['weight_separation'][selected]),weights='true (3,1), rival (1,3)'))
        pulse(family='G3',cell=index+1,cells=len(settings()))
    result=dict(family='G3',rows=rows,scale_only=True,independent_units='worlds',
        separation_definition='literal zero-based last-below step + 1; final-below censored; zero means above on first step')
    write(destination/'SUMMARY.json',result)
    if mode!='reconstruct':write(destination/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    gates=[]
    stock=np.arange(31);empty=np.zeros((31,2),int);p=probabilities(stock,empty)
    gates.append(dict(name='no_need_no_likelihood_update',passed=bool(np.all(p[...,0]==1))))
    k=keep(stock)
    for i,reserve in enumerate((10,15,20)):
        gates.append(dict(name='same_keep_below_reserve_'+str(reserve),passed=bool(np.array_equal(k[stock<=reserve,5],k[stock<=reserve,i+2])),
            qualification='Conditional on a common weight pair; marginal odds can move after type-specific weight histories diverge.'))
    from .g3_reference import infer as reconstruct
    for typ in range(7):
        s=dict(q_in=.8,q_need=.15,truth=typ,steps=2000,initial='uniform_integer_0_30')
        raw=generate(s,'known-answer',100);a=infer(raw,s);b=reconstruct(raw,s)
        gates.append(dict(name='independent_reconstruction_'+CLASSES[typ],passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
        value=float(a['weight_posterior'].mean())
        gates.append(dict(name='true_weight_posterior_'+CLASSES[typ],passed=value>=.99,estimate=value,
            interpretation='Mean true-weight posterior over 100 excluded independent worlds in plentiful regime.',bar=.99))
        pulse(control=CLASSES[typ])
    result=dict(family='G3',passed=all(x['passed'] for x in gates),gates=gates)
    write(Path(destination)/'CONTROLS.json',result);return result
