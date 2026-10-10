"""D02 exact observation-process experiment, with same-artifact counterexamples."""
import numpy as np
from . import world as W,oracle as O
from .studies import retain,controls
from ..v18_3.io import write

FIELDS=(4,6,7,5) # selection, revision, order, inspection

def missing_prob(rows,rate,mechanism):
    out=np.full((len(rows),4),rate)
    if mechanism=='event-dependent':out[:,3]=np.where(rows[:,5]==1,min(1,rate*1.5),rate*.5)
    elif mechanism not in ('independent','squashed','delayed'):raise ValueError('unknown observation process')
    return out

def generate(chosen,rate,mechanism,rng,window=2):
    values=chosen[:,FIELDS].astype(int);miss=rng.random(values.shape)<missing_prob(chosen,rate,mechanism)
    if mechanism=='squashed':
        # One deletion draw per consecutive event group: missingness is correlated.
        for start in range(0,4,window):miss[:,start:start+window]=miss[:,start:start+1]
    if mechanism=='delayed':
        # Whole trailing batch remains unobserved; never code it as a negative event.
        delays=rng.random(len(chosen))<rate;miss[:]=False;miss[delays,4-min(window,4):]=True
    return np.where(miss,-1,values)

def record_posterior(rows,current,records,rate,mechanism,mode,window=2):
    out=[];end=rows[:,11:14].astype(int);events=rows[:,FIELDS].astype(int);missprob=missing_prob(rows,rate,mechanism)
    for observed,rec in zip(current,records):
        mass=rows[:,-1]*np.all(end==observed,axis=1)
        if mode!='endpoint':
            for j,value in enumerate(rec):
                if value>=0:mass*=events[:,j]==value
                elif mode=='naive-negative':mass*=events[:,j]==0
                if mode=='aware' and mechanism in ('independent','event-dependent'):
                    mass*=missprob[:,j] if value<0 else 1-missprob[:,j]
            # Squash and delay mechanisms here are independent of the target.
            # Their pattern likelihood is common to candidates and cancels.
        p=np.bincount(rows[:,10].astype(int),weights=mass,minlength=128)
        if p.sum()==0:
            # Explicit contradiction policy: uniform physically compatible fallback.
            p=np.bincount(rows[:,10].astype(int),weights=rows[:,-1]*np.all(end==observed,axis=1),minlength=128)
        out.append(p/p.sum())
    with np.errstate(divide='ignore'):return np.log(np.array(out))

def run(root,d,pulse):
    # At zero history each coefficient occurs once: independence makes the exact
    # law-marginal joint equal to the joint at coefficient means.
    w=dict(route=4,select=4,revise=4,goal=4,inspect=4,skill=6.5,shift='native',presentation=0,lineage=-1)
    prior=W.enumerate_law(w);population=[];examples=[]
    for lineage in d['lineages']:
        rows=W.enumerate_world(lineage);chosen=W.sampled(rows,d['test_cases'],'v21-records',lineage);y=chosen[:,10].astype(int)
        for mechanism in d.get('mechanisms',['independent','squashed','delayed','event-dependent']):
            for rate in d.get('rates',[0,.25,.5,.75]):
                pulse(phase='record-missingness',lineage=lineage,mechanism=mechanism,rate=rate)
                rng=np.random.default_rng(W.seed('records',lineage,mechanism,rate));rec=generate(chosen,rate,mechanism,rng,d.get('window',2))
                for name in ('endpoint','observed-record','aware','naive-negative'):
                    lp=record_posterior(prior,chosen[:,11:14],rec,rate,mechanism,name,d.get('window',2))
                    result=retain(root,f'{lineage}-{mechanism}-{rate}-{name}',lp,y)
                    population.append(dict(lineage=lineage,mechanism=mechanism,rate=rate,reader=name,metrics=result))
                full=chosen[:,FIELDS].astype(int);lp=record_posterior(prior,chosen[:,11:14],full,0,'independent','observed-record')
                population.append(dict(lineage=lineage,mechanism=mechanism,rate=rate,reader='full-record',metrics=retain(root,f'{lineage}-{mechanism}-{rate}-full-record',lp,y)))
                path=root/'reader'/f'{lineage}-{mechanism}-{rate}_points.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,artifact=chosen[:,11:14].astype(int),observed_records=rec)
                examples.append(dict(lineage=lineage,mechanism=mechanism,rate=rate,artifact=chosen[0,11:14].tolist(),record=rec[0].tolist(),truth=int(y[0]),role='evaluator only'))
    write(root/'CASEBOOK_EVALUATOR.json',examples)
    return dict(card='G21-D02',population=population,controls=controls(),scope='Exact observation-process mechanism; no learned comparison; law-marginal prior at coefficient means; missing records are not mental-process ground truth')
