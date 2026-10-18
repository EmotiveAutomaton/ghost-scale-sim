"""N13: independently calibrated, matched-consistency FX1 falsifier miniature."""
from functools import lru_cache
import math
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.stats import t as student_t
from .common import rng
from . import finite_common as c
FAMILY='N13';CLASSES=4;CONTEXTS=8
AGREEMENT=.4217687074829932
AMPLITUDE=math.sqrt((AGREEMENT-.25)/(.75*np.mean(c.RELIABILITY**2)))
LAW=np.array([np.full((4,4),(1-r*AMPLITUDE)/4)+np.eye(4)*r*AMPLITUDE for r in c.RELIABILITY])

def variants():
    return [dict(id=f'N13-{strength}-{n}',family=FAMILY,settings=dict(strength=strength,history=n)) for strength in (0,1,2) for n in (32,81,250)]

def calibration(v):
    from .falsifier_calibration import CALIBRATION
    return CALIBRATION[v['id']]

def prior(q):
    out=np.full((4,8,4),(1-q)/4)
    for g in range(4):
        for j in range(8):out[g,j,(g+j)%4]+=q
    return out

def predictions(counts,q,goals=None,product=False):
    """Batched exact sum-product; personal context categories are marginalized."""
    counts=np.asarray(counts,float)
    if counts.ndim==2:counts=counts[None]
    ll=np.einsum('bjy,ldy->bljd',counts,np.log(LAW))
    local=softmax(ll,axis=-1);ln=logsumexp(ll,axis=-1)-math.log(4)
    n_law=softmax(ln.sum(-1),axis=-1)
    n=np.einsum('bl,bljd,ldy->bjy',n_law,local,LAW)
    logprior=np.log(prior(q));weighted=ll[:,:,None,:,:]+logprior[None,None]
    lg=logsumexp(weighted,axis=-1)
    joint=lg.sum(-1)
    if goals is not None:
        mask=np.arange(4)[None,None,:]==np.asarray(goals)[:,None,None]
        joint=np.where(mask,joint,-np.inf)
    mass=np.exp(joint-logsumexp(joint,axis=(1,2),keepdims=True))
    post=softmax(weighted,axis=-1)
    g=np.einsum('blg,blgjd,ldy->bjy',mass,post,LAW)
    if product:
        # Independent probability products, normalized per context to avoid underflow.
        scaled=np.exp(ll-np.max(ll,axis=-1,keepdims=True))
        evidence=np.einsum('bljd,gjd->blgj',scaled,prior(q))
        offsets=np.max(ll,axis=-1).sum(-1)
        check=np.prod(evidence,axis=-1)*np.exp(offsets-offsets.max(axis=1,keepdims=True))[:,:,None]
        if goals is not None:check*=mask
        check/=check.sum(axis=(1,2),keepdims=True)
        assert np.allclose(check,mass,atol=2e-12,rtol=0)
    return n,g,mass

def bank(n,q,phase,count=1024):
    r=rng(FAMILY,'calibration-bank',n,phase);world=np.arange(count)%16;goals=r.integers(4,size=count)
    dominant=r.integers(4,size=(count,8));isgoal=r.random((count,8))<q
    dominant=np.where(isgoal,(goals[:,None]+np.arange(8))%4,dominant)
    truth=LAW[world[:,None],dominant]
    contexts=r.integers(8,size=(count,n));u=r.random((count,n))
    outcomes=(u[:,:,None]>np.cumsum(truth[np.arange(count)[:,None],contexts],axis=-1)).sum(-1)
    counts=np.zeros((count,8,4),int)
    for i in range(count):np.add.at(counts[i],(contexts[i],outcomes[i]),1)
    return counts,truth,goals

def expected_loss(pred,truth):return .5*np.mean(1-2*np.sum(pred*truth,axis=-1)+np.sum(pred*pred,axis=-1),axis=-1)
def tempered(p,exponent):return softmax(exponent*np.log(p),axis=-1)

def calibration_risk(n,q,phase,count=1024):
    counts,truth,goals=bank(n,q,phase,count);losses=[]
    for start in range(0,count,64):
        sl=slice(start,start+64);pn,pg,_=predictions(counts[sl],q);_,po,_=predictions(counts[sl],q,goals[sl])
        losses.append(np.c_[expected_loss(pn,truth[sl]),expected_loss(po,truth[sl]),.5*np.mean(1-np.sum(truth[sl]**2,axis=-1),axis=-1)])
    return np.concatenate(losses)

def make_calibration():
    result={}
    for n in (32,81,250):
        for strength in (0,1,2):
            best=None;lo=0.;hi=.98
            for iteration in range(1 if strength==0 else 18):
                q=0. if strength==0 else (lo+hi)/2;loss=calibration_risk(n,q,'strength-fit',1024)
                margin=.1*float(np.mean(loss[:,0]-loss[:,2]))
                gain=float(np.mean(loss[:,0]-loss[:,1]));error=abs(gain-strength*margin)
                if best is None or error<best[0]:best=(error,q,gain,margin)
                if gain<strength*margin:lo=q
                else:hi=q
            error,q,gain,margin=best;check=calibration_risk(n,q,'independent-strength-check',1024)
            counts,truth,_=bank(n,q,'tempering-validation',1024);temperature_risks={}
            pn=np.concatenate([predictions(counts[start:start+64],q)[0] for start in range(0,1024,64)])
            for exponent in (.25,.5,.75,1.,1.25,1.5,2.):
                temperature_risks[str(exponent)]=float(np.mean(expected_loss(tempered(pn,exponent),truth)))
            exponent=float(min(temperature_risks,key=temperature_risks.get))
            result[f'N13-{strength}-{n}']=dict(q=q,margin=margin,fit_oracle_gain=gain,target=strength*margin,
                absolute_calibration_error=error,independent_oracle_gain=float(np.mean(check[:,0]-check[:,1])),
                independent_gain_se=float(np.std(check[:,0]-check[:,1],ddof=1)/32),
                independent_headroom=float(np.mean(check[:,0]-check[:,2])),
                exponent=exponent,validation_losses=temperature_risks,calibration_cases=1024,
                unit='half-Brier',agreement_pooled_over_laws=AGREEMENT)
    return result

def sample(v,phase,world,case):
    n=v['settings']['history'];q=calibration(v)['q'];r=rng(FAMILY,'paired',n,phase,world,case)
    goal=int(r.integers(4));dominant=r.integers(4,size=8);mask=r.random(8)<q
    dominant=np.where(mask,(goal+np.arange(8))%4,dominant);truth=LAW[world%16,dominant]
    contexts=r.integers(8,size=n);outcomes=(r.random(n)[:,None]>np.cumsum(truth[contexts],axis=-1)).sum(-1)
    observations=[dict(context=[int(j)],y=int(y)) for j,y in zip(contexts,outcomes,strict=True)]
    # Every predeclared context is goal-bearing; query order is randomly paired.
    query=[[int(j)] for j in r.permutation(8)];y=[c.draw(truth[j[0]],u) for j,u in zip(query,r.random(8),strict=True)]
    return dict(design=dict(history=n,q=q),observations=observations,query=query),dict(y=y,mechanism=goal,law=world%16,personal=truth.tolist())

def count_records(p):
    counts=np.zeros((8,4),int)
    for row in p['observations']:counts[row['context'][0],row['y']]+=1
    return counts

def evaluate(v,p,t):
    setting=calibration(v);pn,pg,mass=predictions(count_records(p),p['design']['q'],product=True)
    pn,pg=pn[0],pg[0];order=[q[0] for q in p['query']];pn,pg=pn[order],pg[order]
    pt=tempered(pn,setting['exponent']);personal=np.asarray(t['personal'])[order];pooled=np.full((8,4),.25)
    actual=np.eye(4)[t['y']];loss=lambda prob:.5*float(np.mean(np.sum((prob-actual)**2,axis=-1)))
    ln,lg,lt,lp,l0=map(loss,(pn,pg,pt,personal,pooled))
    return dict(records=pn,exact=pg,tempered=pt,personal_oracle=personal,pooled=pooled),dict(goal_posterior=mass.sum(axis=1)[0].tolist()),dict(
        primary=ln-lg,tempered_gain=lt-lg,records_over_pooled=l0-ln,headroom=ln-lp,records_loss=ln,
        oracle_loss=lp,goal_loss=lg,margin=setting['margin'])

def features(v,p):
    return np.r_[count_records(p).ravel()/max(1,p['design']['history']),np.array(p['query']).ravel(),p['design']['history']/250,p['design']['q']]
def emit(p,schema,identity):return c.emit(p,schema,identity,CLASSES)
decode=c.decode

def block_readout(v,world,metrics):
    def estimate(values):
        x=np.array(values);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
        radius=float(student_t.ppf(.975,len(x)-1))*se if len(x)>1 else 0.
        return dict(mean=mean,lower=mean-radius,upper=mean+radius)
    gain=estimate(metrics['primary']);headroom=estimate(metrics['headroom']);pooled=estimate(metrics['records_over_pooled']);margin=calibration(v)['margin']
    return dict(gain=gain,headroom=headroom,records_over_pooled=pooled,margin=margin,
        supported=bool(gain['lower']>0 and gain['mean']>=margin),powered=bool(gain['lower']>0),
        falsifiable_local=bool(headroom['lower']>0 and pooled['mean']>=0),
        low_gain=bool(gain['upper']<margin),cases=len(metrics['primary']))

def study_readout(summaries):
    byid={x['variant']['id']:x for x in summaries};power=sum(x['powered'] for x in byid['N13-1-81']['block_readouts'])
    false=sum(x['supported'] for x in byid['N13-0-81']['block_readouts'])
    cal=calibration(byid['N13-1-81']['variant']);qualified=abs(cal['independent_oracle_gain']-cal['target'])<=3*cal['independent_gain_se']+.1*cal['margin']
    passes=power>=13 and false==0 and qualified
    verdicts={}
    for key,item in byid.items():
        rows=item['block_readouts'];out=[]
        for row in rows:
            out.append('SUPPORTED' if row['supported'] else 'FALSIFIED' if passes and row['falsifiable_local'] and row['low_gain'] else 'INCONCLUSIVE')
        verdicts[key]=dict(verdicts=out,counts={label:out.count(label) for label in ('SUPPORTED','FALSIFIED','INCONCLUSIVE')})
    return dict(primary=dict(powered_blocks=power,required=13,false_supported_blocks=false,allowed=0,P3_passed=passes,independent_strength_calibration_qualified=qualified),
        verdicts=verdicts,unavailable_sensitivities=['N13-1-32-consistency70','N13-1-81-consistency70','N13-1-250-consistency70'],
        interpretation='Conditional method validation only; does not satisfy real-data FX1 P1/P2/P4/P5.')

def controls():
    from .falsifier_calibration import CALIBRATION
    assert set(CALIBRATION)=={v['id'] for v in variants()}
    assert all(0<=row['q']<1 and row['margin']>0 and row['absolute_calibration_error']<.05*row['margin'] for row in CALIBRATION.values())
    assert np.all(LAW>0) and np.allclose(LAW.sum(-1),1)
    agreement=np.sum(LAW[:,0]**2,axis=-1);assert abs(agreement.mean()-AGREEMENT)<1e-14
    # Habit-only: both Bayesian predictors coincide exactly, regardless of records.
    counts=np.zeros((8,4));counts[:,0]=12;pn,pg,_=predictions(counts,0.,product=True);assert np.allclose(pn,pg,atol=1e-13,rtol=0)
    minimum=1.
    for law in range(16):
        for goal in range(4):
            counts=np.zeros((8,4))
            for j in range(8):counts[j]=np.round(80*LAW[law,(goal+j)%4])
            _,_,mass=predictions(counts,CALIBRATION['N13-2-81']['q'],product=True);post=mass.sum(axis=1)[0];minimum=min(minimum,float(post[goal]));assert post[goal]>.25
    return dict(passed=True,habit_alias_exact=True,planted_goal_minimum_posterior=minimum,
        pooled_agreement=AGREEMENT,half_brier_units=True,probability_products_agree=True,
        evaluator_labels_excluded=True,miniature_architecture_untested=True)
