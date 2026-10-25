"""W6: logged and unlogged frustration triggers with inferred and assumed links."""
import math
import numpy as np
from . import pieces
from .common import rng,posterior,supported_posterior,scores,record,close

HYPOTHESES=[(w,3,h) for w in range(15) for h in range(-1,6)]

def signal_laws(e,k,kappa,independent=False):
    result=[]
    for wi,b,h in HYPOTHESES:
        value=math.fsum(float(pieces.WEIGHTS[wi,j]*e[k,j]) for j in range(3)) if independent else float(pieces.WEIGHTS[wi]@e[k])
        result.extend([1-math.exp(-kappa*value),1-math.exp(-kappa*(int(k==h)+.2))])
    return np.array(result).reshape(105,2)

def count_likelihood(probabilities,count,independent=False):
    # Poisson-binomial: sum over every unlogged outcome-block assignment.
    p=np.asarray(probabilities);distribution=np.ones((p.shape[0],1))
    for column in p.T:
        nxt=np.zeros((len(p),distribution.shape[1]+1));nxt[:,:-1]+=distribution*(1-column[:,None]);nxt[:,1:]+=distribution*column[:,None];distribution=nxt
    if not independent:return distribution[:,count]
    # Independent polynomial coefficient via scalar elementary recurrence.
    answer=[]
    for row in p:
        coeff=[1.]
        for q in row:
            coeff=[(coeff[k]*(1-q) if k<len(coeff) else 0)+(coeff[k-1]*q if k else 0) for k in range(len(coeff)+1)]
        answer.append(coeff[count])
    return np.array(answer)

def ends(p):return None if p is None else p.reshape(15,-1).sum(1)

def gain(p,q,truth):return math.log(p[truth])-math.log(q[truth]) if p is not None and p[truth]>0 and q is not None and q[truth]>0 else None

def log_end_mass(logs,weight,independent=False):
    logs=np.asarray(logs).reshape(15,-1)
    def total(x):
        largest=float(np.max(x))
        if not math.isfinite(largest):return -math.inf
        if independent:return largest+math.log(math.fsum(math.exp(float(v-largest)) for v in x.ravel()))
        return largest+math.log(float(np.exp(x-largest).sum()))
    denominator=total(logs)
    return None if not math.isfinite(denominator) else total(logs[weight])-denominator

def block(setting,identity,independent=False):
    r=rng(*identity);e=pieces.library(setting['overlap'],r);b=setting['block_probability'];kappa=setting['kappa'];generator=0 if setting['generator']=='value' else 1;temperament=setting['temperament']
    choice=[pieces.laws(e,a,HYPOTHESES,independent=independent) for a in pieces.TRAIN]
    signals=[signal_laws(e,k,kappa,independent) for k in range(6)];out=[];raw=[]
    length=setting.get('length',512)
    for maker in range(8):
        truth=int(r.integers(105));weight=HYPOTHESES[truth][0];logs=dict(choices=np.zeros(105),joint=np.zeros((105,2)),count=np.zeros((105,2)),assumed=np.zeros(105));history=[];count_probs=[];count_signal=0;blocks=0;first_record=None;first_blocks=None
        for t in range(length):
            si=t%4;local=int(r.choice(len(pieces.TRAIN[si]),p=choice[si][truth]));k=pieces.TRAIN[si][local];blocked=bool(r.random()<b);blocks+=blocked
            pfire=float(signals[k][truth,generator]) if blocked else 0.;pfire=1-(1-pfire)*(1-temperament);fire=bool(r.random()<pfire)
            lc=np.log(choice[si][:,local]);logs['choices']+=lc;logs['joint']+=lc[:,None];logs['count']+=lc[:,None];logs['assumed']+=lc
            probability=signals[k] if blocked else np.zeros((105,2))
            with np.errstate(divide='ignore'):
                logs['joint']+=np.log(probability if fire else 1-probability)
                logs['assumed']+=np.log(probability[:,0] if fire else 1-probability[:,0])
            count_probs.append((b*signals[k]).ravel());count_signal+=fire
            if (t+1)%8==0:
                law=count_likelihood(np.array(count_probs).T,count_signal,independent).reshape(105,2)
                with np.errstate(divide='ignore'):logs['count']+=np.log(law)
                count_probs=[];count_signal=0
            history.append(dict(piece=k,situation=si,blocked=blocked,signal=fire))
            if first_record is None and blocks:
                a=log_end_mass(logs['joint'],weight,independent);base=log_end_mass(logs['choices'],weight,independent)
                if a is not None and base is not None and a-base>=1:
                    first_record=t+1;first_blocks=blocks
            if t+1 in (8,32,128,512):
                p2=ends(posterior(logs['choices'],independent=independent));ps={'choices':p2,'blocks':p2}
                for key in ('joint','count','assumed'):ps[key]=ends(supported_posterior(logs[key].ravel(),independent))
                for reader,p in ps.items():
                    metrics=scores(p,weight) if p is not None else dict(truth_probability=None,log_gain=None,half_brier=None,accuracy=None)
                    metrics['empty_support']=float(p is None);metrics['zero_truth_support']=float(p is not None and p[weight]==0)
                    out.append(record(metrics,maker=maker,reader=reader,history=t+1))
                out.append(record(dict(primary=gain(ps['joint'],p2,weight),assumed_gain=gain(ps['assumed'],p2,weight),unlogged_gain=gain(ps['count'],ps['joint'],weight),outcome_blocks=blocks),maker=maker,reader='contrasts',history=t+1))
        out.append(record(dict(records_to_one_nat=first_record,outcome_blocks_to_one_nat=first_blocks,not_within_512=float(first_record is None)),maker=maker,reader='observed_first_passage'))
        raw.append(dict(truth=truth,generator=generator,observations=history))
    for n in (8,32,128,512):
        vals=[x['metrics'] for x in out if x['labels']['reader']=='contrasts' and x['labels']['history']==n]
        if not vals:continue
        available=all(x['primary'] is not None for x in vals);den=sum(x['outcome_blocks'] for x in vals)
        rate=math.fsum(x['primary'] for x in vals)/den if available and den else None
        out.append(record(dict(nats_per_outcome_block=rate,outcome_blocks=den,unavailable_cases=sum(x['primary'] is None for x in vals)),reader='pooled_event_rate',history=n))
    # Exact signature alias classification on the full beta-three library.
    arrays=np.stack([signals[k][:,0] for k in range(6)],axis=1);train=np.concatenate(choice,axis=1)
    alias=0;broken=0
    for i in range(105):
        for j in range(i):
            if np.max(abs(train[i]-train[j]))<=1e-12:
                alias+=1;broken+=np.max(abs(arrays[i]-arrays[j]))>1e-12
    out.append(record(dict(passive_alias_pairs=alias,signal_broken_alias_pairs=int(broken),signal_broken_alias_share=float(broken/alias) if alias else None),reader='alias_map'))
    return dict(public=dict(effects=e.tolist(),kappa=kappa,block_probability=b),truth_and_observations=raw,readouts=out)

def controls():
    e=np.array([[1.,1.,0],[0,0,1],[.5,.5,0],[0,0,.5],[1,1,0],[0,0,1]])
    a=next(i for i,x in enumerate(pieces.WEIGHTS) if np.allclose(x,[.75,.25,0]));b=next(i for i,x in enumerate(pieces.WEIGHTS) if np.allclose(x,[.25,.75,0]))
    for k in range(6):
        s=signal_laws(e,k,3);assert abs(s[7*a,0]-s[7*b,0])<1e-12;close(s.tolist(),signal_laws(e,k,3,True).tolist(),tol=1e-12)
    p=np.ones(105)/105;joint=np.repeat(p[:,None]/2,2,axis=1)
    assert np.max(abs(joint.sum(1)-p))<1e-12
    x=np.array([[.1,.2,.3],[.2,.7,.8]])
    close(count_likelihood(x,1).tolist(),count_likelihood(x,1,True).tolist(),tol=1e-12)
    logs=np.arange(210,dtype=float).reshape(105,2)/10
    for weight in (0,7,14):
        assert abs(log_end_mass(logs,weight)-log_end_mass(logs,weight,True))<1e-12
        assert abs(log_end_mass(logs,weight)-math.log(ends(posterior(logs.ravel()))[weight]))<1e-12
    assert log_end_mass(np.full((105,2),-math.inf),0) is None
    return dict(passed=True,shared_piece_alias=True,zero_block_identical_posterior=True,independent_signal_and_count_likelihoods=True)
