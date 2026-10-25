"""W13: delayed state evidence versus cue-valuing policies, under fixed dynamics."""
import math
import numpy as np
from .common import rng,softmax,posterior,record,scores,close

HYPOTHESES=[('homeostatic',0.,1.)]+[('learner',lr,1.) for lr in (0.,.05,.2)]+[('proxy',0.,0.)]+[('mixed',0.,w) for w in (.25,.5,.75)]

def policies(restoration,cue,trust,independent=False):
    values=[]
    for i,(kind,lr,weight) in enumerate(HYPOTHESES):
        if kind=='learner':v=trust[i]*cue
        elif kind=='proxy':v=cue
        else:v=weight*restoration+(1-weight)*cue
        values.append(.98*softmax(3*v,independent)+.02/len(v))
    return np.array(values)

def block(setting,identity,independent=False):
    r=rng(*identity);out=[];raw=[];shift=setting['shift'];lr=setting['learning_rate'];truths=[0,1+(0.,.05,.2).index(lr),4,5,6,7]
    for maker,truth in enumerate(truths):
        state=np.array([.3,.3]);trust=np.ones(len(HYPOTHESES));ll=np.zeros(len(HYPOTHESES));history=[];pending=None
        for t in range(160):
            training=t<32;amounts=np.array([[.3,0],[0,.3],[.15,.15]])*r.uniform(.7,1.3,(3,1));restore=np.minimum(amounts,1-state).sum(1);cue=restore.copy();actual=restore.copy()
            if not training:
                if shift=='cue-only':actual=np.zeros(3)
                elif shift=='restoration-only':cue=np.zeros(3)
                elif shift=='harmful-cue':actual=-restore
                # Delayed restoration has its ordinary expected value, but arrives
                # after the following choice. The learner cannot use it early.
            law=policies(actual,cue,trust,independent);choice=int(r.choice(3,p=law[truth]));ll+=np.log(law[:,choice])
            post=posterior(ll,independent=independent);delta=t-31
            if training and t==31 or not training:
                ratio=math.log(post[truth]/post[4]) if truth!=4 else math.log(post[4]/post[1+(0.,.05,.2).index(lr)])
                out.append(record(dict(evidence_vs_rival=ratio,truth_probability=float(post[truth]),log_gain=math.log(float(post[truth])*len(HYPOTHESES))),maker=maker,truth=truth,history=0 if training else delta,reader='type-rate-weight'))
            observed=float(actual[choice]);credited_cue=float(cue[choice]);applied=amounts[choice]*(1 if observed>0 else -1 if observed<0 else 0)
            origin=t
            if not training and shift=='delayed':
                arriving=pending;pending=(t,float(cue[choice]),amounts[choice].copy())
                origin=arriving[0] if arriving else None;credited_cue=arriving[1] if arriving else 0.
                applied=np.minimum(arriving[2],1-state) if arriving else np.zeros(2);observed=float(applied.sum())
            history.append(dict(training=training,state=state.tolist(),restoration=actual.tolist(),cue=cue.tolist(),choice=choice,delayed_state_evidence=observed,evidence_origin=origin))
            # Evidence arrives after this choice and can only influence the next one.
            target=observed/credited_cue if credited_cue>0 else 0.
            for h,(kind,rate,w) in enumerate(HYPOTHESES):
                if kind=='learner' and origin is not None:trust[h]+=rate*(target-trust[h])
            state=np.clip(.95*state+applied,0,1)
        raw.append(dict(truth=truth,observations=history))
    for truth in truths:
        rows=[v for v in out if v['labels']['truth']==truth and v['labels']['history']>0];hit=next((v['labels']['history'] for v in rows if v['metrics']['evidence_vs_rival']>=1),None)
        out.append(record(dict(records_to_one_nat=hit,not_within_128=float(hit is None)),truth=truth,reader='threshold'))
    return dict(public=dict(hypotheses=HYPOTHESES,choice_model='softmax beta 3 plus .02 uniform lapse',delayed_learning='after-choice state evidence used only for next trust update'),truth_and_observations=raw,readouts=out)

def controls():
    restore=np.array([.1,.2,.3]);p=policies(restore,restore,np.ones(8));assert np.max(abs(p-p[0]))<1e-12
    shifted=policies(-restore,restore,np.ones(8));close(shifted[1].tolist(),shifted[4].tolist(),tol=1e-12)
    close(p.tolist(),policies(restore,restore,np.ones(8),True).tolist(),tol=1e-12)
    return dict(passed=True,training_exact_alias=True,zero_learning_rate_proxy_alias=True,independent_softmax=True)
