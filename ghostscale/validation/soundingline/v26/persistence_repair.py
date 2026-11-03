"""R13: paired cue harm with prospective graded misleading frequencies."""
import math
import numpy as np
from .common import rng,posterior,record
from ..v25.persistence import HYPOTHESES,policies,controls

def block(setting,identity,independent=False):
    r=rng(*identity);rate=setting['learning_rate'];truths=[0,1+(0.,.05,.2).index(rate),4,5,6,7];out=[];raw=[]
    for maker,truth in enumerate(truths):
        state=np.array([.3,.3]);trust=np.ones(8);ll=np.zeros(8);obs=[];hit=None
        for t in range(160):
            amounts=np.array([[.3,0],[0,.3],[.15,.15]])*r.uniform(.7,1.3,(3,1));cue=np.minimum(amounts,1-state).sum(1)
            harmful=t>=32 and r.random()<setting['mislead'];actual=-cue if harmful else cue.copy()
            law=policies(actual,cue,trust,independent);y=int(r.choice(3,p=law[truth]));ll+=np.log(law[:,y]);p=posterior(ll,independent=independent)
            rival=4 if truth!=4 else 1+(0.,.05,.2).index(rate);ev=math.log(p[truth]/p[rival]);n=max(0,t-31)
            if t>=32 and ev>=1 and hit is None:hit=n
            if t==31 or t>=32:out.append(record(dict(evidence_vs_rival=ev,truth_probability=float(p[truth])),truth=truth,maker=maker,reader='type-rate-weight',history=n))
            obs.append(dict(state=state.tolist(),cue=cue.tolist(),actual=actual.tolist(),harmful=harmful,choice=y,trust_before=trust.tolist()))
            for i,(kind,lr,w) in enumerate(HYPOTHESES):
                if kind=='learner':trust[i]+=lr*((-1 if harmful else 1)-trust[i])
            state=np.clip(.95*state+amounts[y]*(-1 if harmful else 1),0,1)
        out.append(record(dict(records_to_one_nat=hit,not_within_128=float(hit is None)),truth=truth,maker=maker,reader='threshold'))
        raw.append(dict(truth=truth,records=obs))
    return dict(public=dict(hypotheses=HYPOTHESES,mislead=setting['mislead'],choice_model='inherited W13 .98 softmax beta3 plus .02 lapse; after-choice learning'),truth_and_observations=raw,readouts=out)

def design_check(settings):
    # Enumerated positive-probability harmful branch, not selection of scientific trajectories.
    rows=[]
    for s in settings:
        lr=s['learning_rate'];trust=1+lr*(-1-1)
        rows.append(dict(setting=s,changed=lr==0 or trust!=1,probability_at_least_one=1-(1-s['mislead'])**128))
    return dict(passed=all(x['changed'] for x in rows),checks=rows,seed_domain='enumerated-R13')
