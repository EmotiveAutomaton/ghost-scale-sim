"""Prefix-selected family repair tested on independently generated future records."""
import json
import math
import numpy as np
from . import world as W

KINDS=('in-family','near-family','missing-rule','missing-acquisition','missing-opportunity','outside-menu')


def variants(w):
    # Fixed catalog; the truth is neither inserted nor selected using future outcomes.
    return [dict(w),dict(w,rule='satisficing'),dict(w,coupled=not w['coupled']),
            dict(w,endogenous=not w['endogenous']),dict(w,noise=.15)]


def fit(w,history,power=1.):
    weights=np.ones(len(W.STATES))/len(W.STATES);score=0.
    for obs in history:
        lik=W.likelihood(w,obs)**power
        evidence=float(weights@lik)
        if evidence<=0:return None,-math.inf
        score+=math.log(evidence);weights*=lik;weights/=weights.sum()
    return weights,score


def unit(index,cell=0,kind='missing-rule',order='late',split='test'):
    r=W.rng('revision',split,index,kind)
    supplied=W.make_world(cell,index);truth=dict(supplied)
    if kind=='near-family':truth['noise']=.10
    elif kind=='missing-rule':truth['rule']='satisficing' if truth['rule']=='softmax' else 'softmax'
    elif kind=='missing-acquisition':truth['coupled']=not truth['coupled']
    elif kind=='missing-opportunity':truth['endogenous']=not truth['endogenous']
    elif kind=='outside-menu':truth.update(rule='lexicographic',coupled=not truth['coupled'],noise=.06)
    state=int(r.integers(len(W.STATES)))
    records=[W.observe(truth,state,W.QUERIES[t%len(W.QUERIES)],r,f'prefix-{t}') for t in range(12)]
    # A supplied context cue endorses a predeclared candidate, possibly false.
    endorsed=index%5
    cut=0 if order=='early' else 8
    candidate_worlds=variants(supplied)
    if supplied['rule']=='satisficing':candidate_worlds[1]['rule']='softmax'
    fits=[fit(w,records) for w in candidate_worlds]
    # Cue is a fallible prior, not a truth label. Early/late sequential Bayes
    # must commute for the fixed model; bounded prefix selection need not.
    priors=np.ones(5);priors[endorsed]=4.;priors/=priors.sum()
    # Bounded revision happens once, when the context cue arrives. Later
    # observations reweight states but cannot silently reopen that decision.
    prefit=[fit(w,records[:cut])[1] for w in candidate_worlds]
    selected=int(np.argmax(np.array(prefit)+np.log(priors)))
    scores=np.array([x[1] for x in fits])+np.log(priors)
    mix=np.exp(scores-scores.max());mix/=mix.sum()
    fixed,log_evidence=fits[0];cautious,_=fit(supplied,records,.5)
    future=[W.observe(truth,state,W.FUTURES[j%4],r,f'future-{j}') for j in range(8)]
    rows=[]
    for method in ('fixed','cautious','empirical','revision','mixture','abstain','cue-only'):
        output=[]
        for j,obs in enumerate(future):
            c=obs['context'];q=W.artifact_matrix(truth,c)[state]
            if method=='fixed' or method=='abstain':p=fixed@W.artifact_matrix(supplied,c)
            elif method=='cautious':p=cautious@W.artifact_matrix(supplied,c)
            elif method=='revision':p=fits[selected][0]@W.artifact_matrix(candidate_worlds[selected],c)
            elif method=='cue-only':p=fits[endorsed][0]@W.artifact_matrix(candidate_worlds[endorsed],c)
            elif method=='mixture':p=sum(m*(fit_[0]@W.artifact_matrix(w,c)) for m,fit_,w in zip(mix,fits,candidate_worlds))
            else:
                p=np.ones(16)*.5
                for h in records:
                    distance=sum(h['context'][k]!=c[k] for k in ('goal','signal','budget'))
                    p[h['artifact']]+=math.exp(-distance)
                p/=p.sum()
            abstained=method=='abstain' and log_evidence < -58.
            if abstained:p=np.ones(16)/16
            output.append(dict(probabilities=p.tolist(),truth=q.tolist(),observed=obs['artifact'],
                               expected_loss=W.loss_record(W.cross_entropy(q,p)),
                               observed_loss=W.loss_record(-math.log(p[obs['artifact']])),abstained=abstained))
        rows.append(dict(method=method,forecasts=output,selected=selected if method=='revision' else None,
                         candidate_evaluations=5*len(records)*len(W.STATES) if method in ('revision','mixture') else len(records)*len(W.STATES)))
    return dict(family='D',index=index,cell=cell,kind=kind,order=order,
                public=json.loads(W.packet(supplied,records)),cue=dict(endorsed=endorsed,received_after=cut),
                evaluator=dict(truth_world=truth,state=state,future=future),
                revision=dict(prefix_length=cut,selected=selected,log_evidence=[s for _,s in fits],mixture=mix.tolist()),rows=rows)
