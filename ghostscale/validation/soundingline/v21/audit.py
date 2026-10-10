"""A01 opening: reconstruct the V20 duplicate-factor effect from retained inputs.

Old worlds are explicitly reused data. No V20 runner is invoked or file changed.
"""
from pathlib import Path
import numpy as np
from ..v18_3.io import read,write,file_digest
from ..v20 import world as O,studies as S
from .metrics import scores,check_scores,mean_scores,bootstrap
from .oracle import normalize_log

def counts(x,y):
    prior=np.bincount(y,minlength=128)+.5;prior/=prior.sum()
    c=np.full((128,x.shape[1],3),.5)
    for j in range(x.shape[1]):np.add.at(c[:,j,:],(y,x[:,j]+1),1)
    c/=c.sum(2,keepdims=True)
    return prior,c

def forecast(model,x,drop=(),power=1.,support=None):
    prior,c=model;log=np.tile(np.log(prior),(len(x),1))
    for j in range(x.shape[1]):
        if j%13 not in drop:log+=power*np.log(c[:,j,x[:,j]+1].T)
    if support is not None:log=np.where(support,log,-np.inf)
    return normalize_log(log)

def opening(root,d,pulse):
    oldroot=Path(d['retained_packets']);summaries=[];bindings={}
    for fit,packet in enumerate(d['retained_candidates']):
        src=oldroot/packet;complete=read(src/'COMPLETE.json');plan=read(src/'PLAN.json')
        if file_digest(src/'PLAN.json')!=complete['plan_sha256']:raise ValueError('retained plan changed')
        bindings[packet]=file_digest(src/'COMPLETE.json')
        train=src/'training/examples_points.npz'
        if file_digest(train)!=complete['files']['training/examples_points.npz']:raise ValueError('retained training changed')
        z=np.load(train);model=counts(z['evidence'],z['labels']);n=plan['design']['test_cases']
        for lineage in d['lineages']:
            for shift in ('native','changed-tool'):
                pulse(phase='duplicate-audit',fit=fit,lineage=lineage,shift=shift)
                rows=O.enumerate_world(lineage,shift);sample=O.sampled(rows,n,'test',lineage,shift)
                x,hh=S.inputs(rows,sample,'context',4,('test',lineage,shift));y=sample[:,10].astype(int)
                _,support=S.posterior(S.reference(rows,'context'),x[:,:13],[O.project(rows[h],'context') for h in hh])
                for name,drop in [('naive-bayes',()),('without-evidence',(1,)),('without-claim',(0,))]:
                    lp=forecast(model,x,drop,support=support);a=scores(lp,y);check_scores(lp,y,a)
                    target=root/'raw'/f'{fit}-{lineage}-{shift}-{name}_points.npz';target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,logp=lp,truth=y,**a)
                    if name=='naive-bayes' and shift=='changed-tool':
                        saved=src/'raw'/f'{lineage}-structured-joint_points.npz'
                        if file_digest(saved)!=complete['files'][saved.relative_to(src).as_posix()]:raise ValueError('retained forecast changed')
                        if not np.allclose(np.exp(lp),np.load(saved)['prediction'],rtol=1e-10,atol=1e-12):raise ValueError('V20 reproduction failed')
                    summaries.append(dict(fit=fit,lineage=lineage,shift=shift,reader=name,metrics=mean_scores(a),
                        claim_evidence_agreement=float((x[:,0]==x[:,1]).mean())))
    contrasts={}
    for reader in ('naive-bayes','without-evidence','without-claim'):
        values=[]
        for lineage in d['lineages']:
            changes=[]
            for fit in range(len(d['retained_candidates'])):
                get=lambda shift:next(r['metrics']['log_loss'] for r in summaries if r['lineage']==lineage and r['fit']==fit and r['reader']==reader and r['shift']==shift)
                changes.append(get('changed-tool')-get('native'))
            values.append(np.mean(changes))
        contrasts[reader]=bootstrap(values)
    write(root/'RETAINED_BINDINGS.json',bindings)
    return dict(card='G21-A01',phase='retained duplicate-evidence case',population=summaries,shift_minus_native=contrasts,
        scope='Constructed method; reused V20 worlds; not fresh confirmation; no converged-reader claim',
        controls=dict(retained_prediction_reproduction=True,independent_score_regrouping=True))
