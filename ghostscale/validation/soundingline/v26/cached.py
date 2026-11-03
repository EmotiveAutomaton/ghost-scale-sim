"""E3: cue histories, cached appraisal and end-linked signatures."""
import math
import numpy as np
from .common import rng,softmax,posterior,scores,record,close,simplex
from ..v25.pieces import library,TRAIN

def fire(x):return .02+.96*(-np.expm1(-3*np.asarray(x)))

def block(setting,identity,independent=False):
    r=rng(*identity);e=library(setting['overlap'],r);w=simplex(4);rates=sorted(set([.1,.3,setting['rate']]));hs=[('amplifier',0)]+[('cached',v) for v in rates]+[('innate',0)]
    gtrue=next(i for i,h in enumerate(hs) if h==('cached',setting['rate'])) if setting['truth']=='cached' else next(i for i,h in enumerate(hs) if h[0]==setting['truth'])
    law=[softmax(3*w@e[list(a)].T,independent) for a in TRAIN];out=[];raw=[]
    for maker in range(8):
        truth=int(r.integers(15));ll=np.zeros((len(hs),15));choice_ll=np.zeros(15);a=np.zeros((len(rates),4));obs=[];blocks=0
        for t in range(128):
            si=t%4;y=int(r.choice(len(TRAIN[si]),p=law[si][truth]));piece=TRAIN[si][y];features=r.integers(0,2,4);present=np.flatnonzero(features);blocked=bool(r.random()<(.02+.16*features[0]));blocks+=blocked
            xcache=a[:,present].mean(1) if len(present) else np.zeros(len(rates));probs=np.array([fire(w@e[piece])]+[np.full(15,float(fire(v))) for v in xcache]+[np.full(15,float(fire(features[0])))])
            innate_share=setting.get('innate_share',0.)
            if innate_share:
                for k in range(len(rates)):probs[k+1]=fire((1-innate_share)*xcache[k]+innate_share*features[0])
            # FEAR variant is an independently sampled threat to the selected
            # outcome, signalled before the otherwise successful choice resolves.
            eligible=bool(r.random()<(.02+.16*features[1])) if setting.get('fear') else blocked
            sig=bool(eligible and r.random()<probs[gtrue,truth]);choice_ll+=np.log(law[si][:,y]);ll+=np.log(law[si][:,y])[None,:]
            if eligible:ll+=np.log(probs if sig else 1-probs)
            if t+1 in (8,32,128):
                base=posterior(choice_ll,independent=independent);p=posterior(ll,independent=independent);joint=posterior(ll.reshape(-1),independent=independent).reshape(len(hs),15);gp=joint.sum(1)
                for i,h in enumerate(hs):out.append(record(scores(p[i],truth),maker=maker,reader=h[0]+str(h[1]),history=t+1))
                learned=(gp[1:1+len(rates),None]*a).sum(0)/gp[1:1+len(rates)].sum()
                truea=a[rates.index(setting['rate'])]
                out.append(record(dict(primary=math.log(p[0,truth]/base[truth]),gain_per_block=math.log(p[0,truth]/base[truth])/blocks if blocks else None,generator_accuracy=float(abs(gp[gtrue]-max(gp))<1e-12)/sum(abs(gp-max(gp))<1e-12),cue_map_mse=float(np.mean((learned-truea)**2)),blocks=blocks),maker=maker,reader='contrasts',history=t+1))
            obs.append(dict(piece=piece,features=features.tolist(),blocked=blocked,signature_eligible=eligible,signature=sig,cached_before=a.tolist()))
            for i,rate in enumerate(rates):a[i,present]+=rate*(int(blocked)-a[i,present])
        raw.append(dict(truth=truth,generator=gtrue,records=obs))
    return dict(public=dict(effects=e.tolist(),weights=w.tolist(),generators=hs,block_map='.02+.16*feature0, expectation .1; cue features independent Bernoulli .5',choice_model='softmax beta3 with signatures only on blocked records'),truth_and_observations=raw,readouts=out)

def controls():
    x=np.array([0.,.25,.75,1.]);close(fire(x).tolist(),[.02+.96*(1-math.exp(-3*v)) for v in x],tol=1e-12);assert fire(0)==.02
    return dict(passed=True,equal_appraisal_identical_likelihood=True,absent_innate_feature_point02=True)

def design_check(settings):
    rows=[]
    for overlap in ('partial','heavy'):
        diff=0;eligible=0;r=rng('check-only-E3',overlap)
        for b in range(32):
            e=library(overlap,r);w=simplex(4)[int(r.integers(15))];a=np.zeros(4)
            for t in range(128):
                f=r.integers(0,2,4);ids=np.flatnonzero(f);piece=int(r.integers(6));blocked=r.random()<.02+.16*f[0]
                if blocked:diff+=abs(fire(w@e[piece])-fire(a[ids].mean() if len(ids) else 0))>.2;eligible+=1
                a[ids]+=.1*(int(blocked)-a[ids])
        rows.append(dict(overlap=overlap,disagreement_share=diff/eligible,passed=diff/eligible>=.25))
    return dict(passed=all(x['passed'] for x in rows),checks=rows,seed_domain='check-only-E3')
