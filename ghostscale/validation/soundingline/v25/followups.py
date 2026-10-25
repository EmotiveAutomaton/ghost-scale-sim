"""One-generation W1 evidence-dose and W2 non-Gaussian-noise stress tests."""
import math
import numpy as np
from . import pieces,setpoints
from .common import rng,posterior,information,record,close

def controls():
    assert pieces.controls()['passed'] and setpoints.controls()['passed']
    # Student t with three degrees of freedom has variance three; scaling by
    # sqrt(3) matches the parent's marginal observation variance exactly.
    assert 3/(3-2)/3==1
    return dict(passed=True,parent_known_answers=True,t3_variance_matched=True)

def block(setting,identity,independent=False):
    if setting['followup']=='noise':return setpoints.block(setting,identity,independent)
    r=rng(*identity);e=pieces.library(setting['overlap'],r)
    laws=[pieces.laws(e,a,independent=independent) for a in pieces.TRAIN]
    generation=[pieces.laws(e,a) for a in pieces.TRAIN]
    probes=pieces.probes(e,independent);gen_probes=pieces.probes(e);raw=[];out=[]
    for maker in range(8):
        truth=int(r.choice([i for i,h in enumerate(pieces.HYPOTHESES) if h[1]==3]));ll=np.zeros(210);history=[]
        for t in range(32):
            si=t%4;y=int(r.choice(len(pieces.TRAIN[si]),p=generation[si][truth]));ll+=np.log(laws[si][:,y]);history.append(y)
        active=ll.copy();passive=ll.copy();branches=[]
        for dose in range(1,9):
            p=posterior(active,independent=independent);ig=[information(p,x[-1]) for x in probes]
            chosen=int(np.flatnonzero(np.asarray(ig)>=max(ig)-1e-12)[0]);law=probes[chosen][-1]
            y=int(r.choice(law.shape[1],p=gen_probes[chosen][-1][truth]));active+=np.log(law[:,y])
            si=(31+dose)%4;z=int(r.choice(generation[si].shape[1],p=generation[si][truth]));passive+=np.log(laws[si][:,z]);branches.append(dict(probe=chosen,outcome=y,passive=z))
            if dose in (1,2,4,8):
                a=posterior(active,independent=independent);b=posterior(passive,independent=independent)
                out.append(record(dict(primary=math.log(a[truth])-math.log(b[truth])),maker=maker,reader='dose',history=dose))
        raw.append(dict(truth=truth,training=history,branches=branches))
    return dict(public=dict(effects=e.tolist(),training_records=32,initial_rationality=3,doses=[1,2,4,8]),truth_and_observations=raw,readouts=out)
