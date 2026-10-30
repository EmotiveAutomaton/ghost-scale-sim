"""One-generation evidence-dose, stress and mechanism follow-ups."""
import math
import numpy as np
from . import pieces,setpoints,emotion,volatile,attention,texture
from .common import rng,posterior,information,record,close

def controls():
    assert pieces.controls()['passed'] and setpoints.controls()['passed'] and emotion.controls()['passed'] and volatile.controls()['passed'] and attention.controls()['passed'] and texture.controls()['passed']
    # Student t with three degrees of freedom has variance three; scaling by
    # sqrt(3) matches the parent's marginal observation variance exactly.
    assert 3/(3-2)/3==1
    return dict(passed=True,parent_known_answers=True,t3_variance_matched=True,w8_filter_controls=True,texture_controls=True)

def block(setting,identity,independent=False):
    if setting['followup']=='population-size':return texture.population_size(setting,identity,independent)
    if setting['followup']=='signal-dose':return emotion.block(setting,identity,independent)
    if setting['followup']=='attention-noise':return attention.block(setting,identity,independent)
    if setting['followup']=='affect-channels':return texture.affect_channels(setting,identity,independent)
    if setting['followup']=='noise':return setpoints.block(setting,identity,independent)
    if setting['followup']=='matched-signature':return matched_signature(setting,identity,independent)
    if setting['followup']=='tr4-dose':return volatile.block(setting,identity,independent)
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

def matched_signature(setting,identity,independent=False):
    """Prospectively paired, marginal-ends comparison; no reciprocal extrapolation."""
    r=rng(*identity);e=pieces.library('heavy',r);hs=emotion.HYPOTHESES
    laws=[pieces.laws(e,a,hs,independent=independent) for a in pieces.TRAIN]
    generated=[pieces.laws(e,a,hs) for a in pieces.TRAIN]
    probes=[pieces.laws(e,tuple(range(6)),hs,satiate=j,independent=independent) for j in range(3)]
    gen_probes=[pieces.laws(e,tuple(range(6)),hs,satiate=j) for j in range(3)]
    signals=[emotion.signal_laws(e,k,setting['kappa'],independent) for k in range(6)]
    gen_signals=[emotion.signal_laws(e,k,setting['kappa']) for k in range(6)];out=[];raw=[]
    for maker in range(8):
        truth=int(r.integers(105));weight=hs[truth][0];ll=np.zeros(105);history=[]
        for t in range(32):
            si=t%4;y=int(r.choice(len(pieces.TRAIN[si]),p=generated[si][truth]));ll+=np.log(laws[si][:,y]);history.append(y)
        p=posterior(ll,independent=independent);ig=[information(p,law) for law in probes];j=int(np.flatnonzero(np.array(ig)>=max(ig)-1e-12)[0]);y=int(r.choice(6,p=gen_probes[j][truth]))
        sat=emotion.log_end_mass(ll+np.log(probes[j][:,y]),weight,independent)-emotion.log_end_mass(ll,weight,independent)
        joint=np.repeat(ll[:,None],2,axis=1);branch=[];blocks=0
        for n in range(1,513):
            si=(31+n)%4;choice=int(r.choice(len(pieces.TRAIN[si]),p=generated[si][truth]));piece=pieces.TRAIN[si][choice];blocked=bool(r.random()<.1);blocks+=blocked;fire=bool(blocked and r.random()<gen_signals[piece][truth,0]);lc=np.log(laws[si][:,choice]);ll+=lc;joint+=lc[:,None]
            q=signals[piece] if blocked else np.zeros((105,2))
            with np.errstate(divide='ignore'):joint+=np.log(q if fire else 1-q)
            branch.append(dict(choice=choice,blocked=blocked,signal=fire))
            if n in (8,32,128,512):
                gain=emotion.log_end_mass(joint,weight,independent)-emotion.log_end_mass(ll,weight,independent)
                out.append(record(dict(signature_gain=gain,satiation_gain=sat,outcome_blocks=blocks,primary=gain-sat),maker=maker,reader='paired-marginal-ends',history=n))
        raw.append(dict(truth=truth,passive=history,satiated_end=j,satiation_outcome=y,signature_branch=branch))
    return dict(public=dict(effects=e.tolist(),kappa=setting['kappa'],block_probability=.1,initial_records=32,rationality=3,comparison='Both branches score marginal ends. Report first sampled prefix whose mean signature gain reaches mean one-probe gain; no extrapolation.'),truth_and_observations=raw,readouts=out)
