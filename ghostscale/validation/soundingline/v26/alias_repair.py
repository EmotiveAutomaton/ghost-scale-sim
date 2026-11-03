"""R1: passive split aliases with an explicit distinguishing sixth piece."""
import math
import numpy as np
from .common import rng,posterior,information,scores,record,close
from ..v25.pieces import WEIGHTS,HYPOTHESES,laws,controls as old_controls

TRAIN=((0,1,2,3),(0,1,3,4),(1,2,3,4),(0,1,2,3,4))
SWITCH=((5,0,2),(5,1,4))

def library(r,structure='varying'):
    e=r.choice([0.,.5,1.],(6,3));e[:5,1]=e[:5,0]
    if structure=='constant':e[:5,:2]=.5
    e[5,:2]=[1,0];return e

def hypotheses(beta=3):return [(w,b,h) for w in range(15) for b in sorted(set([1,3,beta])) for h in range(-1,6)]

def probe_set(e,independent=False,hs=HYPOTHESES):
    items=[]
    for k in range(5):items.append(('remove',k,tuple(i for i in range(5) if i!=k),None))
    for k in range(3):items.append(('satiate',k,tuple(range(5)),k))
    for k,a in enumerate(SWITCH):items.append(('switch',k,a,None))
    return [(kind,k,a,s,laws(e,a,hs,satiate=s,independent=independent)) for kind,k,a,s in items]

def split_pairs(beta,hs=HYPOTHESES):
    return [(i,j) for i,a in enumerate(hs) for j,b in enumerate(hs) if i<j and a[1]==b[1]==beta and a[2]==b[2] and WEIGHTS[a[0],2]==WEIGHTS[b[0],2] and WEIGHTS[a[0],0]!=WEIGHTS[b[0],0]]

def block(setting,identity,independent=False):
    r=rng(*identity);e=library(r,setting.get('structure','varying'));hs=hypotheses(setting['beta']);training=TRAIN[:setting.get('training_count',4)];base=[laws(e,a,hs,independent=independent) for a in training];probes=probe_set(e,independent,hs);pairs=split_pairs(setting['beta'],hs);out=[];raw=[]
    # Equal fixed random sample of eight split-alias pairs, with both endpoints.
    selected=r.choice(len(pairs),8,replace=False)
    for maker,pairid in enumerate(selected):
        pair=pairs[int(pairid)];truth=pair[maker%2];ll=np.zeros(len(hs));obs=[];snap=None
        for t in range(128):
            si=t%len(base);y=int(r.choice(base[si].shape[1],p=base[si][truth]));ll+=np.log(base[si][:,y]);obs.append(y)
            if t+1 in (8,32,128):out.append(record(scores(posterior(ll,independent=independent),truth),maker=maker,reader='passive',history=t+1))
            if t==31:snap=ll.copy()
        seqs={}
        for kind in ('remove','satiate','switch'):
            candidates=[x for x in probes if x[0]==kind];active=snap.copy();passive=snap.copy();rows=[];hit=None
            for n in range(1,17):
                p=posterior(active,independent=independent);infos=[information(p,x[-1]) for x in candidates];ix=int(np.flatnonzero(np.asarray(infos)>=max(infos)-1e-12)[0]);typ,k,a,s,law=candidates[ix];y=int(r.choice(len(a),p=law[truth]));active+=np.log(law[:,y]);pa=posterior(active,independent=independent)
                bl=base[(31+n)%len(base)];z=int(r.choice(bl.shape[1],p=bl[truth]));passive+=np.log(bl[:,z]);pb=posterior(passive,independent=independent)
                gain=math.log(pa[truth]/posterior(snap,independent=independent)[truth]);contrast=math.log(pa[truth]/pb[truth]);rows.append(dict(probe=k,choice=y,passive_choice=z))
                if gain>=1 and hit is None:hit=n
                if n in (1,2,4,8,16):out.append(record(dict(primary=contrast,gain_over_32=gain),maker=maker,reader=kind,history=n))
            out.append(record(dict(interventions_to_one_nat=hit,not_within_16=float(hit is None)),maker=maker,reader=kind,history='threshold'));seqs[kind]=rows
        raw.append(dict(truth=truth,split_alias=pair,passive=obs,interventions=seqs))
    sep={}
    for kind in ('remove','satiate','switch'):
        sep[kind]=sum(any(np.max(abs(p[i]-p[j]))>1e-12 for typ,k,a,s,p in probes if typ==kind) for i,j in pairs)/len(pairs)
    out.append(record(sep,reader='by-construction-alias-map'))
    return dict(public=dict(effects=e.tolist(),training=training,switch=SWITCH,choice_model='W1 beta1/3 and additive piece-habit hypothesis library'),truth_and_observations=raw,readouts=out)

def controls():
    old_controls();r=rng('controls-R1');e=library(r,'constant');pair=split_pairs(3)[0];i,j=pair
    for a in TRAIN:assert np.max(abs(laws(e,a)[i]-laws(e,a)[j]))<1e-12
    p=laws(e,tuple(range(5)),satiate=0);assert np.max(abs(p[i]-p[j]))<1e-12
    p=laws(e,SWITCH[0]);assert np.max(abs(p[i]-p[j]))>1e-12
    return dict(passed=True,inherited_tie_and_break=True,constant_unsatiated_end_does_not_break=True,passive_alias=True,switch_break=True)

def design_check(settings):
    for b in range(16):
        e=library(rng('check-only-R1',b))
        for i,j in split_pairs(3):
            for a in TRAIN:assert np.max(abs(laws(e,a)[i]-laws(e,a)[j]))<1e-12
    return dict(passed=True,alias_world_share=1.,floor=.25,seed_domain='check-only-R1')
