"""Finite conditional training targets under an explicitly sampled state prior.

This changes target variance, not the population cross-entropy objective.
The teacher receives public history and law, never the realized hidden state.
"""
import gzip
import json
import math
from pathlib import Path
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import read, file_digest


def prior_for(states):
    states=tuple(states)
    if not states or len(set(states))!=len(states) or any(s not in range(len(W.STATES)) for s in states):
        raise ValueError('invalid teacher support')
    p=np.zeros(len(W.STATES));p[list(states)]=1/len(states)
    return p


def teacher(payload, queries, states):
    public=W.parse(payload)
    weights=W.posterior(payload,prior_for(states))
    if weights is None:raise ValueError('history outside teacher support')
    return np.asarray([weights@W.artifact_matrix(public['world'],q) for q in queries])


def reference_teacher(payload, queries, states):
    """Scalar likelihood products and separate physical-policy implementation."""
    from ..v18_3.verify import reference_policy,execute
    public=W.parse(payload);w=public['world'];seen=set();observations=[]
    for o in public['history']:
        if o['source'] not in seen:observations.append(o)
        seen.add(o['source'])
    weights=[]
    for state in states:
        factors=[float(reference_policy(w,state,o['context'])[W.PROGRAMS.index(tuple(o['program']))]) for o in observations]
        weights.append(math.prod(factors)/len(states))
    total=math.fsum(weights)
    if total<=0:raise ValueError('zero reference evidence')
    weights=[v/total for v in weights];out=[]
    for q in queries:
        probabilities=[]
        for state in states:
            raw=reference_policy(w,state,q);row=np.zeros(16)
            for program,v in zip(W.PROGRAMS,raw):row[execute(program)]+=v
            probabilities.append(row)
        out.append([math.fsum(a*float(p[j]) for a,p in zip(weights,probabilities)) for j in range(16)])
    return np.asarray(out)


def controls():
    states=tuple(i for i,s in enumerate(W.STATES) if s[0]>0)
    w=W.make_world(0,1987001);q=[W.context(),W.context(goal=1)]
    empty=W.packet(w,[])
    observed=W.packet(w,[W.observe(w,states[0],W.context(),W.rng('L2b-control'),'one')])
    return dict(live=bool(np.allclose(teacher(observed,q,states),reference_teacher(observed,q,states),atol=1e-12,rtol=0)),
        placebo=bool(np.allclose(teacher(empty,q,states),[prior_for(states)@W.artifact_matrix(w,c) for c in q],atol=1e-12,rtol=0)))


def audit(root):
    """Verify fixed train/dev cases, targets and IID draws from retained data.

Every generation shard is hash-bound. Reconstruct one case per cell and split
with the independent policy; this is a bounded target audit, not full regeneration.
"""
    from . import neural_data as D
    root=Path(root);config=read(root/'BUILD_PLAN.json')
    if config.get('state_sampling')!='iid':raise ValueError('conditional study must sample its state prior')
    allowed=[s for s in D.NEURAL_STATES if config['support']=='all' or D.parity(s)==(config['support']=='odd')]
    gates=controls();count=0;maximum=0.;counts={};bindings={}
    with np.load(root/'reader/TRAIN.npz',allow_pickle=False) as capsule:
        for part in ('train','dev'):
            datafile=root/f'{part}-DATA.npz';pointsfile=root/f'{part}-points.json.gz';receipt=read(root/f'{part}-PART.json')
            if file_digest(datafile)!=receipt['data_sha256'] or file_digest(pointsfile)!=receipt['points_sha256']:raise ValueError('teacher shard changed')
            bindings[part]=receipt
            truth=json.loads(gzip.decompress(pointsfile.read_bytes()));counts[part]={str(s):sum(t['state']==s for t in truth) for s in allowed}
            with np.load(datafile,allow_pickle=False) as z:
                for key in ('history','length','world','sample','query','target'):
                    if not np.array_equal(z[key],capsule[f'{part}_{key}']):raise ValueError('reader target capsule differs')
                per_cell=config[part+'_per_cell'];split='pilot' if part=='train' and config['pilot'] else part
                for cell in range(16):
                    i=(cell*7)%per_cell;index=cell*per_cell+i;case=truth[index]
                    expected_state=int(W.rng(config.get('namespace','v18.4-neural'),'iid-state',split,cell,i,config['pilot']).choice(allowed))
                    if case['state']!=expected_state:raise ValueError('IID state reconstruction differs')
                    payload=W.packet(case['world'],case['history']);h,n,wf=D.features(payload)
                    if not np.array_equal(h,z['history'][index]) or n!=z['length'][index] or not np.array_equal(wf,z['world'][index]):raise ValueError('teacher public features differ')
                    mode=config.get('dev_query_mode',config['query_mode']) if part=='dev' else config['query_mode']
                    queries=D.training_queries(i,mode)
                    conditional=reference_teacher(payload,queries,allowed)
                    fast=teacher(payload,queries,allowed)
                    maximum=max(maximum,float(np.max(abs(conditional-fast))))
                    if not np.allclose(conditional,fast,atol=1e-12,rtol=0):raise ValueError('independent conditional target differs')
                    if part=='train' and config.get('target_mode')=='conditional':expected=conditional
                    else:
                        from ..v18_3.verify import reference_policy,execute
                        expected=[]
                        for q in queries:
                            row=np.zeros(16)
                            for program,p in zip(W.PROGRAMS,reference_policy(case['world'],case['state'],q)):row[execute(program)]+=p
                            expected.append(row)
                    selected=z['sample']==index
                    if not np.allclose(expected,z['target'][selected],atol=1e-7,rtol=0):raise ValueError('retained supervision differs')
                    count+=len(queries)
    return dict(gates=gates,passed=all(gates.values()),independent_target_rows=count,max_teacher_error=maximum,
        sampled_state_counts=counts,prior=prior_for(allowed).tolist(),bindings=bindings,
        scope='one fixed history per cell in training and development; no test labels or outcomes used; counts descriptive, not a random balance gate')
