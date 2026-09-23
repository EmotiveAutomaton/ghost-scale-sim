"""Complete repeated binary-reply enumeration with explicit source dependence."""
from itertools import product
import math
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D

CHANNELS = ('independent', 'copied')
READERS = ('channel-aware', 'independence-assumed')
FIELDS = ('skill', 'belief')
LENGTHS = (0, 1, 2, 4)
COST_MODES = ('per-return', 'per-acquisition')
RELIABILITY = .75


def likelihoods(length, reliability, mode):
    """Rows are complete lexicographic strings; columns are the true bit."""
    if length not in LENGTHS or mode not in CHANNELS or not .5 <= reliability <= 1:
        raise ValueError('channel specification')
    strings = list(product((0, 1), repeat=length))
    result = np.zeros((len(strings), 2))
    for i, replies in enumerate(strings):
        for bit in (0, 1):
            if not replies:
                value = 1.
            elif mode == 'copied':
                value = (reliability if replies[0] == bit else 1-reliability) if len(set(replies)) == 1 else 0.
            else:
                value = math.prod(reliability if r == bit else 1-reliability for r in replies)
            result[i, bit] = value
    validate_likelihoods(result)
    return strings, result


def validate_likelihoods(table):
    if (table.ndim != 2 or table.shape[1] != 2 or not np.isfinite(table).all()
        or np.any(table < 0) or not np.allclose(table.sum(0), 1, atol=1e-12, rtol=0)):
        raise ValueError('channel likelihood normalization')


def posterior(legal, ends, weights, axis, table):
    """All reply posteriors, including explicit zero-modeled-mass fallbacks."""
    D.distributions(legal, ends, weights)
    validate_likelihoods(table)
    ends = np.asarray(ends, int); weights = np.asarray(weights, float)
    likelihood = table[:, np.array([q[axis] for q in legal])]
    joint = likelihood * weights[None, :]
    mass = joint.sum(1); forecasts = []; fallbacks = []
    for row, like, total in zip(joint, likelihood, mass, strict=True):
        if total:
            p = np.bincount(ends, weights=row, minlength=8)/total
            fallback = None
        else:
            ids = np.flatnonzero(like > 0)
            fallback = 'uniform-likelihood-supported-legal' if len(ids) else 'uniform-whole-legal-group'
            if not len(ids): ids = np.arange(len(legal))
            p = np.bincount(ends[ids], minlength=8)/len(ids)
        forecasts.append(p); fallbacks.append(fallback)
    return np.array(forecasts), mass, fallbacks


def expected_scores(predictions, probabilities, targets, weights):
    n = len(targets)
    if predictions.ndim != 3 or predictions.shape[0] != n or predictions.shape[2] != 8 or probabilities.shape != predictions.shape[:2]:
        raise ValueError('reply shapes')
    for value, axis in ((predictions, 2), (probabilities, 1)):
        if not np.isfinite(value).all() or np.any(value < 0) or not np.allclose(value.sum(axis), 1, atol=1e-12, rtol=0):
            raise ValueError('reply normalization')
    parts = [D.P.proper(predictions[:, j], targets, weights*probabilities[:, j]) for j in range(predictions.shape[1])]
    scores = {k: math.fsum(x[k] for x in parts) for k in parts[0]}
    scores['residual_ambiguous_mass'] = math.fsum(float(np.dot(weights*probabilities[:, j], (predictions[:, j] > 0).sum(1) > 1)) for j in range(predictions.shape[1]))
    return scores


def controls():
    legal = [(s,b,2,1,4,4) for s,b in product(range(2), repeat=2)]
    ends = [0,1,2,3]; weights = [.1,.2,.3,.4]
    one, _, _ = posterior(legal, ends, weights, 0, likelihoods(1,.75,'copied')[1])
    four, _, _ = posterior(legal, ends, weights, 0, likelihoods(4,.75,'copied')[1])
    half, _, _ = posterior(legal, ends, weights, 0, likelihoods(4,.5,'independent')[1])
    naive, _, _ = posterior(legal, ends, weights, 0, likelihoods(4,.75,'independent')[1])
    return {'live:independent_repetition_changes_odds': not np.allclose(naive[0],one[0]),
            'placebo:copied_reply_identity': np.array_equal(four[[0,-1]],one),
            'positive:half_channel_no_information': np.allclose(half[:,:4],weights,atol=1e-12,rtol=0)}


def run(root, plan, pulse):
    cfg=plan['design'];base=root/'inputs';start=time.process_time();checks=controls()
    for key,value in [('channels',CHANNELS),('readers',READERS),('fields',FIELDS),('lengths',LENGTHS),('cost_modes',COST_MODES),('models',D.MODELS),('costs',D.COSTS)]:
        if cfg[key] != list(value): raise ValueError('repeated reply design')
    if cfg['reliability'] != RELIABILITY or not all(checks.values()):raise ValueError('repeated reply admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h:raise ValueError('input binding')
    groups=read(base/'MEMBERSHIP.json')['omit-both'];laws=read(base/'DISCLOSURE_LAWS.json')
    lookup={(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup)!=len(laws):raise ValueError('duplicate laws')
    for folder in ('reader','forecasts','evaluator'):(root/folder).mkdir()
    # All possible strings are exported; zero-probability copied strings remain
    # explicitly zero in evaluator likelihoods, never claimed observed evidence.
    packets={}
    for g in groups.values():
        p=g['reader'];packets[digest(p)]=p
        for field,mode,length in product(FIELDS,CHANNELS,LENGTHS[1:]):
            for replies in likelihoods(length,RELIABILITY,mode)[0]:
                p=dict(g['reader'],requested_field='belief_error' if field=='belief' else field,replies=list(replies),reply_reliability=RELIABILITY,reply_source_mode=mode)
                packets[digest(p)]=p
    write(root/'reader/REPLIES.json',packets)
    write(root/'CONTROLS.json',checks);cells=[];rows=[];used=set();roster=None
    channel_tables={f'{mode}-{length}':dict(strings=[list(s) for s in likelihoods(length,RELIABILITY,mode)[0]],likelihood=likelihoods(length,RELIABILITY,mode)[1].tolist()) for mode,length in product(CHANNELS,LENGTHS)}
    write(root/'evaluator/CHANNELS.json',channel_tables)
    for lineage,rule,model in product(cfg['lineages'],cfg['rules'],D.MODELS):
        pulse(phase='repeated-disclosure',lineage=lineage,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',allow_pickle=False) as parent:
            qs=[tuple(map(int,q)) for q in parent['queries']];mass=parent['mass'].copy();targets=parent['targets'].copy();n=len(qs)
            if n!=cfg['queries'] or len(set(qs))!=n:raise ValueError('query roster')
            if roster is None:roster=qs
            elif qs!=roster:raise ValueError('query identity')
            if D.P.membership(qs)['omit-both']!=groups:raise ValueError('membership')
            if not np.array_equal(targets,[D.P.T.oracle(q,rule) for q in qs]):raise ValueError('targets')
            if not np.isfinite(mass).all() or np.any(mass<0) or not np.isclose(mass.sum(),1,atol=1e-12,rtol=0):raise ValueError('mass')
            for field,mode,length in product(FIELDS,CHANNELS,LENGTHS):
                axis=FIELDS.index(field);strings,actual=likelihoods(length,RELIABILITY,mode)
                probabilities=actual[:,np.array([q[axis] for q in qs])].T
                saved=dict(queries=qs,mass=mass,targets=targets,strings=np.array(strings,dtype=np.int8).reshape(len(strings),length),reply_probabilities=probabilities)
                for reader in READERS:
                    assumed=mode if reader=='channel-aware' else 'independent';table=likelihoods(length,RELIABILITY,assumed)[1]
                    forecasts=np.zeros((n,len(strings),8))
                    for key,g in groups.items():
                        law=lookup[(lineage,rule,model,key)];used.add((lineage,rule,model,key));legal=[tuple(q) for q in law['legal_completions']]
                        if law['legal_completions']!=g['legal_completions'] or law['endpoints']!=[D.P.T.oracle(q,rule) for q in legal]:raise ValueError('legal mechanics')
                        p,reply_mass,fallback=posterior(legal,law['endpoints'],law['conditional_weights'],axis,table)
                        forecasts[g['indices']]=p
                        rows.append(dict(lineage=lineage,rule=rule,model=model,field=field,channel=mode,length=length,reader=reader,reader_id=key,modeled_reply_mass=reply_mass.tolist(),fallbacks=fallback))
                    saved[reader]=forecasts
                    for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                        result=expected_scores(forecasts,probabilities,targets,w)
                        for cost,cost_mode in product(D.COSTS,COST_MODES):
                            acquisitions=int(length>0) if mode=='copied' else length
                            charge=length if cost_mode=='per-return' else acquisitions
                            cells.append(dict(lineage=lineage,rule=rule,model=model,field=field,channel=mode,length=length,reader=reader,weighting=weighting,cost=cost,cost_mode=cost_mode,queries=n,groups=len(groups),returned_replies=length,acquisitions=acquisitions,charged_replies=charge,net_finite_loss=result['finite_loss_contribution']+cost*charge,**result))
                np.savez_compressed(root/'forecasts'/f'{lineage}-{rule}-{model}-{field}-{mode}-{length}_points.npz',**saved)
    if len(used)!=len(laws):raise ValueError('law coverage')
    write(root/'evaluator/REPLY_LAWS.json',rows)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='complete repeated-reply enumeration; no sampled evidence'))
    return dict(controls=checks,cells=cells,law_rows=len(rows),queries=cfg['queries'],reader_packets=len(packets),fits=0,scope='supplied-law and known-provenance method; expected realized scores; per-return versus separate per-acquisition diagnostic; no learned-access or process-correspondence claim')
