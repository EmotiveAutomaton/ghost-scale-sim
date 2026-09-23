"""Two-field reply dependence with fixed marginal reliability; no fitting."""
from itertools import product
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D
from .repeated_disclosure import expected_scores

CHANNELS = ('independent', 'shared-flip')
READERS = ('joint-aware', 'marginals-product')
REQUESTS = ('none', 'skill', 'belief', 'both')
RELIABILITY = .75


def validate(table):
    if (table.ndim != 2 or table.shape[1] != 4 or not np.isfinite(table).all()
        or np.any(table < 0) or not np.allclose(table.sum(0), 1, atol=1e-12, rtol=0)):
        raise ValueError('joint likelihood normalization')


def likelihoods(request, accuracy, mode):
    """Reply rows; truth columns are (skill,belief) in lexicographic order."""
    if request not in REQUESTS or mode not in CHANNELS or not .5 <= accuracy <= 1:
        raise ValueError('joint channel specification')
    axes = {'none': (), 'skill': (0,), 'belief': (1,), 'both': (0, 1)}[request]
    replies = list(product((0, 1), repeat=len(axes)))
    table = np.zeros((len(replies), 4))
    for j, truth in enumerate(product((0, 1), repeat=2)):
        for flips in product((0, 1), repeat=2):
            if mode == 'shared-flip':
                probability = (accuracy if flips[0] == 0 else 1-accuracy) if flips[0] == flips[1] else 0.
            else:
                probability = np.prod([accuracy if f == 0 else 1-accuracy for f in flips])
            bits = tuple(truth[i] ^ flips[i] for i in axes)
            table[replies.index(bits), j] += probability
    validate(table)
    return replies, table


def posterior(legal, ends, weights, table):
    D.distributions(legal, ends, weights)
    validate(table)
    ends = np.asarray(ends, int)
    likes = table[:, [2*q[0]+q[1] for q in legal]]
    joint = likes * np.asarray(weights)[None, :]
    masses = joint.sum(1); forecasts = []; fallbacks = []
    for row, like, mass in zip(joint, likes, masses, strict=True):
        if mass:
            p = np.bincount(ends, weights=row, minlength=8)/mass
            fallback = None
        else:
            ids = np.flatnonzero(like > 0)
            fallback = 'uniform-likelihood-supported-legal' if len(ids) else 'uniform-whole-legal-group'
            if not len(ids): ids = np.arange(len(legal))
            p = np.bincount(ends[ids], minlength=8)/len(ids)
        forecasts.append(p); fallbacks.append(fallback)
    return np.array(forecasts), masses, fallbacks


def controls():
    legal = [(s,b,2,1,4,4) for s,b in product((0,1),repeat=2)]
    _, shared = likelihoods('both', .5, 'shared-flip')
    _, independent = likelihoods('both', .5, 'independent')
    a, _, _ = posterior(legal, [0,1,2,3], [.25]*4, shared)
    b, _, _ = posterior(legal, [0,1,2,3], [.25]*4, independent)
    return {'live:joint_dependence_changes_posterior': not np.array_equal(a,b),
            'positive:half_reliability_shared_parity': np.array_equal(a[0,:4],[.5,0,0,.5]),
            'placebo:half_independent_no_information': np.array_equal(b[:,:4],np.full((4,4),.25))}


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; start = time.process_time(); checks = controls()
    for key, value in [('channels',CHANNELS),('readers',READERS),('requests',REQUESTS),('models',D.MODELS),('costs',D.COSTS)]:
        if cfg[key] != list(value): raise ValueError('joint reply design')
    if cfg['reliability'] != RELIABILITY or not all(checks.values()): raise ValueError('joint reply admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    groups = read(base/'MEMBERSHIP.json')['omit-both']; laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    for folder in ('reader','forecasts','evaluator'): (root/folder).mkdir()
    tables = {f'{mode}-{request}':dict(strings=[list(s) for s in likelihoods(request,RELIABILITY,mode)[0]],likelihood=likelihoods(request,RELIABILITY,mode)[1].tolist()) for mode,request in product(CHANNELS,REQUESTS)}
    write(root/'evaluator/CHANNELS.json',tables)
    packets = {}
    for g in groups.values():
        packets[digest(g['reader'])] = g['reader']
        for request,mode in product(REQUESTS[1:],CHANNELS):
            fields = {'skill':['skill'],'belief':['belief_error'],'both':['skill','belief_error']}[request]
            for bits in likelihoods(request,RELIABILITY,mode)[0]:
                p = dict(g['reader'],requested_fields=fields,replies=list(bits),marginal_reliability=RELIABILITY,joint_source_relation=mode)
                packets[digest(p)] = p
    write(root/'reader/REPLIES.json',packets); write(root/'CONTROLS.json',checks)
    cells = []; rows = []; used = set(); roster = None
    for lineage,rule,model in product(cfg['lineages'],cfg['rules'],D.MODELS):
        pulse(phase='joint-reply',lineage=lineage,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',allow_pickle=False) as parent:
            qs = [tuple(map(int,q)) for q in parent['queries']]; mass = parent['mass'].copy(); targets = parent['targets'].copy(); n = len(qs)
            if n != cfg['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None: roster = qs
            elif qs != roster: raise ValueError('query identity')
            if D.P.membership(qs)['omit-both'] != groups: raise ValueError('membership')
            if not np.array_equal(targets,[D.P.T.oracle(q,rule) for q in qs]): raise ValueError('targets')
            if not np.isfinite(mass).all() or np.any(mass<0) or not np.isclose(mass.sum(),1,atol=1e-12,rtol=0): raise ValueError('mass')
            for mode,request in product(CHANNELS,REQUESTS):
                strings, actual = likelihoods(request,RELIABILITY,mode)
                probabilities = actual[:,[2*q[0]+q[1] for q in qs]].T
                length = len(strings[0]); saved = dict(queries=qs,mass=mass,targets=targets,strings=np.array(strings,dtype=np.int8).reshape(len(strings),length),reply_probabilities=probabilities)
                for reader in READERS:
                    table = likelihoods(request,RELIABILITY,mode if reader=='joint-aware' else 'independent')[1]
                    forecasts = np.zeros((n,len(strings),8))
                    for key,g in groups.items():
                        law = lookup[(lineage,rule,model,key)]; used.add((lineage,rule,model,key))
                        legal = [tuple(q) for q in law['legal_completions']]
                        if law['legal_completions'] != g['legal_completions'] or law['endpoints'] != [D.P.T.oracle(q,rule) for q in legal]: raise ValueError('legal mechanics')
                        p,reply_mass,fallback = posterior(legal,law['endpoints'],law['conditional_weights'],table)
                        forecasts[g['indices']] = p
                        rows.append(dict(lineage=lineage,rule=rule,model=model,channel=mode,request=request,reader=reader,reader_id=key,modeled_reply_mass=reply_mass.tolist(),fallbacks=fallback))
                    saved[reader] = forecasts
                    for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                        result = expected_scores(forecasts,probabilities,targets,w)
                        for cost in D.COSTS:
                            cells.append(dict(lineage=lineage,rule=rule,model=model,channel=mode,request=request,reader=reader,weighting=weighting,cost=cost,queries=n,groups=len(groups),returned_replies=length,net_finite_loss=result['finite_loss_contribution']+cost*length,**result))
                np.savez_compressed(root/'forecasts'/f'{lineage}-{rule}-{model}-{mode}-{request}_points.npz',**saved)
    if len(used) != len(laws): raise ValueError('law coverage')
    write(root/'evaluator/REPLY_LAWS.json',rows)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='complete two-field reply enumeration; no sampled evidence'))
    return dict(controls=checks,cells=cells,law_rows=len(rows),queries=cfg['queries'],reader_packets=len(packets),fits=0,scope='supplied laws and known joint reliability; separate actual probabilities and assumed posteriors; no learned access or historical process claim')
