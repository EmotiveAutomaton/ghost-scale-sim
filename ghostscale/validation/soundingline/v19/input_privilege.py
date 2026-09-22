"""Exact endpoint ambiguity after withholding mechanic metadata; no fitting."""
from collections import defaultdict
from itertools import product
import gzip
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, read, write, file_digest
from . import rollout as R, rollout_transfer as T, missing_tool as M

VIEWS = ('full', 'omit-skill', 'omit-belief', 'omit-both')


def visible(q, view):
    if view not in VIEWS:
        raise ValueError('unadmitted view')
    p = R.visible(q)
    if view in ('omit-skill', 'omit-both'):
        del p['skill']
    if view in ('omit-belief', 'omit-both'):
        del p['belief_error']
    return p


def completions(p):
    """All mechanically legal metadata values, including zero-policy support."""
    result = []
    for skill, belief in product(range(2), repeat=2):
        if p.get('skill', skill) != skill or p.get('belief_error', belief) != belief:
            continue
        full = dict(p, skill=skill, belief_error=belief)
        try:
            R.validate_visible(full)
        except ValueError:
            continue
        result.append((skill, belief, R.code(p['initial']),
                       *(R.L.OPERATIONS.index(o) for o in p['operations'])))
    if not result:
        raise ValueError('no legal completion')
    return result


def membership(queries):
    if len(set(queries)) != len(queries):
        raise ValueError('duplicate query')
    result = {}
    for view in VIEWS:
        groups = {}
        for i, q in enumerate(queries):
            p = visible(q, view)
            R.validate_visible(R.visible(q))
            key = digest(p)
            row = groups.setdefault(key, dict(reader=p, indices=[], legal_completions=[list(v) for v in completions(p)]))
            row['indices'].append(i)
        result[view] = groups
    return result


def entropy(p):
    return -math.fsum(float(x)*math.log(float(x)) for x in p if x > 0)


def evaluate(queries, groups, targets, mass, rule):
    mass = np.asarray(mass, float)
    if mass.shape != (len(queries),) or not np.isfinite(mass).all() or np.any(mass < 0) or not np.isclose(mass.sum(), 1, atol=1e-12):
        raise ValueError('population mass')
    if sorted(i for row in groups.values() for i in row['indices']) != list(range(len(queries))):
        raise ValueError('membership coverage')
    uniform = np.zeros((len(queries), 8)); native = uniform.copy(); rows = []
    for key, row in groups.items():
        indices = row['indices']; legal = [tuple(q) for q in row['legal_completions']]
        if legal != completions(row['reader']) or any(queries[i] not in legal for i in indices):
            raise ValueError('legal membership')
        endpoints = [T.oracle(q, rule) for q in legal]
        p = np.bincount(endpoints, minlength=8).astype(float)/len(legal)
        weights = np.bincount(np.asarray(targets)[indices], weights=mass[indices], minlength=8)
        total = math.fsum(float(mass[i]) for i in indices)
        # Explicit fallback only for a zero-native-mass public group.
        n = weights/total if total else p.copy()
        uniform[indices] = p; native[indices] = n
        legal_set = set(endpoints); supported = set(np.asarray(targets)[indices][mass[indices] > 0].tolist())
        rows.append(dict(reader_id=key, query_indices=indices, legal_completions=[list(q) for q in legal],
                         legal_endpoints=endpoints, native_mass=total, endpoint_mass=weights.tolist(),
                         uniform_prediction=p.tolist(), native_prediction=n.tolist(),
                         legal_ambiguous=len(legal_set)>1, native_ambiguous=len(supported)>1,
                         legal_entropy=entropy(p), native_entropy=entropy(n) if total else None,
                         zero_native_mass=not bool(total), native_zero_mass_fallback='uniform-legal' if not total else None))
    if not np.allclose(uniform.sum(1), 1) or not np.allclose(native.sum(1), 1):
        raise ValueError('normalization')
    return rows, uniform, native


def proper(p, targets, weights):
    truth = p[np.arange(len(targets)), targets]
    positive = truth > 0
    return dict(infinite_loss_mass=float(weights[~positive].sum()),
                finite_loss_contribution=float(np.dot(weights[positive], -np.log(truth[positive]))),
                squared_error=float(np.dot(weights, np.sum((p-np.eye(8)[targets])**2, axis=1))),
                true_probability=float(np.dot(weights, truth)))


def controls():
    qs=[(1,0,2,1,4,4),(1,1,2,1,4,4)]
    targets=np.array([T.oracle(q,'original') for q in qs]); groups=membership(qs)
    rows,u,n=evaluate(qs,groups['omit-belief'],targets,np.array([.25,.75]),'original')
    same=[(1,0,2,4,4,4),(1,1,2,4,4,4)]
    null,_,_=evaluate(same,membership(same)['omit-belief'],np.array([2,2]),np.array([.5,.5]),'original')
    full,_,_=evaluate(qs,groups['full'],targets,np.array([.25,.75]),'original')
    zero,_,_=evaluate(qs,groups['full'],targets,np.array([1.,0.]),'original')
    return {'live:omitted_belief_ambiguity':len(rows)==1 and rows[0]['legal_ambiguous'],
            'positive:known_likelihoods':np.allclose(n[0,targets],[.25,.75]),
            'positive:uniform_completions':np.allclose(u[0,targets],[.5,.5]),
            'placebo:constant_endpoint':not null[0]['legal_ambiguous'] and null[0]['legal_entropy']==0,
            'positive:full_metadata_exact':all(r['legal_entropy']==0 for r in full),
            'placebo:zero_mass_explicit':sum(r['zero_native_mass'] for r in zero)==1,
            'positive:tool_requires_skill':all(q[0]==1 for q in completions(dict(initial=[0,1,0],operations=['accept-tool','inspect','inspect'])))}


def run(root, plan, pulse):
    cfg=plan['design']; base=root/'inputs'; start=time.process_time()
    checks=controls()
    if not all(checks.values()) or cfg['views']!=list(VIEWS) or cfg['rules']!=list(M.RULES):
        raise ValueError('input privilege admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n)!=h: raise ValueError('input binding')
    truth=read(base/'QUERY_TRUTH.json');qs=[tuple(r['query']) for r in truth]
    if len(qs)!=cfg['queries']: raise ValueError('query count')
    groups=membership(qs)
    if groups!=read(base/'MEMBERSHIP.json'): raise ValueError('frozen public membership')
    for folder in ('reader','evaluator','forecasts'):(root/folder).mkdir()
    write(root/'CONTROLS.json',checks)
    for view in VIEWS:
        for key,row in groups[view].items():
            p=root/'reader'/f'{key}.json'
            if not p.exists():write(p,row['reader'])
    write(root/'evaluator/MEMBERSHIP.json',groups)
    cells=[];group_records=[];index={q:i for i,q in enumerate(qs)};paths=0
    for lineage,rule in product(cfg['lineages'],M.RULES):
        pulse(phase='metadata-omission',lineage=lineage,rule=rule)
        targets=np.array([T.oracle(q,rule) for q in qs])
        if any(r['targets'][rule]!=int(t) for r,t in zip(truth,targets,strict=True)):
            raise ValueError('parent targets')
        records=read_gzip(base/'raw'/f'{lineage}-{rule}_points.json.gz');mass=np.zeros(len(qs));paths+=len(records)
        for r in records:
            i=index[R.query(r)];w=r['probability']
            if not math.isfinite(w) or w<0 or R.code(r['final'])!=targets[i]: raise ValueError('parent path')
            mass[i]+=w
        for view in VIEWS:
            rows,u,n=evaluate(qs,groups[view],targets,mass,rule)
            group_records.extend(dict(lineage=lineage,rule=rule,view=view,**r) for r in rows)
            np.savez(root/'forecasts'/f'{lineage}-{rule}-{view}_points.npz',queries=np.array(qs),targets=targets,mass=mass,uniform=u,native=n)
            for weighting,w in (('native',mass),('equal-query',np.ones(len(qs))/len(qs))):
                for arm,p in (('uniform-legal',u),('native-law',n)):
                    cells.append(dict(lineage=lineage,rule=rule,view=view,weighting=weighting,arm=arm,
                                      groups=len(rows),queries=len(qs),zero_mass_groups=sum(r['zero_native_mass'] for r in rows),
                                      legal_ambiguous_groups=sum(r['legal_ambiguous'] for r in rows),
                                      native_ambiguous_groups=sum(r['native_ambiguous'] for r in rows),
                                      legal_ambiguous_mass=math.fsum(float(w[r['query_indices']].sum()) for r in rows if r['legal_ambiguous']),
                                      native_conditional_entropy=math.fsum(r['native_mass']*r['native_entropy'] for r in rows if r['native_mass']),
                                      **proper(p,targets,w)))
    (root/'evaluator/GROUPS_points.json.gz').write_bytes(gzip.compress(canonical(group_records),mtime=0))
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='retained exact-law ambiguity and score enumeration'))
    return dict(controls=checks,cells=cells,queries=len(qs),native_paths=paths,group_rows=len(group_records),fits=0,
                role='constructed-method input privilege; mechanics and policy support distinct; supplied-law oracle; no historical process or human claim')


def read_gzip(path):
    import json
    return json.loads(gzip.decompress(path.read_bytes()))
