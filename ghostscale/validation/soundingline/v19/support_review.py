"""Independent count, support, forecast and score reconstruction for F support.

No producer fitting, forward-propagation, retrieval or scoring code is imported.
The separate crossed-law verifier supplies its independent native-path ruler.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import numpy as np
from ..v18_3.io import read, write, file_digest
from ..v18_3.world import rng
from .crossed_review import population, execute, OPS, near

ARTS = tuple(product(range(2), repeat=3))
ARMS = ('direct', 'retrieval', 'learned-exact', 'rollout-1', 'rollout-4', 'rollout-16', 'oracle-exact')
SUBSETS = ('all', 'query-seen', 'query-unseen', 'heldout-composition', 'heldout-seen-primitives')


def code(a): return 4*a[0]+2*a[1]+a[2]


def query(r):
    return (r['maker'][1], r['maker'][2], code(r['initial']), *(OPS.index(e['operation']) for e in r['steps']))


def held(q): return tuple(q[3:5]) == (3, 1)


def normalize(p):
    p = np.asarray(p, float)
    return p/p.sum(axis=-1, keepdims=True)


def counts_from(records):
    counts = np.ones((2,2,8,8,6,8)); direct = {}
    for r in records:
        q = query(r); direct.setdefault(q, np.ones(8))[code(r['final'])] += 1
        for e in r['steps']:
            counts[q[0],q[1],code(e['before']),code(e['undo_buffer']),OPS.index(e['operation']),code(e['after'])] += 1
    return counts, direct


def nearest(direct, q):
    if not direct: return np.full(8, 1/8)
    distances = {k:sum(a != b for a,b in zip(k,q)) for k in direct}
    distance = min(distances.values())
    return normalize(np.mean([direct[k] for k in direct if distances[k] == distance], axis=0))


def propagate(table, q):
    # Explicit sum over all 8^3 possible intermediate artifact paths, including undo state.
    kernel = table[q[0],q[1]]; a0 = q[2]; o1,o2,o3 = q[3:]
    joint = kernel[a0,a0,o1,:,None,None]*kernel[:,a0,o2,:,None]*kernel[:,:,o3,:].transpose(1,0,2)
    return joint.sum((0,1))


def forecast(counts, direct, q, uniforms):
    table = normalize(counts); kernel = table[q[0],q[1]]; a = b = q[2]; visits = []
    for op in q[3:]:
        visits.append(bool(counts[q[0],q[1],a,b,op].sum() > 8))
        after = code(execute(ARTS[a], ARTS[b], OPS[op], q[0], q[1], 'original'))
        b,a = a,after
    target = a; endpoints = []
    for us in uniforms:
        a = b = q[2]
        for op,u in zip(q[3:],us):
            after = min(int(np.sum(np.cumsum(kernel[a,b,op]) <= u)), 7)
            b,a = a,after
        endpoints.append(a)
    ps = {'direct':normalize(direct.get(q,np.ones(8))), 'retrieval':nearest(direct,q),
          'learned-exact':propagate(table,q), 'oracle-exact':np.eye(8)[target]}
    for n in (1,4,16): ps['rollout-'+str(n)] = np.bincount(endpoints[:n],minlength=8)/n
    ps = {k:31/32*v+np.ones(8)/256 for k,v in ps.items()}
    diagnostic = dict(query_seen=q in direct, all_primitives_seen=all(visits),
                      visited_primitives=sum(visits), withheld_composition=held(q))
    return ps, diagnostic, target


def controls():
    counts = np.ones((2,2,8,8,6,8)); q = (1,0,2,3,1,4)
    ps, diag, target = forecast(counts, {}, q, np.full((16,3), .2))
    rejected = False
    try: near([.25,.75], [.75,.25])
    except ValueError: rejected = True
    return {'live:holdout_order':held(q) and not held((1,0,2,1,3,4)),
            'placebo:unseen_counts':not diag['query_seen'] and not diag['all_primitives_seen'],
            'positive:uniform_propagation':bool(np.allclose(ps['learned-exact'],np.full(8,.125))),
            'positive:common_smoothing':bool(np.array_equal(ps['rollout-1'],ps['rollout-16'])),
            'positive:oracle_smoothing':bool(ps['oracle-exact'][target] == 249/256),
            'positive:corruption_rejected':rejected}


def zipped(path): return json.loads(gzip.decompress(path.read_bytes()))


def interval(values, cfg):
    values = np.asarray(values, float); random = np.random.default_rng(cfg['bootstrap_seed'])
    bs = values[random.integers(len(values),size=(cfg['bootstrap_resamples'],len(values)))].mean(1)
    return dict(mean=float(values.mean()),low=float(np.quantile(bs,.025)),high=float(np.quantile(bs,.975)),lineage_values=values.tolist())


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('support review controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input changed')
    assert file_digest(original/'PLAN.json') == cfg['target_plan_sha256']
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    assert design['arms'] == list(ARMS) and design['epsilon'] == 1/32
    populations = {}; paths = 0; error = 0.; all_queries = set()
    for lineage in design['lineages']:
        pulse(phase='independent-support-paths',lineage=lineage)
        records = zipped(original/'raw'/f'{lineage}_points.json.gz')
        _,_,delta = population(records,lineage,'original'); error = max(error,delta); paths += len(records)
        groups = {}
        for r in records:
            q = query(r); target = code(r['final']); all_queries.add(q)
            if q not in groups: groups[q] = [target,0.]
            assert groups[q][0] == target; groups[q][1] += r['probability']
        populations[lineage] = groups
    qs = sorted(all_queries)
    assert read(original/'reader/QUERIES.json') == [dict(skill=q[0],belief_error=q[1],initial=list(ARTS[q[2]]),operations=[OPS[i] for i in q[3:]]) for q in qs]
    targets = read(original/'QUERY_TRUTH.json'); assert [tuple(t['query']) for t in targets] == qs
    cells = []; rows_checked = 0; support_counts = []; times = read(original/'TIMING.jsonl')['measurements']
    for draw in design['training_draws']:
        selected = read(original/'inputs/auxiliary'/f'selection-{draw}.json')['paths'][:2048]
        kept = [i for i,r in enumerate(selected) if not held(query(r))]
        selection = read(original/'models'/f'{draw}-holdout-selection.json')
        assert selection == dict(retained_indices=kept,original_count=len(selected),retained_count=len(kept))
        support_counts.append(dict(draw=draw,original=len(selected),retained=len(kept)))
        for mode in ('original','composition-holdout'):
            pulse(phase='independent-support-counts',draw=draw,mode=mode)
            records = selected if mode == 'original' else [selected[i] for i in kept]
            counts,direct = counts_from(records); keys = sorted(direct)
            with np.load(original/'models'/f'{draw}-{mode}.npz') as model:
                near(model['transition_counts'],counts,0); near(model['direct_keys'],keys,0); near(model['direct_counts'],[direct[k] for k in keys],0)
            if mode == 'original':
                with np.load(original/'inputs/models'/f'{draw}-2048.npz') as model:
                    near(model['transition_counts'],counts,0); near(model['direct_keys'],keys,0); near(model['direct_counts'],[direct[k] for k in keys],0)
            timing = [t for t in times if t['draw']==draw and t['mode']==mode]
            fits = [t for t in timing if t['phase']=='fit-or-load']; assert len(fits)==1
            assert fits[0]['training_paths']==len(records) and fits[0]['new_fits']==int(mode!='original')
            assert all(t['cpu_seconds']>=0 for t in timing)
            for arm in ARMS: assert len([t for t in timing if t.get('arm')==arm])==len(qs)
            diagnostics = read(original/'forecasts'/f'{draw}-{mode}-support.json'); assert len(diagnostics)==len(qs)
            predictions = {a:[] for a in ARMS}
            with np.load(original/'forecasts'/f'{draw}-{mode}.npz') as saved:
                near(saved['queries'],qs,0)
                for i,q in enumerate(qs):
                    u = rng('v19-rollout-query',draw,*q).uniform(size=(16,3)); near(u,saved['uniforms'][i],0)
                    ps,diag,target = forecast(counts,direct,q,u)
                    assert diag==diagnostics[i] and target==targets[i]['endpoint']
                    for arm,p in ps.items(): error=max(error,near(p,saved[arm][i],1e-12)); predictions[arm].append(p)
            rows = zipped(original/'raw'/f'{draw}-{mode}-support_points.json.gz')
            lookup = {(r['lineage'],r['arm'],r['query_index']):r for r in rows}; assert len(lookup)==len(rows)
            consumed = set()
            for lineage,groups in populations.items():
                for arm in ARMS:
                    entries = []
                    for i,q in enumerate(qs):
                        if q not in groups: continue
                        target,mass = groups[q]; p = predictions[arm][i]; diag=diagnostics[i]
                        loss = float(-np.log(p[target])); square = float(np.sum((p-np.eye(8)[target])**2))
                        row = lookup[lineage,arm,i]; consumed.add((lineage,arm,i))
                        assert row['draw']==draw and row['mode']==mode and all(row[k]==v for k,v in diag.items())
                        error=max(error,near([mass,loss,square],[row['probability_mass'],row['loss'],row['squared_error']]))
                        entries.append(dict(probability_mass=mass,loss=loss,squared_error=square,**diag))
                    subsets = [entries,[r for r in entries if r['query_seen']],[r for r in entries if not r['query_seen']],
                        [r for r in entries if r['withheld_composition']],[r for r in entries if r['withheld_composition'] and r['all_primitives_seen']]]
                    for subset,rr in zip(SUBSETS,subsets):
                        mass=sum(r['probability_mass'] for r in rr)
                        cells.append(dict(lineage=lineage,draw=draw,mode=mode,arm=arm,subset=subset,queries=len(rr),population_mass=mass,
                            **{m:sum(r['probability_mass']*r[m] for r in rr)/mass if mass else None for m in ('loss','squared_error')}))
            assert consumed==set(lookup); rows_checked+=len(rows)
    key=lambda r:tuple(r[k] for k in ('lineage','draw','mode','arm','subset'))
    lookup={key(r):r for r in cells}; old={key(r):r for r in summary['cells']}
    assert len(old)==len(summary['cells']) and set(old)=={key(r) for r in cells if r['population_mass']>0}
    for k,r in old.items():
        s=lookup[k]; assert r['queries']==s['queries']; error=max(error,near([r[m] for m in ('population_mass','loss','squared_error')],[s[m] for m in ('population_mass','loss','squared_error')]))
    assert summary['rows']==rows_checked and summary['queries']==len(qs) and summary['fits']==len(design['training_draws'])
    contrasts=[]
    for mode,subset,arm in product(('original','composition-holdout'),SUBSETS,ARMS[1:]):
        values=[]; draw_means=[]
        for lineage in design['lineages']:
            diffs=[lookup[lineage,d,mode,arm,subset]['loss']-lookup[lineage,d,mode,'direct',subset]['loss'] for d in design['training_draws'] if lookup[lineage,d,mode,arm,subset]['population_mass']>0]
            if len(diffs)==len(design['training_draws']):values.append(float(np.mean(diffs)))
        for d in design['training_draws']:
            diffs=[lookup[l,d,mode,arm,subset]['loss']-lookup[l,d,mode,'direct',subset]['loss'] for l in design['lineages'] if lookup[l,d,mode,arm,subset]['population_mass']>0]
            draw_means.append(float(np.mean(diffs)) if diffs else None)
        contrasts.append(dict(mode=mode,subset=subset,arm=arm,baseline='direct',draw_means=draw_means,complete_lineages=len(values),**(interval(values,cfg) if values else {'mean':None,'low':None,'high':None,'lineage_values':[]})))
    write(root/'INDEPENDENT_REGROUP.json',dict(cells=cells,contrasts=contrasts,training_counts=support_counts,
        population='native weights within stratum, equal paired development lineages, training draws averaged within lineage; absent strata explicit'))
    checks['positive:complete_reconstruction']=True
    return dict(controls=checks,paths=paths,rows=rows_checked,cells=len(cells),queries=len(qs),max_error=error,
        target_plan_sha256=cfg['target_plan_sha256'],scope='independent complete count/support/forecast/score reconstruction; event adjudication pending')
