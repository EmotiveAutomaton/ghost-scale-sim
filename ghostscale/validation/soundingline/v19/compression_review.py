"""Independent checkpoint selection, projection, scoring and paired regroup.

Parent posterior and law validity is inherited from bound, completed reviews.
No producer compression, projection, score or regroup function is called.
"""
from collections import defaultdict
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, file_digest
from .crossed_review import near
from .unknown_review import (AXES, METRICS, hypothesis_roster, current_indices,
    complement, checkpoints, metrics, zipped)

COUNTS = ('1', '16', '64', '256', 'all')
EVIDENCE = ('aware', 'omitted')
ARM = 'unknown-time-type'
MEASURES = (*METRICS, 'removed_mass', 'hypotheses', 'weight_bytes', 'index_bytes')


def selection(weights, count, order=None):
    w = list(map(float, weights))
    if not w or any(not math.isfinite(x) or x < 0 for x in w) or abs(math.fsum(w)-1) > 1e-10:
        raise ValueError('invalid distribution')
    if count == 'all':
        return list(range(len(w))), np.array(w), 0.
    k = int(count)
    if str(k) != str(count) or not 1 <= k <= len(w): raise ValueError('invalid capacity')
    ranked = sorted(range(len(w)), key=lambda i: (-w[i], i)) if order is None else order
    indices = ranked[:k]; mass = math.fsum(w[i] for i in indices)
    if mass <= 0: raise ValueError('empty support')
    return indices, np.array([w[i]/mass for i in indices]), max(0., math.fsum(w)-mass)


def projection(states, indices, weights):
    buckets = [[] for _ in range(16)]
    for i, w in zip(indices, weights): buckets[int(states[i])].append(float(w))
    return np.array([math.fsum(b) for b in buckets])


def controls():
    w = [.5, .5, 0.]; states = [2, 7, 3]
    ix, kept, removed = selection(w, '1')
    full = projection(states, *selection(w, 'all')[:2])
    point = projection(states, ix, kept)
    law = np.full((16, 4, 8), 1/8)
    point_score, _ = metrics(law, point, 7)
    full_score, _ = metrics(law, full, 7)
    return {'positive:all_preserves_weights': np.array_equal(selection(w, 'all')[1], w),
        'placebo:zero_removal_identity': np.array_equal(projection(states, *selection(w, '2')[:2]), full),
        'positive:stable_tie': ix == [0] and removed == .5,
        'live:point_loses_coverage': full_score['coverage90'] == 1 and point_score['coverage90'] == 0,
        'positive:zero_mass_floor': point_score['state_loss'] == -math.log(1e-300),
        'positive:normalized_nonnegative': bool(math.fsum(kept) == 1 and min(kept) >= 0)}


def regroup(cells, cfg):
    key = lambda r: (r['lineage'], r['draw'], *(r[k] for k in AXES), r['evidence'], str(r['count']))
    index = {key(r): r for r in cells}
    if len(index) != len(cells): raise ValueError('duplicate stratum')
    lineages = sorted({r['lineage'] for r in cells}); draws = sorted({r['draw'] for r in cells})
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages), size=(cfg['bootstrap_resamples'], len(lineages)))
    def estimate(values):
        paired = values.mean(1); boot = paired[samples].mean(1)
        return dict(mean=float(paired.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
            lineage_values=paired.tolist(), draw_means=values.mean(0).tolist())
    means = []; contrasts = []
    for stratum in sorted({tuple(r[k] for k in AXES) for r in cells}):
        values = {(e, c): np.array([[[index[l, d, *stratum, e, c][m] for m in MEASURES] for d in draws] for l in lineages]) for e, c in product(EVIDENCE, COUNTS)}
        for evidence, count in product(EVIDENCE, COUNTS):
            v = values[evidence, count]
            means.append(dict(zip(AXES, stratum), evidence=evidence, count=count, **{m: estimate(v[:, :, i]) for i, m in enumerate(MEASURES)}))
            pairs = [(evidence, 'all', 'count-minus-all')]
            if evidence == 'omitted': pairs.append(('aware', count, 'omitted-minus-aware-same-count'))
            for base_evidence, base_count, comparison in pairs:
                delta = v-values[base_evidence, base_count]
                for i, metric in enumerate(MEASURES):
                    contrasts.append(dict(zip(AXES, stratum), evidence=evidence, count=count, baseline_evidence=base_evidence,
                        baseline_count=base_count, comparison=comparison, metric=metric, **estimate(delta[:, :, i])))
    return dict(cells=cells, means=means, contrasts=contrasts, lineages=lineages, draws=draws,
        population='sixteen equally weighted makers within draw; both draws within each paired coefficient lineage; all capacities and strata retained; stationary and independent-source aliases are not replicates',
        uncertainty='conditional paired coefficient-lineage bootstrap; two observation draws do not establish universal sampling uncertainty; no fits or fresh observations')


def verify(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    if tuple(map(str, design['retained_counts'])) != COUNTS: raise ValueError('capacity roster')
    cells = []; error = 0.; total = full_count = marginal_count = alias_count = 0
    for lineage in design['lineages']:
        pulse(phase='independent-checkpoint-selection', lineage=lineage)
        table = np.asarray(read(original/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        if table.shape != (16,4,8) or np.any(~np.isfinite(table)) or np.any(table < 0): raise ValueError('law shape or sign')
        near(table.sum(-1), np.ones((16,4)), 1e-12)
        near(table, read(original/'inputs/omitted/evaluator'/f'{lineage}-law.json'), 0)
        rows = zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed = {(r['evidence'], r['stream'], r['step'], str(r['count'])): r for r in rows}
        if len(indexed) != len(rows): raise ValueError('duplicate output row')
        with np.load(original/'raw'/f'{lineage}-selection_points.npz', allow_pickle=False) as z: arrays = {n:z[n] for n in z.files}
        with np.load(original/'raw'/f'{lineage}-weight_points.npz', allow_pickle=False) as z: weights = {n:z[n] for n in z.files}
        names = {f'{e}-{length}-{kind}-{ARM}-{count}' for e,length,kind,count in product(EVIDENCE,design['lengths'],('purpose','skill'),COUNTS[:-1])}
        if set(arrays) != names or set(weights) != names: raise ValueError('selection array roster')
        roster = list(product(design['draws'],range(16),design['lengths'],('purpose','skill'),(False,True),(False,True)))
        expected = {(si,step) for si,key in enumerate(roster) for step in checkpoints(key[2])}
        acc = defaultdict(list); consumed = set(); mapped = defaultdict(set)
        for evidence in EVIDENCE:
            parent = original/'inputs'/evidence
            oldrows = [r for r in zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz') if r['arm'] == ARM]
            oldindex = {(r['stream'],r['step']):r for r in oldrows}
            if len(oldindex) != len(oldrows) or set(oldindex) != expected: raise ValueError('parent denominator')
            mapping = read(parent/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(parent/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z: joints = {n:z[n] for n in z.files if n.endswith('-'+ARM)}
            parent_names = {f'{length}-{kind}-{ARM}' for length,kind in product(design['lengths'],('purpose','skill'))}
            if set(joints) != parent_names: raise ValueError('parent array roster')
            schedules = {}; used = defaultdict(set)
            for length,kind in product(design['lengths'],('purpose','skill')):
                hs,prior = hypothesis_roster(ARM,length,kind); name = f'{length}-{kind}-{ARM}'
                if mapping[name]['hypotheses'] != [list(h) for h in hs]: raise ValueError('parent hypothesis roster')
                near(prior, mapping[name]['prior'], 1e-14)
                if joints[name].dtype != np.float64 or joints[name].shape != (len(mapping[name]['rows']),len(hs)): raise ValueError('parent dimensions')
                for step in checkpoints(length): schedules[length,kind,step] = current_indices(hs,step)
            for i, old in enumerate(oldrows):
                if i%64 == 0: pulse(phase='independent-compression-scores',lineage=lineage,evidence=evidence,row=i)
                si = old['stream']; step = old['step']; draw,maker,length,kind,switched,duplicates = roster[si]
                actual = complement(maker,kind) if switched and step > length//2 else maker
                meta = dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,kind=kind,switched=switched,duplicates=duplicates,step=step)
                if any(old[k] != v for k,v in meta.items()): raise ValueError('parent assignment')
                name = f'{length}-{kind}-{ARM}'; ji = old['joint_row']
                if old['joint_array'] != name or ji in used[name] or mapping[name]['rows'][ji] != [si,step]: raise ValueError('parent row mapping')
                used[name].add(ji); full = joints[name][ji]; states = schedules[length,kind,step]
                all_ix,all_weights,_ = selection(full,'all'); current = projection(states,all_ix,all_weights)
                parent_values,parent_forecast = metrics(table,current,actual)
                error=max(error,near(current,old['posterior'],1e-10),near(parent_forecast,old['forecast'],1e-10),near([parent_values[m] for m in METRICS],[old[m] for m in METRICS],1e-10))
                marginal_count += 1
                order=sorted(range(len(full)),key=lambda j:(-float(full[j]),j))
                for count in COUNTS:
                    key=(evidence,si,step,count); r=indexed[key]; consumed.add(key)
                    if any(r[k] != v for k,v in meta.items()) or r['parent_joint_array'] != name or r['parent_joint_row'] != ji: raise ValueError('output assignment')
                    ix,kept,removed=selection(full,count,order)
                    if count == 'all':
                        if r['selection_array'] is not None or r['selection_row'] is not None: raise ValueError('full selection metadata')
                        if r['posterior'] != old['posterior'] or r['forecast'] != old['forecast'] or any(r[m] != old[m] for m in METRICS): raise ValueError('full parent identity')
                        full_count += 1
                    else:
                        sn=f'{evidence}-{name}-{count}'; sr=r['selection_row']
                        if r['selection_array'] != sn or sr in mapped[sn] or not isinstance(sr,int) or not 0 <= sr < len(arrays[sn]): raise ValueError('selection mapping')
                        mapped[sn].add(sr)
                        if arrays[sn].dtype != np.int32 or weights[sn].dtype != np.float64 or arrays[sn].shape != weights[sn].shape or arrays[sn].shape[1] != int(count): raise ValueError('selection storage')
                        if not np.array_equal(arrays[sn][sr],ix): raise ValueError('selected indices')
                        error=max(error,near(weights[sn][sr],kept,1e-13))
                    posterior=projection(states,ix,kept); values,forecast=metrics(table,posterior,actual)
                    values.update(removed_mass=removed,hypotheses=len(ix),weight_bytes=8*len(ix),index_bytes=0 if count=='all' else 4*len(ix))
                    if r['current_marginal_weight_bytes'] != 128: raise ValueError('marginal bytes')
                    error=max(error,near(posterior,r['posterior'],1e-10),near(forecast,r['forecast'],1e-10),near([values[m] for m in MEASURES],[r[m] for m in MEASURES],1e-10))
                    if evidence=='omitted' and not duplicates:
                        a=indexed['aware',si,step,count]
                        if any(r[k] != a[k] for k in ('posterior','forecast',*MEASURES)): raise ValueError('independent-source alias')
                        alias_count += 1
                    acc[draw,length,kind,switched,duplicates,step,evidence,count].append(values)
            if any(used[n] != set(range(len(v))) for n,v in joints.items()): raise ValueError('unused parent row')
        if consumed != set(indexed) or any(mapped[n] != set(range(len(v))) for n,v in arrays.items()): raise ValueError('unused output or selection')
        total += len(rows)
        for key,vv in sorted(acc.items()):
            if len(vv) != 16: raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw',*AXES,'evidence','count'),key)),makers=16,**{m:math.fsum(v[m] for v in vv)/16 for m in MEASURES}))
    key=lambda r:(r['lineage'],r['draw'],*(r[k] for k in AXES),r['evidence'],str(r['count']))
    actual={key(r):r for r in summary['cells']}; expected_cells={key(r):r for r in cells}
    if len(actual)!=len(summary['cells']) or set(actual)!=set(expected_cells): raise ValueError('summary roster')
    for k,v in expected_cells.items():
        if actual[k]['makers']!=16: raise ValueError('summary denominator')
        error=max(error,near([v[m] for m in MEASURES],[actual[k][m] for m in MEASURES],1e-10))
    for k,v in dict(rows=total,parent_identity_rows=full_count,current_marginal_controls=marginal_count,fits=0).items():
        if summary[k] != v: raise ValueError('summary total')
    write(root/'INDEPENDENT_REGROUP.json',regroup(cells,cfg))
    checks.update(positive_complete_selection=True,positive_complete_projection_and_scores=True,positive_parent_identity=True,positive_independent_source_identity=True)
    return dict(controls=checks,rows=total,parent_identity_rows=full_count,current_marginal_controls=marginal_count,independent_source_identity_rows=alias_count,
        cells=len(cells),max_error=error,scope='independent retained selection, current projection, forecast, proper-score, storage and paired-regroup reconstruction; parent laws/posteriors inherit prior validation; no recursive or historical-process claim')


def run(root, plan, pulse):
    import os
    from contextlib import redirect_stdout, redirect_stderr
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    import pytest
    reports=[]
    class Progress:
        def pytest_runtest_setup(self,item): pulse(phase='compression-checker-fixture',test=item.nodeid)
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed or report.skipped: reports.append(dict(test=report.nodeid,outcome=report.outcome,when=report.when))
    with (root/'pytest.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code=int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(root/'fixture-temp'),*plan['design']['preflight_tests']],plugins=[Progress()]))
    write(root/'FIXTURE_GATE.json',dict(exit_code=code,reports=reports,log_sha256=file_digest(root/'pytest.log')))
    if code or not reports or any(r['outcome']!='passed' for r in reports): raise ValueError('mandatory fixture gate failed')
    return verify(root,plan,pulse)
