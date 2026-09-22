"""Independent scaled-product audit of fixed likelihood tempering.

Supplied endpoint laws are inherited, hash-bound, previously verified references.
No producer filter, projection, scoring or regroup function is used here.
"""
from collections import defaultdict
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, file_digest
from .crossed_review import near
from .unknown_review import (AXES, METRICS, hypothesis_roster, state_schedule,
    complement, reconstruct_stream, distinct, checkpoints, metrics, zipped)

POWERS = (1., .75, .5)
EVIDENCE = ('aware', 'omitted')
ARM = 'unknown-time-type'


def products(table, observations, length, kind, steps):
    if [r['step'] for r in observations] != list(range(1, length+1)):
        raise ValueError('nonconsecutive stream')
    distinct(observations)
    if any(not 1 <= r['source_step'] <= r['step'] for r in observations):
        raise ValueError('invalid source time')
    hs, prior, schedule = state_schedule(ARM, length, kind)
    weights = np.broadcast_to(prior, (3, len(prior))).copy()
    result = {}; used = 0
    for step in steps:
        available = distinct(observations[:step])
        for row in available[used:]:
            likelihood = table[schedule[row['source_step']], row['context'], row['endpoint']]
            for i, power in enumerate(POWERS):
                weights[i] *= likelihood**power
                total = math.fsum(weights[i])
                if not total > 0 or not math.isfinite(total): raise ValueError('empty support')
                weights[i] /= total
        used = len(available)
        for i, power in enumerate(POWERS):
            current = np.zeros(16)
            np.add.at(current, schedule[step], weights[i])
            result[step, power] = (current, weights[i].copy())
    return result, hs, prior


def controls():
    law = np.full((16, 4, 8), 1/8)
    rows = [dict(step=i, source_step=1, source_id='a', context=0, endpoint=0) for i in range(1, 33)]
    flat, hs, prior = products(law, rows, 32, 'purpose', [32])
    law[:8, 0] = [.6, .4, 0, 0, 0, 0, 0, 0]
    law[8:, 0] = [.2, .8, 0, 0, 0, 0, 0, 0]
    aware, _, _ = products(law, rows, 32, 'purpose', [32])
    doubled = [dict(r, source_id='b' if r['step'] > 1 else 'a') for r in rows]
    missing, _, _ = products(law, doubled, 32, 'purpose', [32])
    expected = prior*np.array([law[m, 0, 0]**1.5 for _, _, m in hs]); expected /= math.fsum(expected)
    return {'placebo:uniform_preserves_prior': all(np.allclose(j, prior, atol=1e-14, rtol=0) for _, j in flat.values()),
        'positive:scalar_three_quarters': bool(np.allclose(missing[32, .75][1], expected, atol=1e-14, rtol=0)),
        'positive:half_two_copies_equals_one': bool(np.allclose(missing[32, .5][1], aware[32, 1.][1], atol=1e-14, rtol=0)),
        'live:copy_weight_changes': bool(np.max(abs(missing[32, 1.][1]-aware[32, 1.][1])) > .001)}


def regroup(cells, cfg):
    key = lambda r: (r['lineage'], r['draw'], *(r[k] for k in AXES), r['evidence'], r['power'])
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
        values = {(e, p): np.array([[[index[l, d, *stratum, e, p][m] for m in METRICS] for d in draws] for l in lineages]) for e, p in product(EVIDENCE, POWERS)}
        for evidence, power in product(EVIDENCE, POWERS):
            v = values[evidence, power]
            means.append(dict(zip(AXES, stratum), evidence=evidence, power=power, **{m: estimate(v[:, :, i]) for i, m in enumerate(METRICS)}))
            pairs = [(evidence, 1., 'power-minus-one')]
            if evidence == 'omitted':
                pairs.extend([('aware', power, 'omitted-minus-aware-same-power'), ('aware', 1., 'omitted-minus-aware-one')])
            for base_evidence, base_power, comparison in pairs:
                delta = v-values[base_evidence, base_power]
                for i, metric in enumerate(METRICS):
                    contrasts.append(dict(zip(AXES, stratum), evidence=evidence, power=power, baseline_evidence=base_evidence,
                        baseline_power=base_power, comparison=comparison, metric=metric, **estimate(delta[:, :, i])))
    return dict(cells=cells, means=means, contrasts=contrasts, lineages=lineages, draws=draws,
        population='sixteen equally weighted makers within draw; two draws within each paired coefficient lineage; all fixed strata retained; independent-source and stationary aliases are not replicates',
        uncertainty='conditional eight-lineage bootstrap; two observation draws do not establish universal sampling uncertainty; no fit or fresh observations')


def verify(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    if design['powers'] != list(POWERS): raise ValueError('power roster differs')
    cells = []; error = 0.; row_count = stream_count = parent_identities = alias_identities = 0
    for lineage in design['lineages']:
        pulse(phase='independent-retained-law', lineage=lineage)
        table = np.asarray(read(original/'inputs/aware/evaluator'/f'{lineage}-law.json'))
        if table.shape != (16,4,8) or np.any(table < 0): raise ValueError('law shape or sign')
        near(table.sum(-1), np.ones((16,4)), 1e-12)
        near(table, read(original/'inputs/omitted/evaluator'/f'{lineage}-law.json'), 0)
        rows = zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed = {(r['evidence'], r['stream'], r['step'], r['power']): r for r in rows}
        if len(indexed) != len(rows): raise ValueError('duplicate row')
        mapping = read(original/'evaluator'/f'{lineage}-joint-map.json')['rows']
        with np.load(original/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as z:
            arrays = {n: z[n] for n in z.files}
        expected_keys = {f'{e}-{length}-{kind}-{power:g}' for e, length, kind, power in product(EVIDENCE, design['lengths'], ('purpose','skill'), POWERS)}
        if set(arrays) != expected_keys or set(mapping) != expected_keys: raise ValueError('joint roster')
        roster = list(product(design['draws'], range(16), design['lengths'], ('purpose','skill'), (False,True), (False,True)))
        mapped = defaultdict(list); acc = defaultdict(list); consumed = set()
        for evidence in EVIDENCE:
            parent = original/'inputs'/evidence
            streams = zipped(parent/'raw'/f'{lineage}-observations_points.json.gz')
            if len(streams) != len(roster): raise ValueError('stream denominator')
            oldrows = zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
            oldindex = {(r['stream'],r['step']): r for r in oldrows if r['arm'] == ARM}
            oldmap = read(parent/'evaluator'/f'{lineage}-joint-map.json')
            with np.load(parent/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as z:
                oldarrays = {n:z[n] for n in z.files if n.endswith('-'+ARM)}
            for si, (s, key) in enumerate(zip(streams, roster)):
                pulse(phase='independent-powered-products', lineage=lineage, evidence=evidence, stream=si)
                draw, maker, length, kind, switched, duplicates = key
                obs = reconstruct_stream(table, lineage, *key)
                if evidence == 'omitted': obs = [dict(r, source_id=f'source-{i:03d}') for i,r in enumerate(obs,1)]
                if s != dict(zip(('draw','maker','length','kind','switched','duplicates'), key), observations=obs): raise ValueError('stream reconstruction')
                outputs, hs, prior = products(table, obs, length, kind, checkpoints(length)); stream_count += 1
                parent_key = f'{length}-{kind}-{ARM}'
                if oldmap[parent_key]['hypotheses'] != [list(h) for h in hs]: raise ValueError('parent hypotheses')
                near(prior, oldmap[parent_key]['prior'], 1e-14)
                for step, power in product(checkpoints(length), POWERS):
                    k = (evidence,si,step,power); r = indexed[k]; consumed.add(k)
                    actual = complement(maker,kind) if switched and step > length//2 else maker
                    meta = dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,kind=kind,switched=switched,duplicates=duplicates,step=step,evidence=evidence,power=power)
                    if any(r[n] != v for n,v in meta.items()): raise ValueError('row assignment')
                    current, joint = outputs[step,power]; values, forecast = metrics(table,current,actual)
                    name = f'{evidence}-{length}-{kind}-{power:g}'; ji = len(mapped[name]); mapped[name].append([si,step])
                    if r['joint_array'] != name or r['joint_row'] != ji: raise ValueError('joint row assignment')
                    error = max(error, near(joint, arrays[name][ji], 1e-10), near(current,r['posterior'],1e-10),
                        near(forecast,r['forecast'],1e-10), near([values[m] for m in METRICS],[r[m] for m in METRICS],1e-10))
                    if power == 1.:
                        old = oldindex.pop((si,step))
                        if not np.array_equal(arrays[name][ji],oldarrays[parent_key][old['joint_row']]) or r['posterior'] != old['posterior'] or r['forecast'] != old['forecast'] or any(r[m] != old[m] for m in METRICS): raise ValueError('parent one identity')
                        parent_identities += 1
                    if evidence == 'omitted' and not duplicates:
                        a = indexed['aware',si,step,power]
                        if any(r[n] != a[n] for n in ('posterior','forecast',*METRICS)): raise ValueError('independent-source identity')
                        alias_identities += 1
                    acc[draw,length,kind,switched,duplicates,step,evidence,power].append(values)
            if oldindex: raise ValueError('unused parent rows')
        if consumed != set(indexed): raise ValueError('unconsumed rows')
        for name in expected_keys:
            if mapped[name] != mapping[name] or arrays[name].shape != (len(mapping[name]),len(hypothesis_roster(ARM,int(name.split('-')[1]),name.split('-')[2])[0])) or arrays[name].dtype != np.float64: raise ValueError('joint map or dimensions')
        for draw,maker,length,duplicates in product(design['draws'],range(16),design['lengths'],(False,True)):
            a = reconstruct_stream(table,lineage,draw,maker,length,'purpose',False,duplicates)
            b = reconstruct_stream(table,lineage,draw,maker,length,'skill',False,duplicates)
            if a != b: raise ValueError('stationary alias')
        row_count += len(rows)
        for key, vv in sorted(acc.items()):
            if len(vv) != 16: raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage,**dict(zip(('draw',*AXES,'evidence','power'),key)),makers=16,
                **{m:math.fsum(v[m] for v in vv)/16 for m in METRICS}))
    key = lambda r:(r['lineage'],r['draw'],*(r[k] for k in AXES),r['evidence'],r['power'])
    actual = {key(r):r for r in summary['cells']}; expected = {key(r):r for r in cells}
    if len(actual) != len(summary['cells']) or set(actual) != set(expected): raise ValueError('summary roster')
    for k,v in expected.items():
        if actual[k]['makers'] != 16: raise ValueError('summary denominator')
        error=max(error,near([v[m] for m in METRICS],[actual[k][m] for m in METRICS],1e-10))
    for k,v in dict(rows=row_count,paired_streams=stream_count,parent_identity_rows=parent_identities,fits=0).items():
        if summary[k] != v: raise ValueError('summary total')
    write(root/'INDEPENDENT_REGROUP.json',regroup(cells,cfg))
    checks.update(positive_complete_reconstruction=True,positive_parent_identity=True,positive_aliases=True)
    return dict(controls=checks,rows=row_count,paired_streams=stream_count,parent_identity_rows=parent_identities,
        independent_source_identity_rows=alias_identities,cells=len(cells),max_error=error,
        scope='independent powered products, joint/current distributions and scores from previously verified supplied laws; no new law validation or historical correspondence')


def run(root, plan, pulse):
    import os
    from contextlib import redirect_stdout, redirect_stderr
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    import pytest
    reports = []
    class Progress:
        def pytest_runtest_setup(self, item): pulse(phase='tempering-checker-fixture',test=item.nodeid)
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' or report.failed or report.skipped: reports.append(dict(test=report.nodeid,outcome=report.outcome,when=report.when))
    with (root/'pytest.log').open('w',encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code=int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(root/'fixture-temp'),*plan['design']['preflight_tests']],plugins=[Progress()]))
    write(root/'FIXTURE_GATE.json',dict(exit_code=code,reports=reports,log_sha256=file_digest(root/'pytest.log')))
    if code or not reports or any(r['outcome'] != 'passed' for r in reports): raise ValueError('mandatory fixture gate failed')
    return verify(root,plan,pulse)
