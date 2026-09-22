"""Independent supplied-provenance reconstruction; no producer filter or score."""
from collections import defaultdict
from itertools import product
from contextlib import redirect_stdout, redirect_stderr
import gzip
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest, canonical
from .crossed_review import population, near
from .unknown_review import (ARMS, METRICS, AXES, complement, endpoint_law,
    distinct, reconstruct_stream, hypothesis_roster, product_checkpoints,
    metrics, checkpoints, zipped)
from .omission_review import omit_identity


def correction_roster(observations):
    distinct(observations)
    first = {}; pairs = []
    for row in observations:
        if row['source_id'] in first:
            pairs.append([row['step'], first[row['source_id']]])
        else:
            first[row['source_id']] = row['step']
    sets = []; aliases = []
    for order in ('ascending', 'descending'):
        ordered = pairs if order == 'ascending' else pairs[::-1]
        for budget in (0, 1, 4, 'all'):
            chosen = sorted(ordered[:budget] if isinstance(budget, int) else ordered)
            if chosen not in sets:
                sets.append(chosen); aliases.append([])
            aliases[sets.index(chosen)].append(dict(order=order, budget=budget))
    return dict(available_pairs=pairs, conditions=[dict(condition=i, pairs=p, aliases=a)
        for i, (p, a) in enumerate(zip(sets, aliases))])


def apply_equivalences(observations, pairs):
    if [r['step'] for r in observations] != list(range(1, len(observations)+1)):
        raise ValueError('nonconsecutive observation times')
    if len(distinct(observations)) != len(observations):
        raise ValueError('input identities must be distinct')
    labels = list(range(len(observations))); seen = set()
    for copy, original in pairs:
        if not 1 <= original < copy <= len(labels) or copy in seen:
            raise ValueError('invalid source equivalence')
        seen.add(copy)
        if any(observations[copy-1][k] != observations[original-1][k]
               for k in ('source_step', 'context', 'endpoint')):
            raise ValueError('conflicting source equivalence')
        labels[copy-1] = labels[original-1]
    return [dict(r, source_id=observations[labels[i]]['source_id'])
            for i, r in enumerate(observations)]


def controls():
    law = np.full((16, 4, 8), 1/8)
    rows = [dict(step=1, source_step=1, source_id='a', context=0, endpoint=0),
            dict(step=2, source_step=1, source_id='b', context=0, endpoint=0)]
    fixed = apply_equivalences(rows, [[2, 1]])
    flat = product_checkpoints(law, fixed, 'unknown-time-type', 32, 'purpose', [2])[2][0]
    law[:8, 0] = [.8, .2, 0, 0, 0, 0, 0, 0]
    law[8:, 0] = [.2, .8, 0, 0, 0, 0, 0, 0]
    once = product_checkpoints(law, fixed, 'static', 32, 'purpose', [2])[2][0]
    twice = product_checkpoints(law, rows, 'static', 32, 'purpose', [2])[2][0]
    return {'placebo:neutral_correction': bool(np.allclose(flat, 1/16)),
            'live:correction_changes_nonuniform_posterior': bool(not np.allclose(once, twice)),
            'positive:undo_counted_twice': bool(np.allclose(once[:8], .8/8) and
                np.allclose(twice[:8], (.8**2/(.8**2+.2**2))/8)),
            'positive:empty_identity': apply_equivalences(rows, []) == rows,
            'positive:aliases_not_replicates': len(correction_roster(fixed)['conditions']) == 2}


def regroup(cells, cfg, roster):
    axes = (*AXES, 'condition')
    index = {(r['lineage'], r['draw'], *(r[k] for k in axes), r['arm']): r for r in cells}
    if len(index) != len(cells):
        raise ValueError('duplicate stratum')
    lineages = sorted({r['lineage'] for r in cells}); draws = sorted({r['draw'] for r in cells})
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages),
        size=(cfg['bootstrap_resamples'], len(lineages)))
    def estimate(values):
        paired = values.mean(1); boot = paired[samples].mean(1)
        return dict(mean=float(paired.mean()), low=float(np.quantile(boot, .025)),
            high=float(np.quantile(boot, .975)), lineage_values=paired.tolist(),
            draw_means=values.mean(0).tolist())
    means = []; contrasts = []
    for stratum in sorted({tuple(r[k] for k in AXES) for r in cells}):
        length, kind, switched, duplicates, step = stratum
        spec = roster[f'{length}-{int(duplicates)}-{step}']; conditions = spec['conditions']
        ids = {tuple((a['order'], a['budget'])): c['condition'] for c in conditions for a in c['aliases']}
        zero = ids['ascending', 0]; full = ids['ascending', 'all']
        for arm in ARMS:
            values = {c['condition']: np.asarray([[[index[l, d, *stratum, c['condition'], arm][m]
                for m in METRICS] for d in draws] for l in lineages]) for c in conditions}
            for c in conditions:
                cid = c['condition']
                means.append(dict(zip(AXES, stratum), arm=arm, condition=cid,
                    revealed_pairs=len(c['pairs']), aliases=c['aliases'],
                    **{m: estimate(values[cid][:, :, i]) for i, m in enumerate(METRICS)}))
                for label, base in (('restoration-minus-omission', zero), ('restoration-minus-full', full)):
                    for i, metric in enumerate(METRICS):
                        contrasts.append(dict(zip(AXES, stratum), arm=arm, condition=cid,
                            baseline_condition=base, metric=metric, comparison=label,
                            identity=cid == base, **estimate(values[cid][:, :, i]-values[base][:, :, i])))
            for budget in (0, 1, 4, 'all'):
                asc, desc = ids['ascending', budget], ids['descending', budget]
                for i, metric in enumerate(METRICS):
                    contrasts.append(dict(zip(AXES, stratum), arm=arm, budget=budget,
                        condition=asc, baseline_condition=desc, metric=metric,
                        comparison='ascending-minus-descending', identity=asc == desc,
                        **estimate(values[asc][:, :, i]-values[desc][:, :, i])))
    return dict(cells=cells, means=means, contrasts=contrasts, lineages=lineages, draws=draws,
        population='sixteen equally weighted makers within draw; draws averaged within paired coefficient lineage; all change/source/checkpoint strata retained; budget/order aliases are not replicates; stationary reference factors never pooled',
        uncertainty='conditional coefficient-lineage intervals; two observation draws do not establish universal sampling uncertainty; no fitted model')


def check_packet(packet, public):
    if packet['schema'] != 'v19.provenance-restoration.reader.1' or packet['cases'] != [public[k] for k in sorted(public)]:
        raise ValueError('reader projection differs')


def verify(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json')
    if any(n not in (32, 128) for n in design['lengths']): raise ValueError('design differs')
    roster = read(original/'inputs/CORRECTION_ROSTER.json')
    contexts = [{'initial': list(a), 'requested_purpose': p} for a in ((0, 1, 0), (1, 0, 1)) for p in range(2)]
    cells = []; public = {}; counts = []; identities = defaultdict(int)
    error = 0.; paths = streams_count = rows_count = condition_count = 0; consumed_roster = set()
    for lineage in design['lineages']:
        pulse(phase='independent-native-paths', lineage=lineage)
        records = zipped(original/'inputs'/f'lineage-{lineage}_points.json.gz')
        _, _, e = population(records, lineage, 'original'); error = max(error, e); paths += len(records)
        table = endpoint_law(records)
        error = max(error, near(table, read(original/'evaluator'/f'{lineage}-law.json'), 1e-12))
        streams = zipped(original/'inputs/parent/raw'/f'{lineage}-observations_points.json.gz')
        parents = {}
        for name in ('parent', 'omission'):
            rr = zipped(original/'inputs'/name/'raw'/f'{lineage}-forecasts_points.json.gz')
            parents[name] = {(r['stream'], r['step'], r['arm']): r for r in rr}
            if len(parents[name]) != len(rr): raise ValueError('duplicate baseline row')
        expected_streams = list(product(design['draws'], range(16), design['lengths'], ('purpose', 'skill'), (False, True), (False, True)))
        if len(streams) != len(expected_streams): raise ValueError('stream denominator')
        rows = zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed = {(r['stream'], r['step'], r['condition'], r['arm']): r for r in rows}
        if len(indexed) != len(rows): raise ValueError('duplicate forecast row')
        mapping = read(original/'evaluator'/f'{lineage}-joint-map.json')
        with np.load(original/'raw'/f'{lineage}-joint_points.npz', allow_pickle=False) as stored:
            arrays = {n: stored[n] for n in stored.files}
        expected_keys = {f'{length}-{kind}-{arm}' for length, kind, arm in product(design['lengths'], ('purpose', 'skill'), ARMS)}
        if set(mapping) != expected_keys or set(arrays) != expected_keys: raise ValueError('joint key roster')
        for length, kind, arm in product(design['lengths'], ('purpose', 'skill'), ARMS):
            name = f'{length}-{kind}-{arm}'; hs, prior = hypothesis_roster(arm, length, kind)
            if mapping[name]['hypotheses'] != [list(h) for h in hs]: raise ValueError('hypothesis mapping')
            error = max(error, near(prior, mapping[name]['prior'], 1e-14))
            if arrays[name].dtype != np.float64 or arrays[name].shape != (len(mapping[name]['rows']), len(hs)):
                raise ValueError('joint array shape or dtype')
        acc = defaultdict(list); mapped = defaultdict(list); consumed = set(); baseline_consumed = set()
        for si, (stream, key) in enumerate(zip(streams, expected_streams)):
            draw, maker, length, kind, switched, duplicates = key
            pulse(phase='independent-correction-products', lineage=lineage, stream=si)
            truth = reconstruct_stream(table, lineage, *key)
            expected = dict(zip(('draw', 'maker', 'length', 'kind', 'switched', 'duplicates'), key), observations=truth)
            if stream != expected: raise ValueError('native stream differs')
            supplied = omit_identity(truth); streams_count += 1
            for step in checkpoints(length):
                rkey = f'{length}-{int(duplicates)}-{step}'; consumed_roster.add(rkey)
                spec = correction_roster(truth[:step])
                if spec != roster[rkey]: raise ValueError('correction roster differs')
                for c in spec['conditions']:
                    cid = c['condition']; pairs = c['pairs']; condition_count += 1
                    observed = apply_equivalences(supplied[:step], pairs)
                    visible = dict(contexts=contexts, observations=supplied[:step], source_equivalences=pairs)
                    ident = digest(visible); public[ident] = dict(input_sha256=ident, inputs=visible)
                    for arm in ARMS:
                        row = indexed[si, step, cid, arm]; consumed.add((si, step, cid, arm))
                        current, joint, hs = product_checkpoints(table, observed, arm, length, kind, [step])[step]
                        actual = complement(maker, kind) if switched and step > length//2 else maker
                        retained = distinct(observed); retained = retained[-16:] if arm == 'reset-16' else retained
                        retained_true = len({truth[r['step']-1]['source_id'] for r in retained})
                        metadata = dict(stream=si, draw=draw, initial_maker=maker, actual_maker=actual,
                            length=length, kind=kind, switched=switched, duplicates=duplicates, step=step,
                            condition=cid, revealed_pairs=len(pairs), arm=arm,
                            supplied_sources=len(distinct(observed)), retained_true_sources=retained_true)
                        if any(row[k] != v for k, v in metadata.items()): raise ValueError('forecast assignment differs')
                        name = f'{length}-{kind}-{arm}'; ji = len(mapped[name]); mapped[name].append([si, step, cid])
                        if row['joint_array'] != name or row['joint_row'] != ji or mapping[name]['rows'][ji] != [si, step, cid]:
                            raise ValueError('joint row mapping differs')
                        types = {k: math.fsum(float(w) for (x, t, m), w in zip(hs, joint) if x == k) for k in ('none', 'purpose', 'skill')}
                        if set(row['type_mass']) != set(types): raise ValueError('type labels differ')
                        values, forecast = metrics(table, current, actual)
                        error = max(error, near(current, row['posterior'], 1e-10), near(joint, arrays[name][ji], 1e-10),
                            near(forecast, row['forecast'], 1e-10), near([values[m] for m in METRICS], [row[m] for m in METRICS], 1e-10),
                            near(list(types.values()), [row['type_mass'][k] for k in types], 1e-10))
                        for label, parent, active in (('zero', 'omission', not pairs), ('full', 'parent', pairs == spec['available_pairs'])):
                            if active:
                                base = parents[parent][si, step, arm]; baseline_consumed.add((si, step, arm))
                                if row['posterior'] != base['posterior'] or row['forecast'] != base['forecast'] or any(row[m] != base[m] for m in METRICS):
                                    raise ValueError('baseline exact identity differs')
                                identities[label] += 1
                        if not pairs:
                            renamed = [dict(r, source_id=f'renamed-{r["step"]:03d}') for r in supplied[:step]]
                            check = product_checkpoints(table, renamed, arm, length, kind, [step])[step]
                            if not np.array_equal(current, check[0]) or not np.array_equal(joint, check[1]):
                                raise ValueError('label-only identity differs')
                            identities['label_only'] += 1
                        counts.append(dict(lineage=lineage, stream=si, step=step, condition=cid, arm=arm,
                            supplied=len(retained), true=retained_true))
                        acc[draw, length, kind, switched, duplicates, step, cid, arm].append(values)
        if consumed != set(indexed): raise ValueError('unconsumed forecast rows')
        if any(baseline_consumed != set(p) for p in parents.values()): raise ValueError('baseline row roster')
        if any(mapped[n] != mapping[n]['rows'] for n in expected_keys): raise ValueError('unconsumed joint rows')
        rows_count += len(rows)
        for key, rr in sorted(acc.items()):
            if len(rr) != 16: raise ValueError('maker denominator')
            cells.append(dict(lineage=lineage, **dict(zip(('draw', *AXES, 'condition', 'arm'), key)), makers=16,
                **{m: math.fsum(v[m] for v in rr)/16 for m in METRICS}))
    key = lambda r: (r['lineage'], r['draw'], *(r[k] for k in AXES), r['condition'], r['arm'])
    actual = {key(r): r for r in summary['cells']}
    if len(actual) != len(summary['cells']) or set(actual) != {key(r) for r in cells}: raise ValueError('summary roster')
    for r in cells:
        if actual[key(r)]['makers'] != 16: raise ValueError('summary maker denominator')
        error = max(error, near([r[m] for m in METRICS], [actual[key(r)][m] for m in METRICS], 1e-10))
    if consumed_roster != set(roster): raise ValueError('unused correction templates')
    if dict(identities) != summary['identities'] or read(original/'IDENTITIES.json')['rows'] != dict(identities):
        raise ValueError('identity denominators')
    check_packet(read(original/'PUBLIC_PACKET.json'), public)
    for k, v in dict(rows=rows_count, streams=streams_count, conditions=condition_count, public_packets=len(public), fits=0).items():
        if summary[k] != v: raise ValueError('overall denominator')
    write(root/'INDEPENDENT_REGROUP.json', regroup(cells, cfg, roster))
    (root/'RETAINED_SOURCE_COUNTS_points.json.gz').write_bytes(gzip.compress(canonical(counts), mtime=0))
    return dict(controls=checks, paths=paths, rows=rows_count, streams=streams_count, conditions=condition_count,
        cells=len(cells), public_packets=len(public), identities=dict(identities), max_error=error,
        scope='independent correction/product verification; partial provenance under supplied laws; no learned trust, historical correspondence or human intent')


def run(root, plan, pulse):
    """Mandatory serial fixture gate before reading the campaign result."""
    import os
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    import pytest
    reports = []
    class Progress:
        def pytest_runtest_setup(self, item): pulse(phase='restoration-checker-fixture', test=item.nodeid)
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' or report.failed or report.skipped:
                reports.append(dict(test=report.nodeid, outcome=report.outcome, phase=report.when))
    with (root/'pytest.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        code = int(pytest.main(['-q', '-p', 'no:cacheprovider', '--basetemp', str(root/'fixture-temp'),
            *plan['design']['preflight_tests']], plugins=[Progress()]))
    write(root/'FIXTURE_GATE.json', dict(exit_code=code, reports=reports, log_sha256=file_digest(root/'pytest.log')))
    if code or any(r['outcome'] != 'passed' for r in reports): raise ValueError('checker fixture gate failed')
    return verify(root, plan, pulse)
