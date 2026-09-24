"""Independent standard-library reconstruction of the paired horizon summaries.

Supply the original retention_horizon_summary_points.json.gz, the independent
checker PAIRED_STRATA.json, and the accepted time_retention_summary_points.json.gz.
No producer or checker implementation is imported.
"""
import argparse
from collections import defaultdict
import gzip
from itertools import product
import json
import math
from pathlib import Path


def regroup(original, checker, inherited):
    keys = ('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha')
    identity = ('initial_maker', 'kind', 'switched', 'duplicates')
    expected_ids = set(product(range(16), ('purpose', 'skill'), (False, True), (False, True)))
    target = {tuple(r[k] for k in keys): r for r in checker}
    assert len(target) == len(checker) == 1120
    metrics = sorted(set(checker[0]) - set(keys) - {'rows'})
    assert len(metrics) == 83
    groups = defaultdict(list)
    for r in original:
        groups[tuple(r[k] for k in keys)].append(r)
    assert len(original) == len(inherited) == 143360 and set(groups) == set(target)

    def average(values):
        missing = [v is None for v in values]
        assert all(missing) or not any(missing)
        if all(missing):
            return None
        assert all(math.isfinite(v) for v in values)
        return math.fsum(values) / len(values)

    discrepancy = 0.0
    strata = []
    for key, values in sorted(groups.items()):
        assert len(values) == 128 and {tuple(r[k] for k in identity) for r in values} == expected_ids
        d = dict(zip(keys, key), rows=128)
        for metric in metrics:
            d[metric] = average([r[metric] for r in values])
            expected = target[key][metric]
            assert (d[metric] is None) == (expected is None)
            if expected is not None:
                discrepancy = max(discrepancy, abs(d[metric] - expected))
        strata.append(d)
    assert discrepancy <= 1e-12
    old_metrics = [m for m in metrics if not m.startswith(('current_', 'near_', 'far_'))]
    assert len(old_metrics) == 50
    inherited_discrepancy = 0.0
    for actual, old in zip(original, inherited):
        assert all(actual[k] == old[k] for k in (*keys, *identity, 'stream', 'report_sources'))
        for m in old_metrics:
            inherited_discrepancy = max(inherited_discrepancy, abs(actual[m] - old[m]))
    assert inherited_discrepancy <= 1e-12

    def reduce(rows, keep, member, expected_members):
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in keep)].append(r)
        out = []
        for key, values in sorted(groups.items()):
            assert len(values) == len(expected_members) and {r[member] for r in values} == expected_members
            out.append(dict(zip(keep, key), **{m: average([r[m] for r in values]) for m in metrics}))
        return out

    laws = reduce(strata, ('lineage', 'evidence', 'length', 'checkpoint', 'alpha'), 'draw', {190201, 190202})
    population = reduce(laws, ('evidence', 'length', 'checkpoint', 'alpha'), 'lineage', set(range(190000, 190008)))
    assert len(laws) == 560 and len(population) == 70

    def contrasts(rows):
        out = []
        for r in rows:
            for size in ('half', 'quarter'):
                d = {k: r[k] for k in keys if k in r}
                d.update(order='spaced', size=size)
                for m in ('expected_squared_regret', 'expected_max_probability_error', 'expected_group_total_variation', 'compact_float64_count', 'compact_int32_count', 'unsupported_mass', 'supported_mass', 'retained_sources', 'forgotten_mass', 'full_hypothesis_float64_count'):
                    d[m] = r['spaced_'+size+'_'+m] - r['recent_'+size+'_'+m]
                for span in ('current', 'near', 'far'):
                    for m in ('expected_squared_regret', 'expected_max_probability_error'):
                        a, b = r[span+'_spaced_'+size+'_'+m], r[span+'_recent_'+size+'_'+m]
                        assert (a is None) == (b is None)
                        d[span+'_'+m] = None if a is None else a-b
                assert d['retained_sources'] == 0
                out.append(d)
        return out

    return dict(strata=strata, laws=laws, population=population,
                paired_draw_contrasts=contrasts(strata), paired_law_contrasts=contrasts(laws),
                paired_population_contrasts=contrasts(population)), dict(
                    max_regroup_discrepancy=discrepancy, max_inherited_discrepancy=inherited_discrepancy)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'checker', 'inherited', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    def load(path):
        data = path.read_bytes()
        return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
    result, proof = regroup(load(args.original), load(args.checker), load(args.inherited))
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
        f.write('\n')
    print(json.dumps(proof))
