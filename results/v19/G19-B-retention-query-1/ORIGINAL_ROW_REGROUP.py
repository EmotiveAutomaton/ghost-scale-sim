"""Regroup original query-context rows without importing either scientific implementation."""
import argparse
from collections import defaultdict
import gzip
from itertools import combinations, product
import json
import math
from pathlib import Path


def regroup(original, checker, inherited):
    keys = ('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha')
    identity = ('initial_maker', 'kind', 'switched', 'duplicates')
    expected = set(product(range(16), ('purpose', 'skill'), (False, True), (False, True)))
    target = {tuple(r[k] for k in keys): r for r in checker}
    assert len(target) == len(checker) == 1120
    metrics = sorted(set(checker[0]) - set(keys) - {'rows'})
    assert len(metrics) == 90
    groups = defaultdict(list)
    for r in original:
        groups[tuple(r[k] for k in keys)].append(r)
    assert len(original) == len(inherited) == 143360 and set(groups) == set(target)

    def average(values):
        assert values and all(math.isfinite(v) for v in values)
        return math.fsum(values) / len(values)

    strata = []
    discrepancy = 0.0
    for key, values in sorted(groups.items()):
        assert len(values) == 128 and {tuple(r[k] for k in identity) for r in values} == expected
        d = dict(zip(keys, key), rows=128)
        for m in metrics:
            d[m] = average([r[m] for r in values])
            discrepancy = max(discrepancy, abs(d[m] - target[key][m]))
        strata.append(d)
    assert discrepancy <= 1e-12
    old_metrics = [m for m in metrics if not m.startswith('context')]
    assert len(old_metrics) == 50
    inherited_discrepancy = 0.0
    mean_recombination = 0.0
    for actual, old in zip(original, inherited):
        assert all(actual[k] == old[k] for k in (*keys, *identity, 'stream', 'report_sources'))
        for m in old_metrics:
            inherited_discrepancy = max(inherited_discrepancy, abs(actual[m] - old[m]))
        for window in ('full', 'recent_half', 'recent_quarter', 'spaced_half', 'spaced_quarter'):
            mean = average([actual[f'context{i}_{window}_expected_squared_regret'] for i in range(4)])
            mean_recombination = max(mean_recombination, abs(mean - actual[window+'_expected_squared_regret']))
    assert max(inherited_discrepancy, mean_recombination) <= 1e-12

    def reduce(rows, keep, member, members):
        grouped = defaultdict(list)
        for r in rows:
            grouped[tuple(r[k] for k in keep)].append(r)
        result = []
        for key, values in sorted(grouped.items()):
            assert len(values) == len(members) and {r[member] for r in values} == members
            result.append(dict(zip(keep, key), **{m: average([r[m] for r in values]) for m in metrics}))
        return result

    laws = reduce(strata, ('lineage', 'evidence', 'length', 'checkpoint', 'alpha'), 'draw', {190201, 190202})
    population = reduce(laws, ('evidence', 'length', 'checkpoint', 'alpha'), 'lineage', set(range(190000, 190008)))
    assert len(laws) == 560 and len(population) == 70

    def contrasts(rows):
        result = []
        for r in rows:
            for size in ('half', 'quarter'):
                d = {k: r[k] for k in keys if k in r}
                d['size'] = size
                for m in ('expected_squared_regret', 'expected_max_probability_error', 'expected_group_total_variation', 'compact_float64_count', 'compact_int32_count', 'unsupported_mass', 'supported_mass', 'retained_sources', 'forgotten_mass', 'full_hypothesis_float64_count'):
                    d[m] = r['spaced_'+size+'_'+m] - r['recent_'+size+'_'+m]
                for i in range(4):
                    for m in ('expected_squared_regret', 'expected_max_probability_error'):
                        d[f'context{i}_'+m] = r[f'context{i}_spaced_{size}_'+m] - r[f'context{i}_recent_{size}_'+m]
                assert d['retained_sources'] == 0
                for i, j in combinations(range(4), 2):
                    for m in ('expected_squared_regret', 'expected_max_probability_error'):
                        d[f'context{i}_minus_context{j}_'+m] = d[f'context{i}_'+m] - d[f'context{j}_'+m]
                result.append(d)
        return result

    return dict(strata=strata, laws=laws, population=population,
                paired_draw_contrasts=contrasts(strata), paired_law_contrasts=contrasts(laws),
                paired_population_contrasts=contrasts(population)), dict(
                    max_regroup_discrepancy=discrepancy,
                    max_inherited_discrepancy=inherited_discrepancy,
                    max_mean_recombination_discrepancy=mean_recombination)


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
