"""Independent standard-library regroup of original fixed-byte retention rows."""
from collections import defaultdict
from itertools import product
import math


def regroup(original, checker):
    keys = ('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha')
    identities = ('initial_maker', 'kind', 'switched', 'duplicates')
    expected = set(product(range(16), ('purpose', 'skill'), (False, True), (False, True)))
    target = {tuple(r[k] for k in keys): r for r in checker}
    assert len(target) == len(checker) == 1120
    metrics = sorted(set(checker[0]) - set(keys) - {'rows'})
    assert len(metrics) == 65 and len(original) == 143360
    groups = defaultdict(list)
    for row in original:
        groups[tuple(row[k] for k in keys)].append(row)
    assert set(groups) == set(target)

    def mean(values):
        assert values and all(math.isfinite(v) for v in values)
        return math.fsum(values) / len(values)

    strata = []
    discrepancy = 0.0
    for key, rows in sorted(groups.items()):
        assert len(rows) == 128
        assert {tuple(r[k] for k in identities) for r in rows} == expected
        for row in rows:
            for label in ('full', 'recent_half', 'recent_quarter', 'spaced_half', 'spaced_quarter'):
                assert row[label+'_storage_budget_bytes'] == row[label+'_storage_used_bytes'] + row[label+'_unused_budget_bytes']
                assert row[label+'_unused_budget_bytes'] >= 0
                assert row[label+'_storage_used_bytes'] == 8*row[label+'_compact_float64_count'] + 4*row[label+'_compact_int32_count'] + 256
            for size in ('half', 'quarter'):
                assert row['recent_'+size+'_storage_budget_bytes'] == row['spaced_'+size+'_storage_budget_bytes']
        result = dict(zip(keys, key), rows=128)
        for metric in metrics:
            result[metric] = mean([r[metric] for r in rows])
            discrepancy = max(discrepancy, abs(result[metric]-target[key][metric]))
        strata.append(result)
    assert discrepancy <= 1e-12

    def reduce(rows, keep, member, members):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in keep)].append(row)
        result = []
        for key, group in sorted(grouped.items()):
            assert len(group) == len(members) and {r[member] for r in group} == members
            result.append(dict(zip(keep, key), **{m: mean([r[m] for r in group]) for m in metrics}))
        return result

    laws = reduce(strata, ('lineage', 'evidence', 'length', 'checkpoint', 'alpha'), 'draw', {190201, 190202})
    population = reduce(laws, ('evidence', 'length', 'checkpoint', 'alpha'), 'lineage', set(range(190000, 190008)))
    assert len(laws) == 560 and len(population) == 70

    def contrasts(rows):
        result = []
        names = [m[len('full_'):] for m in metrics if m.startswith('full_')]
        assert len(names) == 13
        for row in rows:
            for size in ('half', 'quarter'):
                contrast = {k: row[k] for k in keys if k in row}
                contrast['size'] = size
                for name in names:
                    contrast[name] = row['spaced_'+size+'_'+name] - row['recent_'+size+'_'+name]
                assert contrast['storage_budget_bytes'] == 0
                result.append(contrast)
        return result

    return dict(strata=strata, laws=laws, population=population,
                paired_draw_contrasts=contrasts(strata), paired_law_contrasts=contrasts(laws),
                paired_population_contrasts=contrasts(population)), dict(max_regroup_discrepancy=discrepancy)
