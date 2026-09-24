"""Standard-library regroup of original fixed-prior aggregate-state records."""
from collections import defaultdict
from itertools import product
import math


def regroup(original, checker):
    keys = ('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'alpha')
    identity = ('initial_maker', 'kind', 'switched', 'duplicates')
    expected = set(product(range(16), ('purpose', 'skill'), (False, True), (False, True)))
    target = {tuple(r[k] for k in keys):r for r in checker}
    assert len(target) == len(checker) == 1120
    metrics = sorted(set(checker[0])-set(keys)-{'rows'})
    assert len(metrics) == 19 and len(original) == 143360
    groups = defaultdict(list)
    for row in original:groups[tuple(row[k] for k in keys)].append(row)
    assert set(groups) == set(target)
    def mean(values):
        assert values and all(math.isfinite(v) for v in values)
        return math.fsum(values)/len(values)
    strata=[];discrepancy=0.
    for key, rows in sorted(groups.items()):
        assert len(rows)==128 and {tuple(r[k] for k in identity) for r in rows}==expected
        for row in rows:
            assert row['aggregate_state_bytes']==8*row['aggregate_float64_count']+4*row['aggregate_int32_count']
            assert row['source_table_state_bytes']==8*row['source_table_float64_count']+4*row['source_table_int32_count']
            assert row['full_weight_state_bytes']==8*row['full_hypothesis_float64_count']
            assert row['shared_law_float64_count']==512 and row['aggregate_int32_count']==8
        result=dict(zip(keys,key),rows=128)
        for metric in metrics:
            result[metric]=mean([r[metric] for r in rows]);discrepancy=max(discrepancy,abs(result[metric]-target[key][metric]))
        strata.append(result)
    assert discrepancy<=1e-12
    def reduce(rows, keep, member, members):
        grouped=defaultdict(list)
        for row in rows:grouped[tuple(row[k] for k in keep)].append(row)
        result=[]
        for key,rows in sorted(grouped.items()):
            assert len(rows)==len(members) and {r[member] for r in rows}==members
            result.append(dict(zip(keep,key),**{m:mean([r[m] for r in rows]) for m in metrics}))
        return result
    laws=reduce(strata,('lineage','evidence','length','checkpoint','alpha'),'draw',{190201,190202})
    population=reduce(laws,('evidence','length','checkpoint','alpha'),'lineage',set(range(190000,190008)))
    assert len(laws)==560 and len(population)==70
    return dict(strata=strata,laws=laws,population=population),dict(max_regroup_discrepancy=discrepancy)
