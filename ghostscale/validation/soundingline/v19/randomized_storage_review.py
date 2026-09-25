"""Independent full-constraint rational review of source-storage lotteries."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations, permutations
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance


def determinant(matrix):
    total = F()
    for order in permutations(range(len(matrix))):
        inversions = sum(order[i] > order[j] for i in range(len(order)) for j in range(i+1, len(order)))
        term = F((-1) ** inversions)
        for i, j in enumerate(order):
            term *= matrix[i][j]
        total += term
    return total


def vertices(masses, offsets=None):
    """Enumerate zero-weight and active-prior constraints in the full LP.

    Maximize the minimum expected mass minus a prior-specific offset. Cramer's
    rule is independent of the producer's support-based Gauss-Jordan solver.
    """
    n, p = len(masses), len(masses[0])
    offsets = [F()] * p if offsets is None else list(map(F, offsets))
    constraints = [([F(i == j) for i in range(n)] + [F()], F()) for j in range(n)]
    constraints += [([row[j] for row in masses] + [F(-1)], offsets[j]) for j in range(p)]
    found = {}
    for active in combinations(constraints, n):
        matrix = [[F(1)] * n + [F()]] + [row for row, _ in active]
        rhs = [F(1)] + [b for _, b in active]
        det = determinant(matrix)
        if not det:
            continue
        solution = []
        for column in range(n + 1):
            replaced = [list(row) for row in matrix]
            for i, value in enumerate(rhs):
                replaced[i][column] = value
            solution.append(determinant(replaced) / det)
        weights, value = tuple(solution[:-1]), solution[-1]
        if min(weights) < 0:
            continue
        expected = [sum(weights[i] * masses[i][j] for i in range(n)) - offsets[j] for j in range(p)]
        if min(expected) < value:
            continue
        assert sum(weights) == 1 and min(expected) == value
        found[weights] = value
    best = max(found.values())
    return best, sorted(w for w, value in found.items() if value == best), len(found)


def lottery(library):
    rows = sorted(library['candidates'], key=lambda row: row['mask'])
    charges = dict(zip(library['times'], library['costs']))
    for row in rows:
        assert sum(charges[t] for t in row['selected_times']) == row['used_bytes'] <= library['capacity_bytes']
    masses = [[F(*v) for v in row['masses']] for row in rows]
    best, ties, count = vertices(masses)
    weights = ties[-1]
    rational = lambda x: [x.numerator, x.denominator]
    encode = lambda w: [dict(mask=rows[i]['mask'], weight=rational(x)) for i, x in enumerate(w) if x]
    expected = [sum(weights[i] * masses[i][j] for i in range(len(rows))) for j in range(len(masses[0]))]
    deterministic = max(map(min, masses))
    assert F(*library['minimum_mass']) == deterministic
    return dict(optimal_basic_lotteries=[encode(w) for w in ties], selected_lottery=encode(weights),
        optimal_basic_count=len(ties), feasible_basic_count=count, expected_masses=list(map(rational, expected)),
        minimum_expected_mass=rational(best), deterministic_minimum_mass=rational(deterministic),
        expected_gain=rational(best-deterministic), worst_realized_mass=rational(min(min(masses[i]) for i,w in enumerate(weights) if w)),
        expected_used_bytes=rational(sum(weights[i]*row['used_bytes'] for i,row in enumerate(rows))),
        maximum_realized_bytes=max(row['used_bytes'] for i,row in enumerate(rows) if weights[i]),
        expected_retained_sources=rational(sum(weights[i]*len(row['selected_times']) for i,row in enumerate(rows))))


def controls():
    gain, ties, _ = vertices([[F(1), F(0)], [F(0), F(1)]])
    same, same_ties, _ = vertices([[F(1,3)]*3]*3)
    return {'live:known_lottery_gain': gain == F(1,2) and ties == [(F(1,2),F(1,2))],
        'positive:three_way_gain': vertices([[F(i==j) for j in range(3)] for i in range(3)])[0] == F(1,3),
        'placebo:identical_masks': same == F(1,3) and len(same_ties) == 3}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    groups = defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    selections = read(root/'SELECTIONS.json')
    assert len(selections) == len(groups) == 28
    answers, lookup, comparisons = {}, {}, []
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        rebuilt = reconstruct(groups.pop(key))
        rebuilt['lottery'] = lottery(rebuilt)
        assert dict(rebuilt,id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        a = rebuilt['lottery']
        comparisons.append(dict(selection_id=s['id'], **{k:float(F(*a[k])) for k in
            ('minimum_expected_mass','deterministic_minimum_mass','expected_gain','worst_realized_mass')},
            support=len(a['selected_lottery']), optimal_basic_count=a['optimal_basic_count']))
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/randomized_storage_points.json.gz').read_bytes()))
    assert len(rows) == 57344
    axes = ('lineage','evidence','length','checkpoint','draw','budget')
    metrics = ('fixed_bytes','total_budget_bytes','expected_total_used_bytes','maximum_total_realized_bytes',
        'expected_retained_sources','minimum_expected_mass','deterministic_minimum_mass','expected_gain',
        'worst_realized_mass','mass_uniform','mass_time','mass_reciprocal')
    grouped = defaultdict(list); pairs = {}; position = 0
    for r in population:
        st=structures[r['structure']]; times=r['times']; costs=[st['source_costs'][t-1] for t in times]
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware': pairs[pair]=times
        else: assert pairs[pair]==times
        for budget in ('half','quarter'):
            threshold=r['checkpoint']//2 if budget=='half' else 3*r['checkpoint']//4
            recent=[i for i,t in enumerate(times) if t>threshold]; count=len(recent)
            ordered=sorted(range(len(times)),key=times.__getitem__)
            spaced=[ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced))
            sid=lookup[(tuple(times),tuple(costs),capacity)]; s=answers[sid]; a=s['lottery']
            expected=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                expected_total_used_bytes=st['weighted_overhead']+float(F(*a['expected_used_bytes'])),
                maximum_total_realized_bytes=st['weighted_overhead']+a['maximum_realized_bytes'],
                expected_retained_sources=float(F(*a['expected_retained_sources'])),
                minimum_expected_mass=float(F(*a['minimum_expected_mass'])),
                expected_masses=[float(F(*x)) for x in a['expected_masses']],
                deterministic_minimum_mass=float(F(*a['deterministic_minimum_mass'])),
                expected_gain=float(F(*a['expected_gain'])),worst_realized_mass=float(F(*a['worst_realized_mass'])),
                selected_lottery=a['selected_lottery'],optimal_basic_count=a['optimal_basic_count'],
                feasible_basic_count=a['feasible_basic_count'],unique_candidates=len(s['candidates']))
            actual=rows[position]; position+=1
            assert actual==expected, position
            grouped[tuple(actual[k] for k in axes)].append(dict(actual,**dict(zip(
                ('mass_uniform','mass_time','mass_reciprocal'),actual['expected_masses']))))
    def aggregate(groups,names,expected):
        result=[]
        for key, rr in sorted(groups.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    strata=aggregate(grouped,axes,128); by_law=defaultdict(list)
    for r in strata: by_law[tuple(r[k] for k in axes if k!='draw')].append(r)
    law=aggregate(by_law,tuple(k for k in axes if k!='draw'),2); by_cell=defaultdict(list)
    for r in law: by_cell[tuple(r[k] for k in axes if k not in ('draw','lineage'))].append(r)
    cells=aggregate(by_cell,tuple(k for k in axes if k not in ('draw','lineage')),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,source_rosters=len(population),
        structures=len(structures),distinct_allocations=len(selections),candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,allocation_comparisons=comparisons,
        strict_gain_allocations=sum(r['expected_gain']>0 for r in comparisons),checks=controls(),
        scope='finite-library maximum minimum expected source mass;prior fixed before draw;not pointwise improvement or forecast accuracy')
