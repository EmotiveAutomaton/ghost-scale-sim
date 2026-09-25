"""Independent rational enumeration and population joins for prior information."""
from collections import defaultdict, Counter
from fractions import Fraction as F
from itertools import product
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance
from .randomized_storage_review import lottery

MIXTURES = tuple(c for c in product(range(5), repeat=3) if sum(c) == 4)
METRICS = ('fixed_mass', 'revealed_mass', 'information_value',
           'coverage_lottery_mass', 'fixed_over_coverage', 'expected_revealed_bytes')


def value(library, counts, coverage=None):
    assert tuple(counts) in MIXTURES
    rows = sorted(library['candidates'], key=lambda r: r['mask'])
    weights = tuple(F(c, 4) for c in counts)
    mass = {r['mask']: tuple(F(*x) for x in r['masses']) for r in rows}
    scores = {m: sum(weights[j]*v[j] for j in range(3)) for m, v in mass.items()}
    fixed = max(scores.values())
    ties = sorted(m for m in scores if scores[m] == fixed)
    best = [max(v[j] for v in mass.values()) for j in range(3)]
    revealed_ties = [[m for m in mass if mass[m][j] == best[j]] for j in range(3)]
    choices = [max(t) for t in revealed_ties]
    # Independent policy enumeration, including priors with zero probability.
    revealed = max(sum(weights[j]*mass[p[j]][j] for j in range(3))
                   for p in product(mass, repeat=3))
    assert revealed == sum(weights[j]*best[j] for j in range(3))
    cov = lottery(library) if coverage is None else coverage
    average = sum(weights[j]*F(*cov['expected_masses'][j]) for j in range(3))
    assert revealed >= fixed >= average
    charges = {r['mask']: r['used_bytes'] for r in rows}
    assert all(0 <= b <= library['capacity_bytes'] for b in charges.values())
    rat = lambda x: [x.numerator, x.denominator]
    return dict(mixture_counts=list(counts), denominator=4,
        fixed_mask_scores=[dict(mask=m, mass=rat(scores[m])) for m in scores],
        fixed_optimal_masks=ties, selected_fixed_mask=max(ties), fixed_mass=rat(fixed),
        revealed_optimal_masks=revealed_ties, selected_revealed_masks=choices,
        prior_best_masses=list(map(rat, best)), revealed_mass=rat(revealed),
        information_value=rat(revealed-fixed), coverage_lottery_mass=rat(average),
        fixed_over_coverage=rat(fixed-average), coverage_lottery=cov['selected_lottery'],
        fixed_used_bytes=charges[max(ties)], revealed_used_bytes=[charges[m] for m in choices],
        expected_revealed_bytes=rat(sum(weights[j]*charges[choices[j]] for j in range(3))),
        maximum_realized_revealed_bytes=max(charges[choices[j]] for j in range(3) if counts[j]))


def controls():
    from .randomized_source_storage import fixture
    s = fixture([1,2], [1,1], 1, {'a':[1], 'b':[2]},
                [[F(1),F(0)], [F(0),F(1)], [F(1,2),F(1,2)]])
    return {'live:enumerated_policy_value': value(s,(2,2,0))['information_value'] == [1,2],
            'positive:exact_fixed_tie': value(s,(2,2,0))['fixed_optimal_masks'] == [1,2],
            'placebo:point_prior': value(s,(4,0,0))['information_value'] == [0,1]}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    groups = defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    answers, lookup, comparisons = {}, {}, []
    selections = read(root/'SELECTIONS.json')
    assert len(selections) == len(groups) == 28
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        rebuilt = reconstruct(groups.pop(key))
        cov = lottery(rebuilt)
        rebuilt['mixtures'] = [value(rebuilt,c,cov) for c in MIXTURES]
        assert dict(rebuilt,id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        comparisons.extend(dict(selection_id=s['id'],mixture_counts=m['mixture_counts'],
            **{k:float(F(*m[k])) for k in METRICS}) for m in rebuilt['mixtures'])
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/prior_information_points.json.gz').read_bytes()))
    assert len(rows) == 57344
    axes = ('lineage','evidence','length','checkpoint','draw','budget')
    grouped = defaultdict(Counter); pairs = {}; position = 0
    for r in population:
        st = structures[r['structure']]; times = r['times']
        costs = [st['source_costs'][t-1] for t in times]
        pair = tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence'] == 'aware': pairs[pair] = times
        else: assert pairs[pair] == times
        for budget in ('half','quarter'):
            threshold = r['checkpoint']//2 if budget == 'half' else 3*r['checkpoint']//4
            recent = [i for i,t in enumerate(times) if t > threshold]; count = len(recent)
            ordered = sorted(range(len(times)),key=times.__getitem__)
            spaced = [ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity = min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced))
            sid = lookup[(tuple(times),tuple(costs),capacity)]
            expected = dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                mixture_count=15)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            strata.append(dict(zip(axes,key),mixture=mixture,rows=128,
                **{m:fsum(n*float(F(*answers[s]['mixtures'][i][m])) for s,n in counts.items())/128
                   for m in METRICS}))
    def aggregate(records,names,expected):
        grouped = defaultdict(list)
        for r in records: grouped[tuple(tuple(r[k]) if k=='mixture' else r[k] for k in names)].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata,tuple(k for k in axes if k!='draw')+('mixture',),2)
    cells = aggregate(law,tuple(k for k in axes if k not in ('draw','lineage'))+('mixture',),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,mixture_evaluations=position*15,
        source_rosters=len(population),structures=len(structures),distinct_allocations=len(selections),
        distinct_mixture_problems=len(comparisons),candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,allocation_comparisons=comparisons,
        strict_information_problems=sum(r['information_value']>0 for r in comparisons),checks=controls(),
        scope='gross finite-library value of prior revelation;no fee,forecast accuracy or process correspondence')
