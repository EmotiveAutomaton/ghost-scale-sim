"""Independent full-constraint rational review of expected-regret lotteries."""
from collections import defaultdict
from fractions import Fraction as F
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance
from .randomized_storage_review import vertices, lottery as coverage_lottery


SCALARS = ('expected_retained_sources', 'maximum_expected_regret',
    'coverage_lottery_regret', 'regret_improvement', 'deterministic_minimax_regret',
    'gain_over_deterministic', 'minimum_expected_mass', 'minimum_mass_cost',
    'worst_realized_regret')


def lottery(library):
    coverage = coverage_lottery(library)
    rows = sorted(library['candidates'], key=lambda row: row['mask'])
    masses = [[F(*v) for v in row['masses']] for row in rows]
    optima = [F(*v) for v in library['best_prior_mass']]
    assert optima == [max(row[j] for row in masses) for j in range(len(optima))]
    regrets = [[optima[j]-row[j] for j in range(len(optima))] for row in masses]
    for row, regret in zip(rows, regrets):
        assert regret == [F(*v) for v in row['prior_regrets']]
        assert max(regret) == F(*row['maximum_regret'])
    value, ties, count = vertices(masses, optima)
    weights = ties[-1]
    rational = lambda x: [x.numerator, x.denominator]
    encode = lambda w: [dict(mask=rows[i]['mask'], weight=rational(x)) for i,x in enumerate(w) if x]
    expected = [sum(weights[i]*masses[i][j] for i in range(len(rows))) for j in range(len(optima))]
    expected_regrets = [o-e for o,e in zip(optima, expected)]
    coverage_regret = max(o-F(*e) for o,e in zip(optima, coverage['expected_masses']))
    deterministic = min(map(max, regrets))
    assert max(expected_regrets) == -value
    return dict(optimal_basic_lotteries=[encode(w) for w in ties], selected_lottery=encode(weights),
        optimal_basic_count=len(ties), feasible_basic_count=count,
        expected_masses=list(map(rational, expected)), expected_prior_regrets=list(map(rational, expected_regrets)),
        maximum_expected_regret=rational(-value), coverage_lottery_regret=rational(coverage_regret),
        regret_improvement=rational(coverage_regret+value), deterministic_minimax_regret=rational(deterministic),
        gain_over_deterministic=rational(deterministic+value), minimum_expected_mass=rational(min(expected)),
        minimum_mass_cost=rational(F(*coverage['minimum_expected_mass'])-min(expected)),
        worst_realized_regret=rational(max(max(regrets[i]) for i,w in enumerate(weights) if w)),
        expected_used_bytes=rational(sum(w*rows[i]['used_bytes'] for i,w in enumerate(weights))),
        maximum_realized_bytes=max(rows[i]['used_bytes'] for i,w in enumerate(weights) if w),
        expected_retained_sources=rational(sum(w*len(rows[i]['selected_times']) for i,w in enumerate(weights))),
        coverage_lottery=coverage['selected_lottery'])


def controls():
    value, ties, _ = vertices([[F(2,5),F(2,5)],[F(3,5),F(3,10)],[F(0),F(3,10)]], [F(3,5),F(2,5)])
    return {'live:known_minimax_expected_regret': value == -F(1,15) and ties == [(F(1,3),F(2,3),F(0))],
        'positive:asymmetric_prior_offsets': vertices([[F(1),F(0)],[F(0),F(1)]],[F(1),F(1)])[0] == -F(1,2),
        'placebo:no_opportunity_loss': vertices([[F(1,3)]*3],[F(1,3)]*3)[0] == 0}


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
        comparisons.append(dict(selection_id=s['id'], **{k:float(F(*a[k])) for k in SCALARS},
            support=len(a['selected_lottery']), optimal_basic_count=a['optimal_basic_count']))
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/randomized_regret_points.json.gz').read_bytes()))
    assert len(rows) == 57344
    axes = ('lineage','evidence','length','checkpoint','draw','budget')
    metrics = ('fixed_bytes','total_budget_bytes','expected_total_used_bytes','maximum_total_realized_bytes',
        *SCALARS,'mass_uniform','mass_time','mass_reciprocal','regret_uniform','regret_time','regret_reciprocal')
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
                **{k:float(F(*a[k])) for k in SCALARS},
                expected_masses=[float(F(*x)) for x in a['expected_masses']],
                expected_prior_regrets=[float(F(*x)) for x in a['expected_prior_regrets']],
                selected_lottery=a['selected_lottery'],coverage_lottery=a['coverage_lottery'],
                optimal_basic_count=a['optimal_basic_count'],feasible_basic_count=a['feasible_basic_count'],
                unique_candidates=len(s['candidates']))
            actual=rows[position]; position+=1
            assert actual==expected, position
            grouped[tuple(actual[k] for k in axes)].append(dict(actual,**dict(zip(
                ('mass_uniform','mass_time','mass_reciprocal','regret_uniform','regret_time','regret_reciprocal'),
                actual['expected_masses']+actual['expected_prior_regrets']))))
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
        strict_gain_allocations=sum(r['gain_over_deterministic']>0 for r in comparisons),
        changed_objective_gains=sum(r['regret_improvement']>0 for r in comparisons),checks=controls(),
        scope='finite-library minimum maximum expected source regret;prior fixed before draw;not pointwise improvement or forecast accuracy')
