"""Independent rational regret decisions and full original-population review."""
from collections import defaultdict
from fractions import Fraction
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct as reconstruct_library, provenance, PRIORS


def regret_decision(library):
    rational = lambda v: [v.numerator, v.denominator]
    optima = [Fraction(*x) for x in library['best_prior_mass']]
    scores = {}
    minimums = {}
    for row in library['candidates']:
        masses = [Fraction(*x) for x in row['masses']]
        deficits = [optimum - mass for optimum, mass in zip(optima, masses)]
        assert all(x >= 0 for x in deficits)
        assert list(map(rational, deficits)) == row['prior_regrets']
        scores[row['mask']] = max(deficits)
        minimums[row['mask']] = min(masses)
        assert rational(max(deficits)) == row['maximum_regret']
    best = min(scores.values())
    ties = sorted(mask for mask, score in scores.items() if score == best)
    chosen = ties[-1]
    robust = library['selected_mask']
    return dict(regret_selected_mask=chosen, regret_optimal_masks=ties,
        minimax_regret=rational(best), robust_choice_regret=rational(scores[robust]),
        regret_improvement=rational(scores[robust] - best),
        minimum_mass_cost=rational(minimums[robust] - minimums[chosen]),
        regret_choice_minimum_mass=rational(minimums[chosen]))


def reconstruct(records):
    result = reconstruct_library(records)
    result.update(regret_decision(result))
    return result


def controls():
    library = dict(best_prior_mass=[[3,5],[2,5]], selected_mask=1, candidates=[
        dict(mask=1,masses=[[2,5],[2,5]],prior_regrets=[[1,5],[0,1]],maximum_regret=[1,5]),
        dict(mask=2,masses=[[3,5],[3,10]],prior_regrets=[[0,1],[1,10]],maximum_regret=[1,10]),
        dict(mask=4,masses=[[0,1],[3,10]],prior_regrets=[[3,5],[1,10]],maximum_regret=[3,5])])
    r = regret_decision(library)
    return {'live:known_regret_choice': r['regret_selected_mask'] == 2,
        'positive:known_gain': r['regret_improvement'] == [1,10],
        'placebo:coverage_not_free': r['minimum_mass_cost'] == [1,10]}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    parent = read(root/'inputs/PARENT_SELECTIONS.json')
    assert len(parent) == 84
    groups = defaultdict(list)
    for r in parent:
        groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    selections = read(root/'SELECTIONS.json')
    assert len(selections) == len(groups) == 28
    answers = {}; lookup = {}; comparisons = []
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        answer = reconstruct(groups.pop(key))
        assert dict(answer,id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = answer
        comparisons.append(dict(selection_id=s['id'],regret_mask=s['regret_selected_mask'],
            robust_mask=s['selected_mask'],regret=float(Fraction(*s['minimax_regret'])),
            robust_regret=float(Fraction(*s['robust_choice_regret'])),
            improvement=float(Fraction(*s['regret_improvement'])),
            minimum_mass_cost=float(Fraction(*s['minimum_mass_cost']))))
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/source_regret_points.json.gz').read_bytes()))
    assert len(rows) == 57344
    axes = ('lineage','evidence','length','checkpoint','draw','budget')
    metrics = ('minimum_mass','maximum_regret','retained_sources','fixed_bytes',
        'total_budget_bytes','total_used_bytes','robust_choice_regret','regret_improvement',
        'minimum_mass_cost','mass_uniform','mass_time','mass_reciprocal')
    grouped = defaultdict(list); position = 0; pairs = {}
    for r in population:
        st = structures[r['structure']]; times = r['times']
        costs = [st['source_costs'][t-1] for t in times]
        pair = tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence'] == 'aware': pairs[pair] = times
        else: assert pairs[pair] == times
        for budget in ('half','quarter'):
            threshold = r['checkpoint']//2 if budget == 'half' else 3*r['checkpoint']//4
            recent = [i for i,t in enumerate(times) if t>threshold]; count = len(recent)
            ordered = sorted(range(len(times)),key=times.__getitem__)
            spaced = [ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity = min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced))
            sid = lookup[(tuple(times),tuple(costs),capacity)]; s = answers[sid]
            chosen = next(a for a in s['candidates'] if a['mask'] == s['regret_selected_mask'])
            expected = dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                total_used_bytes=st['weighted_overhead']+chosen['used_bytes'],retained_sources=len(chosen['selected_times']),
                minimum_mass=float(Fraction(*chosen['worst_mass'])),masses=[float(Fraction(*x)) for x in chosen['masses']],
                prior_regrets=[float(Fraction(*x)) for x in chosen['prior_regrets']],maximum_regret=float(Fraction(*chosen['maximum_regret'])),
                robust_choice_regret=float(Fraction(*s['robust_choice_regret'])),regret_improvement=float(Fraction(*s['regret_improvement'])),
                minimum_mass_cost=float(Fraction(*s['minimum_mass_cost'])),unique_candidates=len(s['candidates']),
                regret_optimal_masks=s['regret_optimal_masks'],regret_selected_mask=s['regret_selected_mask'],robust_selected_mask=s['selected_mask'])
            actual = rows[position]; position += 1
            assert actual == expected, position
            enriched = dict(actual, **dict(zip(('mass_uniform','mass_time','mass_reciprocal'),actual['masses'])))
            grouped[tuple(actual[k] for k in axes)].append(enriched)
    def aggregate(groups,names,expected):
        result = []
        for key, rr in sorted(groups.items()):
            assert len(rr) == expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    strata = aggregate(grouped,axes,128); by_law = defaultdict(list)
    for r in strata: by_law[tuple(r[k] for k in axes if k!='draw')].append(r)
    law = aggregate(by_law,tuple(k for k in axes if k!='draw'),2); by_cell = defaultdict(list)
    for r in law: by_cell[tuple(r[k] for k in axes if k not in ('draw','lineage'))].append(r)
    cells = aggregate(by_cell,tuple(k for k in axes if k not in ('draw','lineage')),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,source_rosters=len(population),
        structures=len(structures),distinct_allocations=len(selections),candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,allocation_comparisons=comparisons,
        strict_gain_allocations=sum(r['improvement']>0 for r in comparisons),
        changed_masks=sum(r['regret_mask']!=r['robust_mask'] for r in comparisons),checks=controls(),
        scope='finite candidate-library minimax source-prior regret; not unrestricted storage, forecast accuracy or historical correspondence')
