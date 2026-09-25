"""Independent reconstruction of realized-risk-constrained storage."""
from collections import defaultdict
from fractions import Fraction as F
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance
from .randomized_storage_review import lottery, vertices

SCALARS = ('expected_retained_sources','minimum_expected_mass','deterministic_minimum_mass','expected_gain','worst_realized_mass')
LEVEL_SCALARS = ('realized_mass_floor','unrestricted_minimum_expected_mass','expected_mass_cost')


def risk(library, multiplier):
    assert multiplier in (F(0), F(1,2), F(1))
    unrestricted = lottery(library)
    baseline = max(min(F(*x) for x in r['masses']) for r in library['candidates'])
    floor = multiplier*baseline
    keep, drop = [], []
    for row in library['candidates']:
        (keep if all(F(*x)>=floor for x in row['masses']) else drop).append(row)
    assert keep
    constrained = lottery(dict(library,candidates=keep))
    assert F(*constrained['worst_realized_mass']) >= floor
    rational = lambda x: [x.numerator,x.denominator]
    return dict(floor_multiplier=rational(multiplier),realized_mass_floor=rational(floor),
        surviving_masks=sorted(r['mask'] for r in keep),surviving_count=len(keep),
        excluded_masks=sorted(r['mask'] for r in drop),lottery=constrained,
        unrestricted_lottery=unrestricted['selected_lottery'],
        unrestricted_minimum_expected_mass=unrestricted['minimum_expected_mass'],
        expected_mass_cost=rational(F(*unrestricted['minimum_expected_mass'])-F(*constrained['minimum_expected_mass'])))


def controls():
    masses=[[F(7,10),F(1,10)],[F(1,10),F(7,10)],[F(1,5),F(1,5)]]
    return {'live:unrestricted_known_gain': vertices(masses)[0]==F(2,5),
        'positive:full_floor_cost': vertices([masses[-1]])[0]==F(1,5),
        'placebo:empty_mass': vertices([[F(0),F(0)]])[0]==0}


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
        rebuilt['risk_levels'] = [risk(rebuilt, f) for f in (F(0), F(1,2), F(1))]
        assert dict(rebuilt,id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        for level in rebuilt['risk_levels']:
            a = level['lottery']
            comparisons.append(dict(selection_id=s['id'],floor_multiplier=level['floor_multiplier'],
                **{k:float(F(*a[k])) for k in SCALARS},
                **{k:float(F(*level[k])) for k in LEVEL_SCALARS},
                surviving_count=level['surviving_count'],support=len(a['selected_lottery'])))
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/realized_risk_points.json.gz').read_bytes()))
    assert len(rows) == 172032
    axes = ('lineage','evidence','length','checkpoint','draw','budget','floor')
    metrics = ('fixed_bytes','total_budget_bytes','expected_total_used_bytes','maximum_total_realized_bytes',
        *SCALARS,*LEVEL_SCALARS,'mass_uniform','mass_time','mass_reciprocal','surviving_count')
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
            sid=lookup[(tuple(times),tuple(costs),capacity)]; s=answers[sid]
            for level in s['risk_levels']:
                a=level['lottery']
                expected=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,
                    floor_multiplier=level['floor_multiplier'],surviving_masks=level['surviving_masks'],
                    surviving_count=level['surviving_count'],excluded_masks=level['excluded_masks'],
                    fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                    expected_total_used_bytes=st['weighted_overhead']+float(F(*a['expected_used_bytes'])),
                    maximum_total_realized_bytes=st['weighted_overhead']+a['maximum_realized_bytes'],
                    **{k:float(F(*a[k])) for k in SCALARS},
                    **{k:float(F(*level[k])) for k in LEVEL_SCALARS},
                    expected_masses=[float(F(*x)) for x in a['expected_masses']],
                    selected_lottery=a['selected_lottery'],optimal_basic_count=a['optimal_basic_count'],
                    feasible_basic_count=a['feasible_basic_count'],unique_candidates=len(s['candidates']))
                actual=rows[position]; position+=1
                assert actual==expected, position
                enriched=dict(actual,floor=float(F(*actual['floor_multiplier'])),**dict(zip(
                    ('mass_uniform','mass_time','mass_reciprocal'),actual['expected_masses'])))
                grouped[tuple(enriched[k] for k in axes)].append(enriched)
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
        strict_gain_risk_problems=sum(r['expected_gain']>0 for r in comparisons),checks=controls(),
        scope='finite-library expected source coverage subject to realized mass floors;not forecast accuracy or learned access')
