"""Independent policy enumeration for partial source-prior disclosure."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import product
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance

MIXTURES = tuple(c for c in product(range(5), repeat=3) if sum(c) == 4)
PARTITIONS = (((0, 1, 2),), ((0,), (1, 2)), ((1,), (0, 2)),
              ((2,), (0, 1)), ((0,), (1,), (2,)))
METRICS = ('retained_mass', 'information_value', 'full_disclosure_gap',
           'expected_used_bytes', 'maximum_realized_bytes')


def value(library, counts):
    assert tuple(counts) in MIXTURES
    rows = sorted(library['candidates'], key=lambda r: r['mask'])
    mass = {r['mask']: tuple(F(*x) for x in r['masses']) for r in rows}
    charge = {r['mask']: r['used_bytes'] for r in rows}
    assert all(0 <= b <= library['capacity_bytes'] for b in charge.values())
    weights = tuple(F(c, 4) for c in counts)
    rat = lambda x: [x.numerator, x.denominator]
    disclosures = []
    for partition in PARTITIONS:
        # Enumerate complete contingent policies, separately from conditional scores.
        policies = list(product(mass, repeat=len(partition)))
        expected = {p: sum(weights[j] * mass[p[i]][j]
                          for i, cell in enumerate(partition) for j in cell)
                    for p in policies}
        optimum = max(expected.values())
        cells = []
        for cell in partition:
            probability = sum(weights[j] for j in cell)
            scores = {m: sum(weights[j]*v[j] for j in cell) for m, v in mass.items()}
            best = max(scores.values())
            ties = sorted(m for m, score in scores.items() if score == best)
            cells.append(dict(prior_ids=list(cell), probability=rat(probability),
                conditional_prior=[rat(weights[j]/probability) for j in cell] if probability else None,
                joint_mask_scores=[dict(mask=m, mass=rat(scores[m])) for m in mass],
                optimal_masks=ties, selected_mask=max(ties), selected_used_bytes=charge[max(ties)],
                conditional_retained_mass=rat(best/probability) if probability else None,
                contribution=rat(best)))
        chosen = tuple(c['selected_mask'] for c in cells)
        assert expected[chosen] == optimum
        assert sum(F(*c['contribution']) for c in cells) == optimum
        disclosures.append(dict(partition=[list(c) for c in partition], cells=cells,
            retained_mass=rat(optimum),
            expected_used_bytes=rat(sum(F(*c['probability'])*c['selected_used_bytes'] for c in cells)),
            maximum_realized_bytes=max(c['selected_used_bytes'] for c in cells if F(*c['probability']))))
    baseline, full = (F(*disclosures[i]['retained_mass']) for i in (0, -1))
    for d in disclosures:
        score = F(*d['retained_mass'])
        assert baseline <= score <= full
        d.update(information_value=rat(score-baseline), full_disclosure_gap=rat(full-score))
    return dict(mixture_counts=list(counts), denominator=4, disclosures=disclosures)


def controls():
    from .randomized_source_storage import fixture
    s = fixture([1, 2], [1, 1], 1, {'a': [1], 'b': [2]},
                [[F(1), F(0)], [F(0), F(1)], [F(1, 2)]*2])
    v = value(s, (2, 2, 0))['disclosures']
    zero = v[3]['cells'][0]
    return {'live:selective_disclosure': v[1]['information_value'] == [1, 2],
            'placebo:irrelevant_partition': v[3]['information_value'] == [0, 1],
            'positive:undefined_zero_cell': zero['conditional_prior'] is None
                and zero['conditional_retained_mass'] is None
                and zero['optimal_masks'] == [1, 2],
            'placebo:point_prior': all(d['information_value'] == [0, 1]
                for d in value(s, (4, 0, 0))['disclosures'])}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    groups = defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups[(tuple(r['times']), tuple(r['costs']), r['capacity_bytes'])].append(r)
    answers, lookup, comparisons = {}, {}, []
    selections = read(root/'SELECTIONS.json')
    assert len(selections) == len(groups) == 28
    for s in selections:
        key = (tuple(s['times']), tuple(s['costs']), s['capacity_bytes'])
        rebuilt = reconstruct(groups.pop(key))
        rebuilt['mixtures'] = [value(rebuilt, c) for c in MIXTURES]
        assert dict(rebuilt, id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        comparisons.extend(dict(selection_id=s['id'], mixture_counts=m['mixture_counts'],
            partition=d['partition'], **{k: float(F(*d[k])) if isinstance(d[k], list) else d[k]
                                        for k in METRICS})
            for m in rebuilt['mixtures'] for d in m['disclosures'])
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/partial_prior_points.json.gz').read_bytes()))
    assert len(rows) == 57344
    axes = ('lineage', 'evidence', 'length', 'checkpoint', 'draw', 'budget')
    grouped = defaultdict(Counter); pairs = {}; position = 0
    for r in population:
        st = structures[r['structure']]; times = r['times']
        costs = [st['source_costs'][t-1] for t in times]
        pair = tuple(r[k] for k in ('lineage', 'structure', 'draw', 'initial_maker', 'kind', 'switched', 'duplicates'))
        if r['evidence'] == 'aware': pairs[pair] = times
        else: assert pairs[pair] == times
        for budget in ('half', 'quarter'):
            threshold = r['checkpoint']//2 if budget == 'half' else 3*r['checkpoint']//4
            recent = [i for i, t in enumerate(times) if t > threshold]; count = len(recent)
            ordered = sorted(range(len(times)), key=times.__getitem__)
            spaced = [ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity = min(sum(costs[i] for i in recent), sum(costs[i] for i in spaced))
            sid = lookup[(tuple(times), tuple(costs), capacity)]
            expected = dict(**{k: v for k, v in r.items() if k != 'times'}, budget=budget,
                selection_id=sid, fixed_bytes=st['weighted_overhead'],
                total_budget_bytes=st['weighted_overhead']+capacity, mixture_count=15, partition_count=5)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            for j, partition in enumerate(PARTITIONS):
                def number(s, m):
                    x = answers[s]['mixtures'][i]['disclosures'][j][m]
                    return float(F(*x)) if isinstance(x, list) else x
                strata.append(dict(zip(axes, key), mixture=mixture, partition=partition, rows=128,
                    **{m: fsum(n*number(s, m) for s, n in counts.items())/128 for m in METRICS}))
    def aggregate(records, names, expected):
        grouped = defaultdict(list)
        for r in records:
            key = tuple(tuple(r[k]) if k == 'mixture' else tuple(map(tuple, r[k]))
                        if k == 'partition' else r[k] for k in names)
            grouped[key].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names, key), rows=len(rr),
                **{m: fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata, tuple(k for k in axes if k != 'draw')+('mixture', 'partition'), 2)
    cells = aggregate(law, tuple(k for k in axes if k not in ('draw', 'lineage'))+('mixture', 'partition'), 8)
    partial = [r for r in comparisons if 1 < len(r['partition']) < 3]
    return dict(passed=True, numerical_acceptance=True, original_rows=position,
        joined_evaluations=position*75, source_rosters=len(population), structures=len(structures),
        distinct_allocations=len(selections), distinct_disclosure_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections), paired_strata=strata,
        law_strata=law, equal_law_cells=cells, allocation_comparisons=comparisons,
        partial_problems=len(partial), strict_partial_information_problems=sum(r['information_value']>0 for r in partial),
        partial_full_value_problems=sum(r['information_value']>0 and r['full_disclosure_gap']==0 for r in partial),
        checks=controls(), scope='finite-library partial prior information;no fee,forecast accuracy or process correspondence')
