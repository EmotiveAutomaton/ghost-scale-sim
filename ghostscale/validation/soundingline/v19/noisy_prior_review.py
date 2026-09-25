"""Independent complete-policy audit of storage under noisy prior labels."""
from fractions import Fraction as F
from itertools import product
from .partial_prior_review import MIXTURES

RELIABILITIES = ((1, 3), (2, 3), (1, 1))
METRICS = ('fixed_mass', 'informed_mass', 'certainty_mass', 'full_mass',
           'information_value', 'certainty_over_fixed', 'certainty_penalty',
           'expected_used_bytes', 'expected_certainty_bytes',
           'maximum_realized_bytes', 'maximum_realized_certainty_bytes')


def value(library, counts, reliability):
    assert tuple(counts) in MIXTURES
    rows = sorted(library['candidates'], key=lambda r: r['mask'])
    mass = {r['mask']: tuple(F(*x) for x in r['masses']) for r in rows}
    charge = {r['mask']: r['used_bytes'] for r in rows}
    assert all(0 <= b <= library['capacity_bytes'] for b in charge.values())
    weights = tuple(F(c, 4) for c in counts)
    correct = F(*reliability)
    matrix = [[correct if label == prior else (1-correct)/2
               for label in range(3)] for prior in range(3)]
    # Optimize whole policies, not a sequence of conditional maximizations.
    policies = tuple(product(mass, repeat=3))
    def expected(policy):
        return sum(weights[p]*matrix[p][label]*mass[policy[label]][p]
                   for p in range(3) for label in range(3))
    policy_scores = {policy: expected(policy) for policy in policies}
    best = max(policy_scores.values())
    fixed_scores = {m: expected((m, m, m)) for m in mass}
    fixed = max(fixed_scores.values())
    fixed_ties = [m for m, s in fixed_scores.items() if s == fixed]
    rat = lambda x: [x.numerator, x.denominator]
    cues = []
    for label in range(3):
        joint = tuple(weights[p]*matrix[p][label] for p in range(3))
        probability = sum(joint)
        scores = {m: sum(joint[p]*mass[m][p] for p in range(3)) for m in mass}
        optimum = max(scores.values())
        ties = [m for m, s in scores.items() if s == optimum]
        certainty = [m for m in mass if mass[m][label] == max(v[label] for v in mass.values())]
        chosen, naive = max(ties), max(certainty)
        cues.append(dict(label=label, joint_prior=list(map(rat, joint)), probability=rat(probability),
            conditional_prior=list(map(lambda x: rat(x/probability), joint)) if probability else None,
            joint_mask_scores=[dict(mask=m, mass=rat(s)) for m, s in scores.items()],
            optimal_masks=ties, selected_mask=chosen, selected_used_bytes=charge[chosen],
            conditional_retained_mass=rat(optimum/probability) if probability else None,
            contribution=rat(optimum), certainty_optimal_masks=certainty,
            certainty_selected_mask=naive, certainty_used_bytes=charge[naive],
            certainty_contribution=rat(scores[naive])))
    chosen = tuple(c['selected_mask'] for c in cues)
    naive = tuple(c['certainty_selected_mask'] for c in cues)
    assert policy_scores[chosen] == best == sum(F(*c['contribution']) for c in cues)
    certain = policy_scores[naive]
    full = max(sum(weights[p]*mass[policy[p]][p] for p in range(3)) for policy in policies)
    assert fixed <= best <= full and certain <= best
    active = [c for c in cues if F(*c['probability'])]
    return dict(mixture_counts=list(counts), denominator=4,
        channel=[[rat(x) for x in row] for row in matrix], cues=cues,
        fixed_mask_scores=[dict(mask=m, mass=rat(s)) for m, s in fixed_scores.items()],
        fixed_optimal_masks=fixed_ties, selected_fixed_mask=max(fixed_ties), fixed_mass=rat(fixed),
        informed_mass=rat(best), certainty_mass=rat(certain), full_mass=rat(full),
        information_value=rat(best-fixed), certainty_over_fixed=rat(certain-fixed),
        certainty_penalty=rat(best-certain),
        expected_used_bytes=rat(sum(F(*c['probability'])*c['selected_used_bytes'] for c in cues)),
        expected_certainty_bytes=rat(sum(F(*c['probability'])*c['certainty_used_bytes'] for c in cues)),
        maximum_realized_bytes=max(c['selected_used_bytes'] for c in active),
        maximum_realized_certainty_bytes=max(c['certainty_used_bytes'] for c in active))


def controls():
    from .randomized_source_storage import fixture
    s = fixture([1, 2], [1, 1], 1, {'a': [1], 'b': [2]},
                [[F(1), F(0)], [F(0), F(1)], [F(1, 2)]*2])
    weak = value(s, (4, 0, 0), (1, 3))
    perfect = value(s, (2, 2, 0), (1, 1))
    return {'live:selective_noisy_value': value(s, (2, 2, 0), (2, 3))['information_value'] == [1, 4],
            'placebo:uninformative_channel': weak['information_value'] == [0, 1],
            'live:certainty_harms': weak['certainty_over_fixed'] == [-2, 3],
            'positive:perfect_channel': perfect['informed_mass'] == perfect['full_mass'] == [1, 1],
            'positive:zero_label_undefined': perfect['cues'][2]['conditional_prior'] is None}

from collections import Counter, defaultdict
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance

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
        rebuilt['mixtures'] = [dict(mixture_counts=list(c), channels=[value(rebuilt,c,p) for p in RELIABILITIES]) for c in MIXTURES]
        assert dict(rebuilt, id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        comparisons.extend(dict(selection_id=s['id'], mixture_counts=m['mixture_counts'],
            reliability=RELIABILITIES[j], **{k: float(F(*d[k])) if isinstance(d[k], list) else d[k]
                                        for k in METRICS})
            for m in rebuilt['mixtures'] for j,d in enumerate(m['channels']))
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/noisy_prior_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity, mixture_count=15, channel_count=3)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            for j, reliability in enumerate(RELIABILITIES):
                def number(s, m):
                    x = answers[s]['mixtures'][i]['channels'][j][m]
                    return float(F(*x)) if isinstance(x, list) else x
                strata.append(dict(zip(axes, key), mixture=mixture, reliability=reliability, rows=128,
                    **{m: fsum(n*number(s, m) for s, n in counts.items())/128 for m in METRICS}))
    def aggregate(records, names, expected):
        grouped = defaultdict(list)
        for r in records:
            key = tuple(tuple(r[k]) if k in ('mixture','reliability') else r[k] for k in names)
            grouped[key].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names, key), rows=len(rr),
                **{m: fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata, tuple(k for k in axes if k != 'draw')+('mixture', 'reliability'), 2)
    cells = aggregate(law, tuple(k for k in axes if k not in ('draw', 'lineage'))+('mixture', 'reliability'), 8)
    imperfect = [r for r in comparisons if r['reliability'] == (2,3)]
    return dict(passed=True, numerical_acceptance=True, original_rows=position,
        joined_evaluations=position*45, source_rosters=len(population), structures=len(structures),
        distinct_allocations=len(selections), distinct_channel_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections), paired_strata=strata,
        law_strata=law, equal_law_cells=cells, allocation_comparisons=comparisons,
        imperfect_problems=len(imperfect), strict_imperfect_information_problems=sum(r['information_value']>0 for r in imperfect),
        strict_certainty_penalty_problems=sum(r['certainty_penalty']>0 for r in imperfect),
        certainty_below_fixed_problems=sum(r['certainty_over_fixed']<0 for r in imperfect),
        checks=controls(), scope='finite-library noisy prior information;no fee,forecast accuracy or process correspondence')
