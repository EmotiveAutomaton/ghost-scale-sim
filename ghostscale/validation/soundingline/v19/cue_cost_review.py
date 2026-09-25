"""Independent complete-choice reconstruction of priced cue acquisition."""
from fractions import Fraction as F
from itertools import product
from .noisy_prior_review import value, MIXTURES, RELIABILITIES

FEES = ((0, 1), (1, 64), (1, 16), (1, 4))
METRICS = ('gross_mass', 'net_utility', 'net_gain_over_no_access', 'fee_paid',
           'expected_used_bytes', 'maximum_realized_bytes', 'acquire')


def solve(library, counts, reliability, fees=FEES):
    base = value(library, counts, reliability)
    mass = {r['mask']: [F(*x) for x in r['masses']] for r in library['candidates']}
    charges = {r['mask']: r['used_bytes'] for r in library['candidates']}
    masks = sorted(mass)
    correct = F(*reliability)
    joint = [[F(counts[p], 4)*(correct if p == label else (1-correct)/2)
              for p in range(3)] for label in range(3)]
    def score(policy):
        return sum(joint[label][p]*mass[policy[label]][p]
                   for p in range(3) for label in range(3))
    scores = {p: score(p) for p in product(masks, repeat=3)}
    fixed = max(score((m,)*3) for m in masks)
    paid = max(scores.values())
    rat = lambda x: [x.numerator, x.denominator]
    levels = []
    for fee in fees:
        f = F(*fee)
        assert f >= 0
        # Optimize the union of every free and every paid choice directly.
        choices = [(False, (m,)*3, scores[(m,)*3]) for m in masks]
        choices += [(True, p, v-f) for p, v in scores.items()]
        optimum = max(v for _, _, v in choices)
        ties = [(buy, p) for buy, p, v in choices if v == optimum]
        buy, selected = max(ties, key=lambda t: (not t[0], t[1]))
        probability = [sum(row) for row in joint]
        levels.append(dict(fee=list(fee),
            optimal_choices=[dict(acquire=b, policy=list(p)) for b, p in ties],
            selected=dict(acquire=buy, policy=list(selected)),
            gross_mass=rat(scores[selected]), net_utility=rat(optimum),
            net_gain_over_no_access=rat(optimum-fixed), fee_paid=rat(f if buy else F(0)),
            expected_used_bytes=rat(sum(probability[l]*charges[selected[l]] for l in range(3))),
            maximum_realized_bytes=max(charges[selected[l]] for l in range(3) if probability[l])))
    return dict(free_access=base,
        all_paid_policies=[dict(policy=list(p), gross_mass=rat(v)) for p, v in scores.items()],
        paid_optimal_policies=[list(p) for p, v in scores.items() if v == paid],
        break_even_fee=rat(paid-fixed), fee_levels=levels)


def controls():
    s = dict(capacity_bytes=1, candidates=[
        dict(mask=1, used_bytes=1, masses=[[1,1],[0,1],[1,2]]),
        dict(mask=2, used_bytes=1, masses=[[0,1],[1,1],[1,2]])])
    a = solve(s, (2,2,0), (2,3))
    null = solve(s, (4,0,0), (1,3))
    return {'live:priced_value': a['fee_levels'][1]['net_gain_over_no_access']==[15,64],
        'positive:break_even_all_ties': {x['acquire'] for x in a['fee_levels'][-1]['optimal_choices']}=={True,False},
        'positive:tie_avoids_payment': not a['fee_levels'][-1]['selected']['acquire'],
        'placebo:no_information': all(not x['selected']['acquire'] for x in null['fee_levels'])}


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
        rebuilt['mixtures'] = [dict(mixture_counts=list(c), channels=[solve(rebuilt,c,p) for p in RELIABILITIES]) for c in MIXTURES]
        assert dict(rebuilt, id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        comparisons.extend(dict(selection_id=s['id'], mixture_counts=m['mixture_counts'],
            reliability=RELIABILITIES[j], fee=level['fee'],
            break_even_fee=d['break_even_fee'],
            **{k: float(F(*level[k])) if isinstance(level[k], list) else level[k]
               for k in METRICS if k != 'acquire'}, acquire=level['selected']['acquire'])
            for m in rebuilt['mixtures'] for j,d in enumerate(m['channels'])
            for level in d['fee_levels'])
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/cue_cost_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity, mixture_count=15, channel_count=3, fee_count=4)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            for j, reliability in enumerate(RELIABILITIES):
                for f, fee in enumerate(FEES):
                    def number(s, m):
                        level = answers[s]['mixtures'][i]['channels'][j]['fee_levels'][f]
                        if m == 'acquire': return int(level['selected']['acquire'])
                        x = level[m]
                        return float(F(*x)) if isinstance(x, list) else x
                    strata.append(dict(zip(axes, key), mixture=mixture, reliability=reliability,
                        fee=fee, rows=128,
                        **{m: fsum(n*number(s,m) for s,n in counts.items())/128 for m in METRICS}))
    def aggregate(records, names, expected):
        grouped = defaultdict(list)
        for r in records:
            key = tuple(tuple(r[k]) if k in ('mixture','reliability','fee') else r[k] for k in names)
            grouped[key].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names, key), rows=len(rr),
                **{m: fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata, tuple(k for k in axes if k != 'draw')+('mixture', 'reliability', 'fee'), 2)
    cells = aggregate(law, tuple(k for k in axes if k not in ('draw', 'lineage'))+('mixture', 'reliability', 'fee'), 8)
    by_fee = []
    for reliability in RELIABILITIES:
        for fee in FEES:
            rr = [r for r in comparisons if r['reliability']==reliability and r['fee']==list(fee)]
            by_fee.append(dict(reliability=reliability,fee=fee,problems=len(rr),
                acquired=sum(r['acquire'] for r in rr),
                positive_net_gain=sum(r['net_gain_over_no_access']>0 for r in rr),
                maximum_net_gain=max(r['net_gain_over_no_access'] for r in rr),
                break_even_ties=sum(F(*r['break_even_fee'])==F(*fee) for r in rr)))
    return dict(passed=True,numerical_acceptance=True,original_rows=position,
        joined_evaluations=position*180,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_fee_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections),
        paid_policy_scores=sum(len(d['all_paid_policies']) for s in selections for m in s['mixtures'] for d in m['channels']),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,by_fee=by_fee,checks=controls(),
        scope='finite-library priced source cue;hypothetical source-mass utility;no forecast or process correspondence')
