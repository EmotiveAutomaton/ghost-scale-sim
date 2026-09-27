"""Independent complete-policy review of conditional second-cue acquisition."""
from fractions import Fraction as F
from itertools import product
from .repeated_cue_review import solve as integrate, MIXTURES, RELIABILITIES, DEPENDENCES

FEES = ((0, 1), (1, 64), (1, 16), (1, 4))
METRICS = ('buy_probability', 'expected_fee', 'gross_retained_mass', 'net_utility',
           'never_net', 'always_net', 'gain_over_never', 'gain_over_always',
           'gain_over_best_constant', 'expected_used_bytes', 'maximum_realized_bytes',
           'never_expected_bytes', 'always_expected_bytes')


def solve(library, counts, reliability, dependence):
    base = integrate(library, counts, reliability, dependence)
    masses = {r['mask']:tuple(F(*x) for x in r['masses']) for r in library['candidates']}
    costs = {r['mask']:r['used_bytes'] for r in library['candidates']}
    policies = list(product(sorted(masses), repeat=3))
    rat = lambda v: [v.numerator, v.denominator]
    fees = []
    for fee_pair in FEES:
        fee = F(*fee_pair); conditional = []; total = F(0); charge = F(0)
        storage = F(0); acquisition = F(0); gross = F(0); realized = []
        for first in range(3):
            pairs = [x for x in base['pairs'] if x['labels'][0] == first]
            weights = [tuple(F(*v) for v in x['joint_prior']) for x in pairs]
            probabilities = [sum(w) for w in weights]; p = sum(probabilities)
            free = {m:sum(weights[k][j]*masses[m][j] for k in range(3) for j in range(3)) for m in masses}
            # Enumerate entire contingent policies, independently of separable maxima.
            paid = {policy:sum(weights[k][j]*masses[policy[k]][j] for k in range(3) for j in range(3)) for policy in policies}
            one, two = max(free.values()), max(paid.values())
            free_ties = sorted(m for m in free if free[m] == one)
            paid_ties = [x for x in policies if paid[x] == two]
            mask_ties = [sorted({x[k] for x in paid_ties}) for k in range(3)]
            assert set(product(*mask_ties)) == set(paid_ties)
            chosen_free, chosen_paid = max(free_ties), max(paid_ties)
            paid_net = two - p*fee; buy = paid_net > one; best = max(one, paid_net)
            ties = [name for name, value in (('decline', one), ('buy', paid_net)) if value == best]
            used = sum(probabilities[k]*costs[chosen_paid[k]] for k in range(3)) if buy else p*costs[chosen_free]
            selected_mass = paid[chosen_paid] if buy else free[chosen_free]
            total += best; charge += p*fee*buy; acquisition += p*buy
            storage += used; gross += selected_mass
            if p:
                realized.extend([costs[chosen_paid[k]] for k in range(3) if probabilities[k]] if buy else [costs[chosen_free]])
            conditional.append(dict(first_label=first, probability=rat(p),
                decline_optimal_masks=free_ties, decline_selected_mask=chosen_free,
                buy_optimal_masks_by_second=mask_ties, buy_selected_policy=list(chosen_paid),
                decline_joint_mass=rat(one), buy_joint_mass=rat(two), buy_joint_net=rat(paid_net),
                conditional_break_even_fee=rat((two-one)/p) if p else None,
                optimal_decisions=ties, selected_decision='buy' if buy else 'decline',
                gross_contribution=rat(selected_mass), net_contribution=rat(best),
                fee_contribution=rat(p*fee*buy), expected_byte_contribution=rat(used)))
        never = F(*base['single_mass']); always = F(*base['retained_mass'])-fee
        assert total == gross-charge and total >= max(never, always)
        assert charge == acquisition*fee
        if fee and (dependence == (1,1) or reliability in ((1,3), (1,1))): assert acquisition == 0
        fees.append(dict(fee=list(fee_pair), conditional=conditional,
            buy_probability=rat(acquisition), expected_fee=rat(charge), gross_retained_mass=rat(gross),
            net_utility=rat(total), never_net=rat(never), always_net=rat(always),
            gain_over_never=rat(total-never), gain_over_always=rat(total-always),
            gain_over_best_constant=rat(total-max(never,always)), expected_used_bytes=rat(storage),
            maximum_realized_bytes=max(realized), never_expected_bytes=base['single_cue']['expected_used_bytes'],
            always_expected_bytes=base['expected_used_bytes']))
    return dict(repeated=base, fees=fees)


def controls():
    s = dict(capacity_bytes=1, candidates=[dict(mask=1, used_bytes=1, masses=[[1,1],[0,1],[1,2]]),
                                         dict(mask=2, used_bytes=1, masses=[[0,1],[1,1],[1,2]])])
    a = solve(s, (2,2,0), (2,3), (0,1))['fees'][1]
    return {'live:selective_purchase':a['buy_probability']==[1,6],
            'positive:net_gain':a['gain_over_never']==[5,128],
            'placebo:copied_refusal':solve(s,(2,2,0),(2,3),(1,1))['fees'][1]['buy_probability']==[0,1],
            'placebo:uninformative_refusal':solve(s,(2,2,0),(1,3),(0,1))['fees'][1]['buy_probability']==[0,1]}

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
        rebuilt['mixtures'] = [dict(mixture_counts=list(c), channels=[dict(reliability=list(p),levels=[solve(rebuilt,c,p,d) for d in DEPENDENCES]) for p in RELIABILITIES]) for c in MIXTURES]
        assert dict(rebuilt, id=s['id']) == s
        assert key not in lookup and s['id'] not in answers
        lookup[key] = s['id']; answers[s['id']] = rebuilt
        comparisons.extend(dict(selection_id=s['id'],mixture_counts=m['mixture_counts'],
            reliability=RELIABILITIES[j],dependence=DEPENDENCES[k],fee=a['fee'],
            **{n:float(F(*a[n])) if isinstance(a[n],list) else a[n] for n in METRICS})
            for m in rebuilt['mixtures'] for j,d in enumerate(m['channels'])
            for k,level in enumerate(d['levels']) for a in level['fees'])
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/adaptive_second_cue_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity, mixture_count=15, channel_count=3, dependence_count=3,fee_count=4)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            for j, reliability in enumerate(RELIABILITIES):
                for f, dependence in enumerate(DEPENDENCES):
                    for h, fee in enumerate(FEES):
                        def number(s, m):
                            x = answers[s]['mixtures'][i]['channels'][j]['levels'][f]['fees'][h][m]
                            return float(F(*x)) if isinstance(x,list) else x
                        strata.append(dict(zip(axes,key),mixture=mixture,reliability=reliability,
                            dependence=dependence,fee=fee,rows=128,
                            **{m:fsum(n*number(s,m) for s,n in counts.items())/128 for m in METRICS}))
    def aggregate(records, names, expected):
        grouped = defaultdict(list)
        for r in records:
            key = tuple(tuple(r[k]) if k in ('mixture','reliability','dependence','fee') else r[k] for k in names)
            grouped[key].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names, key), rows=len(rr),
                **{m: fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata, tuple(k for k in axes if k != 'draw')+('mixture', 'reliability', 'dependence','fee'), 2)
    cells = aggregate(law, tuple(k for k in axes if k not in ('draw', 'lineage'))+('mixture', 'reliability', 'dependence','fee'), 8)
    by_dependence=[]
    for reliability in RELIABILITIES:
        for dependence in DEPENDENCES:
            for fee in FEES:
                rr=[r for r in comparisons if r['reliability']==reliability and r['dependence']==dependence and r['fee']==list(fee)]
                assert len(rr)==420
                by_dependence.append(dict(reliability=reliability,dependence=dependence,fee=fee,problems=len(rr),
                    gain_over_never=sum(r['gain_over_never']>0 for r in rr),
                    gain_over_best_constant=sum(r['gain_over_best_constant']>0 for r in rr),
                    selective_acquisition=sum(0<r['buy_probability']<1 for r in rr),
                    maximum_gain_over_best_constant=max(r['gain_over_best_constant'] for r in rr),
                    maximum_gain_over_never=max(r['gain_over_never'] for r in rr)))
    return dict(passed=True,numerical_acceptance=True,original_rows=position,
        joined_evaluations=position*540,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_acquisition_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,by_dependence=by_dependence,checks=controls(),
        scope='finite-library conditional second-cue acquisition with supplied dependence;no forecast or process correspondence')
