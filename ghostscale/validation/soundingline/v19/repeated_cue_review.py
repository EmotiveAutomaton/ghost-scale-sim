"""Independent latent-event integration and regroup of repeated source cues."""
from fractions import Fraction as F
from itertools import product
from .noisy_prior_review import value, MIXTURES, RELIABILITIES

DEPENDENCES = ((0,1),(1,2),(1,1))
METRICS = ('retained_mass','single_mass','incremental_value','break_even_second_fee',
           'independent_assumption_mass','independent_gain_over_single','independent_regret',
           'expected_used_bytes','independent_expected_used_bytes',
           'maximum_realized_bytes','independent_maximum_realized_bytes')


def solve(library, counts, reliability, dependence):
    single=value(library, counts, reliability)
    r,d=F(*reliability),F(*dependence)
    assert 0 <= d <= 1
    masks=sorted(x['mask'] for x in library['candidates'])
    mass={x['mask']:[F(*v) for v in x['masses']] for x in library['candidates']}
    charges={x['mask']:x['used_bytes'] for x in library['candidates']}
    assert all(0<=b<=library['capacity_bytes'] for b in charges.values())
    labels=list(product(range(3),repeat=2))
    true={p:[F(0)]*3 for p in labels}; nominal={p:[F(0)]*3 for p in labels}
    # Integrate elementary source/first-label/copy-coin/second-label events.
    # The copy branch has one outcome; the fresh branch has three.
    for source in range(3):
        def likelihood(label): return r if label==source else (1-r)/2
        for a in range(3):
            first=F(counts[source],4)*likelihood(a)
            true[a,a][source]+=first*d
            for b in range(3):
                p=first*likelihood(b)
                true[a,b][source]+=p*(1-d)
                nominal[a,b][source]+=p
    assert sum(sum(x) for x in true.values())==1
    for a in range(3):
        assert sum(sum(true[a,b]) for b in range(3))==F(*single['cues'][a]['probability'])
    rat=lambda x:[x.numerator,x.denominator]
    pairs=[]
    for key in labels:
        joint,independent=true[key],nominal[key];p,q=sum(joint),sum(independent)
        scores={m:sum(w*v for w,v in zip(joint,mass[m])) for m in masks}
        ns={m:sum(w*v for w,v in zip(independent,mass[m])) for m in masks}
        ties=[m for m in masks if scores[m]==max(scores.values())]
        nties=[m for m in masks if ns[m]==max(ns.values())]
        chosen,naive=max(ties),max(nties)
        pairs.append(dict(labels=list(key),joint_prior=list(map(rat,joint)),probability=rat(p),
            conditional_prior=list(map(rat,(x/p for x in joint))) if p else None,
            joint_mask_scores=[dict(mask=m,mass=rat(scores[m])) for m in masks],
            optimal_masks=ties,selected_mask=chosen,used_bytes=charges[chosen],contribution=rat(scores[chosen]),
            independent_joint_prior=list(map(rat,independent)),independent_probability=rat(q),
            independent_conditional_prior=list(map(rat,(x/q for x in independent))) if q else None,
            independent_mask_scores=[dict(mask=m,mass=rat(ns[m])) for m in masks],
            independent_optimal_masks=nties,independent_selected_mask=naive,
            independent_used_bytes=charges[naive],independent_actual_contribution=rat(scores[naive])))
    gross=sum(F(*x['contribution']) for x in pairs)
    naive=sum(F(*x['independent_actual_contribution']) for x in pairs)
    one=F(*single['informed_mass']);full=F(*single['full_mass'])
    assert one<=gross<=full and naive<=gross
    active=[x for x in pairs if F(*x['probability'])]
    return dict(single_cue=single,dependence=list(dependence),pairs=pairs,
        retained_mass=rat(gross),single_mass=rat(one),incremental_value=rat(gross-one),
        break_even_second_fee=rat(gross-one),independent_assumption_mass=rat(naive),
        independent_gain_over_single=rat(naive-one),independent_regret=rat(gross-naive),
        expected_used_bytes=rat(sum(F(*x['probability'])*x['used_bytes'] for x in pairs)),
        independent_expected_used_bytes=rat(sum(F(*x['probability'])*x['independent_used_bytes'] for x in pairs)),
        maximum_realized_bytes=max(x['used_bytes'] for x in active),
        independent_maximum_realized_bytes=max(x['independent_used_bytes'] for x in active))


def controls():
    s=dict(capacity_bytes=1,candidates=[dict(mask=1,used_bytes=1,masses=[[1,1],[0,1],[1,2]]),
                                      dict(mask=2,used_bytes=1,masses=[[0,1],[1,1],[1,2]])])
    fresh=solve(s,(2,2,0),(2,3),(0,1));copy=solve(s,(2,2,0),(2,3),(1,1))
    return {'live:independent_gain':fresh['incremental_value']==[1,24],
            'positive:copy_identity':copy['retained_mass']==copy['single_mass'],
            'placebo:uninformative':solve(s,(2,2,0),(1,3),(0,1))['incremental_value']==[0,1],
            'positive:perfect':solve(s,(2,2,0),(1,1),(0,1))['incremental_value']==[0,1]}

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
            reliability=RELIABILITIES[j],dependence=level['dependence'],
            **{k:float(F(*level[k])) if isinstance(level[k],list) else level[k] for k in METRICS})
            for m in rebuilt['mixtures'] for j,d in enumerate(m['channels']) for level in d['levels'])
    assert not groups
    rows = json.loads(gzip.decompress((root/'raw/repeated_cue_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity, mixture_count=15, channel_count=3, dependence_count=3)
            assert rows[position] == expected, position
            position += 1
            grouped[tuple(expected[k] for k in axes)][sid] += 1
    strata = []
    for key, counts in sorted(grouped.items()):
        assert sum(counts.values()) == 128
        for i, mixture in enumerate(MIXTURES):
            for j, reliability in enumerate(RELIABILITIES):
                for f, dependence in enumerate(DEPENDENCES):
                    def number(s, m):
                        level = answers[s]['mixtures'][i]['channels'][j]['levels'][f]
                        x = level[m]
                        return float(F(*x)) if isinstance(x, list) else x
                    strata.append(dict(zip(axes, key), mixture=mixture, reliability=reliability,
                        dependence=dependence, rows=128,
                        **{m: fsum(n*number(s,m) for s,n in counts.items())/128 for m in METRICS}))
    def aggregate(records, names, expected):
        grouped = defaultdict(list)
        for r in records:
            key = tuple(tuple(r[k]) if k in ('mixture','reliability','dependence') else r[k] for k in names)
            grouped[key].append(r)
        result = []
        for key, rr in sorted(grouped.items()):
            assert len(rr) == expected
            result.append(dict(zip(names, key), rows=len(rr),
                **{m: fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law = aggregate(strata, tuple(k for k in axes if k != 'draw')+('mixture', 'reliability', 'dependence'), 2)
    cells = aggregate(law, tuple(k for k in axes if k not in ('draw', 'lineage'))+('mixture', 'reliability', 'dependence'), 8)
    by_dependence=[]
    for reliability in RELIABILITIES:
        for dependence in DEPENDENCES:
            rr=[r for r in comparisons if r['reliability']==reliability and r['dependence']==list(dependence)]
            by_dependence.append(dict(reliability=reliability,dependence=dependence,problems=len(rr),
                positive_incremental_value=sum(r['incremental_value']>0 for r in rr),
                maximum_incremental_value=max(r['incremental_value'] for r in rr),
                independence_below_single=sum(r['independent_gain_over_single']<0 for r in rr),
                positive_independence_regret=sum(r['independent_regret']>0 for r in rr),
                maximum_independence_regret=max(r['independent_regret'] for r in rr)))
    return dict(passed=True,numerical_acceptance=True,original_rows=position,
        joined_evaluations=position*135,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_dependence_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,by_dependence=by_dependence,checks=controls(),
        scope='finite-library repeated source cues with supplied dependence;no forecast or process correspondence')
