"""Independent rational audit of frozen policies under changed prior mixtures."""
from fractions import Fraction as F
from itertools import product
from collections import Counter, defaultdict
from math import fsum
import gzip
import json
from .noisy_prior_review import value, MIXTURES, RELIABILITIES
from .robust_mass_review import reconstruct, provenance

METRICS = ('retained_mass', 'optimal_informed_mass', 'optimal_fixed_mass',
           'regret', 'gain_over_fixed', 'nominal_fixed_actual_mass',
           'gain_over_nominal_fixed', 'unexpected_label_probability',
           'expected_used_bytes', 'maximum_realized_bytes')


def assess(library, nominal, actual, reliability):
    """Choose using nominal weights only; enumerate all actual policies separately."""
    rows = sorted(library['candidates'], key=lambda r: r['mask'])
    mass = {r['mask']: tuple(F(*x) for x in r['masses']) for r in rows}
    charge = {r['mask']: r['used_bytes'] for r in rows}
    assert all(0 <= b <= library['capacity_bytes'] for b in charge.values())
    correct = F(*reliability)
    matrix = tuple(tuple(correct if p == label else (1-correct)/2
                         for label in range(3)) for p in range(3))
    policies = tuple(product(mass, repeat=3))
    def score(policy, counts):
        return sum(F(counts[p], 4)*matrix[p][label]*mass[policy[label]][p]
                   for p in range(3) for label in range(3))
    nominal_scores = {policy: score(policy, nominal) for policy in policies}
    nominal_best = max(nominal_scores.values())
    # Lexicographically largest optimal complete policy is the specified per-label tie rule.
    chosen = max(p for p, s in nominal_scores.items() if s == nominal_best)
    fixed_scores = {m: score((m, m, m), nominal) for m in mass}
    fixed = max(m for m, s in fixed_scores.items() if s == max(fixed_scores.values()))
    actual_scores = {policy: score(policy, actual) for policy in policies}
    optimum = max(actual_scores.values())
    actual_fixed = max(actual_scores[(m, m, m)] for m in mass)
    retained = actual_scores[chosen]
    fixed_actual = actual_scores[(fixed, fixed, fixed)]
    rat = lambda x: [x.numerator, x.denominator]
    cues = []; used = F(0); unexpected = F(0)
    for label, mask in enumerate(chosen):
        nominal_probability = sum(F(nominal[p],4)*matrix[p][label] for p in range(3))
        probability = sum(F(actual[p],4)*matrix[p][label] for p in range(3))
        contribution = sum(F(actual[p],4)*matrix[p][label]*mass[mask][p] for p in range(3))
        used += probability*charge[mask]
        if not nominal_probability: unexpected += probability
        cues.append(dict(label=label, nominal_probability=rat(nominal_probability),
            actual_probability=rat(probability), nominal_zero=not nominal_probability,
            selected_mask=mask, used_bytes=charge[mask], actual_contribution=rat(contribution)))
    assert retained == sum(F(*c['actual_contribution']) for c in cues)
    return dict(nominal_counts=list(nominal), actual_counts=list(actual), cues=cues,
        retained_mass=rat(retained), optimal_informed_mass=rat(optimum),
        optimal_fixed_mass=rat(actual_fixed), regret=rat(optimum-retained),
        gain_over_fixed=rat(retained-actual_fixed), nominal_fixed_mask=fixed,
        nominal_fixed_actual_mass=rat(fixed_actual), gain_over_nominal_fixed=rat(retained-fixed_actual),
        unexpected_label_probability=rat(unexpected), expected_used_bytes=rat(used),
        maximum_realized_bytes=max(c['used_bytes'] for c in cues if F(*c['actual_probability'])))


def controls():
    library = dict(capacity_bytes=1, candidates=[
        dict(mask=1, used_bytes=1, masses=[[1,1],[0,1],[1,2]]),
        dict(mask=2, used_bytes=1, masses=[[0,1],[1,1],[1,2]])])
    same = assess(library, (2,2,0), (2,2,0), (2,3))
    wrong = assess(library, (4,0,0), (0,4,0), (1,3))
    unexpected = assess(library, (4,0,0), (0,4,0), (1,1))
    return {'positive:matched_identity': same['regret']==[0,1],
        'live:wrong_mixture_harms': wrong['gain_over_fixed']==[-1,1],
        'placebo:no_information': wrong['gain_over_nominal_fixed']==[0,1],
        'positive:unexpected_label': unexpected['unexpected_label_probability']==[1,1]}


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    groups = defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    selections = read(root/'SELECTIONS.json')
    assert len(selections)==len(groups)==28
    lookup = {}; answers = {}; comparisons = []
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        rebuilt = reconstruct(groups.pop(key))
        policies = [[value(rebuilt,c,p) for c in MIXTURES] for p in RELIABILITIES]
        records = [dict(reliability=list(p), nominal_index=i, actual_index=j,
                        **assess(rebuilt,n,a,p))
                   for p in RELIABILITIES for i,n in enumerate(MIXTURES)
                   for j,a in enumerate(MIXTURES)]
        assert dict(rebuilt, policies=policies, comparisons=records, id=s['id'])==s
        assert key not in lookup and s['id'] not in answers
        lookup[key]=s['id']; answers[s['id']]=records
        comparisons.extend(dict(selection_id=s['id'], reliability=r['reliability'],
            nominal_counts=r['nominal_counts'], actual_counts=r['actual_counts'],
            **{m: float(F(*r[m])) if isinstance(r[m],list) else r[m] for m in METRICS}) for r in records)
    assert not groups
    rows=json.loads(gzip.decompress((root/'raw/prior_mixture_points.json.gz').read_bytes()))
    assert len(rows)==57344
    axes=('lineage','evidence','length','checkpoint','draw','budget')
    grouped=defaultdict(Counter); pairs={}; position=0
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
            sid=lookup[(tuple(times),tuple(costs),capacity)]
            expected=dict(**{k:v for k,v in r.items() if k!='times'}, budget=budget,
                selection_id=sid,fixed_bytes=st['weighted_overhead'],
                total_budget_bytes=st['weighted_overhead']+capacity,
                nominal_mixture_count=15,actual_mixture_count=15,channel_count=3)
            assert rows[position]==expected,position
            position+=1; grouped[tuple(expected[k] for k in axes)][sid]+=1
    conditions=[(p,n,a) for p in RELIABILITIES for n in MIXTURES for a in MIXTURES]
    numbers={sid:[{m:float(F(*r[m])) if isinstance(r[m],list) else r[m]
                   for m in METRICS} for r in rr] for sid,rr in answers.items()}
    strata=[]
    for key,counts in sorted(grouped.items()):
        assert sum(counts.values())==128
        for i,(p,n,a) in enumerate(conditions):
            strata.append(dict(zip(axes,key),reliability=p,nominal=n,actual=a,rows=128,
                **{m:fsum(k*numbers[s][i][m] for s,k in counts.items())/128 for m in METRICS}))
    def aggregate(records,names,expected):
        groups=defaultdict(list)
        for r in records: groups[tuple(tuple(r[k]) if k in ('reliability','nominal','actual') else r[k] for k in names)].append(r)
        result=[]
        for key,rr in sorted(groups.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in METRICS}))
        return result
    law=aggregate(strata,tuple(k for k in axes if k!='draw')+('reliability','nominal','actual'),2)
    cells=aggregate(law,tuple(k for k in axes if k not in ('draw','lineage'))+('reliability','nominal','actual'),8)
    by_channel=[]
    for p in RELIABILITIES:
        rr=[r for r in comparisons if r['reliability']==list(p)]
        by_channel.append(dict(reliability=p,problems=len(rr),
            mismatched=sum(r['nominal_counts']!=r['actual_counts'] for r in rr),
            positive_regret=sum(r['regret']>0 for r in rr),
            below_actual_fixed=sum(r['gain_over_fixed']<0 for r in rr),
            below_nominal_fixed=sum(r['gain_over_nominal_fixed']<0 for r in rr),
            unexpected_label=sum(r['unexpected_label_probability']>0 for r in rr),
            maximum_regret=max(r['regret'] for r in rr)))
    return dict(passed=True,numerical_acceptance=True,original_rows=position,
        joined_evaluations=position*675,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_mismatch_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,by_channel=by_channel,checks=controls(),
        scope='finite-library prior-mixture sensitivity;no forecast accuracy or process correspondence')
