"""Independent rational policy audit, including every score-line intersection."""
from fractions import Fraction as F
from itertools import product, combinations
from collections import Counter, defaultdict
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance

MIXTURES = tuple((a, b, 4-a-b) for a in range(5) for b in range(5-a))
METRICS = ('worst_regret', 'fixed_worst_regret', 'nominal_worst_regret',
           'improvement_over_fixed', 'improvement_over_nominal')


def assess(library, counts, interval=((1, 3), (1, 1))):
    """Integrate by observed label; maximize the regret over all breakpoints."""
    assert tuple(counts) in MIXTURES
    lo, hi = (F(*v) for v in interval)
    assert 0 <= lo <= hi <= 1
    rows = sorted(library['candidates'], key=lambda r: r['mask'])
    masses = {r['mask']: tuple(F(*v) for v in r['masses']) for r in rows}
    costs = {r['mask']: r['used_bytes'] for r in rows}
    assert len(masses) == len(rows) and masses
    assert all(0 <= c <= library['capacity_bytes'] for c in costs.values())
    weights = tuple(F(c, 4) for c in counts)
    policies = tuple(product(masses, repeat=3))
    # A label's mass contribution is an affine function of channel correctness.
    intercept = {(label, mask): sum(weights[j]*masses[mask][j]/2
        for j in range(3) if j != label) for label in range(3) for mask in masses}
    slope = {(label, mask): weights[label]*masses[mask][label]-intercept[label, mask]
        for label in range(3) for mask in masses}
    lines = {p: (sum(intercept[k, p[k]] for k in range(3)),
                 sum(slope[k, p[k]] for k in range(3))) for p in policies}
    points = {lo, hi}
    for p, q in combinations(policies, 2):
        a, b = lines[p]; c, d = lines[q]
        if b != d:
            x = (c-a)/(b-d)
            if lo <= x <= hi:
                points.add(x)
    if lo <= F(2, 3) <= hi:
        points.add(F(2, 3))
    scores = {x: {p: a+b*x for p, (a, b) in lines.items()} for x in points}
    optima = {x: max(values.values()) for x, values in scores.items()}
    worst = {p: max(optima[x]-scores[x][p] for x in points) for p in policies}
    assert all(worst[p] == max(optima[x]-scores[x][p] for x in (lo, hi)) for p in policies)
    best = min(worst.values())
    ties = [p for p in policies if worst[p] == best]
    chosen = max(ties)
    at = lambda p, x: lines[p][0]+x*lines[p][1]
    nominal = max(policies, key=lambda p: (at(p, F(2, 3)), p))
    fixed = max((p for p in policies if p[0] == p[1] == p[2]),
                key=lambda p: (at(p, F(2, 3)), p))
    rat = lambda x: [x.numerator, x.denominator]
    diagnostics = []
    for x in sorted({lo, hi, F(2, 3)}):
        optimum = max(at(p, x) for p in policies)
        label_probs = tuple(sum(weights[j]*(x if j == k else (1-x)/2)
                                for j in range(3)) for k in range(3))
        support = [(j, k) for k in range(3) for j in range(3)
                   if weights[j]*(x if j == k else (1-x)/2)]
        arms = {}
        for name, p in (('robust', chosen), ('fixed', fixed), ('nominal', nominal)):
            arms[name] = dict(policy=list(p), retained_mass=rat(at(p, x)),
                regret=rat(optimum-at(p, x)), gain_over_fixed=rat(at(p, x)-at(fixed, x)),
                expected_used_bytes=rat(sum(label_probs[k]*costs[p[k]] for k in range(3))),
                maximum_realized_bytes=max(costs[p[k]] for j, k in support),
                maximum_realized_coverage_loss=rat(max(
                    max(masses[m][j] for m in masses)-masses[p[k]][j] for j, k in support)))
        diagnostics.append(dict(reliability=rat(x), optimal_mass=rat(optimum), arms=arms))
    answer = dict(mixture_counts=list(counts), interval=[rat(lo), rat(hi)],
        policies=[dict(policy=list(p), intercept=rat(lines[p][0]), slope=rat(lines[p][1]),
            worst_regret=rat(worst[p]), used_bytes=[costs[m] for m in p]) for p in policies],
        optimal_policies=[list(p) for p in ties], selected_policy=list(chosen), worst_regret=rat(best),
        fixed_policy=list(fixed), fixed_worst_regret=rat(worst[fixed]),
        nominal_policy=list(nominal), nominal_worst_regret=rat(worst[nominal]),
        improvement_over_fixed=rat(worst[fixed]-best),
        improvement_over_nominal=rat(worst[nominal]-best), diagnostics=diagnostics)
    return answer, dict(policy_count=len(policies), intersection_and_diagnostic_points=len(points),
                        endpoint_reduction_verified=True)


def controls():
    library = dict(capacity_bytes=1, candidates=[
        dict(mask=1, used_bytes=1, masses=[[1,1],[0,1],[1,2]]),
        dict(mask=2, used_bytes=1, masses=[[0,1],[1,1],[1,2]])])
    a, proof = assess(library, (3,1,0))
    point, _ = assess(library, (3,1,0), ((2,3),(2,3)))
    null, _ = assess(library, (4,0,0))
    return {'live:known_compromise': a['worst_regret']==[1,6],
        'positive:strict_fixed_gain': a['improvement_over_fixed']==[1,12],
        'positive:point_interval': point['worst_regret']==[0,1],
        'placebo:single_prior': null['worst_regret']==[0,1],
        'positive:endpoint_reduction': proof['endpoint_reduction_verified'],
        'positive:realized_loss_distinct': a['diagnostics'][0]['arms']['robust']['maximum_realized_coverage_loss']==[1,1]}


def flatten(record):
    result = {k: float(F(*record[k])) for k in METRICS}
    for d in record['diagnostics']:
        label = '_'.join(map(str, d['reliability']))
        result[label+'_optimal_mass'] = float(F(*d['optimal_mass']))
        for arm, values in d['arms'].items():
            for name, value in values.items():
                if name != 'policy':
                    result[label+'_'+arm+'_'+name] = float(F(*value)) if isinstance(value, list) else value
    return result


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    structures, population = provenance(root)
    groups = defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups[(tuple(r['times']), tuple(r['costs']), r['capacity_bytes'])].append(r)
    selections = read(root/'SELECTIONS.json')
    assert len(selections)==len(groups)==28
    lookup = {}; answers = {}; comparisons = []; proofs = []
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        library = reconstruct(groups.pop(key))
        records = []
        for counts in MIXTURES:
            answer, proof = assess(library, counts)
            records.append(answer)
            proofs.append(dict(selection_id=s['id'], mixture=counts, **proof))
            comparisons.append(dict(selection_id=s['id'], mixture=counts, **flatten(answer),
                selected_policy=answer['selected_policy'], fixed_policy=answer['fixed_policy'],
                nominal_policy=answer['nominal_policy'], optimal_policy_count=len(answer['optimal_policies'])))
        assert dict(library, mixtures=records, id=s['id'])==s
        assert key not in lookup and s['id'] not in answers
        lookup[key]=s['id']; answers[s['id']]=records
    assert not groups
    rows=json.loads(gzip.decompress((root/'raw/robust_channel_regret_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity,mixture_count=15,interval_count=1)
            assert rows[position]==expected,position
            position+=1; grouped[tuple(expected[k] for k in axes)][sid]+=1
    numbers={sid:[flatten(r) for r in rr] for sid,rr in answers.items()}
    metrics=tuple(next(iter(numbers.values()))[0])
    strata=[]
    for key,counts in sorted(grouped.items()):
        assert sum(counts.values())==128
        for i,p in enumerate(MIXTURES):
            strata.append(dict(zip(axes,key),mixture=p,rows=128,
                **{m:fsum(k*numbers[s][i][m] for s,k in counts.items())/128 for m in metrics}))
    def aggregate(records,names,expected):
        groups=defaultdict(list)
        for r in records: groups[tuple(tuple(r[k]) if k=='mixture' else r[k] for k in names)].append(r)
        result=[]
        for key,rr in sorted(groups.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    law=aggregate(strata,tuple(k for k in axes if k!='draw')+('mixture',),2)
    cells=aggregate(law,tuple(k for k in axes if k not in ('draw','lineage'))+('mixture',),8)
    return dict(passed=True,numerical_acceptance=True,original_rows=position,
        joined_evaluations=position*15,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_robust_problems=len(comparisons),
        candidate_count=sum(len(s['candidates']) for s in selections),
        paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,intersection_proofs=proofs,
        strict_gain_over_fixed=sum(r['improvement_over_fixed']>0 for r in comparisons),
        strict_gain_over_nominal=sum(r['improvement_over_nominal']>0 for r in comparisons),
        worst_regret_positive=sum(r['worst_regret']>0 for r in comparisons),checks=controls(),
        scope='finite-library robust channel regret;no forecast accuracy or process correspondence')
