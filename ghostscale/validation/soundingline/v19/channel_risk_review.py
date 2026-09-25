"""Independent rational reconstruction of constrained channel lotteries."""
from fractions import Fraction as F
from itertools import product, combinations
from collections import Counter, defaultdict
from math import fsum
import gzip
import json
from .robust_mass_review import reconstruct, provenance

MIXTURES = tuple((a,b,4-a-b) for a in range(5) for b in range(5-a))


def hull_value(points):
    lower=[]
    for p in sorted(set(points)):
        if lower and p[0]==lower[-1][0]: continue
        while len(lower)>1:
            a,b=lower[-2:]
            if (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])>0: break
            lower.pop()
        lower.append(p)
    values=[max(p) for p in lower]
    for a,b in zip(lower,lower[1:]):
        da=a[0]-a[1]; db=b[0]-b[1]
        if da*db<0:
            w=db/(db-da); values.append(w*a[0]+(1-w)*b[0])
    return min(values)


def audit(library, counts, actual):
    assert tuple(counts) in MIXTURES
    rat=lambda x:[x.numerator,x.denominator]
    rows=sorted(library['candidates'],key=lambda r:r['mask'])
    mass={r['mask']:tuple(F(*x) for x in r['masses']) for r in rows}
    cost={r['mask']:r['used_bytes'] for r in rows}
    assert len(mass)==len(rows) and all(0<=x<=library['capacity_bytes'] for x in cost.values())
    policies=list(product(mass,repeat=3)); weights=[F(c,4) for c in counts]
    rates=[F(1,3),F(2,3),F(1)]
    # Separate per-label integration from the producer's scaled integer sum.
    contribution={(r,k,m):sum(weights[j]*(r if j==k else (1-r)/2)*mass[m][j] for j in range(3))
                  for r in rates for k in range(3) for m in mass}
    scores=[[sum(contribution[r,k,p[k]] for k in range(3)) for r in rates] for p in policies]
    optimum=[max(s[t] for s in scores) for t in range(3)]
    endpoints=[(optimum[0]-s[0],optimum[2]-s[2]) for s in scores]
    best_source=[max(m[j] for m in mass.values()) for j in range(3)]
    losses=[max(best_source[j]-mass[p[k]][j] for j in range(3) if counts[j] for k in range(3)) for p in policies]
    fixed=max((i for i,p in enumerate(policies) if p[0]==p[1]==p[2]),key=lambda i:(scores[i][1],policies[i]))
    nominal=max(range(len(policies)),key=lambda i:(scores[i][1],policies[i]))
    det=min(map(max,endpoints)); detties=[i for i,e in enumerate(endpoints) if max(e)==det]
    assert actual['mixture_counts']==list(counts) and actual['interval']==[[1,3],[1,1]]
    assert actual['policy_masks']==[list(p) for p in policies]
    assert [[F(n,actual['score_denominator']) for n in s] for s in actual['policy_score_numerators']]==scores
    assert [tuple(F(n,actual['score_denominator']) for n in e) for e in actual['policy_endpoint_regret_numerators']]==endpoints
    assert [F(n,actual['mass_denominator']) for n in actual['policy_realized_loss_numerators']]==losses
    assert actual['policy_used_bytes']==[[cost[m] for m in p] for p in policies]
    assert [F(n,actual['score_denominator']) for n in actual['unrestricted_optimal_score_numerators']]==optimum
    assert actual['fixed_policy']==fixed and actual['nominal_policy']==nominal
    assert actual['deterministic_optimal_policies']==detties and F(*actual['deterministic_worst_regret'])==det
    thresholds=[losses[fixed],(1+losses[fixed])/2,F(1)]
    assert len(actual['risk_levels'])==3
    seen={}; metrics=[]; candidate_count=0
    for level,threshold in zip(actual['risk_levels'],thresholds):
        survivors=tuple(i for i,v in enumerate(losses) if v<=threshold)
        if survivors not in seen: seen[survivors]=len(seen)
        assert level['lottery_set']==seen[survivors]
        saved=actual['lottery_sets'][seen[survivors]]
        assert saved['surviving_policies']==list(survivors)
        assert saved['excluded_policies']==[i for i in range(len(policies)) if i not in survivors]
        expected={(i,i,F(1)):endpoints[i] for i in survivors}
        for i,j in combinations(survivors,2):
            left=endpoints[i][0]-endpoints[i][1]; right=endpoints[j][0]-endpoints[j][1]
            if left==right: continue
            w=right/(right-left)
            if 0<w<1: expected[i,j,w]=tuple(w*a+(1-w)*b for a,b in zip(endpoints[i],endpoints[j]))
        ordered=sorted(expected); received=saved['candidates']
        assert len(received)==len(ordered)
        for row,key in zip(received,ordered):
            i,j,n,d,a,b,den=row
            assert (i,j,F(n,d))==key and (F(a,den),F(b,den))==expected[key]
        best=min(max(v) for v in expected.values())
        assert best==hull_value([endpoints[i] for i in survivors])
        ties=[i for i,k in enumerate(ordered) if max(expected[k])==best]
        assert saved['optimal_indices']==ties and saved['selected_index']==ties[-1]
        assert F(*saved['worst_expected_regret'])==best
        i,j,w=ordered[ties[-1]]; support=[(i,w)] if i==j else [(i,w),(j,1-w)]
        risk=max(losses[p] for p,q in support if q)
        diagnostics=[]
        for t,r in enumerate(rates):
            joint={(s,k):weights[s]*(r if s==k else (1-r)/2) for s in range(3) for k in range(3)}
            active=[(s,k) for (s,k),q in joint.items() if q]
            value=sum(q*scores[p][t] for p,q in support)
            charge=sum(q*sum(prob*cost[policies[p][k]] for (s,k),prob in joint.items()) for p,q in support)
            realized=max(best_source[s]-mass[policies[p][k]][s] for p,q in support if q for s,k in active)
            diagnostics.append(dict(reliability=rat(r),retained_mass=rat(value),regret=rat(optimum[t]-value),expected_used_bytes=rat(charge),
                maximum_realized_bytes=max(cost[policies[p][k]] for p,q in support if q for s,k in active),maximum_realized_coverage_loss=rat(realized)))
        assert level==dict(threshold=rat(threshold),lottery_set=seen[survivors],maximum_realized_coverage_loss=rat(risk),
            worst_expected_regret=rat(best),improvement_over_unrestricted_deterministic=rat(det-best),diagnostics=diagnostics)
        metric=dict(threshold=float(threshold),worst_expected_regret=float(best),deterministic_worst_regret=float(det),
            improvement_over_deterministic=float(det-best),maximum_realized_coverage_loss=float(risk),surviving_policies=len(survivors),
            extreme_lotteries=len(ordered),optimal_lotteries=len(ties))
        for t,d in enumerate(diagnostics):
            for k,v in d.items():
                if k!='reliability': metric[f'channel_{t}_{k}']=float(F(*v)) if isinstance(v,list) else v
        metrics.append(metric); candidate_count+=len(ordered)
    assert len(actual['lottery_sets'])==len(seen)
    return metrics,candidate_count


def controls():
    return {'live:convex_compromise':hull_value([(F(1),F(0)),(F(0),F(1))])==F(1,2),
            'placebo:identical_zero':hull_value([(F(0),F(0))]*3)==0,
            'positive:dominated_point':hull_value([(F(1),F(0)),(F(0),F(1)),(F(2),F(2))])==F(1,2)}


def verify(root):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    structures,population=provenance(root); groups=defaultdict(list)
    for r in read(root/'inputs/PARENT_SELECTIONS.json'):groups[(tuple(r['times']),tuple(r['costs']),r['capacity_bytes'])].append(r)
    selections=json.loads(gzip.decompress((root/'raw/channel_realized_risk_selections.json.gz').read_bytes()))
    assert len(selections)==len(groups)==28
    answers={};lookup={};comparisons=[];lotteries=0
    for s in selections:
        key=(tuple(s['times']),tuple(s['costs']),s['capacity_bytes']);library=reconstruct(groups.pop(key))
        assert {k:v for k,v in s.items() if k not in ('mixtures','id')}==library
        assert len(s['mixtures'])==15 and key not in lookup and s['id'] not in answers
        lookup[key]=s['id']; answers[s['id']]=[]
        for counts,a in zip(MIXTURES,s['mixtures']):
            metrics,count=audit(library,counts,a);lotteries+=count
            answers[s['id']].extend(metrics)
            comparisons.extend(dict(selection_id=s['id'],mixture=counts,risk_level=i,**r) for i,r in enumerate(metrics))
    assert not groups
    rows=json.loads(gzip.decompress((root/'raw/channel_realized_risk_points.json.gz').read_bytes()))
    assert len(rows)==57344
    axes=('lineage','evidence','length','checkpoint','draw','budget');grouped=defaultdict(Counter);pairs={};position=0
    for r in population:
        st=structures[r['structure']];times=r['times'];costs=[st['source_costs'][t-1] for t in times]
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':pairs[pair]=times
        else:assert pairs[pair]==times
        for budget in ('half','quarter'):
            cutoff=r['checkpoint']//2 if budget=='half' else 3*r['checkpoint']//4
            recent=[i for i,t in enumerate(times) if t>cutoff];count=len(recent);ordered=sorted(range(len(times)),key=times.__getitem__)
            spaced=[ordered[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[i] for i in recent),sum(costs[i] for i in spaced));sid=lookup[tuple(times),tuple(costs),capacity]
            expected=dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=sid,fixed_bytes=st['weighted_overhead'],
                total_budget_bytes=st['weighted_overhead']+capacity,mixture_count=15,risk_levels=3)
            assert rows[position]==expected;position+=1;grouped[tuple(expected[k] for k in axes)][sid]+=1
    metrics=tuple(next(iter(answers.values()))[0]);strata=[]
    for key,counts in sorted(grouped.items()):
        assert sum(counts.values())==128
        for i,mixture in enumerate(MIXTURES):
            for level in range(3):
                strata.append(dict(zip(axes,key),mixture=mixture,risk_level=level,rows=128,
                    **{m:fsum(n*answers[s][3*i+level][m] for s,n in counts.items())/128 for m in metrics}))
    def aggregate(records,names,expected):
        grouped=defaultdict(list)
        for r in records:grouped[tuple(tuple(r[k]) if k=='mixture' else r[k] for k in names)].append(r)
        result=[]
        for key,rr in sorted(grouped.items()):
            assert len(rr)==expected
            result.append(dict(zip(names,key),rows=len(rr),**{m:fsum(r[m] for r in rr)/len(rr) for m in metrics}))
        return result
    law=aggregate(strata,tuple(k for k in axes if k!='draw')+('mixture','risk_level'),2)
    cells=aggregate(law,tuple(k for k in axes if k not in ('draw','lineage'))+('mixture','risk_level'),8)
    counts=[]
    for level in range(3):
        rr=[r for r in comparisons if r['risk_level']==level]
        counts.append(dict(risk_level=level,problems=len(rr),better_than_deterministic=sum(r['improvement_over_deterministic']>0 for r in rr),
            worse_than_deterministic=sum(r['improvement_over_deterministic']<0 for r in rr),maximum_improvement=max(r['improvement_over_deterministic'] for r in rr)))
    return dict(passed=True,numerical_acceptance=True,original_rows=position,joined_evaluations=position*45,source_rosters=len(population),structures=len(structures),
        distinct_allocations=28,distinct_risk_problems=len(comparisons),candidate_count=sum(len(s['candidates']) for s in selections),
        extreme_lotteries_checked_including_repeated_thresholds=lotteries,paired_strata=strata,law_strata=law,equal_law_cells=cells,
        allocation_comparisons=comparisons,risk_level_counts=counts,checks=controls(),scope='finite supplied-law risk constraint;no forecast or process correspondence')
