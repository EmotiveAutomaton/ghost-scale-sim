"""Independent bilinear score reconstruction for directional source cues."""
from fractions import Fraction as F
from itertools import product

MIXTURES=tuple((a,b,4-a-b) for a in range(5) for b in range(5-a))
VERTICES=(((1,3),(0,1)),((1,3),(1,1)),((1,1),(0,1)),((1,1),(1,1)))
DIAGNOSTICS=tuple(product(((1,3),(2,3),(1,1)),((0,1),(1,2),(1,1))))
METRICS=('worst_regret','fixed_worst_regret','nominal_worst_regret',
         'symmetric_worst_regret','improvement_over_fixed','improvement_over_nominal',
         'improvement_over_symmetric')


def assess(library,counts,vertices=VERTICES):
    assert tuple(counts) in MIXTURES
    rows=sorted(library['candidates'],key=lambda x:x['mask'])
    mass={x['mask']:tuple(F(*v) for v in x['masses']) for x in rows}
    cost={x['mask']:x['used_bytes'] for x in rows}
    assert mass and len(mass)==len(rows)
    assert all(0<=v<=library['capacity_bytes'] for v in cost.values())
    policies=tuple(product(mass,repeat=3));weights=tuple(F(x,4) for x in counts)
    # Source-first expansion: c + r*a + b*d - r*b*d.
    coefficients={}
    for p in policies:
        correct=sum(weights[j]*mass[p[j]][j] for j in range(3))
        forward=sum(weights[j]*mass[p[(j+1)%3]][j] for j in range(3))
        reverse=sum(weights[j]*mass[p[(j+2)%3]][j] for j in range(3))
        coefficients[p]=(reverse,correct-reverse,forward-reverse)
    def score(p,r,b):
        c,a,d=coefficients[p]
        return c+r*a+b*d-r*b*d
    channels=set(vertices+DIAGNOSTICS+(((1,3),(1,2)),((1,1),(1,2))))
    tables={key:{p:score(p,F(*key[0]),F(*key[1])) for p in policies} for key in channels}
    opt={k:max(t.values()) for k,t in tables.items()}
    worst={p:max(opt[k]-tables[k][p] for k in vertices) for p in policies}
    best=min(worst.values());ties=[p for p in policies if worst[p]==best];chosen=max(ties)
    symkeys=(((1,3),(1,2)),((1,1),(1,2)))
    symworst={p:max(opt[k]-tables[k][p] for k in symkeys) for p in policies}
    sym=max(p for p in policies if symworst[p]==min(symworst.values()))
    nominal_key=((2,3),(1,2));nominal=max(policies,key=lambda p:(tables[nominal_key][p],p))
    fixed=max((p for p in policies if p[0]==p[1]==p[2]),key=lambda p:(tables[nominal_key][p],p))
    # Check all policies on a separate interior grid against the vertex bound.
    for r,b in product((F(1,3),F(1,2),F(2,3),F(5,6),F(1)),(F(0),F(1,4),F(1,2),F(3,4),F(1))):
        if vertices!=VERTICES:break
        values={p:score(p,r,b) for p in policies};v=max(values.values())
        assert all(v-values[p]<=worst[p] for p in policies)
    rat=lambda v:[v.numerator,v.denominator]
    diagnostics=[]
    for key in DIAGNOSTICS:
        r,b=map(lambda v:F(*v),key);prob=[F(0)]*3;support=[]
        for j in range(3):
            for label,q in ((j,r),((j+1)%3,(1-r)*b),((j+2)%3,(1-r)*(1-b))):
                joint=weights[j]*q;prob[label]+=joint
                if joint:support.append((j,label))
        assert sum(prob)==1
        arms={}
        for name,p in (('robust',chosen),('symmetric_robust',sym),('nominal',nominal),('fixed',fixed)):
            arms[name]=dict(policy=list(p),retained_mass=rat(tables[key][p]),regret=rat(opt[key]-tables[key][p]),
                gain_over_fixed=rat(tables[key][p]-tables[key][fixed]),
                expected_used_bytes=rat(sum(prob[k]*cost[p[k]] for k in range(3))),
                maximum_realized_bytes=max(cost[p[k]] for j,k in support),
                maximum_realized_coverage_loss=rat(max(max(v[j] for v in mass.values())-mass[p[k]][j] for j,k in support)))
        diagnostics.append(dict(reliability=list(key[0]),bias=list(key[1]),optimal_mass=rat(opt[key]),arms=arms))
    return dict(mixture_counts=list(counts),vertices=[[list(r),list(b)] for r,b in vertices],
        vertex_optimal_mass=[rat(opt[k]) for k in vertices],
        policies=[dict(policy=list(p),vertex_scores=[rat(tables[k][p]) for k in vertices],
            diagnostic_scores=[rat(tables[k][p]) for k in DIAGNOSTICS],worst_regret=rat(worst[p]),used_bytes=[cost[m] for m in p]) for p in policies],
        optimal_policies=[list(p) for p in ties],selected_policy=list(chosen),worst_regret=rat(best),
        symmetric_policy=list(sym),symmetric_worst_regret=rat(worst[sym]),
        nominal_policy=list(nominal),nominal_worst_regret=rat(worst[nominal]),
        fixed_policy=list(fixed),fixed_worst_regret=rat(worst[fixed]),
        improvement_over_symmetric=rat(worst[sym]-best),improvement_over_nominal=rat(worst[nominal]-best),
        improvement_over_fixed=rat(worst[fixed]-best),diagnostics=diagnostics)


def controls():
    s=dict(capacity_bytes=1,candidates=[dict(mask=1,used_bytes=1,masses=[[1,1],[0,1],[1,2]]),
                                      dict(mask=2,used_bytes=1,masses=[[0,1],[1,1],[1,2]])])
    a=assess(s,(3,1,0));point=assess(s,(3,1,0),(((2,3),(1,2)),))
    return {'live:positive_regret':F(*a['worst_regret'])>0,
            'positive:point_oracle':point['worst_regret']==[0,1],
            'placebo:single_source':assess(s,(4,0,0))['worst_regret']==[0,1],
            'positive:perfect_duplicate':all(p['vertex_scores'][2]==p['vertex_scores'][3] for p in a['policies'])}


def flatten(record):
    result={k:float(F(*record[k])) for k in METRICS}
    for d in record['diagnostics']:
        label='_'.join(map(str,d['reliability']+d['bias']))
        result[label+'_optimal_mass']=float(F(*d['optimal_mass']))
        for arm,values in d['arms'].items():
            for name,value in values.items():
                if name!='policy':result[label+'_'+arm+'_'+name]=float(F(*value)) if isinstance(value,list) else value
    return result

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
    selections = read(root/'SELECTIONS.json')
    assert len(selections)==len(groups)==28
    lookup = {}; answers = {}; comparisons = []; proofs = []
    for s in selections:
        key = (tuple(s['times']),tuple(s['costs']),s['capacity_bytes'])
        library = reconstruct(groups.pop(key))
        records = []
        for counts in MIXTURES:
            answer = assess(library, counts)
            proof = dict(policy_count=len(answer["policies"]), bilinear_source_expansion=True, all_policy_interior_grid_verified=True)
            records.append(answer)
            proofs.append(dict(selection_id=s['id'], mixture=counts, **proof))
            comparisons.append(dict(selection_id=s['id'], mixture=counts, **flatten(answer),
                selected_policy=answer['selected_policy'], fixed_policy=answer['fixed_policy'],
                nominal_policy=answer['nominal_policy'], symmetric_policy=answer['symmetric_policy'], optimal_policy_count=len(answer['optimal_policies'])))
        assert dict(library, mixtures=records, id=s['id'])==s
        assert key not in lookup and s['id'] not in answers
        lookup[key]=s['id']; answers[s['id']]=records
    assert not groups
    rows=json.loads(gzip.decompress((root/'raw/asymmetric_channel_points.json.gz').read_bytes()))
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
                total_budget_bytes=st['weighted_overhead']+capacity,mixture_count=15,vertex_count=4, diagnostic_count=9)
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
        strict_gain_over_symmetric=sum(r['improvement_over_symmetric']>0 for r in comparisons),
        strict_gain_over_fixed=sum(r['improvement_over_fixed']>0 for r in comparisons),
        strict_gain_over_nominal=sum(r['improvement_over_nominal']>0 for r in comparisons),
        worst_regret_positive=sum(r['worst_regret']>0 for r in comparisons),checks=controls(),
        scope='finite-library robust asymmetric channel regret;no forecast accuracy or process correspondence')
