"""Complete deterministic policies under asymmetric source-cue mistakes."""
from fractions import Fraction as F
from itertools import product
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture
from .robust_channel_regret import solve as symmetric_reference

VERTICES=(((1,3),(0,1)),((1,3),(1,1)),((1,1),(0,1)),((1,1),(1,1)))
DIAGNOSTICS=tuple(product(RELIABILITIES,((0,1),(1,2),(1,1))))


def channel(reliability,bias):
    r,b=F(*reliability),F(*bias)
    if not (0<=r<=1 and 0<=b<=1):raise ValueError('channel parameter')
    return [[r if label==source else (1-r)*(b if label==(source+1)%3 else 1-b)
             for label in range(3)] for source in range(3)]


def solve(selection,counts,vertices=VERTICES):
    if tuple(counts) not in MIXTURES:raise ValueError('mixture')
    rows=sorted(selection['candidates'],key=lambda x:x['mask'])
    mass={x['mask']:tuple(F(*v) for v in x['masses']) for x in rows}
    charge={x['mask']:x['used_bytes'] for x in rows}
    if not mass or any(b<0 or b>selection['capacity_bytes'] for b in charge.values()):raise ValueError('capacity')
    policies=list(product(mass,repeat=3));weights=[F(c,4) for c in counts]
    channels=sorted(set(vertices+DIAGNOSTICS));scores={};joint={}
    for key in channels:
        matrix=channel(*key)
        joint[key]=[[weights[j]*matrix[j][label] for j in range(3)] for label in range(3)]
        conditional={label:{m:sum(joint[key][label][j]*mass[m][j] for j in range(3)) for m in mass} for label in range(3)}
        scores[key]={p:sum(conditional[label][p[label]] for label in range(3)) for p in policies}
    opt={key:max(table.values()) for key,table in scores.items()}
    worst={p:max(opt[key]-scores[key][p] for key in vertices) for p in policies}
    best=min(worst.values());ties=[p for p in policies if worst[p]==best];chosen=max(ties)
    sym=tuple(symmetric_reference(selection,counts)['selected_policy'])
    nominal_key=((2,3),(1,2))
    nominal=max(policies,key=lambda p:(scores[nominal_key][p],p))
    fixed=max((p for p in policies if len(set(p))==1),key=lambda p:(scores[nominal_key][p],p))
    rat=lambda v:[v.numerator,v.denominator]
    diagnostics=[]
    for key in DIAGNOSTICS:
        probability=[sum(x) for x in joint[key]]
        support=[(j,label) for label in range(3) for j in range(3) if joint[key][label][j]]
        arms={}
        for name,p in [('robust',chosen),('symmetric_robust',sym),('nominal',nominal),('fixed',fixed)]:
            arms[name]=dict(policy=list(p),retained_mass=rat(scores[key][p]),regret=rat(opt[key]-scores[key][p]),
                gain_over_fixed=rat(scores[key][p]-scores[key][fixed]),
                expected_used_bytes=rat(sum(probability[label]*charge[p[label]] for label in range(3))),
                maximum_realized_bytes=max(charge[p[label]] for j,label in support),
                maximum_realized_coverage_loss=rat(max(max(v[j] for v in mass.values())-mass[p[label]][j] for j,label in support)))
        diagnostics.append(dict(reliability=list(key[0]),bias=list(key[1]),optimal_mass=rat(opt[key]),arms=arms))
    return dict(mixture_counts=list(counts),vertices=[[list(r),list(b)] for r,b in vertices],
        vertex_optimal_mass=[rat(opt[key]) for key in vertices],
        policies=[dict(policy=list(p),vertex_scores=[rat(scores[key][p]) for key in vertices],
            diagnostic_scores=[rat(scores[key][p]) for key in DIAGNOSTICS],
            worst_regret=rat(worst[p]),used_bytes=[charge[m] for m in p]) for p in policies],
        optimal_policies=[list(p) for p in ties],selected_policy=list(chosen),worst_regret=rat(best),
        symmetric_policy=list(sym),symmetric_worst_regret=rat(worst[sym]),
        nominal_policy=list(nominal),nominal_worst_regret=rat(worst[nominal]),
        fixed_policy=list(fixed),fixed_worst_regret=rat(worst[fixed]),
        improvement_over_symmetric=rat(worst[sym]-best),improvement_over_nominal=rat(worst[nominal]-best),
        improvement_over_fixed=rat(worst[fixed]-best),diagnostics=diagnostics)


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    a=solve(s,(3,1,0));point=solve(s,(3,1,0),(((2,3),(1,2)),))
    null=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    return {'live:directional_channel':channel((1,3),(1,1))[0]==[F(1,3),F(2,3),F(0)],
        'positive:point_oracle':point['worst_regret']==[0,1],
        'placebo:identical_priors':solve(null,(1,1,2))['worst_regret']==[0,1],
        'positive:duplicated_perfect_vertices':all(p['vertex_scores'][2]==p['vertex_scores'][3] for p in a['policies'])}


def compare(selection):return dict(mixtures=[solve(selection,c) for c in MIXTURES])

import gzip
from ..v18_3.io import read, write, canonical, file_digest


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]
        or cfg['vertices'] != [[list(r),list(b)] for r,b in VERTICES]): raise ValueError('design')
    checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h: raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json'); structures=read(root/'inputs/STRUCTURES.json')
    groups={}
    for s in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[]; lookup={}
    for key,records in groups.items():
        pulse(phase='asymmetric-channel-libraries',library=len(selections))
        s=from_parent(records); s.update(compare(s)); s['id']=len(selections)
        selections.append(s); lookup[key]=s
    rows=[]; paired={}
    for i,r in enumerate(population):
        if i%256==0: pulse(phase='asymmetric-channel-rosters',row=i)
        st=structures[r['structure']]; times=tuple(r['times'])
        costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times): raise ValueError('roster')
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware': paired[pair]=times
        elif paired[pair]!=times: raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold]; count=len(recent)
            order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced))
            s=lookup[(times,costs,capacity)]
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                mixture_count=15,vertex_count=4, diagnostic_count=9))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/asymmetric_channel_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,correct mixture,unknown reliability and directional error bias rectangle,structural costs and rosters',
        scope='finite-library asymmetric channel regret;no fee,forecast or learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures with all feasible three-label contingent policies'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),joined_evaluations=len(rows)*15,
        unique_selections=len(selections),distinct_robust_problems=len(selections)*15,
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite-library asymmetric channel regret;not forecast accuracy or process correspondence')
