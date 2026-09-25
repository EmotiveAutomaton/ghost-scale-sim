"""Exact contingent storage lotteries with bounds on every realized coverage loss."""
from fractions import Fraction as F
from itertools import product, combinations
from math import lcm, gcd
import gzip
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture
from ..v18_3.io import read, write, canonical, file_digest


def rational(x): return [x.numerator, x.denominator]


def lotteries(endpoint, survivors, denominator):
    """Retain all extreme candidates with exact integer cross multiplication.

    Rows: policy i,j; weight numerator/denominator; two regret numerators
    and their shared denominator. Zero-weight endpoints use singletons only.
    """
    candidates=[]
    for i in survivors:
        a,b=endpoint[i];candidates.append([i,i,1,1,a,b,denominator])
    for i,j in combinations(survivors,2):
        a,b=endpoint[i];c,d=endpoint[j];n=d-c;q=(a-b)-(c-d)
        if not q:continue
        if q<0:n,q=-n,-q
        if not 0<n<q:continue
        divisor=gcd(n,q);n//=divisor;q//=divisor
        candidates.append([i,j,n,q,n*a+(q-n)*c,n*b+(q-n)*d,q*denominator])
    candidates.sort(key=lambda x:(x[0],x[1],F(x[2],x[3])))
    best_n,best_d=max(candidates[0][4:6]),candidates[0][6];ties=[]
    for k,row in enumerate(candidates):
        n,d=max(row[4:6]),row[6];delta=n*best_d-best_n*d
        if delta<0:best_n,best_d=n,d;ties=[k]
        elif delta==0:ties.append(k)
    return dict(candidates=candidates,optimal_indices=ties,selected_index=ties[-1],worst_expected_regret=rational(F(best_n,best_d)))


def solve(selection, counts):
    if tuple(counts) not in MIXTURES or any(type(x) is not int for x in counts):raise ValueError('mixture')
    rows=sorted(selection['candidates'],key=lambda r:r['mask'])
    masses=[[F(*x) for x in r['masses']] for r in rows]
    if not rows or any(len(m)!=3 or any(x<0 or x>1 for x in m) for m in masses):raise ValueError('masses')
    if any(not 0<=r['used_bytes']<=selection['capacity_bytes'] for r in rows):raise ValueError('charges')
    mass_den=lcm(*(x.denominator for row in masses for x in row))
    mass=[[int(x*mass_den) for x in row] for row in masses]
    policies=list(product(range(len(rows)),repeat=3));score_den=24*mass_den
    scores=[];risk=[];charge=[]
    best_mass=[max(m[j] for m in mass) for j in range(3)]
    for policy in policies:
        # At 1/3, 2/3 and 1, common denominators are 24 times mass_den.
        scores.append([sum(counts[j]*(2*a if j==k else 3-a)*mass[policy[k]][j] for j in range(3) for k in range(3)) for a in (1,2,3)])
        risk.append(max(best_mass[j]-mass[policy[k]][j] for j in range(3) if counts[j] for k in range(3)))
        charge.append([rows[i]['used_bytes'] for i in policy])
    optimal=[max(s[t] for s in scores) for t in range(3)]
    endpoint=[(optimal[0]-s[0],optimal[2]-s[2]) for s in scores]
    fixed=max((i for i,p in enumerate(policies) if len(set(p))==1),key=lambda i:(scores[i][1],policies[i]))
    nominal=max(range(len(policies)),key=lambda i:(scores[i][1],policies[i]))
    deterministic=min(max(p) for p in endpoint)
    detties=[i for i,p in enumerate(endpoint) if max(p)==deterministic]
    thresholds=[F(risk[fixed],mass_den),F(risk[fixed]+mass_den,2*mass_den),F(1)]
    levels=[];sets=[];lookup={}
    for threshold in thresholds:
        survivors=tuple(i for i,v in enumerate(risk) if v*threshold.denominator<=threshold.numerator*mass_den)
        assert fixed in survivors
        if survivors not in lookup:
            lottery=lotteries(endpoint,survivors,score_den)
            lottery.update(surviving_policies=list(survivors),excluded_policies=[i for i in range(len(policies)) if i not in survivors])
            lookup[survivors]=len(sets);sets.append(lottery)
        setid=lookup[survivors];lottery=sets[setid];chosen=lottery['candidates'][lottery['selected_index']]
        i,j,n,d=chosen[:4];support=[(i,F(n,d))] if i==j else [(i,F(n,d)),(j,F(d-n,d))]
        diagnostics=[]
        for t,a in enumerate((1,2,3)):
            probs=[sum(F(counts[p],4)*F(2*a if p==k else 3-a,6) for p in range(3)) for k in range(3)]
            possible=[k for k,p in enumerate(probs) if p]
            value=sum(w*F(scores[p][t],score_den) for p,w in support)
            actual_risk=max(F(best_mass[s]-mass[policies[p][k]][s],mass_den)
                for p,w in support for s in range(3) for k in range(3) if counts[s] and (2*a if s==k else 3-a))
            diagnostics.append(dict(reliability=rational(F(a,3)),retained_mass=rational(value),regret=rational(F(optimal[t],score_den)-value),
                expected_used_bytes=rational(sum(w*sum(probs[k]*charge[p][k] for k in range(3)) for p,w in support)),
                maximum_realized_bytes=max(charge[p][k] for p,w in support for k in possible),maximum_realized_coverage_loss=rational(actual_risk)))
        maxrisk=max(F(risk[p],mass_den) for p,w in support)
        assert maxrisk<=threshold
        levels.append(dict(threshold=rational(threshold),lottery_set=setid,
            maximum_realized_coverage_loss=rational(maxrisk),worst_expected_regret=lottery['worst_expected_regret'],
            improvement_over_unrestricted_deterministic=rational(F(deterministic,score_den)-F(*lottery['worst_expected_regret'])),
            diagnostics=diagnostics))
    return dict(mixture_counts=list(counts),interval=[[1,3],[1,1]],mass_denominator=mass_den,score_denominator=score_den,
        policy_masks=[[rows[i]['mask'] for i in p] for p in policies],policy_score_numerators=scores,
        policy_endpoint_regret_numerators=endpoint,policy_realized_loss_numerators=risk,policy_used_bytes=charge,
        unrestricted_optimal_score_numerators=optimal,fixed_policy=fixed,nominal_policy=nominal,
        deterministic_optimal_policies=detties,deterministic_worst_regret=rational(F(deterministic,score_den)),
        lottery_sets=sets,risk_levels=levels)


def controls():
    endpoint=[(4,0),(0,4),(3,3)]
    free=lotteries(endpoint,[0,1,2],4);bounded=lotteries(endpoint,[2],4)
    null=lotteries([(0,0),(0,0)],[0,1],1)
    return {'live:risk_tradeoff':free['worst_expected_regret']==[1,2] and bounded['worst_expected_regret']==[3,4],
        'placebo:zero_regret':null['worst_expected_regret']==[0,1],
        'positive:singleton_identity':bounded['candidates']==[[2,2,1,1,3,3,4]]}


def run(root,plan,pulse):
    cfg=plan['design']
    if (cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES) or cfg['mixtures']!=[list(x) for x in MIXTURES]
        or cfg['reliabilities']!=[list(x) for x in RELIABILITIES] or cfg['risk_thresholds']!=['fixed','halfway-to-one','one']):raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json');groups={}
    for s in read(root/'inputs/PARENT_SELECTIONS.json'):groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,records in groups.items():
        pulse(phase='channel-realized-risk-library',library=len(selections))
        s=from_parent(records);s['mixtures']=[solve(s,counts) for counts in MIXTURES];s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='channel-realized-risk-roster',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)]
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],fixed_bytes=st['weighted_overhead'],
                total_budget_bytes=st['weighted_overhead']+capacity,mixture_count=15,risk_levels=3))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/channel_realized_risk_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    (root/'raw/channel_realized_risk_selections.json.gz').write_bytes(gzip.compress(canonical(selections),mtime=0))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='supplied source priors,correct mixtures,channel interval,structural costs and roster',
        scope='finite-library risk-constrained expected channel regret;not forecast accuracy or process correspondence',
        raw_schema='roster/budget rows join selection_id to all15mixtures and3risk levels;lottery sets shared only when exact surviving policy lists are identical',
        lottery_row_columns=['first_policy_index','second_policy_index','first_weight_numerator','first_weight_denominator','lower_regret_numerator','upper_regret_numerator','regret_denominator']))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),joined_evaluations=len(rows)*45,unique_selections=len(selections),
        distinct_risk_problems=len(selections)*45,candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite-library realized-risk constrained channel regret;no forecasts or historical correspondence')
