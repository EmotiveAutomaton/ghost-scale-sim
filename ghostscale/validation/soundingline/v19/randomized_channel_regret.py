"""Byte-feasible lotteries over contingent policies for unknown cue reliability."""
from fractions import Fraction as F
from itertools import combinations, product
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture


def deterministic(selection, counts, interval=((1,3),(1,1))):
    """Minimize worst channel-relative regret; freeze a whole contingent policy."""
    assert tuple(counts) in MIXTURES
    lower, upper = map(lambda x:F(*x), interval)
    assert 0 <= lower <= upper <= 1
    rows = sorted(selection['candidates'], key=lambda r:r['mask'])
    masses = {r['mask']:tuple(F(*x) for x in r['masses']) for r in rows}
    charges = {r['mask']:r['used_bytes'] for r in rows}
    assert masses and all(0 <= b <= selection['capacity_bytes'] for b in charges.values())
    weights = tuple(F(c,4) for c in counts)
    policies = list(product(masses, repeat=3))
    # Exact score lines retain their intercept and slope; neither depends on
    # the realized channel. Policies are all feasible before reliability is known.
    def score(policy, correct):
        return sum(weights[p]*(correct if p==label else (1-correct)/2)*masses[policy[label]][p]
                   for p in range(3) for label in range(3))
    lines = {p:(score(p,F(0)),score(p,F(1))-score(p,F(0))) for p in policies}
    at = lambda p,r:lines[p][0]+lines[p][1]*r
    optimal_scores = {r:max(at(p,r) for p in policies) for r in {lower, upper, F(2,3)}}
    optimal = optimal_scores.__getitem__
    worst = {p:max(optimal(lower)-at(p,lower),optimal(upper)-at(p,upper)) for p in policies}
    best = min(worst.values()); ties = [p for p in policies if worst[p]==best]
    chosen = max(ties)
    fixed = max((p for p in policies if len(set(p))==1),key=lambda p:(at(p,F(2,3)),p))
    nominal = max(policies,key=lambda p:(at(p,F(2,3)),p))
    rat = lambda x:[x.numerator,x.denominator]
    diagnostics=[]
    for r in sorted({lower,upper,F(2,3)}):
        best_score=optimal(r);arms={}
        for name,p in [('robust',chosen),('fixed',fixed),('nominal',nominal)]:
            support=[(j,k) for j in range(3) for k in range(3)
                     if weights[j]*(r if j==k else (1-r)/2)>0]
            probability=[sum(weights[j]*(r if j==k else (1-r)/2) for j in range(3)) for k in range(3)]
            # Realized loss compares the selected mask with the best mask for
            # that latent prior; it differs from regret in expected coverage.
            realized=max(max(v[j] for v in masses.values())-masses[p[k]][j] for j,k in support)
            arms[name]=dict(policy=list(p),retained_mass=rat(at(p,r)),regret=rat(best_score-at(p,r)),
                gain_over_fixed=rat(at(p,r)-at(fixed,r)),
                expected_used_bytes=rat(sum(probability[k]*charges[p[k]] for k in range(3))),
                maximum_realized_bytes=max(charges[p[k]] for j,k in support),
                maximum_realized_coverage_loss=rat(realized))
        diagnostics.append(dict(reliability=rat(r),optimal_mass=rat(best_score),arms=arms))
    return dict(mixture_counts=list(counts),interval=[rat(lower),rat(upper)],
        policies=[dict(policy=list(p),intercept=rat(lines[p][0]),slope=rat(lines[p][1]),
                       worst_regret=rat(worst[p]),used_bytes=[charges[m] for m in p]) for p in policies],
        optimal_policies=[list(p) for p in ties],selected_policy=list(chosen),
        worst_regret=rat(best),fixed_policy=list(fixed),fixed_worst_regret=rat(worst[fixed]),
        nominal_policy=list(nominal),nominal_worst_regret=rat(worst[nominal]),
        improvement_over_fixed=rat(worst[fixed]-best),improvement_over_nominal=rat(worst[nominal]-best),
        diagnostics=diagnostics)


def lotteries(endpoint_regrets):
    """All singleton and strictly interior two-policy diagonal intersections."""
    policies=sorted(endpoint_regrets)
    candidates=[(p,p,F(1)) for p in policies]
    for p,q in combinations(policies,2):
        dp=endpoint_regrets[p][0]-endpoint_regrets[p][1]
        dq=endpoint_regrets[q][0]-endpoint_regrets[q][1]
        if dp != dq:
            weight=-dq/(dp-dq)
            if 0 < weight < 1: candidates.append((p,q,weight))
    values={c:tuple(c[2]*endpoint_regrets[c[0]][j]+(1-c[2])*endpoint_regrets[c[1]][j]
                    for j in range(2)) for c in candidates}
    best=min(max(v) for v in values.values())
    ties=sorted(c for c,v in values.items() if max(v)==best)
    return values,best,ties,max(ties)


def solve(selection, counts, interval=((1,3),(1,1))):
    base=deterministic(selection,counts,interval)
    lo,hi=(F(*r) for r in interval)
    lines={tuple(r['policy']):(F(*r['intercept']),F(*r['slope'])) for r in base['policies']}
    score=lambda p,r:lines[p][0]+lines[p][1]*r
    optimal_scores={r:max(score(p,r) for p in lines) for r in {lo,hi,F(2,3)}}
    optimal=optimal_scores.__getitem__
    endpoint={p:tuple(optimal(r)-score(p,r) for r in (lo,hi)) for p in lines}
    candidates,best,ties,chosen=lotteries(endpoint)
    rat=lambda x:[x.numerator,x.denominator]
    def record(c):
        p,q,w=c
        return dict(first_policy=list(p),second_policy=list(q),first_weight=rat(w),
            endpoint_regrets=[rat(x) for x in candidates[c]],
            worst_expected_regret=rat(max(candidates[c])),
            maximum_realized_policy_regret=rat(max(max(endpoint[p]),max(endpoint[q]))))
    mass={r['mask']:tuple(F(*x) for x in r['masses']) for r in selection['candidates']}
    charge={r['mask']:r['used_bytes'] for r in selection['candidates']}
    p,q,w=chosen;support=((p,w),) if p==q else ((p,w),(q,1-w))
    diagnostics=[]
    for r in sorted({lo,hi,F(2,3)}):
        label_probability=[sum(F(counts[j],4)*(r if j==k else (1-r)/2) for j in range(3)) for k in range(3)]
        joint_support=[(j,k) for j in range(3) for k in range(3) if F(counts[j],4)*(r if j==k else (1-r)/2)>0]
        value=sum(weight*score(policy,r) for policy,weight in support)
        diagnostics.append(dict(reliability=rat(r),optimal_mass=rat(optimal(r)),retained_mass=rat(value),
            regret=rat(optimal(r)-value),expected_used_bytes=rat(sum(weight*sum(label_probability[k]*charge[policy[k]] for k in range(3)) for policy,weight in support)),
            maximum_realized_bytes=max(charge[policy[k]] for policy,weight in support for j,k in joint_support),
            maximum_realized_coverage_loss=rat(max(max(m[j] for m in mass.values())-mass[policy[k]][j]
                for policy,weight in support for j,k in joint_support))))
    return dict(mixture_counts=list(counts),interval=base['interval'],deterministic=base,
        endpoint_regrets=[dict(policy=list(p),regrets=[rat(x) for x in endpoint[p]]) for p in sorted(endpoint)],
        candidate_lotteries=[record(c) for c in sorted(candidates)],optimal_lotteries=[record(c) for c in ties],
        selected_lottery=record(chosen),worst_expected_regret=rat(best),
        improvement_over_deterministic=rat(F(*base['worst_regret'])-best),
        improvement_over_fixed=rat(F(*base['fixed_worst_regret'])-best),
        improvement_over_nominal=rat(F(*base['nominal_worst_regret'])-best),diagnostics=diagnostics)


def compare(selection):
    return dict(mixtures=[solve(selection,c) for c in MIXTURES])


def controls():
    endpoint={(1,1,1):(F(1),F(0)),(2,2,2):(F(0),F(1))}
    _,value,ties,chosen=lotteries(endpoint)
    null={(1,1,1):(F(0),F(0)),(2,2,2):(F(0),F(0))}
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    point=solve(s,(3,1,0),((2,3),(2,3)))
    return {'live:known_lottery_gain':value==F(1,2) and chosen[2]==F(1,2),
        'placebo:no_regret':lotteries(null)[1]==0,
        'positive:point_interval':point['worst_expected_regret']==[0,1],
        'positive:feasible_weights':all(0 < c[2] <= 1 for c in ties)}


import gzip
from ..v18_3.io import read, write, canonical, file_digest


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]): raise ValueError('design')
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
        pulse(phase='randomized-channel-regret-libraries',library=len(selections))
        s=from_parent(records); s.update(compare(s)); s['id']=len(selections)
        selections.append(s); lookup[key]=s
    rows=[]; paired={}
    for i,r in enumerate(population):
        if i%256==0: pulse(phase='randomized-channel-regret-rosters',row=i)
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
                mixture_count=15,interval_count=1))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/randomized_channel_regret_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,correct mixture,unknown channel interval,structural costs and rosters',
        scope='finite-library robust channel regret;no fee,forecast or learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures with all feasible three-label policies and extreme one/two-policy lotteries'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),joined_evaluations=len(rows)*15,
        unique_selections=len(selections),distinct_robust_problems=len(selections)*15,
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite-library robust channel regret;not forecast accuracy or process correspondence')
