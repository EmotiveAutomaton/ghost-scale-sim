"""Candidate-library maximin retained source probability, with exact fractions."""
import gzip
from fractions import Fraction
from .source_mass_frontier import PRIORS, POLICIES, weights
from ..v18_3.io import read, write, canonical, file_digest


def choose(times, costs, capacity, candidates, priors):
    if len(times) != len(costs) or len(set(times)) != len(times): raise ValueError('items')
    if any(type(t) is not int or t <= 0 for t in times): raise ValueError('times')
    if type(capacity) is not int or capacity < 0 or any(type(k) is not int or k < 0 for k in costs): raise ValueError('costs')
    if not priors or any(len(p) != len(times) or any(x < 0 for x in p) or sum(p) != (1 if times else 0) for p in priors): raise ValueError('prior normalization')
    rank = {t: i for i, t in enumerate(sorted(times))}; records = {}
    for identity, selected in candidates.items():
        if len(selected) != len(set(selected)) or not set(selected) <= set(times): raise ValueError('candidate')
        indices = [i for i,t in enumerate(times) if t in selected]
        used = sum(costs[i] for i in indices)
        if used > capacity: raise ValueError('candidate bytes')
        mask = sum(1 << rank[t] for t in selected)
        masses = [sum((p[i] for i in indices), Fraction(0)) for p in priors]
        if mask not in records: records[mask] = dict(mask=mask, selected_times=sorted(selected,reverse=True), identities=[], used_bytes=used, masses=masses)
        records[mask]['identities'].append(identity)
    if not records: raise ValueError('empty library')
    best_by_prior = [max(r['masses'][j] for r in records.values()) for j in range(len(priors))]
    score = max(min(r['masses']) for r in records.values())
    tied = sorted(mask for mask,r in records.items() if min(r['masses']) == score)
    rational = lambda x: [x.numerator, x.denominator]
    rows = []
    for mask,r in sorted(records.items()):
        masses = r.pop('masses'); regrets = [b-m for b,m in zip(best_by_prior,masses)]
        rows.append(dict(r, identities=sorted(r['identities']), masses=[rational(m) for m in masses],
            worst_mass=rational(min(masses)), prior_regrets=[rational(v) for v in regrets],
            maximum_regret=rational(max(regrets))))
    return dict(candidates=rows, selected_mask=tied[-1], optimal_masks=tied,
        prior_optimal_masks=[[r['mask'] for r in rows if Fraction(*r['masses'][j]) == value] for j,value in enumerate(best_by_prior)],
        best_prior_mass=[rational(v) for v in best_by_prior], minimum_mass=rational(score))


def from_parent(records):
    if len(records) != 3 or {r['prior'] for r in records} != set(PRIORS): raise ValueError('prior roster')
    first=records[0];times=first['times'];costs=first['costs'];capacity=first['capacity_bytes'];candidates={}
    for r in records:
        if (r['times'],r['costs'],r['capacity_bytes']) != (times,costs,capacity): raise ValueError('paired allocation')
        if set(r['result']) != set(POLICIES): raise ValueError('policies')
        for policy,a in r['result'].items(): candidates[r['prior']+'/'+policy]=a['selected_times']
    priors=[]
    for prior in PRIORS:
        w=weights(times,prior);total=sum(w)
        priors.append([Fraction(v,total) for v in w])
    result=choose(times,costs,capacity,candidates,priors)
    for j,prior in enumerate(PRIORS):
        optimum=next(r for r in records if r['prior']==prior)['result']['optimal']
        if Fraction(*result['best_prior_mass'][j]) != Fraction(optimum['mass_numerator'],optimum['mass_denominator']): raise ValueError('parent optimum')
    return dict(times=times,costs=costs,capacity_bytes=capacity,**result)


def controls():
    x=choose([1,2],[1,1],1,{'a':[1],'b':[2]},[[Fraction(9,10),Fraction(1,10)],[Fraction(2,5),Fraction(3,5)]])
    empty=choose([1,2],[1,1],0,{'none':[]},[[Fraction(1,2)]*2])
    full=choose([1,2],[1,1],2,{'all':[1,2]},[[Fraction(1,2)]*2])
    return {'live:robust_rejects_single_prior_winner':x['selected_mask']==1,
        'placebo:empty_retains_zero':empty['minimum_mass']==[0,1],
        'positive:full_retains_one':full['minimum_mass']==[1,1]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES): raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    parent=read(root/'inputs/PARENT_SELECTIONS.json');groups={}
    for s in parent:groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,rr in groups.items():
        answer=from_parent(rr);answer['id']=len(selections);selections.append(answer);lookup[key]=answer
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='robust-source-mass',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=(r['lineage'],r['structure'],r['draw'],r['initial_maker'],r['kind'],r['switched'],r['duplicates'])
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)]
            chosen=next(a for a in s['candidates'] if a['mask']==s['selected_mask'])
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                total_used_bytes=st['weighted_overhead']+chosen['used_bytes'],retained_sources=len(chosen['selected_times']),
                minimum_mass=float(Fraction(*chosen['worst_mass'])),masses=[float(Fraction(*x)) for x in chosen['masses']],
                prior_regrets=[float(Fraction(*x)) for x in chosen['prior_regrets']],maximum_regret=float(Fraction(*chosen['maximum_regret'])),
                unique_candidates=len(s['candidates']),optimal_masks=s['optimal_masks'],selected_mask=s['selected_mask']))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/robust_mass_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='retained source rosters;structural byte costs;three fixed source priors and verified finite candidate library',scope='candidate-library robust retained mass;not unrestricted minimax knapsack,forecast optimality,learned provenance,process correspondence or human intent'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite candidate-library maximin source probability under convex three-prior uncertainty;no new observations')
