"""Frozen-library minimax source-prior regret with exact rational comparisons."""
import gzip
from fractions import Fraction
from .robust_source_mass import from_parent, choose
from .source_mass_frontier import PRIORS, POLICIES
from ..v18_3.io import read, write, canonical, file_digest


def minimize(selection):
    rows=selection['candidates']
    if not rows:raise ValueError('empty library')
    optima=[Fraction(*v) for v in selection['best_prior_mass']]
    if any(not 0<=v<=1 for v in optima):raise ValueError('prior optimum')
    for j,v in enumerate(optima):
        if max(Fraction(*r['masses'][j]) for r in rows)!=v:raise ValueError('parent optimum')
    scores={};minimums={}
    for r in rows:
        masses=[Fraction(*v) for v in r['masses']]
        if len(masses)!=len(optima) or any(not 0<=v<=1 for v in masses):raise ValueError('mass')
        regrets=[a-b for a,b in zip(optima,masses)]
        if regrets!=[Fraction(*v) for v in r['prior_regrets']] or max(regrets)!=Fraction(*r['maximum_regret']):raise ValueError('regret')
        scores[r['mask']]=max(regrets);minimums[r['mask']]=min(masses)
    if len(scores)!=len(rows):raise ValueError('duplicate mask')
    worst=max(minimums.values());robust_ties=sorted(k for k,v in minimums.items() if v==worst)
    if selection['optimal_masks']!=robust_ties or selection['selected_mask']!=robust_ties[-1]:raise ValueError('robust choice')
    best=min(scores.values());ties=sorted(k for k,v in scores.items() if v==best);chosen=ties[-1]
    rational=lambda v:[v.numerator,v.denominator]
    return dict(regret_selected_mask=chosen,regret_optimal_masks=ties,minimax_regret=rational(best),
        robust_choice_regret=rational(scores[selection['selected_mask']]),
        regret_improvement=rational(scores[selection['selected_mask']]-best),
        minimum_mass_cost=rational(worst-minimums[chosen]),regret_choice_minimum_mass=rational(minimums[chosen]))


def controls():
    priors=[[Fraction(2,5),Fraction(3,5),Fraction(0)],[Fraction(2,5),Fraction(3,10),Fraction(3,10)]]
    s=choose([1,2,3],[1,1,1],1,{'a':[1],'b':[2],'c':[3]},priors);a=minimize(s)
    empty=minimize(choose([1,2],[1,1],0,{'empty':[]},[[Fraction(1,2)]*2]))
    full=minimize(choose([1,2],[1,1],2,{'all':[1,2]},[[Fraction(1,2)]*2]))
    return {'live:regret_differs_from_maximin':s['selected_mask']==1 and a['regret_selected_mask']==2,
        'positive:known_regret':a['minimax_regret']==[1,10] and a['minimum_mass_cost']==[1,10],
        'placebo:empty_no_opportunity':empty['minimax_regret']==[0,1],
        'positive:full_no_regret':full['minimax_regret']==[0,1]}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES):raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    parent=read(root/'inputs/PARENT_SELECTIONS.json');groups={}
    for s in parent:groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,rr in groups.items():
        answer=from_parent(rr);answer.update(minimize(answer));answer['id']=len(selections)
        selections.append(answer);lookup[key]=answer
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='source-prior-regret',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=(r['lineage'],r['structure'],r['draw'],r['initial_maker'],r['kind'],r['switched'],r['duplicates'])
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)]
            chosen=next(a for a in s['candidates'] if a['mask']==s['regret_selected_mask'])
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                total_used_bytes=st['weighted_overhead']+chosen['used_bytes'],retained_sources=len(chosen['selected_times']),
                minimum_mass=float(Fraction(*chosen['worst_mass'])),masses=[float(Fraction(*x)) for x in chosen['masses']],
                prior_regrets=[float(Fraction(*x)) for x in chosen['prior_regrets']],maximum_regret=float(Fraction(*chosen['maximum_regret'])),
                robust_choice_regret=float(Fraction(*s['robust_choice_regret'])),regret_improvement=float(Fraction(*s['regret_improvement'])),
                minimum_mass_cost=float(Fraction(*s['minimum_mass_cost'])),unique_candidates=len(s['candidates']),
                regret_optimal_masks=s['regret_optimal_masks'],regret_selected_mask=s['regret_selected_mask'],robust_selected_mask=s['selected_mask']))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/source_regret_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='source rosters,structural byte costs,three supplied prior definitions and verified prior-optimal finite policy library',scope='candidate-library minimax regret;not unrestricted regret knapsack,forecast optimality,learned provenance,process correspondence or human intent'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),unique_selections=len(selections),
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite candidate-library minimax regret under convex three-prior uncertainty;no new observations')
