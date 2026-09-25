"""Exact value of prior revelation within the frozen source-mask library."""
import gzip
from fractions import Fraction as F
from .randomized_source_storage import randomize, fixture
from .robust_source_mass import from_parent
from .source_mass_frontier import PRIORS, POLICIES
from ..v18_3.io import read, write, canonical, file_digest

MIXTURES = tuple((a, b, 4-a-b) for a in range(5) for b in range(5-a))


def value(selection, counts):
    if tuple(counts) not in MIXTURES or any(type(x) is not int for x in counts):
        raise ValueError('prior mixture')
    coverage = randomize(selection)  # validates every mask and realized byte charge
    rows = sorted(selection['candidates'], key=lambda r:r['mask'])
    masses = [[F(*x) for x in r['masses']] for r in rows]
    if any(len(x)!=3 for x in masses):raise ValueError('three prior vertices required')
    weights = [F(x,4) for x in counts]
    scores = [sum(w*m for w,m in zip(weights,ms)) for ms in masses]
    best = max(scores)
    ties = [r['mask'] for r,s in zip(rows,scores) if s==best]
    chosen = next(r for r in rows if r['mask']==max(ties))
    prior_best = [max(ms[j] for ms in masses) for j in range(3)]
    prior_ties = [[r['mask'] for r,ms in zip(rows,masses) if ms[j]==prior_best[j]] for j in range(3)]
    informed = [next(r for r in rows if r['mask']==max(ts)) for ts in prior_ties]
    revealed = sum(w*x for w,x in zip(weights,prior_best))
    coverage_average = sum(w*F(*x) for w,x in zip(weights,coverage['expected_masses']))
    assert revealed>=best>=coverage_average
    rational = lambda x:[x.numerator,x.denominator]
    return dict(mixture_counts=list(counts),denominator=4,
        fixed_mask_scores=[dict(mask=r['mask'],mass=rational(v)) for r,v in zip(rows,scores)],
        fixed_optimal_masks=ties,selected_fixed_mask=chosen['mask'],fixed_mass=rational(best),
        revealed_optimal_masks=prior_ties,selected_revealed_masks=[r['mask'] for r in informed],
        prior_best_masses=list(map(rational,prior_best)),revealed_mass=rational(revealed),
        information_value=rational(revealed-best),coverage_lottery_mass=rational(coverage_average),
        fixed_over_coverage=rational(best-coverage_average),coverage_lottery=coverage['selected_lottery'],
        fixed_used_bytes=chosen['used_bytes'],revealed_used_bytes=[r['used_bytes'] for r in informed],
        expected_revealed_bytes=rational(sum(w*r['used_bytes'] for w,r in zip(weights,informed))),
        maximum_realized_revealed_bytes=max(r['used_bytes'] for w,r in zip(weights,informed) if w))


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    equal=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    return {'live:strict_information_value':value(s,(2,2,0))['information_value']==[1,2],
        'positive:point_prior_zero':all(value(s,c)['information_value']==[0,1] for c in ((4,0,0),(0,4,0),(0,0,4))),
        'placebo:identical_priors':all(value(equal,c)['information_value']==[0,1] for c in MIXTURES)}


def run(root,plan,pulse):
    cfg=plan['design']
    if cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES) or cfg['mixtures']!=[list(x) for x in MIXTURES]:
        raise ValueError('design')
    checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('input binding')
    population=read(root/'inputs/POPULATION.json');structures=read(root/'inputs/STRUCTURES.json')
    groups={}
    for s in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups.setdefault((tuple(s['times']),tuple(s['costs']),s['capacity_bytes']),[]).append(s)
    selections=[];lookup={}
    for key,records in groups.items():
        pulse(phase='prior-information-libraries',library=len(selections))
        s=from_parent(records);s['mixtures']=[value(s,c) for c in MIXTURES];s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='prior-information-rosters',row=i)
        st=structures[r['structure']];times=tuple(r['times']);costs=tuple(st['source_costs'][t-1] for t in times)
        if len(times)!=r['report_sources'] or any(t>st['checkpoint'] for t in times):raise ValueError('roster')
        pair=tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence']=='aware':paired[pair]=times
        elif paired[pair]!=times:raise ValueError('source pairing')
        for budget,threshold in [('half',st['checkpoint']//2),('quarter',3*st['checkpoint']//4)]:
            recent=[j for j,t in enumerate(times) if t>threshold];count=len(recent);order=sorted(range(len(times)),key=times.__getitem__)
            spaced=[order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity=min(sum(costs[j] for j in recent),sum(costs[j] for j in spaced));s=lookup[(times,costs,capacity)]
            rows.append(dict(**{k:v for k,v in r.items() if k!='times'},budget=budget,selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'],total_budget_bytes=st['weighted_overhead']+capacity,
                mixture_count=len(MIXTURES)))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/prior_information_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='source rosters,structural costs,three supplied priors,finite mask library and denominator-four prior mixtures',
        scope='prior identity revealed before storage versus best fixed mask and robust coverage lottery;no learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15 exact mixture records;no observations dropped'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),mixture_evaluations=len(rows)*len(MIXTURES),
        unique_selections=len(selections),distinct_mixture_problems=len(selections)*len(MIXTURES),
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='exact finite-library value of supplied prior information;no forecasts,learned access or process correspondence')
