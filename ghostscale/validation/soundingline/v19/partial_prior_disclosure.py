"""Finite value of partial disclosure of the source-prior identity."""
import gzip
from fractions import Fraction as F
from .prior_information import MIXTURES, PRIORS, POLICIES, from_parent, fixture, randomize
from ..v18_3.io import read, write, canonical, file_digest

PARTITIONS = (((0,1,2),), ((0,),(1,2)), ((1,),(0,2)), ((2,),(0,1)), ((0,),(1,),(2,)))


def value(selection, counts):
    if tuple(counts) not in MIXTURES or any(type(x) is not int for x in counts):
        raise ValueError('prior mixture')
    randomize(selection)  # Validate masks, masses and individual byte feasibility.
    rows = sorted(selection['candidates'],key=lambda r:r['mask'])
    masses = [[F(*x) for x in r['masses']] for r in rows]
    if any(len(x)!=3 for x in masses):raise ValueError('three priors required')
    weights = [F(x,4) for x in counts]
    rat = lambda x:[x.numerator,x.denominator]
    results = []
    for partition in PARTITIONS:
        cells = []; total = F(0); charge = F(0); realized = []
        for cell in partition:
            probability = sum(weights[j] for j in cell)
            scores = [sum(weights[j]*m[j] for j in cell) for m in masses]
            best = max(scores)
            ties = [r['mask'] for r,x in zip(rows,scores) if x==best]
            chosen = next(r for r in rows if r['mask']==max(ties))
            total += best; charge += probability*chosen['used_bytes']
            if probability:realized.append(chosen['used_bytes'])
            cells.append(dict(prior_ids=list(cell),probability=rat(probability),
                conditional_prior=[rat(weights[j]/probability) for j in cell] if probability else None,
                joint_mask_scores=[dict(mask=r['mask'],mass=rat(v)) for r,v in zip(rows,scores)],
                optimal_masks=ties,selected_mask=chosen['mask'],selected_used_bytes=chosen['used_bytes'],
                conditional_retained_mass=rat(best/probability) if probability else None,
                contribution=rat(best)))
        results.append(dict(partition=[list(c) for c in partition],cells=cells,retained_mass=rat(total),
            expected_used_bytes=rat(charge),maximum_realized_bytes=max(realized)))
    baseline=F(*results[0]['retained_mass']);full=F(*results[-1]['retained_mass'])
    for r in results:
        score=F(*r['retained_mass']);assert baseline<=score<=full
        r.update(information_value=rat(score-baseline),full_disclosure_gap=rat(full-score))
    return dict(mixture_counts=list(counts),denominator=4,disclosures=results)


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    same=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1,2)]*2]*3)
    r=value(s,(2,2,0))['disclosures']
    return {'live:selective_partition':r[1]['information_value']==[1,2] and r[3]['information_value']==[0,1],
        'positive:full_revelation':r[-1]['retained_mass']==[1,1],
        'placebo:identical_priors':all(x['information_value']==[0,1] for x in value(same,(1,1,2))['disclosures']),
        'placebo:point_mixture':all(x['information_value']==[0,1] for x in value(s,(4,0,0))['disclosures'])}


def run(root,plan,pulse):
    cfg=plan['design']
    if (cfg['source_priors']!=list(PRIORS) or cfg['policies']!=list(POLICIES)
        or cfg['mixtures']!=[list(x) for x in MIXTURES]
        or cfg['partitions']!=[[list(c) for c in p] for p in PARTITIONS]):raise ValueError('design')
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
        pulse(phase='partial-prior-libraries',library=len(selections))
        s=from_parent(records);s['mixtures']=[value(s,c) for c in MIXTURES];s['id']=len(selections)
        selections.append(s);lookup[key]=s
    rows=[];paired={}
    for i,r in enumerate(population):
        if i%256==0:pulse(phase='partial-prior-rosters',row=i)
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
                mixture_count=15,partition_count=5))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/partial_prior_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,mixture and disclosed prior partition;structural costs and rosters',
        scope='finite-library partial prior information;no fee,forecast outcome or learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures and all5partitions'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),joined_evaluations=len(rows)*75,
        unique_selections=len(selections),distinct_disclosure_problems=len(selections)*75,
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='gross finite-library partial disclosure value;no forecasts or process correspondence')
