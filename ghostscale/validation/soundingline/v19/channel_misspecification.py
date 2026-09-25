"""Frozen nominal cue reliability evaluated under different actual reliability."""
import gzip
from fractions import Fraction as F
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, channel, value, from_parent, fixture
from ..v18_3.io import read, write, canonical, file_digest


def evaluate(selection, nominal, actual):
    """Evaluate an already selected policy; actual mixture cannot change it."""
    assert nominal['mixture_counts'] == actual['mixture_counts']
    rows = {r['mask']: r for r in selection['candidates']}
    assert all(0 <= r['used_bytes'] <= selection['capacity_bytes'] for r in rows.values())
    rat = lambda x: [x.numerator, x.denominator]
    weights = [F(c, 4) for c in actual['mixture_counts']]
    matrix = [[F(*x) for x in row] for row in actual['channel']]
    mass = F(0); used = F(0); unsupported = F(0); cues = []
    for label, cue in enumerate(nominal['cues']):
        row = rows[cue['selected_mask']]
        joint = [weights[p]*matrix[p][label] for p in range(3)]
        probability = sum(joint)
        contribution = sum(joint[p]*F(*row['masses'][p]) for p in range(3))
        mass += contribution; used += probability*row['used_bytes']
        nominal_zero = cue['probability'] == [0, 1]
        if nominal_zero: unsupported += probability
        cues.append(dict(label=label, nominal_probability=cue['probability'],
            actual_probability=rat(probability), nominal_zero=nominal_zero,
            selected_mask=row['mask'], used_bytes=row['used_bytes'],
            actual_contribution=rat(contribution)))
    fixed_row = rows[nominal['selected_fixed_mask']]
    fixed_mass = sum(weights[p]*F(*fixed_row['masses'][p]) for p in range(3))
    optimum, ignore = F(*actual['informed_mass']), F(*actual['fixed_mass'])
    assert mass <= optimum
    return dict(mixture_counts=nominal['mixture_counts'], nominal_channel=nominal['channel'], actual_channel=actual['channel'],
        cues=cues, retained_mass=rat(mass), optimal_informed_mass=rat(optimum),
        optimal_fixed_mass=rat(ignore), regret=rat(optimum-mass), gain_over_fixed=rat(mass-ignore),
        nominal_fixed_mask=nominal['selected_fixed_mask'], nominal_fixed_actual_mass=rat(fixed_mass),
        gain_over_nominal_fixed=rat(mass-fixed_mass),
        unexpected_label_probability=rat(unsupported), expected_used_bytes=rat(used),
        maximum_realized_bytes=max(c['used_bytes'] for c in cues if F(*c['actual_probability'])))


def compare(selection):
    policies = [[value(selection, c, channel(p)) for p in RELIABILITIES] for c in MIXTURES]
    records = [dict(mixture_counts=list(c), nominal_reliability=list(n), actual_reliability=list(a),
                    **{k:v for k,v in evaluate(selection, choices[i], choices[j]).items() if k != 'mixture_counts'})
               for c, choices in zip(MIXTURES, policies)
               for i,n in enumerate(RELIABILITIES) for j,a in enumerate(RELIABILITIES)]
    return dict(policies=policies, comparisons=records)


def controls():
    s = fixture([1,2], [1,1], 1, {'a':[1], 'b':[2]},
                [[F(1),F(0)], [F(0),F(1)], [F(1,2)]*2])
    n = value(s,(3,1,0),channel((1,1))); a = value(s,(3,1,0),channel((1,3)))
    same = evaluate(s,n,n); wrong = evaluate(s,n,a)
    return {'positive:matched_channel': same['regret']==[0,1],
            'live:harmful_overconfidence': wrong['gain_over_fixed']==[-1,3],
            'placebo:actual_uninformative': a['information_value']==[0,1],
            'positive:unexpected_label': wrong['unexpected_label_probability']==[1,3]}


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
        pulse(phase='channel-misspecification-libraries',library=len(selections))
        s=from_parent(records); s.update(compare(s)); s['id']=len(selections)
        selections.append(s); lookup[key]=s
    rows=[]; paired={}
    for i,r in enumerate(population):
        if i%256==0: pulse(phase='channel-misspecification-rosters',row=i)
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
                mixture_count=15,nominal_channel_count=3,actual_channel_count=3))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/channel_misspecification_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,correct source mixture,nominal and actual cue channels,structural costs and rosters',
        scope='finite-library cue-channel sensitivity;no fee,forecast or learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures and3nominal by3actual channels'))
    return dict(controls=checks,population_rows=len(population),rows=len(rows),joined_evaluations=len(rows)*135,
        unique_selections=len(selections),distinct_mismatch_problems=len(selections)*135,
        candidate_count=sum(len(s['candidates']) for s in selections),numerical_acceptance=False,
        scope='finite-library cue-channel sensitivity;not forecast accuracy or process correspondence')
