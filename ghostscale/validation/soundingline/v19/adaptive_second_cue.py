"""Exact conditional purchase of a second cue under a supplied copy law."""
from fractions import Fraction as F
from .repeated_cue import solve as repeated, DEPENDENCES
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture

FEES=((0,1),(1,64),(1,16),(1,4))


def price(base,fee):
    f=F(*fee)
    if f<0:raise ValueError('negative fee')
    rat=lambda x:[x.numerator,x.denominator]
    conditional=[];gross=F(0);used=F(0);buy_probability=F(0);realized=[]
    for first,single in enumerate(base['single_cue']['cues']):
        pairs=[p for p in base['pairs'] if p['labels'][0]==first]
        p=F(*single['probability']);one=F(*single['contribution'])
        two=sum(F(*x['contribution']) for x in pairs)
        assert p==sum(F(*x['probability']) for x in pairs) and two>=one
        paid=two-p*f;best=max(one,paid)
        ties=[name for name,v in (('decline',one),('buy',paid)) if v==best]
        buy=paid>one  # Decline on exact ties, including impossible first labels.
        mass=two if buy else one
        cost=sum(F(*x['probability'])*x['used_bytes'] for x in pairs) if buy else p*single['selected_used_bytes']
        gross+=mass;used+=cost;buy_probability+=p*buy
        if p:
            realized.extend([x['used_bytes'] for x in pairs if F(*x['probability'])] if buy else [single['selected_used_bytes']])
        conditional.append(dict(first_label=first,probability=rat(p),
            decline_optimal_masks=single['optimal_masks'],decline_selected_mask=single['selected_mask'],
            buy_optimal_masks_by_second=[x['optimal_masks'] for x in pairs],
            buy_selected_policy=[x['selected_mask'] for x in pairs],
            decline_joint_mass=rat(one),buy_joint_mass=rat(two),buy_joint_net=rat(paid),
            conditional_break_even_fee=rat((two-one)/p) if p else None,
            optimal_decisions=ties,selected_decision='buy' if buy else 'decline',
            gross_contribution=rat(mass),net_contribution=rat(best),
            fee_contribution=rat(p*f if buy else F(0)),expected_byte_contribution=rat(cost)))
    expected_fee=f*buy_probability;net=gross-expected_fee
    never=F(*base['single_mass']);always=F(*base['retained_mass'])-f
    assert net>=max(never,always)
    return dict(fee=list(fee),conditional=conditional,buy_probability=rat(buy_probability),
        expected_fee=rat(expected_fee),gross_retained_mass=rat(gross),net_utility=rat(net),
        never_net=rat(never),always_net=rat(always),gain_over_never=rat(net-never),
        gain_over_always=rat(net-always),gain_over_best_constant=rat(net-max(never,always)),
        expected_used_bytes=rat(used),maximum_realized_bytes=max(realized),
        never_expected_bytes=base['single_cue']['expected_used_bytes'],
        always_expected_bytes=base['expected_used_bytes'])


def solve(selection,counts,reliability,dependence):
    base=repeated(selection,counts,reliability,dependence)
    return dict(repeated=base,fees=[price(base,f) for f in FEES])


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    a=solve(s,(2,2,0),(2,3),(0,1))['fees'][1]
    copy=solve(s,(2,2,0),(2,3),(1,1))['fees'][1]
    null=solve(s,(2,2,0),(1,3),(0,1))['fees'][1]
    return {'live:selective_purchase':a['buy_probability']==[1,6],
            'positive:known_net_gain':a['gain_over_never']==[5,128],
            'positive:copied_refusal':copy['buy_probability']==[0,1],
            'placebo:uninformative_refusal':null['buy_probability']==[0,1]}


import gzip
from ..v18_3.io import read, write, canonical, file_digest

def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]
        or cfg['dependences'] != [list(x) for x in DEPENDENCES]
        or cfg['fees'] != [list(x) for x in FEES]): raise ValueError('design')
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('controls')
    for name, h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name) != h: raise ValueError('input binding')
    population = read(root/'inputs/POPULATION.json'); structures = read(root/'inputs/STRUCTURES.json')
    groups = {}
    for s in read(root/'inputs/PARENT_SELECTIONS.json'):
        groups.setdefault((tuple(s['times']), tuple(s['costs']), s['capacity_bytes']), []).append(s)
    selections = []; lookup = {}
    for key, records in groups.items():
        pulse(phase='adaptive-second-cue-libraries', library=len(selections))
        s = from_parent(records)
        s['mixtures'] = [dict(mixture_counts=list(c), channels=[dict(reliability=list(p),levels=[solve(s,c,p,d) for d in DEPENDENCES]) for p in RELIABILITIES])
                         for c in MIXTURES]
        s['id'] = len(selections); selections.append(s); lookup[key] = s
    rows = []; paired = {}
    for i, r in enumerate(population):
        if i%256 == 0: pulse(phase='adaptive-second-cue-rosters', row=i)
        st = structures[r['structure']]; times = tuple(r['times'])
        costs = tuple(st['source_costs'][t-1] for t in times)
        if len(times) != r['report_sources'] or any(t > st['checkpoint'] for t in times): raise ValueError('roster')
        pair = tuple(r[k] for k in ('lineage','structure','draw','initial_maker','kind','switched','duplicates'))
        if r['evidence'] == 'aware': paired[pair] = times
        elif paired[pair] != times: raise ValueError('source pairing')
        for budget, threshold in [('half',st['checkpoint']//2), ('quarter',3*st['checkpoint']//4)]:
            recent = [j for j,t in enumerate(times) if t > threshold]; count = len(recent)
            order = sorted(range(len(times)), key=times.__getitem__)
            spaced = [order[(2*j+1)*len(times)//(2*count)] for j in range(count)]
            capacity = min(sum(costs[j] for j in recent), sum(costs[j] for j in spaced))
            s = lookup[(times,costs,capacity)]
            rows.append(dict(**{k:v for k,v in r.items() if k != 'times'}, budget=budget, selection_id=s['id'],
                fixed_bytes=st['weighted_overhead'], total_budget_bytes=st['weighted_overhead']+capacity,
                mixture_count=15, channel_count=3, dependence_count=3,fee_count=4))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/adaptive_second_cue_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,mixtures,known cue channel,known dependence,conditional acquisition fees;structural costs and rosters',
        scope='finite-library conditional cue acquisition;no forecast or process claim',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures,all3channels,3dependences and4fees'))
    return dict(controls=checks, population_rows=len(population), rows=len(rows), joined_evaluations=len(rows)*540,
        unique_selections=len(selections), distinct_acquisition_problems=len(selections)*540,
        candidate_count=sum(len(s['candidates']) for s in selections), numerical_acceptance=False,
        scope='net finite-library conditional acquisition value;not forecast accuracy or process correspondence')
