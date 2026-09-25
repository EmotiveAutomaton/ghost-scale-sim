"""Exact ex-ante choice to acquire a priced source cue or store without it."""
import gzip
from fractions import Fraction as F
from itertools import product
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture, channel, value
from ..v18_3.io import read, write, canonical, file_digest

FEES=((0,1),(1,64),(1,16),(1,4))


def solve(selection, counts, reliability, fees=FEES):
    base=value(selection,counts,channel(reliability))
    if any(len(f)!=2 or type(f[0]) is not int or type(f[1]) is not int or f[1]<=0 or f[0]<0 for f in fees):
        raise ValueError('nonnegative rational fees required')
    rat=lambda x:[x.numerator,x.denominator]
    masks=sorted(r['mask'] for r in selection['candidates'])
    charges={r['mask']:r['used_bytes'] for r in selection['candidates']}
    label_scores=[{r['mask']:F(*r['mass']) for r in c['joint_mask_scores']} for c in base['cues']]
    policies=[dict(policy=list(p),gross_mass=rat(sum(label_scores[k][p[k]] for k in range(3)))) for p in product(masks,repeat=3)]
    fixed=F(*base['fixed_mass']);paid=F(*base['informed_mass']);threshold=paid-fixed
    best_policies=[r['policy'] for r in policies if F(*r['gross_mass'])==paid]
    levels=[]
    for fee in fees:
        f=F(*fee);best=max(fixed,paid-f);ties=[]
        if fixed==best:ties.extend(dict(acquire=False,policy=[m,m,m]) for m in base['fixed_optimal_masks'])
        if paid-f==best:ties.extend(dict(acquire=True,policy=p) for p in best_policies)
        # Preserve all ties; select free access when acquisition is exactly indifferent.
        chosen=max((t for t in ties if not t['acquire']),key=lambda t:t['policy']) if any(not t['acquire'] for t in ties) else max(ties,key=lambda t:t['policy'])
        p=chosen['policy']; gross=paid if chosen['acquire'] else fixed
        expected=sum(F(*c['probability'])*charges[p[k]] for k,c in enumerate(base['cues']))
        active=[k for k,c in enumerate(base['cues']) if F(*c['probability'])]
        levels.append(dict(fee=list(fee),optimal_choices=ties,selected=chosen,gross_mass=rat(gross),net_utility=rat(best),
            net_gain_over_no_access=rat(best-fixed),fee_paid=rat(f if chosen['acquire'] else F(0)),
            expected_used_bytes=rat(expected),maximum_realized_bytes=max(charges[p[k]] for k in active)))
    return dict(free_access=base,all_paid_policies=policies,paid_optimal_policies=best_policies,break_even_fee=rat(threshold),fee_levels=levels)


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    a=solve(s,(2,2,0),(2,3));null=solve(s,(4,0,0),(1,3))
    return {'live:positive_net_value':a['fee_levels'][1]['net_gain_over_no_access']==[15,64],
        'positive:break_even':a['break_even_fee']==[1,4] and {t['acquire'] for t in a['fee_levels'][-1]['optimal_choices']}=={False,True},
        'placebo:uninformative_no_purchase':all(not l['selected']['acquire'] for l in null['fee_levels']),
        'positive:free_identity':a['fee_levels'][0]['gross_mass']==a['free_access']['informed_mass']}


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]
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
        pulse(phase='cue-cost-libraries', library=len(selections))
        s = from_parent(records)
        s['mixtures'] = [dict(mixture_counts=list(c), channels=[solve(s,c,p) for p in RELIABILITIES])
                         for c in MIXTURES]
        s['id'] = len(selections); selections.append(s); lookup[key] = s
    rows = []; paired = {}
    for i, r in enumerate(population):
        if i%256 == 0: pulse(phase='cue-cost-rosters', row=i)
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
                mixture_count=15, channel_count=3, fee_count=4))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/cue_cost_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,mixtures,known cue channel,hypothetical utility fee;structural costs and rosters',
        scope='finite-library net cue value;fees in source-mass units,not measured cost;no forecast or process claim',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures,all3channels and4fees'))
    return dict(controls=checks, population_rows=len(population), rows=len(rows), joined_evaluations=len(rows)*180,
        unique_selections=len(selections), distinct_fee_problems=len(selections)*180,
        candidate_count=sum(len(s['candidates']) for s in selections), numerical_acceptance=False,
        scope='net finite-library cue value;not forecast accuracy or process correspondence')
