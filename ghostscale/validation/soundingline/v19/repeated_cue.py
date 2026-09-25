"""Exact value and misspecification of a second source cue with copied errors."""
from fractions import Fraction as F
from itertools import product
from .noisy_prior_disclosure import MIXTURES, PRIORS, POLICIES, RELIABILITIES, from_parent, fixture, channel, value

DEPENDENCES = ((0,1), (1,2), (1,1))


def solve(selection, counts, reliability, dependence):
    matrix = channel(reliability)
    single = value(selection, counts, matrix)
    d = F(*dependence)
    if not 0 <= d <= 1: raise ValueError('dependence must be in [0,1]')
    rows = sorted(selection['candidates'], key=lambda r:r['mask'])
    rat = lambda x:[x.numerator,x.denominator]
    mass = {r['mask']:[F(*x) for x in r['masses']] for r in rows}
    charge = {r['mask']:r['used_bytes'] for r in rows}
    pairs=[];gross=F(0);naive_gross=F(0);used=F(0);naive_used=F(0)
    for first,second in product(range(3),repeat=2):
        independent=[F(counts[j],4)*matrix[j][first]*matrix[j][second] for j in range(3)]
        joint=[(1-d)*independent[j]+d*F(counts[j],4)*matrix[j][first]*(first==second) for j in range(3)]
        probability=sum(joint);nominal_probability=sum(independent)
        scores={m:sum(joint[j]*mass[m][j] for j in range(3)) for m in mass}
        nominal={m:sum(independent[j]*mass[m][j] for j in range(3)) for m in mass}
        optimum=max(scores.values());nominal_best=max(nominal.values())
        ties=[m for m,v in scores.items() if v==optimum];naive_ties=[m for m,v in nominal.items() if v==nominal_best]
        chosen=max(ties);naive=max(naive_ties)
        gross+=optimum;naive_gross+=scores[naive];used+=probability*charge[chosen];naive_used+=probability*charge[naive]
        pairs.append(dict(labels=[first,second],joint_prior=list(map(rat,joint)),probability=rat(probability),
            conditional_prior=[rat(x/probability) for x in joint] if probability else None,
            joint_mask_scores=[dict(mask=m,mass=rat(v)) for m,v in scores.items()],
            optimal_masks=ties,selected_mask=chosen,used_bytes=charge[chosen],contribution=rat(optimum),
            independent_joint_prior=list(map(rat,independent)),independent_probability=rat(nominal_probability),
            independent_conditional_prior=[rat(x/nominal_probability) for x in independent] if nominal_probability else None,
            independent_mask_scores=[dict(mask=m,mass=rat(v)) for m,v in nominal.items()],
            independent_optimal_masks=naive_ties,independent_selected_mask=naive,
            independent_used_bytes=charge[naive],independent_actual_contribution=rat(scores[naive])))
    one=F(*single['informed_mass']);full=F(*single['full_mass'])
    assert one<=gross<=full and naive_gross<=gross
    active=[p for p in pairs if F(*p['probability'])]
    return dict(single_cue=single,dependence=list(dependence),pairs=pairs,
        retained_mass=rat(gross),single_mass=rat(one),incremental_value=rat(gross-one),
        break_even_second_fee=rat(gross-one),independent_assumption_mass=rat(naive_gross),
        independent_gain_over_single=rat(naive_gross-one),independent_regret=rat(gross-naive_gross),
        expected_used_bytes=rat(used),independent_expected_used_bytes=rat(naive_used),
        maximum_realized_bytes=max(p['used_bytes'] for p in active),
        independent_maximum_realized_bytes=max(p['independent_used_bytes'] for p in active))


def controls():
    s=fixture([1,2],[1,1],1,{'a':[1],'b':[2]},[[F(1),F(0)],[F(0),F(1)],[F(1,2)]*2])
    independent=solve(s,(2,2,0),(2,3),(0,1));copied=solve(s,(2,2,0),(2,3),(1,1))
    return {'live:independent_value':independent['incremental_value']==[1,24],
        'positive:copied_identity':copied['incremental_value']==[0,1],
        'placebo:uninformative':solve(s,(2,2,0),(1,3),(0,1))['incremental_value']==[0,1],
        'positive:perfect':solve(s,(2,2,0),(1,1),(0,1))['incremental_value']==[0,1],
        'positive:copied_off_diagonal_undefined':all(p['conditional_prior'] is None for p in copied['pairs'] if p['labels'][0]!=p['labels'][1])}


import gzip
from ..v18_3.io import read, write, canonical, file_digest

def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]
        or cfg['dependences'] != [list(x) for x in DEPENDENCES]): raise ValueError('design')
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
        pulse(phase='repeated-cue-libraries', library=len(selections))
        s = from_parent(records)
        s['mixtures'] = [dict(mixture_counts=list(c), channels=[dict(reliability=list(p),levels=[solve(s,c,p,d) for d in DEPENDENCES]) for p in RELIABILITIES])
                         for c in MIXTURES]
        s['id'] = len(selections); selections.append(s); lookup[key] = s
    rows = []; paired = {}
    for i, r in enumerate(population):
        if i%256 == 0: pulse(phase='repeated-cue-rosters', row=i)
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
                mixture_count=15, channel_count=3, dependence_count=3))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/repeated_cue_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,mixtures,known cue channel,known dependence;structural costs and rosters',
        scope='finite-library repeated cue value and mistaken independence;no forecast or process claim',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures,all3channels and3dependences'))
    return dict(controls=checks, population_rows=len(population), rows=len(rows), joined_evaluations=len(rows)*135,
        unique_selections=len(selections), distinct_dependence_problems=len(selections)*135,
        candidate_count=sum(len(s['candidates']) for s in selections), numerical_acceptance=False,
        scope='gross finite-library repeated-cue value;not forecast accuracy or process correspondence')
