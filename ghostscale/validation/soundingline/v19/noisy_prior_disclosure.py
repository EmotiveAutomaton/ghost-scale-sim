"""Storage decisions under a supplied noisy prior-identity channel."""
import gzip
from fractions import Fraction as F
from .prior_information import MIXTURES, PRIORS, POLICIES, from_parent, fixture, randomize
from ..v18_3.io import read, write, canonical, file_digest

RELIABILITIES = ((1, 3), (2, 3), (1, 1))


def channel(correct):
    p = F(*correct)
    if not F(0) <= p <= F(1):
        raise ValueError('invalid reliability')
    return [[p if i == j else (1-p)/2 for j in range(3)] for i in range(3)]


def value(selection, counts, matrix):
    if tuple(counts) not in MIXTURES or any(type(x) is not int for x in counts):
        raise ValueError('prior mixture')
    if len(matrix) != 3 or any(len(row) != 3 or any(type(x) is not F or x < 0 for x in row)
                               or sum(row) != 1 for row in matrix):
        raise ValueError('rational stochastic three-label channel required')
    randomize(selection)
    rows = sorted(selection['candidates'], key=lambda r: r['mask'])
    masses = [[F(*x) for x in r['masses']] for r in rows]
    if any(len(m) != 3 for m in masses): raise ValueError('three priors required')
    weights = [F(c, 4) for c in counts]
    rat = lambda x: [x.numerator, x.denominator]
    fixed_scores = [sum(w*m for w, m in zip(weights, mass)) for mass in masses]
    fixed = max(fixed_scores)
    fixed_ties = [r['mask'] for r, s in zip(rows, fixed_scores) if s == fixed]
    cues = []; informed = F(0); certain = F(0); informed_bytes = F(0); certain_bytes = F(0)
    realized, certain_realized = [], []
    for label in range(3):
        joint = [weights[j]*matrix[j][label] for j in range(3)]
        probability = sum(joint)
        scores = [sum(j*m for j, m in zip(joint, mass)) for mass in masses]
        best = max(scores)
        ties = [r['mask'] for r, s in zip(rows, scores) if s == best]
        selected = next(i for i, r in enumerate(rows) if r['mask'] == max(ties))
        certainty_best = max(m[label] for m in masses)
        certainty_ties = [r['mask'] for r, m in zip(rows, masses) if m[label] == certainty_best]
        naive = next(i for i, r in enumerate(rows) if r['mask'] == max(certainty_ties))
        informed += best; certain += scores[naive]
        informed_bytes += probability*rows[selected]['used_bytes']
        certain_bytes += probability*rows[naive]['used_bytes']
        if probability:
            realized.append(rows[selected]['used_bytes']); certain_realized.append(rows[naive]['used_bytes'])
        cues.append(dict(label=label, joint_prior=list(map(rat, joint)), probability=rat(probability),
            conditional_prior=[rat(x/probability) for x in joint] if probability else None,
            joint_mask_scores=[dict(mask=r['mask'], mass=rat(s)) for r, s in zip(rows, scores)],
            optimal_masks=ties, selected_mask=rows[selected]['mask'],
            selected_used_bytes=rows[selected]['used_bytes'],
            conditional_retained_mass=rat(best/probability) if probability else None,
            contribution=rat(best), certainty_optimal_masks=certainty_ties,
            certainty_selected_mask=rows[naive]['mask'], certainty_used_bytes=rows[naive]['used_bytes'],
            certainty_contribution=rat(scores[naive])))
    full = sum(weights[j]*max(m[j] for m in masses) for j in range(3))
    assert fixed <= informed <= full and certain <= informed
    return dict(mixture_counts=list(counts), denominator=4,
        channel=[[rat(x) for x in row] for row in matrix], cues=cues,
        fixed_mask_scores=[dict(mask=r['mask'], mass=rat(s)) for r, s in zip(rows, fixed_scores)],
        fixed_optimal_masks=fixed_ties, selected_fixed_mask=max(fixed_ties), fixed_mass=rat(fixed),
        informed_mass=rat(informed), certainty_mass=rat(certain), full_mass=rat(full),
        information_value=rat(informed-fixed), certainty_over_fixed=rat(certain-fixed),
        certainty_penalty=rat(informed-certain), expected_used_bytes=rat(informed_bytes),
        expected_certainty_bytes=rat(certain_bytes), maximum_realized_bytes=max(realized),
        maximum_realized_certainty_bytes=max(certain_realized))


def controls():
    s = fixture([1,2], [1,1], 1, {'a':[1], 'b':[2]},
                [[F(1),F(0)], [F(0),F(1)], [F(1,2)]*2])
    weak = value(s, (4,0,0), channel((1,3)))
    perfect = value(s, (2,2,0), channel((1,1)))
    return {'live:imperfect_cue_value': value(s,(2,2,0),channel((2,3)))['information_value'] == [1,4],
            'positive:perfect_disclosure': perfect['informed_mass'] == perfect['full_mass'] == [1,1],
            'placebo:uninformative_cue': weak['information_value'] == [0,1],
            'live:misleading_certainty': weak['certainty_over_fixed'] == [-2,3]}


def run(root, plan, pulse):
    cfg = plan['design']
    if (cfg['source_priors'] != list(PRIORS) or cfg['policies'] != list(POLICIES)
        or cfg['mixtures'] != [list(x) for x in MIXTURES]
        or cfg['reliabilities'] != [list(x) for x in RELIABILITIES]): raise ValueError('design')
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
        pulse(phase='noisy-prior-libraries', library=len(selections))
        s = from_parent(records)
        s['mixtures'] = [dict(mixture_counts=list(c), channels=[value(s,c,channel(p)) for p in RELIABILITIES])
                         for c in MIXTURES]
        s['id'] = len(selections); selections.append(s); lookup[key] = s
    rows = []; paired = {}
    for i, r in enumerate(population):
        if i%256 == 0: pulse(phase='noisy-prior-rosters', row=i)
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
                mixture_count=15, channel_count=3))
    (root/'raw').mkdir(exist_ok=True)
    (root/'raw/noisy_prior_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    write(root/'SELECTIONS.json',selections)
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',
        evaluator='supplied source priors,mixtures and known cue channel;structural costs and rosters',
        scope='finite-library noisy prior information and mistaken certainty;no fee,forecast or learned provenance',
        raw_schema='each roster/budget row joins by selection_id to all15mixtures and all3channels'))
    return dict(controls=checks, population_rows=len(population), rows=len(rows), joined_evaluations=len(rows)*45,
        unique_selections=len(selections), distinct_channel_problems=len(selections)*45,
        candidate_count=sum(len(s['candidates']) for s in selections), numerical_acceptance=False,
        scope='gross finite-library noisy prior disclosure;not forecast accuracy or process correspondence')
