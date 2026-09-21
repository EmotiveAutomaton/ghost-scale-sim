"""Independent native execution, probability and support review of crossed laws.

No producer execution, projection, posterior or scoring routine is imported.
This consumes completed evidence and never fits or samples a new population.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, canonical, digest, file_digest
from ..v18_3.world import rng

RULES = ('original', 'presentation-tool', 'inverted-repair', 'both')
GOALS = ('meaning', 'dependency', 'presentation')
OPS = ('edit-claim', 'repair-evidence', 'replace-presentation', 'accept-tool', 'inspect', 'undo')
TIERS = ('E0', 'E1', 'E2-sparse', 'E2-full')
FAMILIES = {'base': (0,), 'tool-expanded': (0, 1), 'repair-expanded': (0, 2),
            'single-changes': (0, 1, 2), 'complete': (0, 1, 2, 3)}
METRICS = ('abstain', 'coverage', 'infinite_loss_mass', 'finite_loss_contribution', 'entropy', 'hypotheses')


def near(a, b, tol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid shape or value')
    error = float(np.max(abs(a-b))) if a.size else 0.
    if error > tol:
        raise ValueError(f'crossed-rule reconstruction mismatch: {error}')
    return error


def execute(a, previous, operation, skill, belief, rule):
    if rule not in RULES or operation not in OPS:
        raise ValueError('unknown rule or operation')
    x = list(a)
    if operation == 'edit-claim': x[0] ^= 1
    elif operation == 'repair-evidence':
        x[1] = (a[0] ^ belief ^ int(rule in ('inverted-repair', 'both'))) if skill else 1-a[1]
    elif operation == 'replace-presentation': x[2] ^= 1
    elif operation == 'accept-tool':
        if not skill: raise ValueError('illegal tool')
        if rule in ('presentation-tool', 'both'): x[2] ^= 1
        else: x[0] ^= 1; x[1] = x[0]
    elif operation == 'undo': x = list(previous)
    return tuple(x)


def packet(r, tier):
    inputs = {'artifact': r['final']}
    if tier != 'E0': inputs.update(initial=r['initial'], requested_purpose=r['requested_purpose'])
    if tier.startswith('E2'):
        events = r['steps'][:1] if tier == 'E2-sparse' else r['steps']
        inputs['observations'] = [{k:e[k] for k in ('step', 'operation', 'before', 'after', 'tool_proposal')} for e in events]
    return dict(schema='v19.local.public.1', tier=tier, inputs=inputs)


def score(truth, prediction):
    near(sum(truth.values()), 1.)
    if prediction: near(sum(prediction.values()), 1.)
    missing = sum(p for k, p in truth.items() if prediction.get(k, 0.) == 0.)
    return dict(abstain=float(not prediction), coverage=1.-missing, infinite_loss_mass=missing,
        finite_loss_contribution=-sum(p*math.log(prediction[k]) for k,p in truth.items() if prediction.get(k, 0.) > 0.),
        entropy=-sum(p*math.log(p) for p in prediction.values()), hypotheses=len(prediction))


def union(groups, key, indices):
    weights = defaultdict(float)
    for i in indices:
        for process, value in groups[i].get(key, {}).items(): weights[process] += value
    mass = sum(weights.values())
    return {k:v/mass for k,v in weights.items()} if mass else {}


def controls():
    truth = {'a': .25, 'b': .75}; known = score(truth, truth); absent = score(truth, {})
    corrupt = False
    try: near([.25, .75], [.75, .25])
    except ValueError: corrupt = True
    alias = all(execute(a, a, 'repair-evidence', 1, b, 'both') == execute(a, a, 'repair-evidence', 1, 1-b, 'presentation-tool')
                for a in product(range(2), repeat=3) for b in (0, 1))
    return {'live:changed_execution': execute((0,1,0),(0,1,0),'accept-tool',1,0,'both') == (0,1,1),
        'positive:known_entropy': abs(known['finite_loss_contribution']+.25*math.log(.25)+.75*math.log(.75)) < 1e-14,
        'positive:missing_mass': score(truth, {'a':1.})['infinite_loss_mass'] == .75,
        'positive:empty_abstention': absent['abstain'] == 1 and absent['infinite_loss_mass'] == 1,
        'positive:belief_alias': alias, 'positive:corruption_rejected': corrupt,
        'positive:undo': execute((0,1,0),(1,0,1),'undo',1,0,'both') == (1,0,1),
        'placebo:identical_family': union([{'x':truth},{'x':truth}], 'x', (0,1)) == truth}


def population(records, lineage, rule):
    random = rng('v19-local-world', lineage)
    strength, rate, routine_weight = random.uniform(.8,1.2), random.uniform(.65,.85), random.uniform(.1,.3)
    makers = tuple(product(range(2), repeat=4)); seen = defaultdict(set); masses = defaultdict(float)
    groups = {tier:defaultdict(lambda:defaultdict(float)) for tier in TIERS}; packets = {}; error = 0.
    if len(records) != 13824: raise ValueError('incomplete population')
    for r in records:
        ci, mi = r['context_index'], r['maker_index']; purpose, skill, belief, routine = makers[mi]
        initial = ((0,1,0),(1,0,1))[ci//2]; request = ci%2
        assert r['maker'] == list(makers[mi]) and r['initial'] == list(initial) and r['requested_purpose'] == request
        a = previous = initial; probability = 1/64; sequence = []
        if len(r['steps']) != 3: raise ValueError('wrong horizon')
        for step, e in enumerate(r['steps']):
            weights = np.array([1.5 if purpose == 0 else .6, 1.4 if a[0]^belief != a[1] else .5, 1.8 if purpose == 1 else .5])
            weights[2 if routine else 0] += routine_weight; weights[2 if request else 0] += .2
            weights = weights**strength
            gi = GOALS.index(e['goal']); op = e['operation']
            action = ('accept-tool' if skill else 'edit-claim', 'repair-evidence', 'replace-presentation')[gi]
            alternative = 'undo' if gi == 1 and step == 2 else 'inspect'
            assert op in (action, alternative)
            probability *= weights[gi]/weights.sum()*(rate if op == action else 1-rate)
            after = execute(a, previous, op, skill, belief, rule)
            assert e == dict(step=step, goal=GOALS[gi], operation=op, before=list(a), after=list(after), undo_buffer=list(previous),
                dependency_edges=[[0,1]], tool_proposal=list(after) if op == 'accept-tool' else None, perceived_claim=a[0]^belief)
            previous, a = a, after; sequence.append(2*gi+int(op == alternative))
        assert list(a) == r['final'] and tuple(sequence) not in seen[ci,mi]
        seen[ci,mi].add(tuple(sequence)); masses[ci,mi] += probability
        error = max(error, near(probability, r['probability'], 1e-14))
        identity = (tuple(e['goal'] for e in r['steps']), tuple(e['operation'] for e in r['steps']))
        for tier in TIERS:
            p = packet(r, tier); key = digest(p); packets[key] = p
            groups[tier][key][identity] += probability
    assert len(seen) == 64 and all(len(v) == 216 for v in seen.values())
    near(list(masses.values()), np.full(64,1/64), 1e-12)
    return groups, packets, error


def distribution(serialized):
    result = {(tuple(g),tuple(o)):p for g,o,p in serialized}
    if len(result) != len(serialized): raise ValueError('duplicate process')
    return result


def compare_distribution(a, b):
    if set(a) != set(b): raise ValueError('support mismatch')
    return near([a[k] for k in sorted(a)], [b[k] for k in sorted(a)])


def run(root, plan, pulse):
    checks = controls(); write(root/'CONTROLS.json', checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg = plan['design']; original = root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('review input changed')
    original_plan = read(original/'PLAN.json'); target = read(original/'SUMMARY.json')
    assert file_digest(original/'PLAN.json') == cfg['target_plan_sha256']
    lineages = original_plan['design']['lineages']; all_cells = []; public = {}; max_error = 0.; paths = rows_count = 0
    for lineage in lineages:
        groups = {tier:[] for tier in TIERS}
        for rule in RULES:
            pulse(phase='independent-crossed-population', lineage=lineage, rule=rule)
            records = json.loads(gzip.decompress((original/'raw'/f'{lineage}-{rule}_points.json.gz').read_bytes()))
            by_tier, packets, error = population(records, lineage, rule)
            max_error = max(max_error, error); public.update(packets); paths += len(records)
            for tier in TIERS: groups[tier].append(by_tier[tier])
        for tier in TIERS:
            pulse(phase='independent-crossed-posterior', lineage=lineage, tier=tier)
            saved_scores = json.loads(gzip.decompress((original/'raw'/f'{lineage}-{tier}-scores_points.json.gz').read_bytes()))
            saved_posts = json.loads(gzip.decompress((original/'raw'/f'{lineage}-{tier}-posterior_points.json.gz').read_bytes()))
            scores = {(r['truth_rule'],r['frame'],r['arm']):r for r in saved_scores}
            posts = {(r['truth_rule'],r['frame'],r['arm']):r for r in saved_posts}
            assert len(scores) == len(saved_scores) == len(posts) == len(saved_posts)
            consumed = set(); accum = defaultdict(lambda:np.zeros(len(METRICS)+1)); frames = defaultdict(int)
            for ti, rule in enumerate(RULES):
                for key, mass_table in groups[tier][ti].items():
                    mass = sum(mass_table.values()); truth = {k:v/mass for k,v in mass_table.items()}
                    for arm in (*FAMILIES, 'known-rule-oracle'):
                        identity = (rule,key,arm); pred = union(groups[tier],key,(ti,) if arm == 'known-rule-oracle' else FAMILIES[arm])
                        post, old = posts[identity], scores[identity]; consumed.add(identity)
                        max_error = max(max_error, compare_distribution(truth,distribution(post['truth'])), compare_distribution(pred,distribution(post['prediction'])))
                        metrics = score(truth,pred); max_error = max(max_error,near([mass]+[metrics[k] for k in METRICS], [old['probability_mass']]+[old[k] for k in METRICS]))
                        accum[rule,arm] += mass*np.array([1.]+[metrics[k] for k in METRICS]); frames[rule,arm] += 1
                        rows_count += 1
            assert consumed == set(scores) == set(posts)
            for (rule,arm), values in accum.items():
                near(values[0],1.,1e-10)
                all_cells.append(dict(lineage=lineage,truth_rule=rule,tier=tier,arm=arm,frames=frames[rule,arm],**dict(zip(METRICS,map(float,values[1:]/values[0])))))
    assert public == read(original/'PUBLIC_PACKET.json')['frames']
    key = lambda r:(r['lineage'],r['truth_rule'],r['tier'],r['arm'])
    a,b = {key(r):r for r in all_cells}, {key(r):r for r in target['cells']}
    assert set(a) == set(b) and len(b) == len(target['cells'])
    for k in a:
        assert a[k]['frames'] == b[k]['frames']
        max_error = max(max_error,near([a[k][m] for m in METRICS],[b[k][m] for m in METRICS]))
    contrasts = []
    for rule,tier,arm in product(RULES,TIERS,('base','tool-expanded','repair-expanded','single-changes','known-rule-oracle')):
        for metric in ('coverage','infinite_loss_mass','finite_loss_contribution','entropy'):
            values = np.array([a[l,rule,tier,arm][metric]-a[l,rule,tier,'complete'][metric] for l in lineages])
            random = np.random.default_rng(cfg['bootstrap_seed']); bs=values[random.integers(0,len(values),(cfg['bootstrap_resamples'],len(values)))].mean(1)
            finite = all(a[l,rule,tier,x]['infinite_loss_mass'] < 1e-14 for l in lineages for x in (arm,'complete'))
            contrasts.append(dict(truth_rule=rule,tier=tier,arm=arm,metric=metric,mean=float(values.mean()),low=float(np.quantile(bs,.025)),high=float(np.quantile(bs,.975)),lineage_values=values.tolist(),finite_expected_loss_contrast=finite))
    write(root/'INDEPENDENT_REGROUP.json',dict(cells=all_cells,contrasts=contrasts,population='native weights within lineage; equal paired coefficient lineages; no fit or seed replication'))
    checks['positive:complete_reconstruction'] = True
    return dict(controls=checks,paths=paths,rows=rows_count,cells=len(all_cells),anonymous_packets=len(public),max_error=max_error,
        target_plan_sha256=cfg['target_plan_sha256'],scope='independent full transient execution, probability, joint support and score reconstruction; finite loss contributions remain separate from infinite expected loss; event adjudication pending')
