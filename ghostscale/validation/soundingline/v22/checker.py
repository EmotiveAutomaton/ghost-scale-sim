"""Independent arithmetic: no producer posterior, scoring or feature calls."""
import gzip
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..v18_3.io import read, write, file_digest


def normalized(values):
    values = np.array(values, float)
    values = np.exp(values-np.max(values))
    return values/values.sum()


def binary(prob):
    return np.array([[1-p, p] for p in prob])


def independent_a1(p):
    qs = [i*.75-3 for i in range(9)]
    qp = normalized([-.5*q*q for q in qs])
    terms = []
    def ln(x, mu, sd):
        return -math.log(sd*math.sqrt(2*math.pi))-(x-mu)**2/(2*sd*sd)
    for goal, sigma, agenda, law in itertools.product((0, 1), (.35, 1.2), (-1.5, 1.5), (.8, 1.2)):
        history = 0.
        seen = set()
        for q, c, identity in zip(p['qh'], p['ch'], p['roots']):
            if p['condition'] == 'unrelated':
                continue
            if identity not in seen:
                history += ln(c, q if goal == 0 else agenda, sigma*law)
                seen.add(identity)
        terms.append([history+math.log(prior)+ln(p['own'], q, p['sr'])+
                      ln(p['claim'], q if goal == 0 else agenda, sigma*law) for q, prior in zip(qs, qp)])
    weights = normalized(np.array(terms).ravel()).reshape(-1, 9)
    return weights.sum(0)[None, :]


def independent_b1(p):
    points = np.array(list(itertools.product((-1., 0., 1.), repeat=p['d'])))
    # Independently form all action utilities from after-state drive. The common
    # before-state drive cancels inside the softmax; no producer choice routine.
    laws = ((2, .8), (2, 3.), (4, .8), (4, 3.))
    logweights, forecasts = [], []
    for exponent, gamma in laws:
        ll = np.zeros(len(points))
        for record in p['history']+[p['current']]:
            after = np.asarray(record['h'])[None, None, :]+np.asarray(record['effects'])[None, :, :]
            distance = np.power(np.abs(after-points[:, None, :]), exponent).sum(-1)**.5
            utility = -gamma*(distance+np.asarray(record['costs']))
            prob = np.exp(utility-utility.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            if 'picked' in record:
                with np.errstate(divide='ignore'):
                    ll += np.log(prob[:, record['picked']])
            else:
                forecasts.append(prob)
        logweights.append(ll)
    weight = normalized(np.array(logweights).ravel()).reshape(4, -1)
    return np.sum(weight[:, :, None]*np.array(forecasts), axis=(0, 1))[None, :]


def independent_a2(p):
    states, likelihood = [], []
    for q, goal, technique, agenda, law in itertools.product((0, 1), repeat=5):
        rel = [.7, .9][law]
        mass = .75 if p['own'] == q else .25
        mass *= rel if p['claim'] == (q if goal == 0 else agenda) else 1-rel
        for obs in p['selection']:
            expected = goal ^ int(p['swapped'])
            mass *= (rel if obs == expected else 1-rel) if p['signature'] else .5
        for obs in p['operations']:
            mass *= rel if obs == technique else 1-rel
        states.append([q, goal, technique])
        likelihood.append(mass)
    weights = np.array(likelihood)/math.fsum(likelihood)
    return binary(weights@np.array(states))


def independent_c2(p):
    masses, targets = [], []
    for library, preference, law in itertools.product((0, 1), repeat=3):
        rel = [.7, .9][law]
        mass = 1.
        for obs in p['witnesses']:
            mass *= rel if obs == library else 1-rel
        if not p['unrelated']:
            for obs in p['records']:
                mass *= rel if obs == preference else 1-rel
        if p['coupled']:
            mass *= rel if p['preference_witness'] == preference else 1-rel
        if p['disclosed']:
            right = (library == (p['own_lib'] if p['share_lib'] else 1-p['own_lib']) and
                     preference == (p['own_pref'] if p['share_pref'] else 1-p['own_pref']))
            mass *= int(right)
        masses.append(mass)
        targets.append([rel if (p['rendered'] ^ library) else 1-rel, preference])
    weights = np.array(masses)/math.fsum(masses)
    return binary(weights@np.array(targets))


def independent_b2(p):
    masses, targets = [], []
    for route, law in itertools.product((0, 1), repeat=2):
        rate = [.2, .5][law]
        qx = qother = 0.
        mass = 1.
        for i, (action, reward) in enumerate(p['experience']):
            if action:
                qx = (1-rate)*qx+rate*reward
            else:
                qother = (1-rate)*qother+rate*reward
            value = qx-(qother if route == 0 else 0)
            if i in p['checkpoints']:
                prob = 1/(1+math.exp(-5*(value-.2)))
                observed = p['choices'][p['checkpoints'].index(i)]
                mass *= prob if observed else 1-prob
        final = 0. if p['devalued'] and route == 0 else value
        masses.append(mass)
        targets.append([route, 1/(1+math.exp(-5*(final-.2)))])
    weights = np.array(masses)/math.fsum(masses)
    return binary(weights@np.array(targets))


def independent_a3(p):
    states, masses = [], []
    for g, v, t, law in itertools.product((0, 1), repeat=4):
        rel = [.7, .9][law]
        mass = float(not p['coupled'] or v == t)
        for i in range(p['prefix']):
            mass *= rel if p['flags'][i] == g else 1-rel
            for j, target in enumerate((g, v, t)):
                if j == 2 or i >= 8:
                    mass *= rel if p['observations'][i][j] == target else 1-rel
        states.append([g, v, t])
        masses.append(mass)
    return binary((np.array(masses)/math.fsum(masses))@np.array(states))


def independent_c1(p):
    branch = p['branching']
    parents = [-1]+[(i-1)//branch for i in range(1, len(p['evidence']))]
    neighbors = [[j for j in range(len(parents)) if parents[j] == i or parents[i] == j] for i in range(len(parents))]
    law_forecasts, evidence = [], []
    for rel in (.65, .8, .95):
        def emission(i, value):
            return 1. if p['evidence'][i] < 0 else .9 if value == p['evidence'][i] else .1
        @lru_cache(None)
        def message(source, dest, dest_value):
            return math.fsum((rel if value == dest_value else 1-rel)*emission(source, value)*
                             math.prod(message(n, source, value) for n in neighbors[source] if n != dest)
                             for value in (0, 1))
        marginal = []
        for i in range(len(parents)):
            mass = [emission(i, v)*math.prod(message(n, i, v) for n in neighbors[i]) for v in (0, 1)]
            marginal.append(np.array(mass)/math.fsum(mass))
            if i == 0:
                evidence.append(math.fsum(mass)/2)
        law_forecasts.append(marginal)
    weights = np.array(evidence)/math.fsum(evidence)
    out = np.full((121, 2), .5)
    out[:len(parents)] = np.tensordot(weights, law_forecasts, axes=1)
    return out


INDEPENDENT = dict(A1=independent_a1, B1=independent_b1, C2=independent_c2,
                   A2=independent_a2, A3=independent_a3, B2=independent_b2, C1=independent_c1)


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def physical_execution(world, program):
    state, stopped, legal = [-1, -1, -1], False, True
    for action in program:
        if stopped or action not in range(10):
            legal = False
            break
        if action == 9:
            stopped = True
            continue
        part = action % 3
        attached_child = any(parent == part and state[j] >= 0 for j, parent in enumerate(world['parents']))
        if action < 3:
            parent = world['parents'][part]
            legal = state[part] < 0 and (parent < 0 or state[parent] >= 0)
            if legal:
                state[part] = world['defaults'][part]
        elif action < 6:
            legal = state[part] >= 0 and not attached_child
            if legal:
                state[part] = -1
        else:
            legal = state[part] >= 0 and not attached_child
            if legal:
                state[part] = 1-state[part]
        if not legal:
            break
    return state, stopped and legal, legal


def verify(root, destination):
    root = Path(root)
    summary = read(root/'SUMMARY.json')
    manifest = read(root/'SCIENTIFIC_MANIFEST.json')
    for name, digest in manifest.items():
        if file_digest(root/name) != digest:
            raise ValueError('scientific file differs: '+name)
    cases = forecasts = 0
    contrasts = read(root/'CONTRASTS.json')
    contrast_checks = 0
    for world in summary['worlds']:
        path = root/f'world-{world}'
        observations = records(path/'records.jsonl.gz')
        details = records(path/'details.jsonl.gz')
        per_head = {}
        if summary['card'] == 'A3':
            physical = read(path/'G5_PHYSICAL.json')
            for row in physical['baseline']['rows']+physical['withdrawal']:
                for output in row['outputs']:
                    state, stopped, legal = physical_execution(row['world'], output['plan']['program'])
                    saved = output['execution']
                    if state != saved['state'] or stopped != saved['successfully_stopped'] or legal != saved['legal']:
                        raise ValueError('physical assembly differs')
                    if output['charged_total'] > 256:
                        raise ValueError('physical consumer budget differs')
            for row in read(path/'E56_ACCOUNTING.json'):
                for metric in ('process_uptake', 'goal_uptake', 'value_uptake'):
                    if not math.isclose(row['early-guard'][metric], .2*row['no-gate'][metric], abs_tol=1e-10):
                        raise ValueError('historical per-step uniform gate accounting differs')
        with np.load(path/'inputs.npz') as arrays:
            y, mask = arrays['y'], arrays['mask']
        with np.load(path/'law-marginal.npz') as exact:
            for i, record in enumerate(observations):
                independently = INDEPENDENT[summary['card']](record['public'])
                if not np.allclose(independently, exact['p'][i], rtol=1e-8, atol=1e-10):
                    raise ValueError(f'independent posterior differs: {world}/{i}')
                if record['truth']['y'] != y[i].tolist():
                    raise ValueError('target differs from truth record')
                cases += 1
        for forecast in sorted(path.glob('*.npz')):
            if forecast.name == 'inputs.npz':
                continue
            with np.load(forecast) as a:
                p = a['p']
                if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-8):
                    raise ValueError('bad probability')
                for i in range(len(y)):
                    loss, brier, accuracy, error = [], [], [], []
                    for head in np.flatnonzero(mask[i]):
                        row = [float(v) for v in p[i, head]]
                        target = int(y[i, head])
                        loss.append(-math.log(row[target]) if row[target] else float('inf'))
                        brier.append(math.fsum(v*v for v in row)-2*row[target]+1)
                        mode = max(range(len(row)), key=row.__getitem__)
                        accuracy.append(float(mode == target))
                        error.append(float(max(row) >= .9 and mode != target))
                    for name, values in [('log_loss', loss), ('brier', brier), ('accuracy', accuracy), ('confident_error', error)]:
                        value = math.fsum(values)/len(values)
                        if not math.isclose(value, float(a[name][i]), rel_tol=1e-8, abs_tol=1e-9):
                            raise ValueError('score mismatch '+str(forecast)+' '+name)
                    forecasts += 1
                with np.errstate(divide='ignore'):
                    per_head[(forecast.stem, 'log_loss')] = -np.log(np.take_along_axis(p, y[..., None], -1)[..., 0])
                per_head[(forecast.stem, 'brier')] = np.sum(p*p, -1)-2*np.take_along_axis(p, y[..., None], -1)[..., 0]+1
                expected = next(r['metrics'] for r in summary['rows'] if r['world'] == world and r['reader'] == forecast.stem)
                for name, result in expected.items():
                    mean = np.mean(a[name])
                    if result['mean'] is None:
                        if np.isfinite(mean):
                            raise ValueError('finite summary discarded')
                    elif not math.isclose(result['mean'], float(mean), rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError('summary mismatch')
        labels = {'A1': ['property'], 'B1': ['future-choice'], 'C2': ['operation', 'preference'],
                  'A2': ['content-property', 'top-goal', 'technique'], 'A3': ['purpose', 'value', 'technique'],
                  'B2': ['acquisition-route', 'future-choice']}
        for estimate in contrasts['estimates']:
            stored = [r for r in estimate['world_values'] if r['world'] == world]
            if not stored:
                continue
            ids = [i for i, detail in enumerate(details) if all(detail[k] == v for k, v in estimate['group'].items())]
            if summary['card'] == 'C1':
                depth = int(estimate['target'].split('-')[1])
                branch = observations[ids[0]]['public']['branching']
                lo = sum(branch**d for d in range(depth))
                heads = list(range(lo, lo+branch**depth))
            else:
                heads = [labels[summary['card']].index(estimate['target'])]
            values = per_head[(estimate['reader'], estimate['metric'])][np.ix_(ids, heads)]
            expected = float(values.mean())
            actual = stored[0]['value']
            if actual is None:
                if np.isfinite(expected):
                    raise ValueError('lost finite stratum')
            elif not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('target/stratum mismatch')
            contrast_checks += 1
    receipt = dict(passed=True, card=summary['card'], independent_posteriors=cases,
                   independently_scored_forecasts=forecasts, independently_regrouped_strata=contrast_checks, files=len(manifest),
                   manifest_sha256=file_digest(root/'SCIENTIFIC_MANIFEST.json'),
                   scope='all law-marginal posteriors and all scores; rival semantics also require family controls and full replay')
    write(destination, receipt)
    return receipt


def compare_replay(original, replay, destination):
    a = read(Path(original)/'SCIENTIFIC_MANIFEST.json')
    b = read(Path(replay)/'SCIENTIFIC_MANIFEST.json')
    if a != b:
        differences = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError('scientific replay differs: '+repr(differences[:20]))
    result = dict(passed=True, files=len(a), complete_scientific_replay=True,
                  original_manifest_sha256=file_digest(Path(original)/'SCIENTIFIC_MANIFEST.json'),
                  replay_manifest_sha256=file_digest(Path(replay)/'SCIENTIFIC_MANIFEST.json'))
    write(destination, result)
    return result
