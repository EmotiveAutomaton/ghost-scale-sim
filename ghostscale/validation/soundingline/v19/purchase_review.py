"""Independent R3 reconstruction from complete prefix products and saved evidence.

No purchase, calibration, or distinct-family reader/scorer is imported. The native
world is the stipulated likelihood ruler; its generative assumptions are not tested.
"""
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import read, write, file_digest

METHODS = ('fixed', 'select', 'mixture', 'expanded-select', 'expanded-mixture',
           'paid-raw-1.5', 'paid-raw-3', 'calibrated-raw', 'calibrated-centered')
METRICS = ('expected_loss', 'final_loss', 'expected_match', 'expanded',
           'purchase_step', 'likelihood_evaluations', 'net_match_low',
           'net_match_high', 'model_entropy')


def close(actual, expected, tolerance=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid reconstruction shape or value')
    error = float(np.max(np.abs(a - b))) if a.size else 0.
    if error > tolerance:
        raise ValueError(f'independent reconstruction differs: {error}')
    return error


def joint(likelihoods, priors, prefix):
    """Rebuild from scratch; do not recurse on the producer's saved posterior."""
    models, _, states = likelihoods.shape
    logs = np.array([[math.log(priors[k] / states) + math.fsum(
        math.log(float(likelihoods[k, j, s])) for j in range(prefix))
        for s in range(states)] for k in range(models)])
    mass = np.exp(logs - logs.max())
    return mass / math.fsum(mass.flat)


def forecast(mass, laws, select=False):
    class_mass = np.array([math.fsum(row) for row in mass])
    used = mass.copy()
    if select:
        chosen = int(np.argmax(class_mass))
        used[np.arange(len(used)) != chosen] = 0
        used /= math.fsum(used.flat)
    return np.sum([used[k] @ laws[k] for k in range(len(used))], axis=0), class_mass


def entropy(probabilities):
    return -math.fsum(float(p) * math.log(float(p)) for p in probabilities if p > 0)


def loss(target, prediction):
    return -math.fsum(float(a) * math.log(float(b))
                      for a, b in zip(target, prediction) if a > 0)


def crossing(values, threshold):
    return next((8 + j for j, v in enumerate(values) if v > threshold), 33)


def prices(match, expanded, work):
    return (match - (expanded + .00001 * work) / 32,
            match - (4 * expanded + .0001 * work) / 32)


def controls():
    likelihoods = np.array([[[.25, .75], [.5, .5]]])
    p = joint(likelihoods, [1.], 1)
    uniform = joint(np.full((1, 2, 2), .5), [1.], 2)
    mixed = np.array([[.45, .05], [.1, .4]])
    laws = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    pred, _ = forecast(mixed, laws)
    selected, _ = forecast(mixed, laws, True)
    corrupt = False
    try:
        close([.5, .5], [.4, .6])
    except ValueError:
        corrupt = True
    low, high = prices(.5, 0, 1536)
    paid_low, paid_high = prices(.5, 1, 2304)
    return {
        'live:known_posterior': bool(np.allclose(p, [[.25, .75]], atol=1e-14)),
        'placebo:uniform_likelihood': bool(np.array_equal(uniform, [[.5, .5]])),
        'positive:known_mixture': bool(np.allclose(pred, [.85, .15], atol=1e-14)),
        'positive:selection_tie': bool(np.allclose(selected, [.9, .1], atol=1e-14)),
        'live:known_score': abs(loss([1., 0.], [.25, .75]) - math.log(4)) < 1e-14,
        'placebo:strict_threshold_tie': crossing([1.] * 24, 1.) == 33,
        'live:past_only_first_crossing': crossing([1., 2.] + [0.] * 22, 1.) == 9,
        'positive:inert_purchase_prices': abs(low - paid_low - (1 + .00001 * 768) / 32) < 1e-14
            and abs(high - paid_high - (4 + .0001 * 768) / 32) < 1e-14,
        'positive:corruption_rejected': corrupt,
    }


def verify_unit(unit, public, thresholds):
    assert set(public) == {'world', 'history', 'queries', 'prior_mode'}
    history = public['history']
    assert len(history) == 32 and len({o['source'] for o in history}) == 32
    for obs in history:
        assert set(obs) == {'context', 'program', 'artifact', 'source'}
        assert W.execute(tuple(obs['program'])).artifact == obs['artifact']
    assert public['prior_mode'] == unit['prior_mode']
    w = public['world']
    worlds = [w, dict(w, rule='satisficing' if w['rule'] == 'softmax' else 'softmax'),
              dict(w, rule='lexicographic')]
    prior = [2., 1., 1.] if unit['prior_mode'] == 'duplicate-supplied' else [1., 1., 1.]
    programs = np.array([[W.matrix(v, o['context']) for o in history] for v in worlds])
    observed = [W.PROGRAMS.index(tuple(o['program'])) for o in history]
    likelihoods = np.stack([programs[:, t, :, a] for t, a in enumerate(observed)], axis=1)
    queries = public['queries'] + list(W.FUTURES)
    query_keys = [json.dumps(q, sort_keys=True, separators=(',', ':')) for q in queries]
    unique_queries = dict(zip(query_keys, queries))
    laws = {key: np.array([W.artifact_matrix(v, q) for v in worlds]) for key, q in unique_queries.items()}
    truth = {key: W.artifact_matrix(unit['evaluator']['world'], q)[unit['evaluator']['state']]
             for key, q in unique_queries.items()}
    masses = {n: [joint(likelihoods[:n], prior[:n], t) for t in range(33)] for n in (1, 2, 3)}
    sensor_predictions = [forecast(masses[2][t], programs[:2, t])[0] for t in range(32)]
    surprise = [-math.log(float(p[a])) for p, a in zip(sensor_predictions, observed)]
    entropies = [entropy(p) for p in sensor_predictions]
    values = {key: [math.fsum(surprise[j] - (entropies[j] if key == 'centered' else 0.)
               for j in range(t - 4, t)) / 4 for t in range(8, 32)] for key in ('raw', 'centered')}
    assert [r['method'] for r in unit['rows']] == list(METHODS)
    result = []
    max_error = 0.
    for row in unit['rows']:
        method = row['method']
        paid = method.startswith(('paid-', 'calibrated-'))
        sensor = row['sensor']
        if paid:
            max_error = max(max_error, close(sensor['predictions'], sensor_predictions))
            close(sensor['surprises'], surprise)
            close(sensor['entropies'], entropies)
            close(row['sensor_surprises'], surprise)
            for key in values:
                close(sensor['values'][key], values[key])
                close(sensor['maxima'][key], max(values[key]))
            assert sensor['likelihood_evaluations'] == 1536
            key = 'centered' if method == 'calibrated-centered' else 'raw'
            threshold = float(method.rsplit('-', 1)[1]) if method.startswith('paid-') else thresholds[key]
            assert row['sensor_kind'] == key and row['threshold'] == threshold
            # Exact identity uses frozen arithmetic after independent tolerance validation.
            stored = sensor['values'][key]
            purchase = crossing(stored, threshold)
            near = sum(abs(v - threshold) <= 1e-12 for v in stored)
            assert row['near_threshold_opportunities'] == near
            sensitivity = {label: int(crossing(stored, threshold + offset) < 33)
                           for label, offset in (('minus_1e12', -1e-12), ('unchanged', 0.), ('plus_1e12', 1e-12))}
        else:
            assert sensor is None and row['threshold'] is None and row['sensor_kind'] is None
            assert row['near_threshold_opportunities'] == 0
            assert row['sensor_surprises'] == []
            purchase = 0 if method.startswith('expanded-') else 33
            near = 0
            sensitivity = {}
        assert row['purchase_step'] == purchase
        assert row['expanded'] == float(purchase < 33)
        assert len(row['trace']) == 32 and len(row['final']) == 4
        losses, matches, model_entropies, final_losses = [], [], [], []
        for t, item in enumerate(row['trace'] + row['final']):
            count = 1 if method == 'fixed' else 2 + int(min(t, 32) >= purchase)
            mass = masses[count][min(t, 32)]
            q = queries[t]
            qkey = query_keys[t]
            assert item['query'] == q
            p, model_weights = forecast(mass, laws[qkey][:count], method == 'fixed' or 'select' in method)
            max_error = max(max_error, close(item['prediction'], p))
            # Scores use the verified stored probabilities: preserve their exact tie decisions.
            saved = np.asarray(item['prediction'])
            assert (saved >= 0).all() and abs(math.fsum(saved) - 1) < 1e-10
            expected_loss = loss(truth[qkey], saved)
            close(item['expected_loss'], expected_loss)
            if t < 32:
                assert item['source'] == history[t]['source'] and item['step'] == t
                assert item['expanded'] == (t >= purchase)
                close(item['model_weights'], model_weights)
                legal = sorted(set(map(int, W.ARTIFACTS)))
                chosen = min(a for a in legal if saved[a] >= max(saved[legal]) - 1e-10)
                canonical = min((p for p in W.PROGRAMS if W.execute(p).artifact == chosen), key=lambda p: (len(p), p))
                assert item['program'] == list(canonical) and item['constructed_artifact'] == chosen
                close(item['expected_match'], truth[qkey][chosen])
                losses.append(expected_loss)
                matches.append(float(truth[qkey][chosen]))
                model_entropies.append(entropy(item['model_weights']))
            else:
                final_losses.append(expected_loss)
        count = 1 if method == 'fixed' else 2 + int(purchase < 33)
        close(row['class_priors'], prior[:count])
        for k in range(count):
            logs = [-math.log(24) + math.fsum(math.log(float(v)) for v in likelihoods[k, :, s]) for s in range(24)]
            top = max(logs)
            total = math.fsum(math.exp(v - top) for v in logs)
            close(row['final_state_weights'][k], [math.exp(v - top) / total for v in logs])
            close(row['final_log_evidence'][k], top + math.log(total))
        predictor = count * 24 * 32
        sensor_work = 1536 if paid else 0
        assert row['predictor_likelihood_evaluations'] == predictor
        assert row['sensor_likelihood_evaluations'] == sensor_work
        assert row['sensor_probability_terms'] == sensor_work * len(W.PROGRAMS)
        work = predictor + sensor_work
        assert row['likelihood_evaluations'] == work
        match = math.fsum(matches) / 32
        low, high = prices(match, float(purchase < 33), work)
        metrics = dict(expected_loss=math.fsum(losses) / 32, final_loss=math.fsum(final_losses) / 4,
                       expected_match=match, expanded=float(purchase < 33), purchase_step=purchase,
                       likelihood_evaluations=work, net_match_low=low, net_match_high=high,
                       model_entropy=math.fsum(model_entropies) / 32)
        for key, value in metrics.items():
            close(row[key], value)
        result.append(dict(method=method, **metrics, near_threshold_opportunities=near,
                           purchase_sensitivity=sensitivity))
    return result, max_error


def aggregate(rows, cfg):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('cell', 'kind', 'order', 'prior', 'method'))
        groups.setdefault(key, []).append(row)
    strata = [dict(zip(('cell', 'kind', 'order', 'prior', 'method'), key), streams=len(v),
                   **{m: math.fsum(r[m] for r in v) / len(v) for m in METRICS},
                   near_threshold_opportunities=sum(r['near_threshold_opportunities'] for r in v))
              for key, v in sorted(groups.items())]
    lookup = {tuple(r[k] for k in ('index', 'cell', 'kind', 'order', 'prior', 'method')): r for r in rows}
    # Reuse the same resampled coefficient indices for every paired contrast.
    draws = np.random.default_rng(190501).integers(0, len(cfg['evaluation_indices']), (10000, len(cfg['evaluation_indices'])))
    means, contrasts = [], []
    for kind, prior in product(cfg['kinds'], cfg['priors']):
        per_method = {}
        for method in METHODS:
            per_index = []
            for index in cfg['evaluation_indices']:
                paired = [lookup[index, cell, kind, order, prior, method]
                          for cell, order in product(cfg['cells'], cfg['orders'])]
                per_index.append([math.fsum(r[m] for r in paired) / len(paired) for m in METRICS])
            per_method[method] = np.asarray(per_index)
            means.append(dict(kind=kind, prior=prior, method=method, indices=len(per_index),
                              **dict(zip(METRICS, np.mean(per_index, axis=0).tolist()))))
        for a, b in (('calibrated-centered', 'calibrated-raw'),
                     ('calibrated-centered', 'mixture'), ('calibrated-raw', 'mixture')):
            delta = per_method[a] - per_method[b]
            samples = delta[draws].mean(axis=1)
            contrasts.append(dict(kind=kind, prior=prior, first=a, second=b,
                lineage_values=delta.tolist(), metrics={m: dict(mean=float(delta[:, j].mean()),
                    interval95=np.quantile(samples[:, j], [.025, .975]).tolist()) for j, m in enumerate(METRICS)}))
    return dict(strata=strata, equal_cell_order_means=means, paired_contrasts=contrasts,
                metrics=list(METRICS), bootstrap_resamples=10000, bootstrap_seed=190501,
                uncertainty='twenty paired coefficient indices; cells and orders averaged within index, truth and prior kept separate; conditional on fixed calibration and supplied-law family')


def run(out, plan, pulse):
    checks = {k: bool(v) for k, v in controls().items()}
    write(out / 'CONTROLS.json', checks)
    if not all(checks.values()):
        raise ValueError('independent known-answer controls failed before outcome access')
    for name, h in plan['design']['input_files'].items():
        if file_digest(out / 'inputs' / name) != h:
            raise ValueError('frozen verification input changed: ' + name)
    root = out / 'inputs/original'
    target, done = read(root / 'PLAN.json'), read(root / 'COMPLETE.json')
    assert file_digest(root / 'PLAN.json') == done['plan_sha256'] == plan['design']['target_plan_sha256']
    assert file_digest(root / 'COMPLETE.json') == plan['design']['target_complete_sha256']
    for name, h in {**done['files'], **done.get('execution_measurements', {})}.items():
        assert file_digest(root / name) == h, name
    cfg = target['design']
    assert cfg['evaluation_indices'] == list(range(197000, 197020))
    assert cfg['cells'] == [0, 2, 8, 10] and cfg['methods'] == list(METHODS)
    bars = read(root / 'inputs/THRESHOLDS.json')
    assert file_digest(root / 'inputs/THRESHOLDS.json') == cfg['thresholds_sha256']
    mapping = {(b['cell'], b['order'], b['prior'], b['sensor']): b['threshold'] for b in bars['strata']}
    manifest = read(root / 'READER_MANIFEST.json')['files']
    rows, seen, max_error = [], set(), 0.
    for index in cfg['evaluation_indices']:
        pulse(phase='independent-purchase-reconstruction',completed_streams=len(seen))
        units = json.loads(gzip.decompress((root / 'raw' / f'evaluation-{index}_points.json.gz').read_bytes()))
        for unit in units:
            key = tuple(unit[k] for k in ('index', 'cell', 'kind', 'order', 'prior_mode'))
            assert unit['index'] == index and key not in seen
            seen.add(key)
            path = 'reader/' + unit['public_sha256'] + '.json'
            assert manifest[path] == unit['public_sha256'] == file_digest(root / path)
            thresholds = {s: mapping[unit['cell'], unit['order'], unit['prior_mode'], s] for s in ('raw', 'centered')}
            verified, error = verify_unit(unit, read(root / path), thresholds)
            max_error = max(max_error, error)
            rows.extend(dict(index=index, cell=unit['cell'], kind=unit['kind'], order=unit['order'],
                             prior=unit['prior_mode'], **r) for r in verified)
    assert seen == set(product(cfg['evaluation_indices'], cfg['cells'], cfg['kinds'], cfg['orders'], cfg['priors']))
    result = aggregate(rows, cfg)
    original = {tuple(r[k] for k in ('cell', 'kind', 'order', 'prior', 'method')): r for r in read(root / 'STRATA.json')['strata']}
    assert len(result['strata']) == len(original) == 864
    for r in result['strata']:
        saved = original[tuple(r[k] for k in ('cell', 'kind', 'order', 'prior', 'method'))]
        assert r['streams'] == saved['streams'] == 20
        for m in METRICS:
            close(r[m], saved[m])
    write(out / 'INDEPENDENT_REGROUP.json', result)
    (out / 'review_points.json.gz').write_bytes(gzip.compress(
        json.dumps(rows, sort_keys=True, separators=(',', ':')).encode(), mtime=0))
    checks['positive:every_forecast_sensor_action_cost_and_stratum'] = True
    return dict(controls=checks, streams=len(seen), methods=len(METHODS), rows=len(rows),
                strata=len(result['strata']), paired_contrasts=len(result['paired_contrasts']),
                max_probability_error=max_error, thresholds_sha256=cfg['thresholds_sha256'],
                target_plan_sha256=plan['design']['target_plan_sha256'],
                scope='independent constructed-method numerical review; replay comparison and scientific interpretation remain event-review duties; no new fit or population')
