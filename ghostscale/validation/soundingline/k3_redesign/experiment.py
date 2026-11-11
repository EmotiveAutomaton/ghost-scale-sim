"""Frozen excluded fixtures and all-pair checks for K3-RD1; no production lane."""
import itertools
import math
import numpy as np
from ..v27 import registry, setpoints, setpoint_resolution
from ..v27.common import posteriors, margin, read, write, digest
from . import filter as adapted, reference

CALIBRATION = (0, 1, 5, 8, 11, 14)
COUNTS = (1024, 4096, 16384)
LIMITS = dict(setpoint_resolution.LIMITS, step=1e-12, reference_path=1e-9)


def first_deficit(raw, setting):
    """Evaluator-only, pre-update edge; P0/P+ need their own clock counter."""
    edge = np.asarray(raw['point'], dtype=float)-setting['width']/2
    streak = np.zeros(setting['needs'], dtype=int)
    first = None
    for t, row in enumerate(raw['records']):
        streak = np.where(np.asarray(row['stock']) < edge, streak+1, 0)
        if first is None and np.any(streak >= 20):
            first = t
        edge = np.asarray(row['edge'])
    if setting['dynamic'] in ('S', 'P-', 'G') and first != raw['first_long_deficit']:
        raise ArithmeticError('successor event clock changed an existing dynamic clock')
    return first


def fixture(index, lane):
    setting = registry.primary_settings('K3')[index]
    identity = (['excluded-dynamic-resolution-20261107', index] if lane == 'old'
                else ['excluded-k3-rd1-validation-20261110', index])
    if lane not in ('old', 'fresh') or index not in range(15):
        raise ValueError('unregistered excluded fixture')
    raw = setpoints.generate(setting | dict(inference='checked_particles'), identity)
    return setting, raw, first_deficit(raw, setting)


def reconstruct_paths(hypotheses, log_rows, independent=False):
    logs = np.asarray(log_rows).T
    prior = np.full(len(hypotheses), 1/len(hypotheses))
    p = posteriors(logs, prior, independent)
    marginal = np.column_stack([p[:, [law == d for _, law in hypotheses]].sum(axis=1)
                                for d in adapted.LAWS])
    contrasts = {}
    for law in ('S', 'P-', 'G'):
        yes = np.array([d == law for _, d in hypotheses])
        no = np.array([d in ('S', 'P-', 'G') and d != law for _, d in hypotheses])
        contrasts[law] = margin(logs, prior, yes, no)
    if not np.isfinite(marginal).all() or not all(np.isfinite(x).all() for x in contrasts.values()):
        raise ArithmeticError('nonfinite numerical comparison; no invented likelihood floor')
    return dict(posterior=marginal.tolist(), margins={k: v.tolist() for k, v in contrasts.items()})


def paths(index, lane, count, replicate, destination, pulse=lambda **kw: None, independent=False):
    setting, raw, begin = fixture(index, lane)
    observed = setpoint_resolution.hidden_observations(raw)
    hypotheses = list(itertools.product(itertools.product((8, 11, 14), repeat=setting['needs']), adapted.LAWS))
    logs, bindings = [], []
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination/'fixture_points.json'
    if raw_path.exists():
        if read(raw_path) != raw:
            raise ValueError('retained excluded fixture changed')
    else:
        write(raw_path, raw)
    for h, (point, law) in enumerate(hypotheses):
        path = destination/f'hypothesis-{h:03d}.json'
        if path.exists():
            # Checkpoints are immutable and bound to the job plan by the worker.
            row = read(path)
            if row['point'] != list(point) or row['law'] != law:
                raise ValueError('hypothesis checkpoint identity changed')
        else:
            if law == 'P0':
                ll, peak = setpoints.likelihood(observed, setting | dict(inference='exact'),
                    point, law, setting['width'], setting['beta'], independent=independent)
                diagnostics = dict(exact_static=True, peak_states=peak)
            else:
                ll, diagnostics = adapted.likelihood(observed, setting, point, law, count,
                    replicate, independent=independent, pulse=pulse)
            row = dict(point=list(point), law=law, prefix_log_likelihood=ll.tolist(), diagnostics=diagnostics)
            write(path, row)
        logs.append(row['prefix_log_likelihood'])
        bindings.append(dict(file=path.name, sha256=digest(path)))
        pulse(hypothesis=h+1, hypotheses=len(hypotheses), fixture=index, lane=lane)
    result = dict(index=index, lane=lane, particles=count, replicate=replicate,
        first_long_deficit=begin, scientific_draws=0, configuration=setting,
        paths=reconstruct_paths(hypotheses, logs, independent), hypotheses=bindings,
        raw_sha256=digest(raw_path))
    write(destination/'RESULT.json', result)
    return result


def as_arrays(paths):
    return dict(posterior=np.asarray(paths['posterior']),
                margins={k: np.asarray(v) for k, v in paths['margins'].items()})


def all_pairs(rows):
    checks = []
    for a, b in itertools.combinations(rows, 2):
        if (a['index'], a['lane'], a['first_long_deficit']) != (b['index'], b['lane'], b['first_long_deficit']):
            raise ValueError('incompatible numerical fixture pair')
        comparison = setpoint_resolution.compare(as_arrays(a['paths']), as_arrays(b['paths']),
                                                  a['first_long_deficit'] or 0)
        checks.append(dict(left=[a['particles'], a['replicate']], right=[b['particles'], b['replicate']], **comparison))
    return dict(passed=bool(checks) and all(x['passed'] for x in checks), pairs=checks,
                maximum_posterior_error=max((x['posterior_error'] for x in checks), default=None),
                maximum_margin_error=max((x['margin_error'] for x in checks), default=None))


def controls(pulse=lambda **kw: None):
    """Record gates before asserting: failures are retained by the controller."""
    gates = []
    def record(name, passed, **details):
        gates.append(dict(name=name, passed=bool(passed), **details))
        pulse(control=name)
    s = dict(needs=2, dynamic='P0', beta=0., width=0, consumption=3., probe=0., lapse=.02, records=3)
    worst = 0.
    for needs in (1, 2, 3):
        states = np.array([[0., 20., 7.][:needs]+[8., 11., 14.][:needs]+[.1, 10., 1.][:needs]+[19., 20., 0.][:needs],
                           [20., 0., 15.][:needs]+[8., 11., 14.][:needs]+[1., .10001, 9.99][:needs]+[20., 19., 20.][:needs]])
        losses, probability = adapted.branch_law(needs, 3.)
        for law, shock, beta in itertools.product(range(5), (-1, 0), (0., .5)):
            args = (states, 0, 5., 0, 5., shock, law, 2., beta, .02, losses, probability)
            x, w = adapted.candidates(*args)
            y, v = reference.enumerate_candidates(*args)
            worst = max(worst, float(np.max(abs(x-y))), float(np.max(abs(w-v))))
    record('independent_transition_weights', worst <= 1e-12, maximum_error=worst,
           cases='1/2/3 needs, five laws, shock/clipping, beta 0/.5, precision limits, twentieth deficit')
    for law in adapted.LAWS:
        setting = s | dict(dynamic=law)
        raw = setpoints.generate(setting | dict(inference='checked_particles'), ['k3-rd1-exact-prefix', law])
        obs = setpoint_resolution.hidden_observations(raw)
        exact, peak = reference.exact(obs, setting, (8, 11), adapted.LAWS.index(law))
        old, _ = setpoints.likelihood(obs, setting, (8, 11), law, 0, 0.)
        error = float(np.max(abs(exact-old)))
        record('independent_exact_prefix_'+law, error <= 1e-12, maximum_error=error, peak_states=peak)
        apf, _ = adapted.likelihood(obs, setting, (8, 11), law, 4096)
        scalar, _ = adapted.likelihood(obs, setting, (8, 11), law, 4096, independent=True)
        record('same_rng_reference_path_'+law, float(np.max(abs(apf-scalar))) <= 1e-9,
               maximum_error=float(np.max(abs(apf-scalar))))
    # Every step is shocked to zero. Hidden loss has no influence on any choice;
    # the entire dynamic history, including caps/floors, is analytically known.
    for law in adapted.LAWS:
        setting = s | dict(needs=1, dynamic=law, records=256)
        obs = dict(inference_key=['k3-rd1-reset', law], records=[dict(action=1, size=0., shock=0)]*256)
        state = np.array([0., 11., 1., 0.])
        expected, acc = [], 0.
        for row in obs['records']:
            state, p = reference.observe_scalar.py_func(state, 1, 0., 0, adapted.LAWS.index(law), 0., 0., .02)
            acc += math.log(p)
            expected.append(acc)
        ll, _ = adapted.likelihood(obs, setting, (11,), law, 64)
        error = float(np.max(abs(ll-expected)))
        record('long_reset_absolute_'+law, error <= 1e-9, maximum_error=error, records=256)
    # Static absolute controls are distinct from the exact deployed P0 reader.
    for k, point in enumerate(((8, 8), (11, 14), (14, 8))):
        setting = s | dict(records=12)
        raw = setpoints.generate(setting | dict(inference='checked_particles'), ['excluded-static-resolution', k])
        obs = setpoint_resolution.hidden_observations(raw)
        exact, _ = setpoints.likelihood(obs, setting, point, 'P0', 0, 0.)
        ll, _ = adapted.likelihood(obs, setting, point, 'P0', 16384)
        error = float(np.max(abs(ll-exact)))
        record('static_absolute_'+str(k), error <= .1, maximum_error=error)
    for index in range(15):
        setting, raw, first = fixture(index, 'old')
        record('evaluator_clock_'+str(index), setting['dynamic'] not in ('S', 'P-', 'G') or first == raw['first_long_deficit'],
               old=raw['first_long_deficit'], successor=first)
    impossible = dict(inference_key=['k3-rd1-zero'], records=[dict(action=2, size=1., shock=-1)])
    rejected = False
    try:
        adapted.likelihood(impossible, s, (8, 11), 'S', 32)
    except ArithmeticError:
        rejected = True
    record('placebo_impossible_observation_rejected', rejected)
    truth_rejected = False
    try:
        adapted.likelihood(impossible | dict(point=[8, 11]), s, (8, 11), 'S', 32)
    except ValueError:
        truth_rejected = True
    record('evaluator_truth_channel_rejected', truth_rejected)
    # Visible-stock ceiling reconstructs known stock, separate from hidden input.
    setting = s | dict(dynamic='S', records=32)
    raw = setpoints.generate(setting, ['k3-rd1-visible-ceiling'])
    stock_state = np.array([0., 0., 8., 11., 1., 1., 0., 0.])
    acc, expected = 0., []
    for row in raw['records']:
        stock_state[:2] = row['stock']
        stock_state, cp = reference.observe_scalar.py_func(stock_state, row['action'], row['size'], -1, 3, 0., 0., .02)
        acc += math.log(cp)
        expected.append(acc)
    ceiling, _ = setpoints.likelihood(raw, setting, (8, 11), 'S', 0, 0., visible=True)
    record('visible_noiseless_ceiling', float(np.max(abs(ceiling-expected))) <= 1e-12,
           maximum_error=float(np.max(abs(ceiling-expected))))
    return dict(passed=all(x['passed'] for x in gates), gates=gates, scientific_draws=0,
                method_only=True, exact_state_cap=200000,
                limitation='Known-answer checks do not establish 512-record numerical convergence.')
