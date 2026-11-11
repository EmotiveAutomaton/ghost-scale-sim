"""Prospectively frozen, excluded full-reader cost probes after validation.

Expanded-setting probes measure engineering cost only. They never certify
overshoot, three-need or noisy-observation scientific validity.
"""
import itertools
import time
import numpy as np
from ..v27 import setpoints, setpoint_resolution, registry
from ..v27.common import read, write, digest, close
from . import filter as adapted


def prepare(c, plan_job):
    if not read(c/'VALIDATION_RESULT.json')['passed']:
        raise ValueError('primary numerical validation must pass before cost probes')
    count = read(c/'VALIDATION_FREEZE.json')['counts'][-1]
    settings = [registry.primary_settings('K3')[i] for i in (9, 10, 11)]
    settings += [s | dict(needs=3, beta=.5, width=2, consumption=6., lapse=.05) for s in settings]
    jobs = []
    # Independent full passes are separate jobs so each has native accounting.
    # All plans and seed identities are written before the first cost draw.
    for i, setting in enumerate(settings):
        for label in ('original', 'replay', 'independent'):
            identity = f'cost-{i:02d}-{label}'
            plan_job(c, identity, 'C', 'cost', 1500., setting=setting, cost_index=i,
                particles=count, label=label, validity='primary validated' if i < 3 else 'expanded law unvalidated',
                validation_result_sha256=digest(c/'VALIDATION_RESULT.json'))
            jobs.append(identity)
    write(c/'COST_FREEZE.json', dict(jobs=jobs, settings=settings, particles=count,
        scientific_draws=0, readers=['naive', 'aware', 'band', 'dynamic', 'visible'],
        all_passes=['original', 'replay', 'independent'],
        validation_result_sha256=digest(c/'VALIDATION_RESULT.json'),
        limitation='Expanded laws remain unvalidated even when their cost probe completes.'))


def run(plan, root, pulse):
    s = plan['setting']
    raw = setpoints.generate(s | dict(inference='checked_particles'), ['excluded-k3-rd1-full-cost', plan['cost_index']])
    observed = setpoint_resolution.hidden_observations(raw)
    points = list(itertools.product((8, 11, 14), repeat=s['needs']))
    libraries = dict(naive=[(p, 'P0', 0, None, False) for p in points],
        aware=[(p, 'P0', 0, s['beta'], False) for p in points],
        band=[(p, 'P0', w, s['beta'], False) for p in points for w in (0, 2, 4)],
        dynamic=[(p, d, s['width'], s['beta'], False) for p in points for d in adapted.LAWS],
        visible=[(p, s['dynamic'], s['width'], s['beta'], True) for p in points])
    independent = plan['label'] == 'independent'
    cached, readers = {}, {}
    for reader, hypotheses in libraries.items():
        rows = []
        for h, (point, law, width, beta, visible) in enumerate(hypotheses):
            key = (point, law, width, beta, visible)
            if key not in cached:
                if visible or law == 'P0':
                    ll, _ = setpoints.likelihood(raw if visible else observed, s | dict(inference='exact'),
                        point, law, width, beta, visible, independent)
                else:
                    ll, _ = adapted.likelihood(observed, s | dict(width=width, beta=beta),
                        point, law, plan['particles'], 0, independent, pulse)
                cached[key] = [None if np.isneginf(v) else float(v) for v in ll]
            rows.append(cached[key])
            pulse(reader=reader, hypothesis=h+1, hypotheses=len(hypotheses))
        readers[reader] = rows
    write(root/'PATHS.json', dict(readers=readers, scientific_draws=0))
    return dict(readers={k: len(v) for k, v in readers.items()}, unique_hypotheses=len(cached),
                scientific_draws=0, path_sha256=digest(root/'PATHS.json'), label=plan['label'],
                validity=plan['validity'])


def verify_cost(c):
    receipts = []
    for i in range(len(read(c/'COST_FREEZE.json')['settings'])):
        roots = [c/'jobs'/f'cost-{i:02d}-{label}' for label in ('original', 'replay', 'independent')]
        a, b, d = [read(p/'PATHS.json') for p in roots]
        if a != b:
            raise ArithmeticError('full cost probe replay differs')
        close(a, d, tol=1e-9)
        receipts.append(dict(index=i, full_replay=True, independent=True,
            hashes=[digest(p/'PATHS.json') for p in roots],
            cpu_seconds=sum(read(c/'accounting'/(p.name+'.json'))['cpu_seconds'] for p in roots)))
    return dict(passed=True, probes=receipts, scientific_draws=0,
                all_reader_passes_counted=True, production_authorized=False)
