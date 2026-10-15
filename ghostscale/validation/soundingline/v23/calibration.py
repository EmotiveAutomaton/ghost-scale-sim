"""Bounded tuning/validation of the complete seven-look family policy."""
from itertools import product
import time
import numpy as np
from . import adaptive
from .common import canonical, digest, read, write


def conditions():
    return [dict(variants=n, mean=mu, spread=tau, base_se=se, slow=slow)
            for n, mu, tau, se, slow in product((27, 81), (-.02, 0, .01, .02, .03),
                 (.005, .01, .02), (.01, .02, .03), (False, True))]


def one_policy(namespace, prior, kill, repetitions, pulse, deadline_cpu):
    rows, counts = [], dict(strong=0, killed_strong=0, null=0, nominated_null=0)
    counts['counting_kills_strong'] = {str(k): 0 for k in (2, 5, 7)}
    start = time.process_time()
    for index, setting in enumerate(conditions()):
        cell = dict(strong=0, killed_strong=0, null=0, nominated_null=0)
        for repeat in range(repetitions):
            if time.process_time() >= deadline_cpu:
                raise TimeoutError('calibration CPU cap reached; no threshold accepted')
            # Common calibration worlds across policy comparisons, disjoint held-out
            # namespace for final validation. No family experimental seeds enter.
            result = adaptive.calibration_trial([namespace, index, repeat], setting['variants'],
                setting['mean'], setting['spread'], setting['base_se'], setting['slow'], kill, prior)
            if result['strong_alternative']:
                cell['strong'] += 1
                cell['killed_strong'] += result['state'] == 'STOPPED_BELOW_MARGIN'
                for k in (2, 5, 7):
                    counts['counting_kills_strong'][str(k)] += result['counting_killed'][k]
            if result['null']:
                cell['null'] += 1
                cell['nominated_null'] += result['state'] == 'NOMINATED'
        for key in cell:
            counts[key] += cell[key]
        rows.append(dict(setting=setting, counts=cell,
                         kill=adaptive.wilson(cell['killed_strong'], cell['strong']),
                         nominate=adaptive.wilson(cell['nominated_null'], cell['null'])))
        pulse(phase='stopping-calibration', namespace=namespace, prior=list(prior),
              kill=kill, condition=index+1, conditions=len(conditions()))
    return dict(namespace=namespace, prior=list(prior), kill_threshold=kill,
                repetitions_per_condition=repetitions, cells=rows, counts=counts,
                false_kill=adaptive.wilson(counts['killed_strong'], counts['strong']),
                false_nomination=adaptive.wilson(counts['nominated_null'], counts['null']),
                counting_comparators={k: adaptive.wilson(value, counts['strong'])
                                      for k, value in counts['counting_kills_strong'].items()},
                cpu_seconds=time.process_time()-start)


def qualifies(result):
    kill, nominate = result['false_kill'], result['false_nomination']
    return (kill['trials'] > 0 and nominate['trials'] > 0 and
            kill['upper'] <= .1 and nominate['upper'] <= .2)


def run(root, pulse=lambda **kw: None, tuning_repetitions=16, validation_repetitions=128,
        cpu_cap_seconds=7200):
    root.mkdir(parents=True, exist_ok=False)
    start = time.process_time(); limit = start+cpu_cap_seconds
    config = dict(priors=list(adaptive.PRIOR_GRID), kills=list(adaptive.KILL_GRID),
        world_rungs=list(adaptive.RUNGS), tuning_repetitions=tuning_repetitions,
        validation_repetitions=validation_repetitions, conditions=conditions(),
        acceptance='pooled_grid_95_percent_Wilson_upper_bounds_at_most_.10_and_.20',
        per_cell_limits='reported stress strata, not universal guarantees',
        prior_choice='qualified tuning policy with lowest false-kill upper bound; then lowest false-nomination upper bound; fixed grid order',
        failure='calibration unresolved; no experimental batch admission')
    write(root/'DESIGN.json', config)
    tuning = []
    for index, (prior, kill) in enumerate(product(adaptive.PRIOR_GRID, adaptive.KILL_GRID)):
        result = one_policy('tuning', prior, kill, tuning_repetitions, pulse, limit)
        cpu = result.pop('cpu_seconds'); write(root/f'tuning-{index}.json', result)
        tuning.append(result)
        write(root/f'tuning-{index}-cost.json', dict(cpu_seconds=cpu))
    candidates = [result for result in tuning if qualifies(result)]
    if not candidates:
        summary = dict(state='CALIBRATION_UNRESOLVED', reason='no permitted tuning policy meets both error targets', design_sha256=digest(root/'DESIGN.json'))
    else:
        chosen = min(candidates, key=lambda x: (x['false_kill']['upper'], x['false_nomination']['upper']))
        write(root/'CHOSEN_BEFORE_VALIDATION.json', chosen)
        validation = one_policy('heldout-validation', chosen['prior'], chosen['kill_threshold'],
                                validation_repetitions, pulse, limit)
        cpu = validation.pop('cpu_seconds'); write(root/'VALIDATION.json', validation)
        write(root/'validation-cost.json', dict(cpu_seconds=cpu))
        summary = dict(state='CALIBRATION_PASSED' if qualifies(validation) else 'CALIBRATION_UNRESOLVED',
                       prior=chosen['prior'], kill_threshold=chosen['kill_threshold'],
                       design_sha256=digest(root/'DESIGN.json'), validation_sha256=digest(root/'VALIDATION.json'),
                       scope='pooled declared Gaussian/slow-starter grid conditional on fixed fits; not a real-data error guarantee',
                       per_cell_stress_failures=[i for i, cell in enumerate(validation['cells'])
                         if (cell['kill']['rate'] is not None and cell['kill']['rate'] > .1) or
                            (cell['nominate']['rate'] is not None and cell['nominate']['rate'] > .2)])
    write(root/'SUMMARY.json', summary)
    return dict(summary, cpu_seconds=time.process_time()-start)
