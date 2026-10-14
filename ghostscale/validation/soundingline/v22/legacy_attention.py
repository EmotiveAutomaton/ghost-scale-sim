"""A3 reuses E56's pure uptake scorer and G5's physical learner, with new seeds.

No versioned run() entry point is called and no historical output is written.
The sequential channel-precision experiment remains separately identified.
"""
from types import SimpleNamespace
import numpy as np
from ....v10.e56_selective_gate import _uptake
from ..v18_2.selective_assembly import evaluate as physical_case
from ..v16 import assembly
from .attention import trajectory


def e56_accounting(public, truth):
    _, path = trajectory(public, 'no-gate')
    goal = [np.array([1-row['posterior'][0], row['posterior'][0]]) for row in path]
    process = [np.array([1-row['posterior'][2], row['posterior'][2]]) for row in path]
    enc = SimpleNamespace(goal_posteriors_by_step=goal, goal_prior=np.array([.5, .5]),
                          true_goal=truth['y'][0], subgoal_posteriors=process,
                          true_modes=[truth['y'][2]]*len(path), goal_posterior=goal[-1],
                          process={'process_error_reduction': float(np.log(2*process[-1][truth['y'][2]]))})
    values_map = np.array([[.9, .1], [.1, .9]])
    return {name: _uptake(enc, [weight]*len(path), values_map, 2, 2)
            for name, weight in [('no-gate', 1.), ('early-guard', .2)]}


def physical(namespace, world):
    baseline = physical_case('v22-'+namespace, world)
    supplements = []
    for row in baseline['rows']:
        if row['method'] != 'all-weight-1.0-apply-1':
            continue
        w = assembly.World(tuple(row['world']['parents']), tuple(row['world']['defaults']))
        for threshold in (.7, .9):
            selected = []
            fired = False
            cumulative_odds = 1.
            for record in row['observations']:
                # A visible goal-bearing operation is a process signature. Equal
                # .8/.2 flag likelihoods define this supplied diagnostic channel.
                signature = 6 in record['program']
                cumulative_odds *= (.8/.2) if signature else (.2/.8)
                flag = cumulative_odds/(1+cumulative_odds)
                fired |= flag >= threshold
                if not fired or record['topic'] == 'procedure':
                    selected.append(record)
            acquisition = assembly.fragments(selected, w)
            observation_cost = sum(r['execution']['primitive_cost'] for r in row['observations'])+len(row['observations'])
            hits = sum(r['execution']['state'][0] == 1 for r in selected)
            uptake = (1+hits)/(10+len(selected))
            budget = max(0, 256-observation_cost-acquisition['training_primitives']-7)
            outputs = []
            for target in ([0, 0, 0], [int(uptake >= .5), 0, 0]):
                plan = assembly.plan(w, target, library=acquisition['library'], budget=budget, max_steps=7)
                result = assembly.execute(w, plan['program'])
                charged = observation_cost+acquisition['training_primitives']+plan['successor_evaluations']+result['primitive_cost']
                if charged > 256:
                    raise ValueError('physical consumer exceeded common envelope')
                outputs.append(dict(target=target, plan=plan, execution=result, charged_total=charged))
            supplements.append(dict(condition=row['condition'], threshold=threshold, fired=fired,
                                    selected_sources=[r['source'] for r in selected], library=acquisition['library'],
                                    unwanted_goal_uptake=uptake, outputs=outputs, world=row['world']))
    return dict(baseline=baseline, withdrawal=supplements,
                scope='actual G5 fragments/planner on fresh identities; .8/.2 flag heuristic with all inspections charged; separate from sequential Bayes study')
