"""Fully adapted finite-branch filter. No change to the V27 world or likelihood.

Particles are equally weighted states immediately AFTER the current dynamic
update, before its action/consumption transition. The next proposal enumerates
previous consumption, applies the next observed shock/update, and conditions on
the next action and size. Joint ancestor/branch systematic sampling has exactly
the fully adapted marginal; its normalizer is the predictive likelihood.
Duplicate destinations within an ancestor are merged exactly (no rounding).
Duplicate ancestors are additive atoms of the same empirical distribution.
"""
import itertools
import math
import numpy as np
from numba import njit
from ..v27.common import rng

LAWS = ('P0', 'P+', 'P-', 'S', 'G')


@njit(cache=True)
def observe_state(state, action, size, shock, law, width, beta, lapse):
    needs = len(state) // 4
    out = state.copy()
    urgency = np.empty(needs)
    top = 1.
    for j in range(needs):
        if shock == j:
            out[j] = 0.
        x, edge, precision, streak = out[j], out[needs+j], out[2*needs+j], out[3*needs+j]
        streak = min(20., streak+1.) if x < edge else 0.
        if law == 1 and x >= edge and (width == 0 or x <= edge+width):
            precision = min(10., precision*1.03)
        if streak >= 20:
            if law == 2:
                precision = max(.1, precision*.97)
            elif law == 3:
                edge = max(x, edge-.1)
            elif law == 4:
                precision = min(10., precision*1.03)
        if law == 0 or law == 1:
            streak = 0.
        out[needs+j], out[2*needs+j], out[3*needs+j] = edge, precision, streak
        urgency[j] = min(5., max(0., edge-x)*precision)
        top = max(top, urgency[j])
    denominator = math.exp(1.-top)
    for j in range(needs):
        denominator += math.exp(urgency[j]-top)
    selected = 1. if action == needs else urgency[action]
    weight = (1-lapse)*math.exp(selected-top)/denominator + lapse/(needs+1)
    expected_size = 0. if action == needs else 5.+beta*urgency[action]
    if abs(size-expected_size) > 1e-10:
        weight = 0.
    return out, weight


@njit(cache=True)
def candidates(states, previous_action, previous_size, action, size, shock,
               law, width, beta, lapse, losses, probabilities):
    """Enumerate and exactly merge clipped/shocked branches per ancestor."""
    parents, dimensions = states.shape
    needs = dimensions//4
    branches = len(losses)
    destinations = np.empty((parents*branches, dimensions))
    weights = np.zeros(parents*branches)
    for i in range(parents):
        for b in range(branches):
            slot = i*branches+b
            state = states[i].copy()
            for j in range(needs):
                restore = previous_size if previous_action == j else 0.
                state[j] = min(20., max(0., state[j]-losses[b,j]+restore))
                if shock == j:
                    state[j] = 0.
            duplicate = -1
            # Stock equality before the deterministic update implies equality
            # of the complete destination for this ancestor.
            for earlier in range(b):
                equal = True
                for j in range(needs):
                    if destinations[i*branches+earlier,j] != state[j]:
                        equal = False
                        break
                if equal:
                    duplicate = i*branches+earlier
                    break
            out, likelihood = observe_state(state, action, size, -1, law, width, beta, lapse)
            destinations[slot] = out
            if duplicate >= 0:
                weights[duplicate] += probabilities[b]*likelihood
            else:
                weights[slot] = probabilities[b]*likelihood
    return destinations, weights


def branch_law(needs, consumption):
    events = list(itertools.product(range(3), repeat=needs))
    losses = np.array([[((0., 1., 5.)[j])*consumption for j in row] for row in events])
    probability = np.array([math.prod((.45, .5, .05)[j] for j in row) for row in events])
    return losses, probability


def initial(point, width):
    stocks = np.array(list(itertools.product(range(21), repeat=len(point))), dtype=float)
    return np.column_stack((stocks, np.tile(np.asarray(point)-width/2, (len(stocks), 1)),
                            np.ones_like(stocks), np.zeros_like(stocks)))


def validate_observations(observed, needs):
    if set(observed) != {'inference_key', 'records'}:
        raise ValueError('hidden reader accepts only a seed identity and observed records')
    for row in observed['records']:
        if set(row) != {'action', 'size', 'shock'}:
            raise ValueError('hidden/evaluator state is not an observation')
        if row['action'] not in range(needs+1) or row['shock'] not in range(-1, needs):
            raise ValueError('invalid observed category')
        if not math.isfinite(row['size']) or row['size'] < 0:
            raise ValueError('invalid observed restoration')


def likelihood(observed, setting, point, law, particles, replicate=0,
               independent=False, pulse=lambda **kw: None):
    validate_observations(observed, setting['needs'])
    if particles < 1 or law not in LAWS or len(point) != setting['needs']:
        raise ValueError('invalid numerical hypothesis')
    from .reference import observe_scalar, enumerate_candidates
    code = LAWS.index(law)
    # Hypotheses, populations and replicas have disjoint deterministic streams.
    r = rng('k3-rd1-adapted-v1', observed['inference_key'], list(point), law,
            setting['width'], setting['beta'], particles, replicate)
    states = initial(point, setting['width'])
    losses, probabilities = branch_law(setting['needs'], setting['consumption'])
    logs, ess, diversity, initial_diversity, support, increments = [], [], [], [], [], []
    acc = 0.
    roots = None
    previous = None
    kernel = enumerate_candidates if independent else candidates
    observer = observe_scalar if independent else observe_state
    for t, row in enumerate(observed['records']):
        args = (row['action'], row['size'], row['shock'], code,
                setting['width'], setting['beta'], setting['lapse'])
        if previous is None:
            pairs = [observer(state, *args) for state in states]
            destinations = np.array([p[0] for p in pairs])
            weight = np.array([p[1] for p in pairs])
            branches = 1
        else:
            destinations, weight = kernel(states, previous['action'], previous['size'],
                                           *args, losses, probabilities)
            branches = len(losses)
        normalizer = math.fsum(weight) if independent else float(weight.sum())
        if normalizer <= 0 or not math.isfinite(normalizer):
            raise ArithmeticError(f'zero/nonfinite observation support at record {t+1}; no floor')
        increment = math.log(normalizer/len(states))
        acc += increment
        logs.append(acc)
        increments.append(increment)
        # Ancestor ESS is the relevant auxiliary-filter diagnostic.
        parent_mass = weight.reshape(len(states), branches).sum(axis=1)/normalizer
        ess.append(float(1/(parent_mass@parent_mass)))
        support.append(int(np.count_nonzero(parent_mass == 0)))
        cumulative = np.cumsum(weight/normalizer)
        indices = np.searchsorted(cumulative, (np.arange(particles)+r.random())/particles, side='left')
        indices = np.minimum(indices, len(weight)-1)
        parent = indices//branches
        diversity.append(int(len(np.unique(parent))))
        roots = parent if roots is None else roots[parent]
        initial_diversity.append(int(len(np.unique(roots))))
        states = destinations[indices].copy()
        previous = row
        if t % 16 == 0 or t+1 == len(observed['records']):
            pulse(record=t+1, particles=particles, replicate=replicate, law=law, point=list(point))
    return np.asarray(logs), dict(particles=particles, replicate=replicate,
        ancestor_effective_sample_size=ess, surviving_parents=diversity,
        surviving_initial_ancestors=initial_diversity, zero_support_parents=support,
        log_normalizer_increments=increments, exact_initial_states=21**len(point),
        proposal='fully_adapted_joint_systematic', independent_reference=independent)
