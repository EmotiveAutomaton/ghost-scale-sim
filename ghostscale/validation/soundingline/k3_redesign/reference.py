"""Independent scalar arithmetic and exhaustive small-state reference.

Does not call the production transition, update, or observation kernel.
The compiled scalar implementation permits full reference-path reconstruction.
"""
import itertools
import math
from collections import defaultdict
import numpy as np
from numba import njit


@njit(cache=True)
def observe_scalar(state, action, size, shock, law, width, beta, lapse):
    result = state.copy()
    n = len(state)//4
    utility = np.ones(n+1)
    for need in range(n):
        x = 0. if shock == need else state[need]
        e, p = state[n+need], state[2*n+need]
        run = min(20., state[3*n+need]+1) if x < e else 0.
        if law == 1:
            if x >= e and (width == 0 or x <= e+width):
                p = min(p*1.03, 10.)
        elif run == 20:
            if law == 2:
                p = max(p*.97, .1)
            if law == 3:
                e = max(e-.1, x)
            if law == 4:
                p = min(p*1.03, 10.)
        result[need] = x
        result[n+need] = e
        result[2*n+need] = p
        result[3*n+need] = 0. if law < 2 else run
        utility[need] = min(max(e-x, 0.)*p, 5.)
    # Deliberately use unshifted exponentials; utilities lie in [0, 5].
    denominator = 0.
    for u in utility:
        denominator += math.exp(u)
    probability = (1-lapse)*math.exp(utility[action])/denominator+lapse/(n+1)
    expected = 0. if action == n else 5.+beta*utility[action]
    if abs(expected-size) > 1e-10:
        probability = 0.
    return result, probability


@njit(cache=True)
def enumerate_candidates(states, previous_action, previous_size, action, size,
                         shock, law, width, beta, lapse, losses, probabilities):
    n = states.shape[1]//4
    bcount = len(losses)
    out = np.empty((len(states)*bcount, states.shape[1]))
    weights = np.zeros(len(out))
    for i in range(len(states)):
        for b in range(bcount):
            value = states[i].copy()
            for j in range(n):
                value[j] += (previous_size if previous_action == j else 0.)-losses[b,j]
                value[j] = max(0., min(20., value[j]))
            value, cp = observe_scalar(value, action, size, shock, law, width, beta, lapse)
            index = i*bcount+b
            out[index] = value
            first = index
            for earlier in range(i*bcount, index):
                same = True
                for j in range(len(value)):
                    if out[earlier,j] != value[j]:
                        same = False
                        break
                if same:
                    first = earlier
                    break
            weights[first] += probabilities[b]*cp
    return out, weights


def exact(observed, setting, point, law, state_cap=200000):
    """Unrounded exhaustive filter; cap failure is explicit, never a pass."""
    n = len(point)
    states = {tuple(x)+tuple(p-setting['width']/2 for p in point)+(1.,)*n+(0.,)*n:
              1/21**n for x in itertools.product(range(21), repeat=n)}
    events = list(itertools.product((0., 1., 5.), repeat=n))
    probabilities = [math.prod({0.: .45, 1.: .5, 5.: .05}[x] for x in row) for row in events]
    logs, total_log, peak = [], 0., len(states)
    for t, row in enumerate(observed['records']):
        weighted = defaultdict(float)
        for state, mass in states.items():
            dest, cp = observe_scalar.py_func(np.array(state), row['action'], row['size'],
                row['shock'], law, setting['width'], setting['beta'], setting['lapse'])
            weighted[tuple(dest)] += mass*cp
        z = math.fsum(weighted.values())
        if z == 0:
            return np.r_[logs, np.full(len(observed['records'])-t, -np.inf)], peak
        total_log += math.log(z)
        logs.append(total_log)
        if t+1 == len(observed['records']):
            break
        following = defaultdict(float)
        for dest, mass in weighted.items():
            if mass == 0:
                continue
            for losses, p in zip(events, probabilities):
                x = tuple(max(0., min(20., dest[j]-setting['consumption']*losses[j]+
                          (row['size'] if row['action'] == j else 0.))) for j in range(n))
                following[x+dest[n:]] += mass/z*p
                if len(following) > state_cap:
                    raise MemoryError(f'independent exact state cap {state_cap} at record {t+1}')
        states = following
        peak = max(peak, len(states))
    return np.array(logs), peak
