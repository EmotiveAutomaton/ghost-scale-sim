"""A3: causal per-step attention and uptake; no closed-version run calls."""
import itertools
import numpy as np
from .common import rng, normalize, bern

STATES = np.array(list(itertools.product((0, 1), repeat=4)))  # goal, value, technique, law
POLICIES = ('no-gate', 'early-guard', 'procedure-only', 'withdraw-.7', 'withdraw-.9')


def sample(namespace, world, index):
    r = rng('A3', namespace, world, index)
    state = r.integers(2, size=4)
    state[-1] = rng('A3-law', namespace, world).integers(2)
    coupled = bool(index % 2)
    if coupled:
        state[1] = state[2]
    rel = (.7, .9)[state[-1]]
    observations = r.binomial(1, np.where(state[:3], rel, 1-rel), (16, 3))
    flags = r.binomial(1, rel if state[0] else 1-rel, 16)
    return dict(observations=observations.tolist(), flags=flags.tolist(), coupled=coupled,
                prefix=int((4, 8, 16)[(index//2) % 3])), dict(y=state[:3].tolist(), state=state.tolist())


def features(p):
    x = np.zeros((16, 5))
    for i in range(p['prefix']):
        x[i] = p['observations'][i]+[p['flags'][i], 1]
        if i < 8:
            x[i, :2] = 0  # same unavailable early content as the exact reader
    return np.r_[x.ravel(), p['coupled']]


def trajectory(p, policy, known_law=None):
    s = STATES
    rel = np.where(s[:, 3] == 0, .7, .9)
    ll = np.zeros(len(s))
    if p['coupled']:
        ll = np.where(s[:, 1] == s[:, 2], ll, -np.inf)
    if known_law is not None:
        ll = np.where(s[:, 3] == known_law, ll, -np.inf)
    belief = np.full(3, .5)
    memory = np.full(3, .5)
    path = []
    flag_p = .5
    fired = False
    # Fixed threshold decisions see only the flag prefix, never future rows/truth.
    for i in range(p['prefix']):
        previous = normalize(ll)
        flag_prob = np.where(s[:, 0], rel, 1-rel)
        flag = p['flags'][i]
        ll += np.log(np.where(flag, flag_prob, 1-flag_prob))
        flag_p = float(normalize(ll)@s[:, 0])
        precision = np.ones(3)
        if policy == 'early-guard':
            precision[:] = .2
        elif policy == 'procedure-only':
            precision[:2] = 0.
        elif policy.startswith('withdraw'):
            fired |= flag_p >= float(policy.split('-')[1])
            if fired:
                precision[:2] = .05
                # Withdrawal has an explicit processing cost for shared operations.
                precision[2] = .5 if p['coupled'] else .9
        # Content resolves later; procedure evidence is available from the start.
        available = np.array([i >= 8, i >= 8, True], float)
        precision *= available
        obs = p['observations'][i]
        for j in range(3):
            channel = np.where(s[:, j], rel, 1-rel)
            ll += precision[j]*np.log(np.where(obs[j], channel, 1-channel))
        belief = normalize(ll)@s[:, :3]
        # V18.2 shared update rule: the same .25 learning coefficient in every arm.
        # Procedure-only selection does not sever actual dependency in the generator.
        memory += .25*precision*(belief-memory)
        path.append(dict(posterior=belief.tolist(), memory=memory.tolist(),
                         precision=precision.tolist(), flag_probability=flag_p, fired=bool(fired)))
    return bern(belief), path


def evaluate(p, truth):
    readers, trajectories = {}, {}
    for policy in POLICIES:
        readers[policy], trajectories[policy] = trajectory(p, policy)
    readers['law-marginal'] = readers['no-gate']
    readers['oracle-law'] = trajectory(p, 'no-gate', truth['state'][-1])[0]
    draw = rng('A3-production', repr(p)).random()
    production = {name: int(draw < path[-1]['memory'][2]) for name, path in trajectories.items()}
    return readers, dict(coupled=p['coupled'], prefix=p['prefix'], benign=truth['y'][0] == 0,
                         production=production, technique_truth=truth['y'][2],
                         trajectories=trajectories,
                         scope='new sequential apparatus; E56 timing and G5 attention principles, not their historical numbers')
