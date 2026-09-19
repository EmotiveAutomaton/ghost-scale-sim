"""Finite acquired-craft worlds; reader likelihoods consume public records only.

DESIGN CHECK: native execution and acquisition, identical-observation ambiguity,
exact probability normalization, no private/future fields in the reader packet.
State roles are a declared simulator construction, not identified human entities.
"""
from functools import lru_cache
from itertools import permutations, product
import json
import math
import numpy as np
from ..v16.learning import learn, encoding_cost
from ..v16.world import execute
from ..v16.records import canonical, seed_for

STATES = tuple(product(range(3), range(2), range(2), range(2)))
PROGRAMS = tuple(p for n in range(4) for p in permutations(range(4), n))
ARTIFACTS = np.array([execute(p).artifact for p in PROGRAMS])
FACTORS = tuple(product(range(2), repeat=4))
WORLD_KEYS = {'groups', 'price', 'temperature', 'rule', 'coupled', 'endogenous', 'shared', 'noise'}
CONTEXT_KEYS = {'goal', 'signal', 'budget', 'price_scale', 'offered', 'uninformative'}


def rng(*parts):
    return np.random.default_rng(seed_for('v18.3', *parts))


def make_world(cell=0, draw=0):
    r = rng('world', draw)
    cells = r.permutation(4).tolist()
    rule, coupled, endogenous, shared = FACTORS[cell]
    return dict(groups=[cells[:2], cells[2:]], price=float(r.uniform(.12, .35)),
                temperature=float(r.uniform(.55, 1.0)), rule='satisficing' if rule else 'softmax',
                coupled=bool(coupled), endogenous=bool(endogenous), shared=bool(shared), noise=.02)


def context(goal=None, signal=None, budget=2, price_scale=1., offered=None, uninformative=False):
    return dict(goal=goal, signal=signal, budget=budget, price_scale=price_scale,
                offered=offered, uninformative=uninformative)


QUERIES = (context(), context(goal=0), context(goal=1), context(signal=0),
           context(signal=1), context(goal=0, signal=0, budget=1),
           context(goal=1, signal=0, budget=1), context(budget=3, price_scale=2.))
FUTURES = (context(goal=0, price_scale=.7), context(goal=1, price_scale=1.4),
           context(signal=0, budget=3), context(signal=1, budget=1))


@lru_cache(maxsize=256)
def acquisition(groups, skill, coupled):
    if skill == 0:
        return learn([], [], capacity=1)
    own = tuple(groups[skill-1]); other = tuple(groups[2-skill])
    attempts = [own]*4 + ([other]*5 if coupled else [])
    targets = [execute(p).artifact for p in attempts]
    return learn(attempts, targets, capacity=1)


def repertoire(w, state):
    return acquisition(tuple(map(tuple, w['groups'])), int(state[0]), w['coupled']).library


@lru_cache(maxsize=32768)
def _matrix(world_json, context_json):
    w, c = json.loads(world_json), json.loads(context_json)
    rows = []
    for state in STATES:
        skill, goal, belief, tradeoff = state
        goal = goal if c['goal'] is None else c['goal']
        belief = belief if c['signal'] is None else c['signal']
        target = sum(1 << x for x in w['groups'][goal ^ belief])
        costs = np.array([encoding_cost(p, repertoire(w, state)) for p in PROGRAMS])
        allowed = costs <= c['budget']
        if c['offered'] is not None:
            allowed &= np.array([all(x in c['offered'] for x in p) for p in PROGRAMS])
        if w['endogenous']:
            # The considered set depends on purpose, separately from physical feasibility.
            excluded = w['groups'][1-goal][tradeoff]
            allowed &= np.array([excluded not in p for p in PROGRAMS])
        error = np.array([(int(a) ^ target).bit_count() for a in ARTIFACTS])
        imbalance = np.array([sum(bool(int(a) & (1 << x)) for x in w['groups'][1]) -
                              sum(bool(int(a) & (1 << x)) for x in w['groups'][0]) for a in ARTIFACTS])
        utility = -1.6*error + .5*(2*tradeoff-1)*imbalance - w['price']*c['price_scale']*costs
        if c['uninformative']:
            p = np.zeros(len(PROGRAMS)); p[0] = 1.
        elif w['rule'] == 'softmax':
            z = np.where(allowed, utility/w['temperature'], -np.inf)
            p = np.exp(z-z.max()); p /= p.sum()
        elif w['rule'] == 'satisficing':
            # First acceptable program in a public deterministic search order.
            acceptable = np.flatnonzero(allowed & (error <= 1))
            chosen = int(acceptable[0]) if len(acceptable) else int(np.argmax(np.where(allowed, utility, -np.inf)))
            p = np.zeros(len(PROGRAMS)); p[chosen] = 1.
        elif w['rule'] == 'lexicographic':
            options = np.flatnonzero(allowed)
            chosen = min(options, key=lambda j: (error[j], costs[j], -imbalance[j]*(2*tradeoff-1), int(j)))
            p = np.zeros(len(PROGRAMS)); p[chosen] = 1.
        else:
            raise ValueError('unknown decision rule')
        # Lapses choose among physically offered programs, regardless of learned code cost.
        lapse = np.array([c['offered'] is None or all(x in c['offered'] for x in p) for p in PROGRAMS], float)
        lapse /= lapse.sum()
        p = (1-w['noise'])*p + w['noise']*lapse
        rows.append(p)
    return np.array(rows)


def matrix(w, c):
    return _matrix(canonical(w).decode(), canonical(c).decode()).copy()


def artifact_matrix(w, c):
    m = matrix(w, c)
    return np.stack([m[:, ARTIFACTS == a].sum(axis=1) for a in range(16)], axis=1)


def observe(w, state, c, random, source):
    j = int(random.choice(len(PROGRAMS), p=matrix(w, c)[state]))
    return dict(context=c, program=list(PROGRAMS[j]), artifact=int(ARTIFACTS[j]), source=str(source))


def packet(w, history):
    return canonical(dict(schema='v18.3.public.1', world=w, history=history))


def parse(payload):
    data = json.loads(payload)
    if set(data) != {'schema', 'world', 'history'} or data['schema'] != 'v18.3.public.1':
        raise ValueError('private public-packet field')
    if set(data['world']) != WORLD_KEYS or len(data['history']) > 128:
        raise ValueError('invalid public world/history')
    if data['world']['rule'] not in ('softmax', 'satisficing', 'lexicographic'):
        raise ValueError('undeclared public rule')
    for h in data['history']:
        if set(h) != {'context', 'program', 'artifact', 'source'} or set(h['context']) != CONTEXT_KEYS:
            raise ValueError('private observation field')
        c = h['context']
        if c['goal'] not in (None, 0, 1) or c['signal'] not in (None, 0, 1) or c['budget'] not in (1, 2, 3):
            raise ValueError('invalid public context')
        if tuple(h['program']) not in PROGRAMS or execute(h['program']).artifact != h['artifact']:
            raise ValueError('invalid public execution')
    return data


def likelihood(w, observation):
    return matrix(w, observation['context'])[:, PROGRAMS.index(tuple(observation['program']))]


def posterior(payload, prior=None, deduplicate=True):
    data = parse(payload)
    weights = np.ones(len(STATES))/len(STATES) if prior is None else np.array(prior, float)
    seen = set()
    for observation in data['history']:
        if deduplicate and observation['source'] in seen:
            continue
        seen.add(observation['source'])
        weights *= likelihood(data['world'], observation)
        total = weights.sum()
        if total <= 0:
            return None
        weights /= total
    return weights


def entropy(p):
    p = np.asarray(p, float)
    return float(-np.sum(p[p > 0]*np.log(p[p > 0])))


def cross_entropy(truth, prediction):
    q, p = np.asarray(truth, float), np.asarray(prediction, float)
    if np.any((q > 0) & (p <= 0)):
        return math.inf
    keep = q > 0
    return float(-np.sum(q[keep]*np.log(p[keep])))


def loss_record(value):
    return {'value': float(value) if math.isfinite(value) else None, 'infinite': bool(math.isinf(value))}


def predict(w, weights, contexts=FUTURES):
    return np.array([weights @ artifact_matrix(w, c) for c in contexts])


def scoring(w, weights, state, contexts=FUTURES):
    p = predict(w, weights, contexts)
    truth = np.array([artifact_matrix(w, c)[state] for c in contexts])
    losses = [cross_entropy(q, v) for q, v in zip(truth, p)]
    return dict(expected_loss=loss_record(float(np.mean(losses))), brier=float(np.mean(np.sum((truth-p)**2, axis=1))),
                state_mass=float(weights[state]), entropy=entropy(weights))


@lru_cache(maxsize=4096)
def enact(program, target, code_budget=2):
    """The reader actually practices four times, acquires, plans and executes."""
    acq = learn([program]*4, [execute(program).artifact]*4, capacity=1)
    options = [p for p in PROGRAMS if encoding_cost(p, acq.library) <= code_budget]
    choice = min(options, key=lambda p: ((execute(p).artifact ^ target).bit_count(),
                                       encoding_cost(p, acq.library), len(p), p))
    result = execute(choice)
    return dict(success=bool(result.legal and result.artifact == target), program=list(choice),
                artifact=result.artifact, library=[list(x) for x in acq.library],
                practice_actions=acq.processing_cost, definition_actions=acq.definition_cost,
                planning_evaluations=len(options), execution_actions=result.primitive_cost,
                code_cost=encoding_cost(choice, acq.library))


def initial_history(w, state, random, length=6):
    history=[]
    for t in range(length):
        if w['shared'] and t % 3:
            history.append(dict(history[-1]))
        else:
            history.append(observe(w, state, context(), random, f'root-{t}'))
    return history


def transition(kind, rate):
    out = np.zeros((len(STATES), len(STATES)))
    axes = {'goal':(1,), 'belief':(2,), 'skill':(0,), 'fast':(1,2), 'all':(0,1,2,3)}[kind]
    for i, a in enumerate(STATES):
        peers = [j for j,b in enumerate(STATES) if all(a[k] == b[k] for k in range(4) if k not in axes)]
        out[i, i] += 1-rate
        out[i, peers] += rate/len(peers)
    return out
