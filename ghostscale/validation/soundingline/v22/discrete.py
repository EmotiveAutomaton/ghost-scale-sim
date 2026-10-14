"""A2, B2 and C2: explicit finite latent models and public observation channels."""
import itertools
import numpy as np
from scipy.special import expit
from .common import rng, bern, normalize

A_STATES = np.array(list(itertools.product((0, 1), repeat=5)))  # q, goal, technique, agenda, law
C_STATES = np.array(list(itertools.product((0, 1), repeat=3)))  # library, preference, law
B_STATES = np.array(list(itertools.product((0, 1), repeat=2)))  # route, learning law


def binary_ll(obs, p):
    return np.log(np.where(np.asarray(obs) == 1, p, 1-p))


def sample_a2(namespace, world, index):
    r = rng('A2', namespace, world, index)
    s = r.integers(0, 2, 5)
    s[-1] = rng('A2-law', namespace, world).integers(2)
    q, goal, technique, agenda, law = s
    rel = (.7, .9)[law]
    signature = bool(index % 2)
    swapped = bool((index//2) % 2)
    own = int(r.random() < (.75 if q else .25))
    mu = q if goal == 0 else agenda
    claim = int(r.random() < (rel if mu else 1-rel))
    selection = r.binomial(1, (rel if goal else 1-rel) if signature else .5, 8)
    if swapped:
        selection = 1-selection
    operations = r.binomial(1, rel if technique else 1-rel, 8)
    return dict(own=own, claim=claim, selection=selection.tolist(), operations=operations.tolist(),
                signature=signature, swapped=swapped), dict(y=[int(q), int(goal), int(technique)], state=s.tolist())


def infer_a2(p, library=True, known_law=None, ignore_mapping=False):
    s = A_STATES
    q, g, t, a, law = s.T
    rel = np.where(law == 0, .7, .9)
    ll = binary_ll(p['own'], np.where(q, .75, .25))
    mu = np.where(g == 0, q, a)
    ll += binary_ll(p['claim'], np.where(mu, rel, 1-rel))
    sig = np.where(g, rel, 1-rel) if p['signature'] else np.full(len(s), .5)
    if p['swapped'] and not ignore_mapping:
        sig = 1-sig
    for obs in p['selection']:
        ll += binary_ll(obs, sig)
    for obs in p['operations']:
        ll += binary_ll(obs, np.where(t, rel, 1-rel))
    if not library:
        ll = np.where(g == 0, ll, -np.inf)
    if known_law is not None:
        ll = np.where(law == known_law, ll, -np.inf)
    post = normalize(ll)
    return bern(post@s[:, :3])


def evaluate_a2(p, truth):
    exact = infer_a2(p)
    missing = infer_a2(p, library=False)
    own = .75 if p['own'] else .25
    readers = {'law-marginal': exact, 'missing-process': missing,
               'wrong-signature-map': infer_a2(p, ignore_mapping=True),
               'oracle-law': infer_a2(p, known_law=truth['state'][-1])}
    return readers, dict(signature=p['signature'], swapped=p['swapped'], source_goal=truth['y'][1],
                         content_uptake={k: float(v[0, 1]-own) for k, v in readers.items()},
                         # A sampled production policy copies the inferred operation distribution.
                         technique_production={k: float(v[2, truth['y'][2]]) for k, v in readers.items()},
                         goal_probability={k: float(v[1, 1]) for k, v in readers.items()})


def sample_c2(namespace, world, index):
    r = rng('C2', namespace, world, index)
    own_lib, own_pref = r.integers(0, 2, 2)
    share_lib = bool(index % 2)
    share_pref = bool((index//2) % 2)
    library = int(own_lib if share_lib else 1-own_lib)
    pref = int(own_pref if share_pref else 1-own_pref)
    law = int(rng('C2-law', namespace, world).integers(2))
    rel = (.7, .9)[law]
    k = (0, 2, 8, 32)[(index//4) % 4]
    coupled = bool((index//16) % 2)
    disclosed = bool((index//32) % 2)
    unrelated = bool((index//64) % 2)
    # A noisy operation rendered through a library-specific invertible code.
    operation = int(r.integers(2))
    rendered = (operation ^ library) ^ int(r.random() > rel)
    witnesses = r.binomial(1, rel if library else 1-rel, k)
    records = r.binomial(1, .5 if unrelated else rel if pref else 1-rel, k)
    preference_witness = int(r.random() < (rel if pref else 1-rel)) if coupled else -1
    return dict(own_lib=int(own_lib), own_pref=int(own_pref), rendered=rendered,
                witnesses=witnesses.tolist(), records=records.tolist(), k=k,
                coupled=coupled, preference_witness=preference_witness, disclosed=disclosed,
                share_lib=share_lib if disclosed else None, share_pref=share_pref if disclosed else None,
                unrelated=unrelated), dict(y=[operation, pref], state=[library, pref, law],
                                          share_lib=share_lib, share_pref=share_pref)


def infer_c2(p, known_law=None, project_self=False, no_records=False):
    s = C_STATES
    lib, pref, law = s.T
    rel = np.where(law == 0, .7, .9)
    ll = np.zeros(len(s))
    # Route witnesses remain available to both heads, making negative controls real.
    for witness in p['witnesses']:
        ll += binary_ll(witness, np.where(lib, rel, 1-rel))
    if not no_records and not p['unrelated']:
        for record in p['records']:
            ll += binary_ll(record, np.where(pref, rel, 1-rel))
    if p['coupled']:
        ll += binary_ll(p['preference_witness'], np.where(pref, rel, 1-rel))
    if p['disclosed']:
        want_lib = p['own_lib'] if p['share_lib'] else 1-p['own_lib']
        want_pref = p['own_pref'] if p['share_pref'] else 1-p['own_pref']
        ll = np.where((lib == want_lib) & (pref == want_pref), ll, -np.inf)
    if project_self:
        # Explicit similarity heuristic, not extra evidence or the exact reader.
        ll += np.log(np.where(lib == p['own_lib'], .9, .1))
        ll += np.log(np.where(pref == p['own_pref'], .9, .1))
    if known_law is not None:
        ll = np.where(law == known_law, ll, -np.inf)
    post = normalize(ll)
    op = np.where((p['rendered'] ^ lib) == 1, rel, 1-rel)
    return bern([post@op, post@pref])


def evaluate_c2(p, truth):
    readers = {'law-marginal': infer_c2(p), 'self-projection': infer_c2(p, project_self=True),
               'without-records': infer_c2(p, no_records=True),
               'oracle-law': infer_c2(p, known_law=truth['state'][-1])}
    return readers, dict(k=p['k'], share_lib=truth['share_lib'], share_pref=truth['share_pref'],
                         coupled=p['coupled'], disclosed=p['disclosed'], unrelated=p['unrelated'])


def acquisition(experience, route, rate):
    """Learn both alternatives causally, or retain X's acquired association."""
    q = np.zeros(2)
    association = 0.
    values = []
    for action, reward in experience:
        q[action] += rate*(reward-q[action])
        if action == 1:
            association += rate*(reward-association)
        values.append(q[1]-q[0] if route == 0 else association)
    return np.array(values)


def sample_b2(namespace, world, index):
    r = rng('B2', namespace, world, index)
    route = int(r.integers(2))
    law = int(rng('B2-law', namespace, world).integers(2))
    rate = (.2, .5)[law]
    condition = ('contingent', 'degraded', 'alias')[index % 3]
    experience = []
    for step in range(32):
        action = 1 if condition == 'alias' else int(r.integers(2))
        chance = .85 if action or condition == 'degraded' else .15
        experience.append([action, int(r.random() < chance)])
    val = acquisition(experience, route, rate)
    checkpoints = np.array([3, 7, 15, 31])
    choices = r.binomial(1, expit(5*(val[checkpoints]-.2)))
    devalued = bool((index//3) % 2)
    need = 0. if devalued else 1.
    final = (need if route == 0 else 1.)*val[-1]
    y = int(r.random() < expit(5*(final-.2)))
    return dict(experience=experience, checkpoints=checkpoints.tolist(), choices=choices.tolist(),
                devalued=devalued, condition=condition), dict(y=[route, y], state=[route, law])


def infer_b2(p, only_route=None, known_law=None, ignore_choices=False):
    ll, probs = [], []
    for route, law in B_STATES:
        values = acquisition(p['experience'], route, (.2, .5)[law])
        pre = expit(5*(values[p['checkpoints']]-.2))
        score = 0. if ignore_choices else float(binary_ll(p['choices'], pre).sum())
        if only_route is not None and route != only_route or known_law is not None and law != known_law:
            score = -np.inf
        need = 0. if p['devalued'] and route == 0 else 1.
        probs.append(expit(5*(need*values[-1]-.2)))
        ll.append(score)
    post = normalize(ll)
    return bern([post@B_STATES[:, 0], post@np.array(probs)])


def evaluate_b2(p, truth):
    readers = {'law-marginal': infer_b2(p), 'model-based-only': infer_b2(p, only_route=0),
               'association-only': infer_b2(p, only_route=1), 'prior-mixture': infer_b2(p, ignore_choices=True),
               'oracle-law': infer_b2(p, known_law=truth['state'][-1])}
    return readers, dict(condition=p['condition'], devalued=p['devalued'], route=truth['state'][0])


def features(card, p):
    if card == 'A2':
        return np.r_[p['own'], p['claim'], p['selection'], p['operations'], p['signature'], p['swapped']]
    if card == 'C2':
        h = np.zeros((32, 3))
        for i in range(p['k']):
            h[i] = (p['witnesses'][i], p['records'][i], 1)
        return np.r_[h.ravel(), p['own_lib'], p['own_pref'], p['rendered'], p['coupled'],
                     p['preference_witness'], p['disclosed'],
                     -1 if p['share_lib'] is None else p['share_lib'],
                     -1 if p['share_pref'] is None else p['share_pref'], p['unrelated']]
    return np.r_[np.array(p['experience']).ravel(), p['choices'], p['devalued']]


def sample(card, namespace, world, index):
    return {'A2': sample_a2, 'B2': sample_b2, 'C2': sample_c2}[card](namespace, world, index)


def evaluate(card, p, truth):
    return {'A2': evaluate_a2, 'B2': evaluate_b2, 'C2': evaluate_c2}[card](p, truth)
