"""R1--R4: new implementations of the repaired V22 observation channels."""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.special import logsumexp
from .common import rng, normalize, bern, binary_ll

REL = np.linspace(.60, .90, 16)
A_STATES = np.array(list(product(range(2), range(2), range(2), range(2), range(16))))
G_STATES = np.array(list(product(range(2), range(2), range(2), range(16))))


def sample_r1(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    q, t, a = r.integers(2, size=3); g = int(r.random() < s['prevalence'])
    mu = q if not g else a
    p = dict(own=int(r.random() < (.75 if q else .25)),
             claim=int(r.random() < (REL[law] if mu else 1-REL[law])),
             selection=r.binomial(1, s['strength'] if g else 1-s['strength'], s['bits']).tolist(),
             operations=r.binomial(1, REL[law] if t else 1-REL[law], 8).tolist())
    return p, dict(y=[int(q)], goal=g, technique=int(t), agenda=int(a), law=int(law))


def r1_posterior(v, p, masked=False, missing=False, prevalence=None, law=None, full=False):
    s = v['settings']; q, g, t, a, l = A_STATES.T
    prior = s['prevalence'] if prevalence is None else prevalence
    ll = np.log(np.where(g, prior, 1-prior))
    ll += binary_ll(p['own'], np.where(q, .75, .25))
    ll += binary_ll(p['claim'], np.where(np.where(g, a, q), REL[l], 1-REL[l]))
    if not masked:
        count, n = sum(p['selection']), len(p['selection'])
        probability = np.where(g, s['strength'], 1-s['strength'])
        ll += count*np.log(probability)+(n-count)*np.log1p(-probability)
    count = sum(p['operations'])
    probability = np.where(t, REL[l], 1-REL[l])
    ll += count*np.log(probability)+(8-count)*np.log1p(-probability)
    if missing:
        ll = np.where(g == 0, ll, -np.inf)
    if law is not None:
        ll = np.where(l == law, ll, -np.inf)
    post = normalize(ll)
    return post if full else bern(post@A_STATES[:, :3])


def evaluate_r1(v, p, truth=None):
    specs = [
        ('law-marginal', {}), ('masked', {'masked': True}),
        ('missing-process', {'missing': True}), ('rare-steering-prior', {'prevalence': .01})]
    if truth is not None:
        specs.append(('oracle-law', {'law': truth['law']}))
    posts = {name: r1_posterior(v, p, full=True, **kw) for name, kw in specs}
    all_heads = {name: bern(post@A_STATES[:, :3]) for name, post in posts.items()}
    return {k: a[:1] for k, a in all_heads.items()}, dict(heads={k: a.tolist() for k, a in all_heads.items()},
        posteriors={k: post.tolist() for k, post in posts.items()})


def features_r1(v, p):
    selection = np.pad(np.asarray(p['selection'])*2-1, (0, 32-len(p['selection'])))
    return np.r_[p['own'], p['claim'], selection, np.asarray(p['operations'])*2-1]


def r2_policies():
    out = [('no-gate', None), ('procedure-only', None), ('early-guard', None), ('withdraw-content', None)]
    for floor, threshold in product((.5, .2, .05, 0), (.5, .7, .9)):
        for release in ('none', 'release') if floor else ('none',):
            out.append((f'withdraw-all-{floor:g}-{threshold:g}-{release}', (floor, threshold, release)))
    return out


def sample_r2(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    state = r.integers(2, size=3); state[0] = r.random() >= s['benign_share']
    if s['coupled']:
        state[1] = state[2]
    rel = np.linspace(.65, .95, 16)[law]
    p = dict(observations=r.binomial(1, np.where(state, rel, 1-rel), (s['prefix'], 3)).tolist(),
             flags=r.binomial(1, rel if state[0] else 1-rel, s['prefix']).tolist())
    # The first eight content/value marks are unavailable to every reader.
    for row in p['observations'][:8]:
        row[:2] = [0, 0]
    return p, dict(y=[int(state[0])], state=state.tolist(), law=int(law))


def r2_trajectory(v, p, policy, law=None):
    s = v['settings']; state = G_STATES
    rel = np.linspace(.65, .95, 16)[state[:, 3]]
    ll = np.log(np.where(state[:, 0], 1-s['benign_share'], s['benign_share']))
    if s['coupled']:
        ll = np.where(state[:, 1] == state[:, 2], ll, -np.inf)
    if law is not None:
        ll = np.where(state[:, 3] == law, ll, -np.inf)
    rule = dict(r2_policies())[policy]
    fired = False; memory = np.full(3, .5); path = []
    for step in range(s['prefix']):
        flag_before = float(normalize(ll)@state[:, 0])
        if rule:
            floor, threshold, release = rule
            # Resolve mathematical boundary ties identically in log-space and
            # probability-space calculations; this is below the checker tolerance.
            if release == 'release' and fired and flag_before < .5-1e-12:
                fired = False
            fired = fired or flag_before >= threshold-1e-12
        elif policy == 'withdraw-content':
            fired = fired or flag_before >= .7-1e-12
        precision = np.ones(3); flag_precision = 1.
        if policy == 'early-guard':
            precision[:] = .2; flag_precision = .2
        elif policy == 'procedure-only':
            precision[:2] = 0; flag_precision = 0
        elif fired:
            precision[:2] = rule[0] if rule else .05
            precision[2] = .5 if s['coupled'] else .9
            flag_precision = rule[0] if rule else 1.
        flagprob = np.where(state[:, 0], rel, 1-rel)
        ll += flag_precision*binary_ll(p['flags'][step], flagprob)
        precision *= np.array([step >= 8, step >= 8, True])
        for head in range(3):
            ll += precision[head]*binary_ll(p['observations'][step][head], np.where(state[:, head], rel, 1-rel))
        belief = normalize(ll)@state[:, :3]
        memory += .25*precision*(belief-memory)
        path.append(dict(purpose=float(belief[0]), memory=memory.tolist(), fired=bool(fired),
                         flag_precision=float(flag_precision), precision=precision.tolist()))
    return bern(belief), path


def evaluate_r2(v, p, truth=None):
    heads, paths = {}, {}
    for name, _ in r2_policies():
        heads[name], paths[name] = r2_trajectory(v, p, name)
    heads['law-marginal'] = heads['no-gate']
    if truth is not None:
        heads['oracle-law'] = r2_trajectory(v, p, 'no-gate', truth['law'])[0]
    return {k: a[:1] for k, a in heads.items()}, dict(trajectories=paths, heads={k: a.tolist() for k, a in heads.items()})


def features_r2(v, p):
    x = np.zeros((32, 5))
    for i in range(v['settings']['prefix']):
        x[i] = [*p['observations'][i], p['flags'][i], 1]
        if i < 8:
            x[i, :2] = 0
    return x.ravel()


@lru_cache(None)
def setpoints(d):
    return np.array(list(product((-1., 0., 1.), repeat=d)))


R3_LAWS = tuple((float(n), 2., float(beta)) for n, beta in product(np.linspace(2, 4, 4), np.linspace(.8, 3, 4)))


def choice3(hstar, h, effects, costs, law, cost_scale):
    n, m, beta = R3_LAWS[law]
    hstar = np.atleast_2d(hstar)
    before = np.sum(np.abs(hstar-h)**n, -1)**(1/m)
    after = np.sum(np.abs(hstar[:, None, :]-h-effects[None, :, :])**n, -1)**(1/m)
    return normalize(beta*(before[:, None]-after-cost_scale*costs))


def sample_r3(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    states = setpoints(s['dimensions']); target = states[r.integers(len(states))]
    cost_scale = {'relevant': 1., 'irrelevant': 0., 'half': .5}[s['cost_role']]
    rows = []
    for step in range(s['records']+1):
        scale = 3 if s['shift'] and step == s['records'] else 1
        h = r.uniform(-scale, scale, s['dimensions'])
        effects = r.choice((-.75, 0., .75), (3, s['dimensions']))
        costs = r.uniform(0, 1, 3)
        choice = int(r.choice(3, p=choice3(target, h, effects, costs, law, cost_scale)[0]))
        rows.append(dict(h=h.tolist(), effects=effects.tolist(), costs=costs.tolist(), picked=choice))
    current = rows.pop(); y = current.pop('picked')
    return dict(history=rows, current=current), dict(y=[y], setpoint=target.tolist(), law=int(law), cost_role=cost_scale)


def r3_likelihoods(v,p):
    """Vectorize fixed finite menus; reuse within one case only."""
    states=setpoints(v['settings']['dimensions']);rows=p['history']+[p['current']]
    h=np.array([row['h'] for row in rows]);effects=np.array([row['effects'] for row in rows])
    costs=np.array([row['costs'] for row in rows]);picked=np.array([row['picked'] for row in p['history']])
    distances=np.abs(states[None,:,None,:]-h[:,None,None,:]-effects[:,None,:,:])
    deficits={n:np.sum(distances**n,axis=-1)**.5 for n,_,_ in R3_LAWS}
    result={}
    for role in (0.,.5,1.):
        for law,(n,m,beta) in enumerate(R3_LAWS):
            # Common before-action drive cancels analytically from softmax.
            probabilities=normalize(-beta*(deficits[n]+role*costs[:,None,:]))
            with np.errstate(divide='ignore'):
                loghistory=np.log(probabilities[np.arange(len(picked))[:,None],np.arange(len(states))[None,:],picked[:,None]])
            result[role,law]=(loghistory,probabilities[-1])
    return result


def r3_infer(v, p, role=None, weighting='unit', known_law=None, _cache=None):
    s = v['settings']; states = setpoints(s['dimensions'])
    roles = (0., .5, 1.) if role is None else (role,)
    laws = list(range(16)) if known_law is None else [known_law]
    ll = np.zeros((len(roles), len(laws), len(states)))
    previous, weights = [], []
    for row in p['history']:
        picked = row['picked']; habit = max(range(3), key=previous.count) if previous else picked
        weights.append(1+row['costs'][picked]*(picked != habit)); previous.append(picked)
    if weighting == 'tempered':
        weights = [float(np.mean(weights))]*len(weights)
    elif weighting == 'unit':
        weights = [1.]*len(weights)
    cache=r3_likelihoods(v,p) if _cache is None else _cache
    for i,r in enumerate(roles):
        for j,law in enumerate(laws):
            ll[i,j]=np.sum(np.asarray(weights)[:,None]*cache[r,law][0],axis=0)
    post = normalize(ll.ravel()).reshape(ll.shape)
    cur = p['current']; pred = np.zeros(3)
    for i, r in enumerate(roles):
        for j, law in enumerate(laws):
            pred += post[i, j]@cache[r,law][1]
    return pred[None, :], post.sum((0, 1)), post.sum((1, 2))


def evaluate_r3(v, p, truth=None):
    true_role = {'relevant': 1., 'irrelevant': 0., 'half': .5}[v['settings']['cost_role']]
    specs = [('law-marginal', true_role, 'unit'), ('unit', true_role, 'unit'),
             ('relevance-aware', None, 'unit'), ('assumes-relevant', 1., 'unit'),
             ('effort-weighted', true_role, 'effort'), ('matched-tempering', true_role, 'tempered'),
             ('cost-ignorant', 0., 'unit')]
    pred, details = {}, {};cache=r3_likelihoods(v,p)
    for name, role, weighting in specs:
        pred[name], post, roles = r3_infer(v, p, role, weighting,_cache=cache)
        details[name] = dict(setpoint_posterior=post.tolist(), role_posterior=roles.tolist())
    counts = np.bincount([row['picked'] for row in p['history']], minlength=3)+.5
    pred['frequency'] = (counts/counts.sum())[None, :]
    if truth is not None:
        pred['oracle-law'] = r3_infer(v, p, true_role, known_law=truth['law'],_cache=cache)[0]
    return pred, details


def features_r3(v, p):
    d = v['settings']['dimensions']; x = np.zeros((129, 24))
    for i, row in enumerate(p['history']+[p['current']]):
        at = i if 'picked' in row else 128
        x[at, :d] = row['h']; effect = np.zeros((3, 4)); effect[:, :d] = row['effects']
        x[at, 4:16] = effect.ravel(); x[at, 16:19] = row['costs']
        if 'picked' in row:
            x[at, 19+row['picked']] = 1
        x[at, 22] = 1; x[at, 23] = d/4
    return x.ravel()


@lru_cache(None)
def tree(branching, depth=4):
    parents, levels = [-1], [0]
    for i in range(sum(branching**d for d in range(depth))):
        parents.extend([i]*branching); levels.extend([levels[i]+1]*branching)
    return np.array(parents), np.array(levels)


def sample_r4(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    parent, levels = tree(s['branching']); state = np.zeros(len(parent), int)
    state[0] = r.integers(2); reliability = np.linspace(.60, .98, 16)[law]
    for i in range(1, len(parent)):
        state[i] = state[parent[i]] ^ int(r.random() > reliability)
    visible = (levels == 4) & (r.random(len(parent)) < s['coverage'])
    observations = np.where(visible, state ^ (r.random(len(parent)) > .9), -1)
    cue_node = 0 if s['cue_position'] == 'root' else 1
    cue = int(state[cue_node] ^ (r.random() > s['cue_reliability']))
    return dict(evidence=observations.tolist(), cue=cue, cue_node=cue_node), dict(y=[int(state[0])], nodes=state.tolist(), law=int(law))


def tree_messages(parent, emission, reliability, mode='joint'):
    logt = np.log([[reliability, 1-reliability], [1-reliability, reliability]])
    below, messages = emission.copy(), np.zeros_like(emission)
    for i in range(len(parent)-1, 0, -1):
        messages[i] = logsumexp(logt+below[i][None, :], axis=1)
        below[parent[i]] += messages[i]
    evidence = float(logsumexp(below[0])-np.log(2))
    if mode == 'top-down':
        probs = np.zeros_like(emission); probs[0] = normalize(emission[0])
        for i in range(1, len(parent)):
            probs[i] = normalize(np.log(probs[parent[i]]@np.exp(logt))+emission[i])
        return probs, evidence
    above = np.zeros_like(emission)-np.log(2)
    for i in range(1, len(parent)):
        outside = above[parent[i]]+below[parent[i]]-messages[i]
        above[i] = logsumexp(outside[:, None]+logt, axis=0)
    return normalize(above+below), evidence


def r4_infer(v, p, mode='joint', cue=True, marginal_rho=False, law=None, fixed_trust=False):
    s = v['settings']; parent, levels = tree(s['branching'])
    rhos = (.3, .5, .7, .9, .99) if marginal_rho else (.9 if fixed_trust else s['cue_reliability'],)
    combinations=list(product(range(16) if law is None else [law],rhos))
    reliability=np.array([np.linspace(.60,.98,16)[ell] for ell,_ in combinations])
    emission=np.zeros((len(parent),len(combinations),2))
    for i,obs in enumerate(p['evidence']):
        if obs>=0 and mode!='top-down':emission[i]=np.log([.9 if obs==0 else .1,.9 if obs==1 else .1])
    if cue:
        probability=np.array([rho for _,rho in combinations])
        emission[p['cue_node']]+=np.log(np.stack([probability if p['cue']==0 else 1-probability,
            probability if p['cue']==1 else 1-probability],axis=1))
    transition=np.log(np.stack([np.stack([reliability,1-reliability],axis=1),
        np.stack([1-reliability,reliability],axis=1)],axis=1))
    below=emission.copy();messages=np.zeros_like(emission)
    for level in range(4,0,-1):
        nodes=np.flatnonzero(levels==level);ancestors=np.flatnonzero(levels==level-1)
        messages[nodes]=logsumexp(transition[None]+below[nodes][:,:,None,:],axis=3)
        below[ancestors]+=messages[nodes].reshape(len(ancestors),s['branching'],len(combinations),2).sum(1)
    likelihood=logsumexp(below[0],axis=1)-np.log(2)
    if mode=='top-down':
        posterior=np.zeros_like(emission);posterior[0]=normalize(emission[0])
        for level in range(1,5):
            nodes=np.flatnonzero(levels==level)
            predicted=np.einsum('nki,kij->nkj',posterior[parent[nodes]],np.exp(transition))
            posterior[nodes]=normalize(np.log(predicted)+emission[nodes])
    else:
        above=np.zeros_like(emission)-np.log(2)
        for level in range(1,5):
            nodes=np.flatnonzero(levels==level)
            outside=above[parent[nodes]]+below[parent[nodes]]-messages[nodes]
            above[nodes]=logsumexp(outside[:,:,:,None]+transition[None],axis=2)
        posterior=normalize(above+below)
    return np.einsum('k,nkj->nj',normalize(likelihood),posterior)


def evaluate_r4(v, p, truth=None):
    full = {name: r4_infer(v, p, **kw) for name, kw in [
        ('law-marginal', {}), ('bottom-up', {'cue': False}),
        ('top-down', {'mode': 'top-down'}), ('without-cue', {'cue': False}),
        ('rho-marginal', {'marginal_rho': True}), ('fixed-trust', {'fixed_trust': True})]}
    if truth is not None:
        full['oracle-law'] = r4_infer(v, p, law=truth['law'])
    return {k: a[:1] for k, a in full.items()}, dict(nodes={k: a.tolist() for k, a in full.items()})


def features_r4(v, p):
    x = np.zeros((121, 2)); evidence = np.array(p['evidence'])
    x[:len(evidence), 0] = np.where(evidence >= 0, 2*evidence-1, 0)
    x[:len(evidence), 1] = evidence >= 0
    return np.r_[x.ravel(), p['cue'], p['cue_node']/121]


def sample(v, phase, world, case, law=None):
    return globals()['sample_'+v['family'].lower()](v, phase, world, case, law)


def evaluate(v, p, truth=None):
    return globals()['evaluate_'+v['family'].lower()](v, p, truth)


def features(v, p):
    return globals()['features_'+v['family'].lower()](v, p)
