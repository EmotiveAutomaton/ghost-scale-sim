"""Separate probability-space reconstruction; no production posterior imports."""
from itertools import product
import math
import numpy as np


def r1(v, public, *, masked=False, missing=False, prevalence=None, law=None):
    s = v['settings']; prevalence = s['prevalence'] if prevalence is None else prevalence
    weighted = []
    for q, g, t, a, ell in product(range(2), range(2), range(2), range(2), range(16)):
        reliability = .6+.3*ell/15
        mass = prevalence if g else 1-prevalence
        mass *= .75 if public['own'] == q else .25
        implied = a if g else q
        mass *= reliability if public['claim'] == implied else 1-reliability
        if not masked:
            for bit in public['selection']:
                mass *= s['strength'] if bit == g else 1-s['strength']
        for bit in public['operations']:
            mass *= reliability if bit == t else 1-reliability
        if missing and g or law is not None and ell != law:
            mass = 0.
        weighted.append(((q, g, t, a, ell), mass))
    total = math.fsum(mass for _, mass in weighted)
    if not total > 0: raise ValueError('empty independent support')
    post = np.array([mass/total for _, mass in weighted])
    heads = np.zeros((3, 2))
    for state, mass in weighted:
        for h in range(3): heads[h, state[h]] += mass/total
    return heads, post


def verify_r1(v, public, truth, predictions, details):
    specs = {'law-marginal': {}, 'masked': {'masked': True},
             'missing-process': {'missing': True}, 'rare-steering-prior': {'prevalence': .01}}
    if 'oracle-law' in predictions: specs['oracle-law'] = {'law': truth['law']}
    error = 0.
    for name, kw in specs.items():
        heads, posterior = r1(v, public, **kw)
        error = max(error, float(np.max(np.abs(heads-np.asarray(details['heads'][name])))),
                    float(np.max(np.abs(heads[:1]-predictions[name]))))
        if 'posteriors' in details:
            error = max(error, float(np.max(np.abs(posterior-np.asarray(details['posteriors'][name])))))
    if error > 1e-12: raise ValueError('independent R1 reconstruction mismatch')
    return error


def r2(v, public, policy, law=None):
    """Probability-mass sequential update, separately enumerated state support."""
    states = np.array(list(product((0, 1), (0, 1), (0, 1), range(16))))
    rel = .65+.30*states[:, 3]/15
    s = v['settings']; mass = np.where(states[:, 0] == 0, s['benign_share'], 1-s['benign_share'])
    if s['coupled']: mass *= states[:, 1] == states[:, 2]
    if law is not None: mass *= states[:, 3] == law
    mass /= mass.sum(); fired = False; memory = np.full(3, .5); paths = []
    rule = None
    if policy.startswith('withdraw-all-'):
        parts = policy.split('-'); rule = float(parts[2]), float(parts[3]), parts[4]
    for index in range(s['prefix']):
        prior_flag = float(mass[states[:, 0] == 1].sum())
        if rule:
            if rule[2] == 'release' and prior_flag < .5-1e-12: fired = False
            fired = fired or prior_flag >= rule[1]-1e-12
        if policy == 'withdraw-content': fired = fired or prior_flag >= .7-1e-12
        attention, flag = np.ones(3), 1.
        if policy == 'early-guard': attention[:], flag = .2, .2
        elif policy == 'procedure-only': attention[:2], flag = 0., 0.
        elif fired:
            attention[:2] = rule[0] if rule else .05
            attention[2] = .5 if s['coupled'] else .9
            flag = rule[0] if rule else 1.
        mass *= np.where(states[:, 0] == public['flags'][index], rel, 1-rel)**flag
        if index < 8: attention[:2] = 0
        for head, observed in enumerate(public['observations'][index]):
            mass *= np.where(states[:, head] == observed, rel, 1-rel)**attention[head]
        mass /= mass.sum()
        p = np.array([mass[states[:, h] == 1].sum() for h in range(3)])
        memory = memory+.25*attention*(p-memory)
        paths.append(dict(purpose=float(p[0]), memory=memory.copy(), fired=fired, flag_precision=flag, precision=attention.copy()))
    return np.stack([1-p, p], axis=1), paths


def verify_r2(v, public, truth, predictions, details):
    error = 0.
    for name in details['trajectories']:
        heads, paths = r2(v, public, name)
        error = max(error, float(np.max(np.abs(heads-np.asarray(details['heads'][name])))))
        for a, b in zip(paths, details['trajectories'][name], strict=True):
            for key in a:
                if key == 'fired':
                    if a[key] != b[key]: raise ValueError('independent R2 gate trajectory mismatch')
                else: error = max(error, float(np.max(np.abs(np.asarray(a[key])-b[key]))))
    if 'oracle-law' in predictions:
        error = max(error, float(np.max(np.abs(r2(v, public, 'no-gate', truth['law'])[0][:1]-predictions['oracle-law']))))
    if error > 1e-12: raise ValueError('independent R2 reconstruction mismatch')
    return error


def r4(v, public, *, cue=True, marginal_rho=False, fixed_trust=False, law=None, topdown=False):
    """Conditional subtree probabilities and ancestral posteriors, in probability space."""
    b = v['settings']['branching']; count = sum(b**d for d in range(5))
    parents = [-1]+[(i-1)//b for i in range(1, count)]
    children = [[j for j in range(1, count) if parents[j] == i] for i in range(count)]
    mixed, weights = [], []
    for ell in range(16) if law is None else [law]:
        r = .6+.38*ell/15; transition = np.array([[r, 1-r], [1-r, r]])
        rhos = (.3, .5, .7, .9, .99) if marginal_rho else ((.9 if fixed_trust else v['settings']['cue_reliability']),)
        for rho in rhos:
            emission = np.ones((count, 2))
            if not topdown:
                for i, obs in enumerate(public['evidence']):
                    if obs >= 0: emission[i] = [.9 if state == obs else .1 for state in (0, 1)]
            if cue:
                emission[public['cue_node']] *= [rho if state == public['cue'] else 1-rho for state in (0, 1)]
            subtree = emission.copy()
            for i in range(count-1, -1, -1):
                for child in children[i]: subtree[i] *= transition@subtree[child]
            likelihood = float(subtree[0].sum()/2)
            posterior = np.zeros((count, 2))
            if topdown:
                posterior[0] = emission[0]/emission[0].sum()
                for i in range(1, count):
                    p = (posterior[parents[i]]@transition)*emission[i]; posterior[i] = p/p.sum()
            else:
                posterior[0] = subtree[0]/subtree[0].sum()
                for i in range(1, count):
                    conditioned = transition*subtree[i]
                    conditioned /= conditioned.sum(1, keepdims=True)
                    posterior[i] = posterior[parents[i]]@conditioned
            mixed.append(posterior); weights.append(likelihood)
    weights = np.asarray(weights)/math.fsum(weights)
    return sum(w*p for w, p in zip(weights, mixed, strict=True))


def r4_grid(v,public,*,cue=True,marginal_rho=False,fixed_trust=False,law=None,topdown=False):
    """Independent probability-space ancestral recursion, laws vectorized."""
    b=v['settings']['branching'];count=sum(b**d for d in range(5))
    parents=[-1]+[(i-1)//b for i in range(1,count)]
    rhos=(.3,.5,.7,.9,.99) if marginal_rho else ((.9 if fixed_trust else v['settings']['cue_reliability']),)
    settings=list(product(range(16) if law is None else [law],rhos))
    matrices=np.array([[[.6+.38*ell/15,.4-.38*ell/15],[.4-.38*ell/15,.6+.38*ell/15]] for ell,_ in settings])
    emission=np.ones((count,len(settings),2))
    for i,obs in enumerate(public['evidence']):
        if obs>=0 and not topdown:emission[i]=[.9 if state==obs else .1 for state in (0,1)]
    if cue:
        emission[public['cue_node']]*=np.array([[rho if state==public['cue'] else 1-rho for state in (0,1)] for _,rho in settings])
    subtree=emission.copy()
    layers=[np.arange(sum(b**d for d in range(level)),sum(b**d for d in range(level+1))) for level in range(5)]
    for level in range(4,0,-1):
        child_mass=np.sum(matrices[None]*subtree[layers[level]][:,:,None,:],axis=3)
        subtree[layers[level-1]]*=child_mass.reshape(len(layers[level-1]),b,len(settings),2).prod(1)
    likelihood=subtree[0].sum(1)/2
    post=np.zeros_like(subtree)
    post[0]=(emission[0]/emission[0].sum(1,keepdims=True)) if topdown else (subtree[0]/subtree[0].sum(1,keepdims=True))
    for level in range(1,5):
        nodes=layers[level];parent_nodes=np.array(parents)[nodes]
        if topdown:
            mass=np.sum(post[parent_nodes][:,:,:,None]*matrices[None],axis=2)*emission[nodes]
            post[nodes]=mass/mass.sum(2,keepdims=True)
        else:
            conditioned=matrices[None]*subtree[nodes][:,:,None,:]
            conditioned/=conditioned.sum(3,keepdims=True)
            post[nodes]=np.sum(post[parent_nodes][:,:,:,None]*conditioned,axis=2)
    weights=likelihood/math.fsum(likelihood)
    return np.sum(post*weights[None,:,None],axis=1)


def verify_r4(v, public, truth, predictions, details):
    specs = {'law-marginal': {}, 'bottom-up': {'cue': False}, 'without-cue': {'cue': False},
             'top-down': {'topdown': True}, 'rho-marginal': {'marginal_rho': True}, 'fixed-trust': {'fixed_trust': True}}
    if 'oracle-law' in predictions: specs['oracle-law'] = {'law': truth['law']}
    error = max(float(np.max(np.abs(r4_grid(v, public, **kw)-np.asarray(details['nodes'][name])))) for name, kw in specs.items())
    if error > 1e-12: raise ValueError('independent R4 reconstruction mismatch')
    return error


def r3(v, public, role=None, weighting='unit', law=None, _cache=None):
    d = v['settings']['dimensions']; points = np.array(list(product((-1., 0., 1.), repeat=d)))
    law_menu = [(2+i*2/3, .8+j*2.2/3) for i in range(4) for j in range(4)]
    roles = (0., .5, 1.) if role is None else (role,)
    laws = range(16) if law is None else [law]
    counts, row_weights = [0, 0, 0], []
    for index, row in enumerate(public['history']):
        choice = row['picked']; habitual = counts.index(max(counts)) if index else choice
        row_weights.append(1+row['costs'][choice]*float(choice != habitual)); counts[choice] += 1
    if weighting == 'unit': row_weights = [1.]*len(row_weights)
    elif weighting == 'tempered': row_weights = [math.fsum(row_weights)/len(row_weights)]*len(row_weights)
    def probabilities(row, role, ell):
        key=(id(row),role,ell)
        if _cache is not None and key in _cache:return _cache[key]
        exponent, beta = law_menu[ell]
        h = np.asarray(row['h']); e = np.asarray(row['effects']); costs = np.asarray(row['costs'])
        deficit = np.sqrt(np.sum(np.abs(points[:, None, :]-h-e)**exponent, axis=2))
        # The before-action deficit is the same for every offered action and
        # cancels from the softmax, unlike the production formula.
        z = -beta*(deficit+role*costs); z -= z.max(1, keepdims=True)
        z = np.exp(z); value=z/z.sum(1, keepdims=True)
        if _cache is not None:_cache[key]=value
        return value
    joint, future = [], []
    for role in roles:
        row_laws, next_laws = [], []
        for ell in laws:
            logmass = np.zeros(len(points))
            for row, w in zip(public['history'], row_weights, strict=True):
                logmass += w*np.log(probabilities(row, role, ell)[:, row['picked']])
            row_laws.append(logmass); next_laws.append(probabilities(public['current'], role, ell))
        joint.append(row_laws); future.append(next_laws)
    logmass = np.asarray(joint); mass = np.exp(logmass-logmass.max()); mass /= mass.sum()
    predicted = np.sum(mass[..., None]*np.asarray(future), axis=(0, 1, 2))[None, :]
    return predicted, mass.sum((0, 1)), mass.sum((1, 2))


def verify_r3(v, public, truth, predictions, details):
    role = {'relevant': 1., 'irrelevant': 0., 'half': .5}[v['settings']['cost_role']]
    specifications = {'law-marginal': (role, 'unit'), 'unit': (role, 'unit'),
        'relevance-aware': (None, 'unit'), 'assumes-relevant': (1., 'unit'),
        'effort-weighted': (role, 'effort'), 'matched-tempering': (role, 'tempered'), 'cost-ignorant': (0., 'unit')}
    error = 0.;cache={}
    # Independent probability-space tables, shared only within this verification.
    # Retain r3's scalar path for adversarial comparisons and standalone checks.
    points=np.array(list(product((-1.,0.,1.),repeat=v['settings']['dimensions'])))
    rows=public['history']+[public['current']]
    for exponent_index in range(4):
        exponent=2+exponent_index*2/3
        after=np.asarray([np.sqrt(np.sum(np.abs(points[:,None,:]-np.asarray(row['h'])-
            np.asarray(row['effects']))**exponent,axis=2)) for row in rows])
        costs=np.asarray([row['costs'] for row in rows])
        for beta_index in range(4):
            ell=exponent_index*4+beta_index;beta=.8+beta_index*2.2/3
            for cost_role in (0.,.5,1.):
                mass=np.exp(-beta*(after+cost_role*costs[:,None,:]))
                mass/=mass.sum(axis=2,keepdims=True)
                for index,row in enumerate(rows):cache[id(row),cost_role,ell]=mass[index]
    for name, (cost_role, weighting) in specifications.items():
        probability, post, roles = r3(v, public, cost_role, weighting,_cache=cache)
        error = max(error, float(np.max(np.abs(probability-predictions[name]))),
                    float(np.max(np.abs(post-np.asarray(details[name]['setpoint_posterior'])))),
                    float(np.max(np.abs(roles-np.asarray(details[name]['role_posterior'])))))
    if 'oracle-law' in predictions:
        error = max(error, float(np.max(np.abs(r3(v, public, role, law=truth['law'],_cache=cache)[0]-predictions['oracle-law']))))
    if error > 1e-12: raise ValueError('independent R3 reconstruction mismatch')
    return error


def scores(probabilities, truth):
    """Independent scalar score consumer, preserving zero probability loss."""
    rows = []
    for prediction, y in zip(probabilities, truth, strict=True):
        p = list(map(float, prediction)); total = math.fsum(p)
        if abs(total-1) > 1e-10 or min(p) < 0 or not all(math.isfinite(x) for x in p):
            raise ValueError('invalid checked probability')
        rows.append(dict(log_loss=-math.log(p[y]) if p[y] else math.inf,
            half_brier=math.fsum((a-float(i == y))**2 for i, a in enumerate(p))/2,
            accuracy=float(max(range(len(p)), key=p.__getitem__) == y)))
    return {name: np.array([r[name] for r in rows]) for name in rows[0]}
