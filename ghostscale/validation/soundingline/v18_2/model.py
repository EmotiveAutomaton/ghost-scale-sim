"""Thin policy wrapper over V16's executable board and acquired-fragment learner.

The finite grid is a declared model, not a human ontology. A stochastic action is
one draw. Evaluator truth never enters the serialized reader interface.
"""
from functools import lru_cache
from itertools import permutations, product
import json
import math
import random
import zlib
import numpy as np
from ..v16.world import execute
from ..v16.learning import learn, encoding_cost

STATES = tuple(product(range(3), range(3), range(2), range(2)))
PROGRAMS = tuple(p for n in range(4) for p in permutations(range(4), n))
ARTIFACTS = tuple(execute(p).artifact for p in PROGRAMS)
TIERS = ('one-artifact', 'artifact-history', 'process-history')
SCHEMA = 'v18.2.public.1'


def seed(*parts):
    return zlib.crc32(json.dumps(parts, sort_keys=True).encode())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def world(split, index, family='board'):
    # Disjoint task/feature incidence structures, not independent new physics.
    groups = {'train': ((0, 1), (2, 3)), 'dev': ((0, 2), (1, 3)),
              'test': ((0, 3), (1, 2)), 'export': ((0, 3), (1, 2))}[split]
    return dict(groups=[list(x) for x in groups], price=round(.18 + .03*(index % 5), 3),
                family=family, max_code=2, temperature=.7)


@lru_cache(maxsize=512)
def library(groups, skill):
    if skill == 0:
        return ()
    motif = groups[skill-1]
    target = sum(1 << x for x in motif)
    return learn([motif]*4, [target]*4, capacity=1).library


def context(goal, signal, reader_fact=None, noticed=1.0):
    return dict(goal=goal, signal=signal, reader_fact=reader_fact, noticed=noticed)


@lru_cache(maxsize=32768)
def _policy(encoded_world, state, goal, signal, noticed):
    w = json.loads(encoded_world)
    skill, preference, default_goal, prior_belief = state
    groups = tuple(map(tuple, w['groups']))
    lib = library(groups, skill)
    intended = default_goal if goal is None else goal
    # A signal names the current routing convention, not evaluator world truth.
    beliefs = [(prior_belief, 1.0)] if signal is None else [(signal, noticed), (prior_belief, 1-noticed)]
    probabilities = np.zeros(len(PROGRAMS))
    costs = np.array([encoding_cost(p, lib) for p in PROGRAMS], float)
    for belief, mass in beliefs:
        target = 15 if intended == 2 else sum(1 << x for x in groups[intended ^ belief])
        rewards = []
        for p, artifact, cost in zip(PROGRAMS, ARTIFACTS, costs):
            # Three tradeoffs: group A, neutral, group B. Goal reward is separate.
            a = sum(bool(artifact & (1 << x)) for x in groups[0])
            b = sum(bool(artifact & (1 << x)) for x in groups[1])
            utility = -1.6*(artifact ^ target).bit_count() + .65*(preference-1)*(b-a) - w['price']*cost
            allowed = cost <= w['max_code']
            if w['family'] == 'restricted':
                allowed &= 3 not in p
            rewards.append(utility/w['temperature'] if allowed else -np.inf)
        rewards = np.array(rewards)
        values = np.exp(rewards-np.max(rewards)); values /= values.sum()
        probabilities += mass*values
    return tuple(probabilities)


def policy(w, state, ctx):
    return np.array(_policy(canonical(w).decode(), tuple(state), ctx['goal'], ctx['signal'], ctx['noticed']))


def artifacts(probabilities):
    result = np.zeros(16)
    for artifact, p in zip(ARTIFACTS, probabilities):
        result[artifact] += p
    return result


def draw(w, state, ctx, rng):
    p = policy(w, state, ctx)
    choice = rng.choices(range(len(PROGRAMS)), weights=p, k=1)[0]
    program = list(PROGRAMS[choice])
    result = execute(program)
    return dict(context=ctx, artifact=result.artifact, program=program), choice


def public_packet(case, tier, probe=0, history=None):
    records = case['history'] if history is None else history
    if tier == 'one-artifact': records = records[-1:]
    allowed = [dict(context=x['context'], artifact=x['artifact'],
                    program=x['program'] if tier == 'process-history' else None) for x in records]
    return canonical(dict(schema=SCHEMA, case_id=case['case_id'], world=case['world'],
                          tier=tier, history=allowed, current=case['probes'][probe]['context']))


def parse(payload):
    p = json.loads(payload)
    if set(p) != {'schema', 'case_id', 'world', 'tier', 'history', 'current'} or p['schema'] != SCHEMA:
        raise ValueError('public boundary violation')
    if p['tier'] not in TIERS or len(p['history']) > 64:
        raise ValueError('invalid evidence tier/history')
    if set(p['world']) != {'groups', 'price', 'family', 'max_code', 'temperature'}:
        raise ValueError('private world fields')
    for c in [p['current'], *[h['context'] for h in p['history']]]:
        if set(c) != {'goal', 'signal', 'reader_fact', 'noticed'} or c['goal'] not in (None,0,1,2) or c['signal'] not in (None,0,1) or c['reader_fact'] not in (None,0,1) or not 0 <= c['noticed'] <= 1:
            raise ValueError('private or invalid context fields')
    for h in p['history']:
        if set(h) != {'context','artifact','program'} or h['artifact'] not in range(16):
            raise ValueError('private observation fields')
        if p['tier'] != 'process-history' and h['program'] is not None:
            raise ValueError('process evidence leaked')
        if h['program'] is not None and (tuple(h['program']) not in PROGRAMS or execute(h['program']).artifact != h['artifact']):
            raise ValueError('bad public execution')
    return p


def likelihood(w, states, observation):
    rows = np.array([policy(w, s, observation['context']) for s in states])
    if observation['program'] is not None:
        return rows[:, PROGRAMS.index(tuple(observation['program']))]
    return rows[:, np.array(ARTIFACTS) == observation['artifact']].sum(axis=1)


def infer(payload, method='persistent', states=STATES, transition=0.):
    p = parse(payload)
    if method == 'raw':
        count=np.ones(16)*.025
        for obs in p['history']:
            distance=sum(obs['context'][k]!=p['current'][k] for k in ('goal','signal','noticed'))
            count[obs['artifact']]+=math.exp(-distance)
        return dict(mismatch=False,probabilities=(count/count.sum()).tolist(),posterior=None,
                    evaluations=3*len(p['history'])+16,log_evidence=None)
    weights = np.ones(len(states))/len(states)
    work = 0; log_evidence = 0.; mismatch = False
    history = [] if method == 'direct' else p['history']
    if method == 'rebuilt': history = history[-1:]
    for obs in history:
        if transition:
            # Only preference can change in this declared alternative; no intervention labels.
            mixed = np.zeros_like(weights)
            for i, a in enumerate(states):
                peers = [j for j,b in enumerate(states) if (a[0],a[2],a[3]) == (b[0],b[2],b[3])]
                mixed[i] = weights[peers].sum()/len(peers)
            weights = (1-transition)*weights + transition*mixed
        likelihoods = likelihood(p['world'], states, obs)
        weights *= likelihoods; total = float(weights.sum()); work += len(states)*len(PROGRAMS)
        if total <= 0:
            mismatch = True; break
        log_evidence += math.log(total); weights /= total
    if mismatch:
        return dict(mismatch=True, probabilities=None, posterior=None, evaluations=work, log_evidence=None)
    distributions = np.array([artifacts(policy(p['world'], s, p['current'])) for s in states])
    prediction = weights @ distributions
    return dict(mismatch=False, probabilities=prediction.tolist(), posterior=weights.tolist(),
                evaluations=work+len(states)*len(PROGRAMS), log_evidence=log_evidence)


def make_case(namespace, index, split='test', length=8, change=False, family='board',probe_mode='standard'):
    rng = random.Random(seed(namespace, split, index))
    state = STATES[rng.randrange(len(STATES))]
    w = world(split,index,family)
    history=[]; truth=[]
    for t in range(length):
        s = list(state)
        if change and t >= length//2: s[1] = (s[1]+1)%3
        c = context(rng.choice([0,1,None]),rng.choice([0,1,None]),rng.choice([0,1,None]))
        obs, choice = draw(w,s,c,rng); history.append(obs); truth.append(dict(state=s,choice=choice))
    future_state = list(state)
    if change: future_state[1] = (state[1]+1)%3
    probes=[]
    for j in range(4):
        c=context(j%2, j//2)
        if probe_mode=='uncertain' or (probe_mode=='mixed' and index%2):
            c=context([None,0,1,None][j],[None,None,None,1][j])
        obs, choice=draw(w,future_state,c,rng)
        probes.append(dict(context=c,observed=obs,choice=choice))
    return dict(case_id=f'{namespace}:{split}:{index:05d}', maker_id=f'{namespace}:{split}:maker-{index}',
                world_id=f'{split}:incidence-{index%5}', world=w, history=history, probes=probes,
                truth=dict(initial_state=list(state), future_state=future_state, history=truth),
                split=split, independent_unit='maker/world lineage')


def score(prediction, truth, observed):
    p=np.array(prediction,float); q=np.array(truth,float)
    return dict(log_loss=-math.log(max(p[observed],1e-12)),
                brier=float(np.sum((p-np.eye(16)[observed])**2)),
                expected_log_loss=float(-np.sum(q*np.log(np.maximum(p,1e-12)))),
                expected_brier=float(np.sum((p-q)**2)),
                action_accuracy=float(np.argmax(p)==observed),
                oracle_action_mass=float(q[np.argmax(p)]))


def evaluate(case, methods=('direct','raw','persistent','oracle'), tiers=TIERS):
    rows=[]
    for tier in tiers:
        for j, probe in enumerate(case['probes']):
            payload=public_packet(case,tier,j)
            true=artifacts(policy(case['world'],case['truth']['future_state'],probe['context']))
            for method in methods:
                if method=='oracle':
                    result=dict(probabilities=true.tolist(),evaluations=len(PROGRAMS),posterior=None,mismatch=False)
                else:
                    result=infer(payload,method,transition=.15 if method=='adaptive' else 0.)
                if result['mismatch']:
                    rows.append(dict(tier=tier,probe=j,method=method,instrument='model_mismatch',result=result));continue
                scores=score(result['probabilities'],true,probe['observed']['artifact'])
                if result.get('posterior') is not None:
                    weights=np.array(result['posterior'])
                    true_index=STATES.index(tuple(case['truth']['future_state']))
                    # Distribution-equivalent states in the supplied future context;
                    # joint likelihood remains full-sized, no arbitrary MAP biography.
                    equivalent=np.array([np.allclose(artifacts(policy(case['world'],s,probe['context'])),true,atol=1e-12) for s in STATES])
                    ordering=np.argsort(-weights);selected=ordering[np.cumsum(weights[ordering])-weights[ordering]<.95]
                    scores['state_credible_coverage']=float(true_index in selected)
                    scores['equivalence_coverage']=float(any(equivalent[k] for k in selected))
                    scores['equivalence_mass']=float(weights[equivalent].sum())
                rows.append(dict(tier=tier,probe=j,method=method,instrument='valid',result=result,scores=scores))
    return rows
