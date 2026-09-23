"""Enumerated binary reply channels under supplied mechanical group laws."""
from itertools import product
import math
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D

POLICIES = ('none', 'skill', 'belief', 'entropy-choice')
RELIABILITIES = (.5, .75, 1.)
READERS = ('channel-aware', 'trusting')


def channel(legal, endpoints, weights, reliability):
    """Condition on each noisy reply, using only the complete visible-group law."""
    if not math.isfinite(reliability) or not .5 <= reliability <= 1:
        raise ValueError('reply reliability')
    # Reuse the parent's validated group-law normalization, not its field choice.
    prior, _, _, _ = D.distributions(legal, endpoints, weights)
    endpoints = np.asarray(endpoints, int); weights = np.asarray(weights, float)
    tables = {}; reply_mass = {}; entropy = {}; alternate_entropy = {}; fallbacks = {}
    for field, axis in (('skill', 0), ('belief', 1)):
        tables[field] = {}; reply_mass[field] = {}; fallbacks[field] = {}
        terms = []; alt_terms = []
        for reply in (0, 1):
            likelihood = np.array([reliability if q[axis] == reply else 1-reliability for q in legal])
            raw = weights*likelihood; mass = math.fsum(float(x) for x in raw)
            if mass:
                p = np.bincount(endpoints, weights=raw, minlength=8)/mass
                fallback = None
            else:
                possible = np.flatnonzero(likelihood > 0)
                # Uniform over possible legal completions; empty literal support
                # uses the whole legal group. This affects the trusting rival.
                if not len(possible): possible = np.arange(len(legal)); fallback = 'uniform-whole-legal-group'
                else: fallback = 'uniform-likelihood-supported-legal'
                p = np.bincount(endpoints[possible], minlength=8)/len(possible)
            tables[field][reply] = p; reply_mass[field][reply] = mass; fallbacks[field][reply] = fallback
            terms.append(mass*D.P.entropy(p))
            positive = p[p > 0]; alt_terms.append(mass*float(-np.sum(positive*np.log(positive))))
        entropy[field] = math.fsum(terms); alternate_entropy[field] = float(np.sum(alt_terms))
    choice = 'skill' if entropy['skill'] <= entropy['belief'] else 'belief'
    alternate = 'skill' if alternate_entropy['skill'] <= alternate_entropy['belief'] else 'belief'
    return dict(prior=prior, tables=tables, reply_mass=reply_mass, expected_entropy=entropy,
                choice=choice, alternate_choice=alternate, alternate_entropy=alternate_entropy,
                fallbacks=fallbacks)


def validate_channel(result):
    for field in ('skill', 'belief'):
        if not math.isclose(sum(result['reply_mass'][field].values()), 1, abs_tol=1e-12):
            raise ValueError('reply mass')
        for p in result['tables'][field].values():
            if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(sum(p), 1, atol=1e-12, rtol=0):
                raise ValueError('reply normalization')


def expected_scores(predictions, probabilities, targets, weights):
    if predictions.shape != (len(targets), 2, 8) or probabilities.shape != (len(targets), 2):
        raise ValueError('reply arrays')
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError('reply probabilities')
    if (not np.isfinite(predictions).all() or np.any(predictions < 0)
        or not np.allclose(predictions.sum(2), 1, atol=1e-12, rtol=0)):
        raise ValueError('forecast normalization')
    parts = [D.P.proper(predictions[:, b], targets, weights*probabilities[:, b]) for b in (0, 1)]
    result = {k: math.fsum(p[k] for p in parts) for k in parts[0]}
    result['residual_ambiguous_mass'] = math.fsum(float(np.dot(weights*probabilities[:, b], (predictions[:, b] > 0).sum(1) > 1)) for b in (0, 1))
    return result


def controls():
    legal = [(s,b,2,1,4,4) for s,b in product(range(2), repeat=2)]
    informative = channel(legal, [0,1,0,1], [.25]*4, .75)
    null = channel(legal, [2]*4, [.25]*4, .75)
    half = channel(legal, [0,1,2,3], [.1,.2,.3,.4], .5)
    return {'live:belief_reply_informative': informative['choice']=='belief',
            'placebo:constant_endpoint': null['choice']=='skill' and all(v==0 for v in null['expected_entropy'].values()),
            'positive:half_channel_no_information': all(np.allclose(p, half['prior'], atol=1e-12, rtol=0) for t in half['tables'].values() for p in t.values())}


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; start = time.process_time(); checks = controls()
    if (not all(checks.values()) or cfg['policies'] != list(POLICIES)
        or cfg['reliabilities'] != list(RELIABILITIES) or cfg['readers'] != list(READERS)
        or cfg['models'] != list(D.MODELS) or cfg['costs'] != list(D.COSTS)):
        raise ValueError('noisy disclosure admission')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate law')
    groups = read(base/'MEMBERSHIP.json')['omit-both']; cells = []; rows = []; sensitivity = []; used = set()
    for folder in ('reader','forecasts','evaluator'): (root/folder).mkdir()
    for g in groups.values():
        packet = g['reader']; write(root/'reader'/f'{digest(packet)}.json',packet)
        for field, reliability, reply in product(('skill','belief_error'), RELIABILITIES, (0,1)):
            packet = dict(g['reader'],requested_field=field,reply_value=reply,reply_reliability=reliability)
            write(root/'reader'/f'{digest(packet)}.json',packet)
    write(root/'CONTROLS.json',checks)
    for lineage,rule,model in product(cfg['lineages'],cfg['rules'],D.MODELS):
        pulse(phase='noisy-disclosure',lineage=lineage,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',allow_pickle=False) as data:
            queries = [tuple(map(int,q)) for q in data['queries']]; mass=data['mass'].copy(); targets=data['targets'].copy()
            if len(queries)!=cfg['queries'] or len(set(queries))!=len(queries): raise ValueError('queries')
            if D.P.membership(queries)['omit-both']!=groups: raise ValueError('membership')
            if not np.array_equal(targets,[D.P.T.oracle(q,rule) for q in queries]): raise ValueError('targets')
            if np.any(mass<0) or not np.isfinite(mass).all() or not np.isclose(sum(mass),1,atol=1e-12): raise ValueError('mass')
            n=len(queries)
            for reliability in RELIABILITIES:
                before=np.zeros((n,8)); choices=np.zeros(n,dtype=np.int8); alternate_choices=choices.copy()
                tables={reader:{field:np.zeros((n,2,8)) for field in ('skill','belief')} for reader in READERS}
                for key,g in groups.items():
                    law=lookup[(lineage,rule,model,key)];used.add((lineage,rule,model,key))
                    legal=[tuple(q) for q in law['legal_completions']]
                    if law['legal_completions']!=g['legal_completions'] or law['endpoints']!=[D.P.T.oracle(q,rule) for q in legal]: raise ValueError('law mechanics')
                    aware=channel(legal,law['endpoints'],law['conditional_weights'],reliability)
                    trusting=channel(legal,law['endpoints'],law['conditional_weights'],1.)
                    validate_channel(aware);validate_channel(trusting)
                    indices=g['indices'];before[indices]=aware['prior'];choices[indices]=int(aware['choice']=='belief');alternate_choices[indices]=int(aware['alternate_choice']=='belief')
                    for reader,dist in [('channel-aware',aware),('trusting',trusting)]:
                        for field in ('skill','belief'):
                            tables[reader][field][indices]=np.stack([dist['tables'][field][b] for b in (0,1)])
                    rows.append(dict(lineage=lineage,rule=rule,model=model,reliability=reliability,reader_id=key,
                        expected_entropy=aware['expected_entropy'],choice=aware['choice'],alternate_choice=aware['alternate_choice'],alternate_entropy=aware['alternate_entropy'],reply_mass=aware['reply_mass'],fallbacks={r:d['fallbacks'] for r,d in [('channel-aware',aware),('trusting',trusting)]}))
                for reader in READERS:
                    saved=dict(queries=queries,mass=mass,targets=targets,choices=choices,alternate_choices=alternate_choices)
                    scores={}
                    for policy in (*POLICIES,'alternate-choice'):
                        if policy=='none':
                            pred=np.repeat(before[:,None,:],2,axis=1);probs=np.tile([1.,0.],(n,1));axis=np.zeros(n,dtype=int)
                        else:
                            axis=np.zeros(n,dtype=int) if policy=='skill' else np.ones(n,dtype=int) if policy=='belief' else alternate_choices if policy=='alternate-choice' else choices
                            pred=np.where(axis[:,None,None]==0,tables[reader]['skill'],tables[reader]['belief'])
                            bits=np.array([q[int(a)] for q,a in zip(queries,axis,strict=True)])
                            probs=np.where(bits[:,None]==np.arange(2)[None,:],reliability,1-reliability)
                        saved[policy]=pred;saved[policy+'-reply-probabilities']=probs
                        for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                            score=expected_scores(pred,probs,targets,w);scores[(policy,weighting)]=score
                            if policy=='alternate-choice':continue
                            fields=int(policy!='none')
                            for cost in D.COSTS:
                                cells.append(dict(lineage=lineage,rule=rule,model=model,reliability=reliability,reader=reader,policy=policy,weighting=weighting,cost=cost,queries=n,groups=len(groups),request_rate=fields,
                                    skill_request_rate=float(np.dot(w,axis==0))*fields,belief_request_rate=float(np.dot(w,axis==1))*fields,net_finite_loss=score['finite_loss_contribution']+cost*fields,**score))
                    for weighting in ('native','equal-query'):
                        sensitivity.append(dict(lineage=lineage,rule=rule,model=model,reliability=reliability,reader=reader,weighting=weighting,changed_queries=int(np.count_nonzero(choices!=alternate_choices)),score_changes={k:scores[('alternate-choice',weighting)][k]-scores[('entropy-choice',weighting)][k] for k in scores[('entropy-choice',weighting)]}))
                    np.savez_compressed(root/'forecasts'/f'{lineage}-{rule}-{model}-{reliability:g}-{reader}_points.npz',**saved)
    if len(used)!=len(laws):raise ValueError('law coverage')
    write(root/'evaluator/CHANNEL_LAWS.json',rows);write(root/'evaluator/TIE_SENSITIVITY.json',sensitivity)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='complete reply enumeration; no sampled observations'))
    return dict(controls=checks,cells=cells,queries=cfg['queries'],law_rows=len(rows),fits=0,
        scope='supplied-law binary mechanical replies; trusting reader shares channel-aware field selection; full expected proper scores,not averaged forecasts; no learned access or historical process claim')
