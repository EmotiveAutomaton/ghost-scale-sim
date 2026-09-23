"""Fixed actual versus stated binary reply accuracy; exhaustive, no fitting."""
from itertools import product
import time
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import metadata_disclosure as D
from .noisy_disclosure import channel, validate_channel, expected_scores

ACCURACIES = (.5, .75, 1.)
REQUESTS = ('none', 'skill', 'belief')


def reply_probabilities(queries, request, actual):
    if request not in REQUESTS or actual not in ACCURACIES:
        raise ValueError('actual reply specification')
    if request == 'none':
        return np.tile([1., 0.], (len(queries), 1))
    axis = int(request == 'belief')
    truth = np.array([q[axis] for q in queries])
    return np.where(truth[:, None] == np.arange(2)[None, :], actual, 1-actual)


def controls():
    legal = [(s,b,2,1,4,4) for s,b in product((0,1),repeat=2)]
    half = channel(legal,[0,1,2,3],[.1,.2,.3,.4],.5)
    calibrated = channel(legal,[0,1,2,3],[.1,.2,.3,.4],.75)
    certain = channel(legal,[0,1,2,3],[.1,.2,.3,.4],1.)
    predictions = np.tile(np.stack([certain['tables']['skill'][r] for r in (0,1)]),(4,1,1))
    score = expected_scores(predictions,reply_probabilities(legal,'skill',.75),np.arange(4),np.full(4,.25))
    return {'live:stated_accuracy_changes_forecast': not np.array_equal(calibrated['tables']['skill'][0],certain['tables']['skill'][0]),
            'positive:half_assumed_channel_ignores_reply': all(np.allclose(p,half['prior'],atol=1e-15,rtol=0) for t in half['tables'].values() for p in t.values()),
            'placebo:overconfidence_has_explicit_infinite_loss': score['infinite_loss_mass']==.25}


def run(root, plan, pulse):
    cfg = plan['design']; base = root/'inputs'; start = time.process_time(); checks = controls()
    for key, expected in [('actual_accuracies',ACCURACIES),('assumed_accuracies',ACCURACIES),('requests',REQUESTS),('models',D.MODELS),('costs',D.COSTS)]:
        if cfg[key] != list(expected): raise ValueError('mismatch design')
    if not all(checks.values()): raise ValueError('mismatch controls')
    for n,h in cfg['input_files'].items():
        if file_digest(base/n) != h: raise ValueError('input binding')
    groups = read(base/'MEMBERSHIP.json')['omit-both']; laws = read(base/'DISCLOSURE_LAWS.json')
    lookup = {(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    if len(lookup) != len(laws): raise ValueError('duplicate laws')
    for folder in ('reader','forecasts','evaluator'): (root/folder).mkdir()
    packets = {}
    for g in groups.values():
        packets[digest(g['reader'])] = g['reader']
        for field,stated,bit in product(('skill','belief_error'),ACCURACIES,(0,1)):
            packet = dict(g['reader'],requested_field=field,reply_value=bit,stated_accuracy=stated)
            packets[digest(packet)] = packet
    write(root/'reader/REPLIES.json',packets); write(root/'CONTROLS.json',checks)
    cells = []; rows = []; used = set(); roster = None
    for lineage,rule,model in product(cfg['lineages'],cfg['rules'],D.MODELS):
        pulse(phase='reliability-mismatch',lineage=lineage,rule=rule,model=model)
        with np.load(base/'forecasts'/f'{lineage}-{rule}-{model}_points.npz',allow_pickle=False) as parent:
            qs = [tuple(map(int,q)) for q in parent['queries']]; mass = parent['mass'].copy(); targets = parent['targets'].copy(); n = len(qs)
            if n != cfg['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if roster is None: roster = qs
            elif qs != roster: raise ValueError('query identity')
            if D.P.membership(qs)['omit-both'] != groups: raise ValueError('membership')
            if not np.array_equal(targets,[D.P.T.oracle(q,rule) for q in qs]): raise ValueError('targets')
            if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass<0) or not np.isclose(mass.sum(),1,atol=1e-12,rtol=0): raise ValueError('mass')
            for assumed in ACCURACIES:
                saved = dict(queries=qs,mass=mass,targets=targets,strings=np.array([0,1],dtype=np.int8))
                forecasts = {request:np.zeros((n,2,8)) for request in REQUESTS}
                for key,g in groups.items():
                    ident = (lineage,rule,model,key); law = lookup[ident]; used.add(ident)
                    legal = [tuple(q) for q in law['legal_completions']]
                    if law['legal_completions'] != g['legal_completions'] or law['endpoints'] != [D.P.T.oracle(q,rule) for q in legal]: raise ValueError('legal mechanics')
                    dist = channel(legal,law['endpoints'],law['conditional_weights'],assumed); validate_channel(dist)
                    forecasts['none'][g['indices']] = np.tile(dist['prior'],(2,1))
                    for request in REQUESTS[1:]:
                        forecasts[request][g['indices']] = np.stack([dist['tables'][request][b] for b in (0,1)])
                        rows.append(dict(lineage=lineage,rule=rule,model=model,assumed_accuracy=assumed,request=request,reader_id=key,
                                         modeled_reply_mass=[dist['reply_mass'][request][b] for b in (0,1)],fallbacks=[dist['fallbacks'][request][b] for b in (0,1)]))
                if not np.allclose(forecasts['none'][:,0],parent['none'],atol=1e-12,rtol=0): raise ValueError('no-request identity')
                for request,predictions in forecasts.items():
                    saved[request] = predictions
                    for actual in ACCURACIES:
                        probabilities = reply_probabilities(qs,request,actual); saved[f'{request}-actual-{actual:g}'] = probabilities
                        for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                            result = expected_scores(predictions,probabilities,targets,w)
                            for cost in D.COSTS:
                                count = int(request != 'none')
                                cells.append(dict(lineage=lineage,rule=rule,model=model,actual_accuracy=actual,assumed_accuracy=assumed,request=request,
                                                  weighting=weighting,cost=cost,queries=n,groups=len(groups),request_rate=count,
                                                  net_finite_loss=result['finite_loss_contribution']+cost*count,**result))
                np.savez_compressed(root/'forecasts'/f'{lineage}-{rule}-{model}-assumed-{assumed:g}_points.npz',**saved)
    if used != set(lookup): raise ValueError('law coverage')
    write(root/'evaluator/REPLY_LAWS.json',rows)
    write(root/'TIMING.jsonl',dict(cpu_seconds=time.process_time()-start,fits=0,scope='complete fixed actual/assumed reply enumeration; calibrated diagonal is an identity reference'))
    return dict(controls=checks,cells=cells,queries=cfg['queries'],law_rows=len(rows),reader_packets=len(packets),fits=0,
                scope='supplied-law calibration method; actual accuracy evaluator-only; no new sampled evidence, learned access or historical-process claim')
