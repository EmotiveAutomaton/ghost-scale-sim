"""Independent scalar audit of repeated replies; no producer routines imported."""
from itertools import product
import math
import numpy as np
from ..v18_3.io import read, write, digest, file_digest
from . import disclosure_review as D

CHANNELS = ('independent', 'copied')
READERS = ('channel-aware', 'independence-assumed')
FIELDS = ('skill', 'belief')
LENGTHS = (0, 1, 2, 4)
COST_MODES = ('per-return', 'per-acquisition')
AXES = ('rule', 'model', 'field', 'channel', 'length', 'reader', 'weighting', 'cost', 'cost_mode')
METRICS = ('finite_loss_contribution', 'infinite_loss_mass', 'squared_error',
           'true_probability', 'residual_ambiguous_mass', 'net_finite_loss')


def channel(length, accuracy, mode):
    if length not in LENGTHS or mode not in CHANNELS or not .5 <= accuracy <= 1:
        raise ValueError('channel specification')
    strings = list(product((0, 1), repeat=length)); rows = []
    for bits in strings:
        values = []
        for truth in (0, 1):
            matches = sum(bit == truth for bit in bits)
            if mode == 'independent':
                value = accuracy**matches * (1-accuracy)**(length-matches)
            elif not bits:
                value = 1.
            elif 0 < matches < length:
                value = 0.
            else:
                value = accuracy if matches else 1-accuracy
            values.append(value)
        rows.append(values)
    result = np.array(rows)
    D.V.V.near([math.fsum(result[:,i]) for i in (0,1)], [1.,1.])
    return strings, result


def conditional(legal, ends, weights, axis, table):
    if not legal or len(legal) != len(ends) or len(ends) != len(weights):
        raise ValueError('law shape')
    if axis not in (0,1) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('law weights')
    D.V.V.near([math.fsum(weights)], [1.])
    table = np.asarray(table)
    if table.ndim != 2 or table.shape[1] != 2 or not np.isfinite(table).all() or np.any(table < 0):
        raise ValueError('likelihood')
    D.V.V.near([math.fsum(table[:,i]) for i in (0,1)], [1.,1.])
    forecasts = []; masses = []; fallbacks = []
    for reply in table:
        joint = [float(w)*float(reply[q[axis]]) for q,w in zip(legal,weights,strict=True)]
        total = math.fsum(joint); masses.append(total)
        if total:
            forecasts.append([math.fsum(w for w,e in zip(joint,ends) if e == t)/total for t in range(8)])
            fallbacks.append(None)
        else:
            supported = [i for i,q in enumerate(legal) if reply[q[axis]] > 0]
            fallbacks.append('uniform-likelihood-supported-legal' if supported else 'uniform-whole-legal-group')
            if not supported: supported = list(range(len(legal)))
            forecasts.append([sum(ends[i] == t for i in supported)/len(supported) for t in range(8)])
    return np.array(forecasts), masses, fallbacks


def scores(predictions, probabilities, targets, weights):
    predictions = np.asarray(predictions); probabilities = np.asarray(probabilities)
    if predictions.ndim != 3 or predictions.shape[0] != len(targets) or predictions.shape[2] != 8 or probabilities.shape != predictions.shape[:2]:
        raise ValueError('reply shape')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError('reply probability')
    D.V.V.near([math.fsum(p) for p in probabilities], np.ones(len(targets)))
    parts = [D.V.scores(predictions[:,j],targets,[float(w)*float(p[j]) for w,p in zip(weights,probabilities,strict=True)]) for j in range(predictions.shape[1])]
    result = {k:math.fsum(p[k] for p in parts) for k in parts[0]}
    result['residual_ambiguous_mass'] = math.fsum(float(w)*float(pr[j]) for p,pr,w in zip(predictions,probabilities,weights,strict=True) for j in range(len(pr)) if np.count_nonzero(p[j]) > 1)
    return result


def regroup(cells, lineages, cfg):
    lookup = {(r['lineage'], *(r[k] for k in AXES)):r for r in cells}
    if len(lookup) != len(cells): raise ValueError('duplicate cells')
    estimates = []
    for key in sorted({tuple(r[k] for k in AXES) for r in cells}):
        bases = [('mean',None)]
        if key[4] > 0: bases.append(('minus-zero', (*key[:4],0,*key[5:])))
        if key[4] > 1: bases.append(('minus-one', (*key[:4],1,*key[5:])))
        if key[5] == 'independence-assumed': bases.append(('minus-channel-aware', (*key[:5],'channel-aware',*key[6:])))
        if key[3] == 'copied': bases.append(('copied-minus-independent', (*key[:3],'independent',*key[4:])))
        for label,base in bases:
            for metric in METRICS:
                values = [lookup[(lin,*key)][metric] - (lookup[(lin,*base)][metric] if base else 0.) for lin in lineages]
                estimates.append(dict(zip(AXES,key),contrast=label,metric=metric,**D.V.V.interval(values,cfg)))
    return dict(estimates=estimates,lineages=lineages,bootstrap_seed=cfg['bootstrap_seed'],bootstrap_resamples=cfg['bootstrap_resamples'],
                population='paired development laws; native and equal-query populations kept separate',
                limitation='conditional on retained queries and supplied laws; finite loss excludes explicit infinite mass; no fits or confirmation')


def review(original, output, cfg, pulse=lambda **kw:None):
    design = read(original/'PLAN.json')['design']; summary = read(original/'SUMMARY.json'); base = original/'inputs'
    for key,expected in [('channels',CHANNELS),('readers',READERS),('fields',FIELDS),('lengths',LENGTHS),('cost_modes',COST_MODES),('models',D.MODELS),('costs',D.COSTS),('rules',D.V.T.RULES)]:
        if design[key] != list(expected): raise ValueError('design')
    if design['reliability'] != .75: raise ValueError('reliability')
    tables = {f'{mode}-{length}':dict(strings=[list(s) for s in channel(length,.75,mode)[0]],likelihood=channel(length,.75,mode)[1].tolist()) for mode,length in product(CHANNELS,LENGTHS)}
    error = D.V.compare(read(original/'evaluator/CHANNELS.json'),tables)
    laws = read(base/'DISCLOSURE_LAWS.json'); lookup = {(r['lineage'],r['rule'],r['model'],r['reader_id']):r for r in laws}
    recorded = read(original/'evaluator/REPLY_LAWS.json')
    law_keys = ('lineage','rule','model','field','channel','length','reader','reader_id')
    replies = {tuple(r[k] for k in law_keys):r for r in recorded}
    if len(lookup) != len(laws) or len(replies) != len(recorded): raise ValueError('duplicate laws')
    cells = []; used = set(); reply_used = set(); seen = set(); input_seen = set(); queries = None; vectors = 0; fallbacks = 0
    for lin,rule,model in product(design['lineages'],design['rules'],D.MODELS):
        pulse(phase='independent-repeated-disclosure',lineage=lin,rule=rule,model=model)
        fname = f'{lin}-{rule}-{model}_points.npz'; input_seen.add(fname)
        with np.load(base/'forecasts'/fname,allow_pickle=False) as parent:
            qs = [tuple(map(int,q)) for q in parent['queries']]; n = len(qs)
            if n != design['queries'] or len(set(qs)) != n: raise ValueError('query roster')
            if queries is None:
                queries = qs; groups = D.V.groups_for(qs,'omit-both')
                if groups != read(base/'MEMBERSHIP.json')['omit-both']: raise ValueError('membership')
            elif qs != queries: raise ValueError('query identity')
            index = {q:i for i,q in enumerate(qs)}; targets = np.array([D.V.T.endpoint(q,rule) for q in qs]); mass = parent['mass'].copy()
            if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0): raise ValueError('population')
            error = max(error,D.V.V.near(parent['targets'],targets,0),D.V.V.near([math.fsum(mass)],[1.]))
            baseline = {}
            for field,mode,length in product(FIELDS,CHANNELS,LENGTHS):
                axis = FIELDS.index(field); strings,actual = channel(length,.75,mode)
                probabilities = np.array([[row[q[axis]] for row in actual] for q in qs])
                fname = f'{lin}-{rule}-{model}-{field}-{mode}-{length}_points.npz'; seen.add(fname)
                with np.load(original/'forecasts'/fname,allow_pickle=False) as saved:
                    if set(saved.files) != {'queries','mass','targets','strings','reply_probabilities',*READERS}: raise ValueError('forecast schema')
                    for k,v in dict(queries=qs,mass=mass,targets=targets,strings=np.array(strings,dtype=np.int8).reshape(len(strings),length),reply_probabilities=probabilities).items():
                        error = max(error,D.V.V.near(saved[k],v,0))
                    for reader in READERS:
                        assumed = mode if reader == 'channel-aware' else 'independent'; table = channel(length,.75,assumed)[1]
                        forecasts = np.zeros((n,len(strings),8))
                        for key,g in groups.items():
                            ident = (lin,rule,model,key); law = lookup[ident]; used.add(ident); legal = g['legal_completions']; ends = [D.V.T.endpoint(q,rule) for q in legal]
                            raw = [float(mass[index[tuple(q)]]) if tuple(q) in index else 0. for q in legal]; total = math.fsum(raw)
                            weights = [w/total for w in raw] if model == 'native-law' and total else [1/len(legal)]*len(legal)
                            if law['legal_completions'] != legal or law['endpoints'] != ends: raise ValueError('mechanical law')
                            error = max(error,D.V.V.near(law['conditional_weights'],weights))
                            p,reply_mass,fallback = conditional(legal,ends,weights,axis,table); forecasts[g['indices']] = p
                            ident = (lin,rule,model,field,mode,length,reader,key); reply_used.add(ident)
                            expected = dict(zip(law_keys,ident),modeled_reply_mass=reply_mass,fallbacks=fallback)
                            error = max(error,D.V.compare(replies[ident],expected)); fallbacks += sum(f is not None for f in fallback)
                        error = max(error,D.V.V.near(saved[reader],forecasts)); vectors += n*len(strings)
                        if length == 0: error = max(error,D.V.V.near(forecasts[:,0],parent['none']))
                        if mode == 'copied' and reader == 'channel-aware':
                            if length == 1: baseline[field] = forecasts.copy()
                            elif length > 1: error = max(error,D.V.V.near(forecasts[:,[0,-1]],baseline[field]))
                        if mode == 'independent' or length < 2:
                            error = max(error,D.V.V.near(saved['channel-aware'],saved['independence-assumed']))
                        for weighting,w in (('native',mass),('equal-query',np.full(n,1/n))):
                            result = scores(forecasts,probabilities,targets,w)
                            for cost,cost_mode in product(D.COSTS,COST_MODES):
                                acquisitions = int(length > 0) if mode == 'copied' else length
                                charge = length if cost_mode == 'per-return' else acquisitions
                                cells.append(dict(lineage=lin,rule=rule,model=model,field=field,channel=mode,length=length,reader=reader,weighting=weighting,cost=cost,cost_mode=cost_mode,queries=n,groups=len(groups),returned_replies=length,acquisitions=acquisitions,charged_replies=charge,net_finite_loss=result['finite_loss_contribution']+cost*charge,**result))
    if set(replies) != reply_used or set(lookup) != used: raise ValueError('law coverage')
    if {p.name for p in (original/'forecasts').glob('*.npz')} != seen or {p.name for p in (base/'forecasts').glob('*.npz')} != input_seen: raise ValueError('forecast coverage')
    error = max(error,D.V.compare(summary['cells'],cells))
    packets = {digest(g['reader']):g['reader'] for g in groups.values()}
    for g in groups.values():
        for field,mode,length in product(FIELDS,CHANNELS,LENGTHS[1:]):
            for bits in channel(length,.75,mode)[0]:
                p = dict(g['reader'],requested_field='belief_error' if field == 'belief' else field,replies=list(bits),reply_reliability=.75,reply_source_mode=mode); packets[digest(p)] = p
    if {p.name for p in (original/'reader').iterdir()} != {'REPLIES.json'} or read(original/'reader/REPLIES.json') != packets: raise ValueError('reader projection')
    checks = read(original/'CONTROLS.json')
    if not checks or not all(checks.values()) or summary['controls'] != checks: raise ValueError('controls')
    if summary['queries'] != len(queries) or summary['law_rows'] != len(replies) or summary['reader_packets'] != len(packets) or summary['fits'] != 0: raise ValueError('totals')
    timing = read(original/'TIMING.jsonl')
    if timing['fits'] != 0 or timing['cpu_seconds'] < 0: raise ValueError('timing')
    write(output/'RECONSTRUCTED_CELLS.json',cells); write(output/'INDEPENDENT_REGROUP.json',regroup(cells,design['lineages'],cfg))
    result = dict(passed=True,cells=len(cells),law_rows=len(replies),queries=len(queries),reader_packets=len(packets),forecast_vectors=vectors,max_error=error,zero_model_mass_fallbacks=fallbacks,
                  scope='independent mechanics,scalar channel and Bayes sums,complete reply strings,actual probabilities,all branch proper scores,copied and zero-request identities,costs,all strata and reader allowlists; native mass inherits verified parent')
    write(output/'NUMERICAL_REVIEW.json',result); return result


def run(root,plan,pulse):
    cfg = plan['design']; original = root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n) != h: raise ValueError('input binding')
    if file_digest(original/'PLAN.json') != cfg['target_plan_sha256']: raise ValueError('target binding')
    result = review(original,root,cfg,pulse)
    return dict(result,controls={'live:complete_repeated_reply_reconstruction':True,'positive:copied_and_zero_request_identities':True,'placebo:zero_mass_and_infinity_preserved':True})
