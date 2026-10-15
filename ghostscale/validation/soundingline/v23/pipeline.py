"""Repair-family batches: retained observations, rivals, checking and replay."""
import gzip
import hashlib
import hmac
from pathlib import Path
import time
import numpy as np
from . import registry, repairs, checker, ledger
from .common import Direct, canonical, digest, interval, losses, read, rng, seed, write


def features(v, p, masked=False):
    x = repairs.features(v, p).copy()
    if masked:
        if v['family'] != 'R1': raise ValueError('mask contract only defined for R1')
        x[2:34] = 0
    return np.r_[x, registry.variant_features(v)]


def training(family, phase, count, masked=False, pulse=lambda **kw: None, law=None):
    variants = registry.variants(family); x, y = [], []
    for i in range(count):
        v = variants[i % len(variants)]
        # Every variant spans the law menu; variant index does not encode law.
        p, truth = repairs.sample(v, phase, i//len(variants), i, law=(i//len(variants)) % 16 if law is None else law)
        x.append(features(v, p, masked)); y.append(truth['y'])
        if i % 1024 == 0: pulse(phase='training-generation', labels=i, target_labels=count)
    return np.asarray(x), np.asarray(y)


def fit_family(root, family, pulse):
    root.mkdir(parents=True, exist_ok=False); fitted, metadata = {}, {}
    for masked in (False, True) if family == 'R1' else (False,):
        suffix = '-masked' if masked else ''
        vx, vy = training(family, 'validation', 4096, masked, pulse)
        np.savez_compressed(root/('validation'+suffix+'.npz'), x=vx, y=vy)
        for labels in (2048, 8192):
            x, y = training(family, 'training', labels, masked, pulse)
            np.savez_compressed(root/(f'training-{labels}'+suffix+'.npz'), x=x, y=y)
            for fit in (0, 1):
                for fresh in (False, True) if labels == 8192 else (False,):
                    name = f'direct-{labels}-fit{fit}'+('-fresh' if fresh else '')+suffix
                    def provider(epoch, offset, size, fit=fit):
                        law = int(rng('fresh-law', family, fit, epoch, offset).integers(16))
                        return training(family, f'fresh-{fit}-{epoch}-{offset}', size, masked, pulse, law)
                    classes = 3 if family == 'R3' else 2
                    model = Direct(x, y, vx, vy, classes, family+suffix, fit,
                                   fresh=provider if fresh else None, pulse=pulse)
                    model.save(root/(name+'.npz')); fitted[name] = model
                    metadata[name] = model.metadata
                    write(root/(name+'-cost.json'), dict(cpu_seconds=model.cpu_seconds))
        pulse(phase='training-fits-complete', family=family, masked=masked)
    write(root/'FITS.json', metadata)
    return fitted, metadata


class Stream:
    def __init__(self, path):
        self.file = path.open('xb')
        self.zip = gzip.GzipFile(fileobj=self.file, mode='wb', filename='', mtime=0)
    def write(self, value): self.zip.write(canonical(value)+b'\n')
    def close(self): self.zip.close(); self.file.close()


def secondary(v, truth, detail):
    """Predeclared nonprimary readouts, independently reconstructed state included."""
    family=v['family'];out={}
    if family=='R1':
        target=[truth['y'][0],truth['goal'],truth['technique']]
        for name,heads in detail['heads'].items():
            metrics=checker.scores(heads,target)
            for index,label in enumerate(('content','purpose','technique')):
                for metric,values in metrics.items():out[name+':'+label+':'+metric]=float(values[index])
    elif family=='R2':
        for name,path in detail['trajectories'].items():
            metrics=checker.scores(detail['heads'][name],truth['state'])
            for index,label in enumerate(('purpose','value','technique')):
                for metric,values in metrics.items():out[name+':'+label+':'+metric]=float(values[index])
                out[name+':'+label+':memory_change']=abs(path[-1]['memory'][index]-.5)
            out[name+':fired_any']=float(any(r['fired'] for r in path))
            out[name+':benign_false_alarm']=float(truth['state'][0]==0 and any(r['fired'] for r in path))
            out[name+':releases']=float(sum(a['fired'] and not b['fired'] for a,b in zip(path,path[1:])))
    elif family=='R3':
        states=repairs.setpoints(v['settings']['dimensions'])
        for name,row in detail.items():
            post=np.array(row['setpoint_posterior']);coverage=[];width=[]
            for index in range(states.shape[1]):
                marginal=np.array([post[states[:,index]==value].sum() for value in (-1,0,1)])
                cumulative=np.cumsum(marginal)
                low=(-1,0,1)[min(2,int(np.searchsorted(cumulative,.025)))];high=(-1,0,1)[min(2,int(np.searchsorted(cumulative,.975)))]
                coverage.append(low<=truth['setpoint'][index]<=high);width.append(high-low)
            out[name+':setpoint_coverage_95']=float(np.mean(coverage));out[name+':setpoint_width_95']=float(np.mean(width))
            if name=='relevance-aware':
                index=(0.,.5,1.).index(truth['cost_role'])
                out[name+':true_role_probability']=row['role_posterior'][index]
    elif family=='R4':
        _,levels=repairs.tree(v['settings']['branching'])
        for name,post in detail['nodes'].items():
            metrics=checker.scores(post,truth['nodes'])
            for level in range(5):
                for metric,values in metrics.items():out[f'{name}:depth-{level}:{metric}']=float(values[levels==level].mean())
    return out


def headroom_groups(headroom_worlds,fitmeta,v,batch):
    groups={}
    for name in fitmeta:
        if '-fit0' not in name:continue
        other=name.replace('-fit0','-fit1')
        if other not in fitmeta:raise ValueError('headroom requires both frozen fits')
        pair=[name,other];intervals=[interval(headroom_worlds[n],[v['id'],batch,'headroom',n]) for n in pair]
        converged=all(fitmeta[n]['early_stopped'] for n in pair)
        passed=converged and all(i['finite'] and i['lower']>.02 for i in intervals)
        parity=converged and all(i['finite'] and i['upper']<=.02 for i in intervals)
        groups[name.replace('-fit0','')]=dict(state='HEADROOM' if passed else 'PARITY' if parity else 'UNRESOLVED',
            fits=pair,both_validation_stopped=converged,fit_intervals=intervals,
            information='masked' if name.endswith('-masked') else 'visible',
            extra_label_privilege=not fitmeta[name]['matched_label_budget'])
    return groups


def run_batch(root, plan, pulse):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    family, batch = plan['family'], plan['batch']
    if family not in ('R1', 'R2', 'R3', 'R4'): raise ValueError('full independent batch checker not admitted for '+family)
    schema, _ = ledger.canonical_schema(plan['schema_directory'], plan['schema_pin'])
    variants = [v for v in registry.variants(family) if v['id'] in plan['variants']]
    if not variants or len(variants) != len(plan['variants']): raise ValueError('unknown/repeated variant')
    if plan.get('fit_directory'):
        fit_root = Path(plan['fit_directory']); fitmeta = read(fit_root/'FITS.json')
        for relative, expected in plan['fit_hashes'].items():
            if digest(fit_root/relative) != expected: raise ValueError('frozen fit changed')
        fitted = {name: Direct.load(fit_root/(name+'.npz')) for name in fitmeta}
    else:
        if batch != 1: raise ValueError('only first batch may fit across family')
        fitted, fitmeta = fit_family(root/'fits', family, pulse)
    write(root/'DESIGN.json', {k: plan[k] for k in ('family', 'batch', 'variants', 'worlds', 'cases', 'schema_pin')})
    summaries, index = [], {}
    for v in variants:
        folder = root/v['id']; folder.mkdir()
        public_stream, truth_stream = Stream(folder/'public_points.jsonl.gz'), Stream(folder/'truth_points.jsonl.gz')
        score_stream, detail_stream = Stream(folder/'prediction_points.jsonl.gz'), Stream(folder/'posterior_points.jsonl.gz')
        ledger_stream = Stream(folder/'ledger_points.jsonl.gz')
        reader_worlds, gain_worlds, headroom_worlds, strata_worlds, secondary_worlds = {}, [], {}, {}, {}
        max_check, max_roundtrip = 0., 0.
        try:
            for world in range(plan['worlds']):
                rows, predictions, details, truths, bundles = [], [], [], [], []
                for case in range(plan['cases']):
                    p, truth = repairs.sample(v, ['discovery', batch], world, case)
                    prediction, detail = repairs.evaluate(v, p, truth)
                    max_check = max(max_check, getattr(checker, 'verify_'+family.lower())(v, p, truth, prediction, detail))
                    identity = [family, v['id'], batch, world, case]
                    # A separate namespace makes the random canary independent of
                    # observations. It is never passed into a reader.
                    canary = hmac.new(bytes.fromhex(plan['canary_key']), canonical(identity), hashlib.sha256).hexdigest()
                    bundle = ledger.emit(v, p, schema, identity)
                    ledger.validate(bundle, schema, canaries=[canary])
                    decoded = ledger.decode(v, bundle, schema)
                    from_ledger = repairs.evaluate(v, decoded)[0]['law-marginal']
                    difference = float(np.max(np.abs(from_ledger-prediction['law-marginal'])))
                    if difference > 1e-12: raise ValueError('ledger/native forecast mismatch')
                    max_roundtrip = max(max_roundtrip, difference)
                    rows.append(p); predictions.append(prediction); details.append(detail); truths.append(truth); bundles.append(bundle)
                    public_stream.write(dict(identity=identity, public=p))
                    truth_stream.write(dict(identity=identity, truth=truth, canary=canary))
                    ledger_stream.write(dict(identity=identity, ledger=bundle))
                for name, model in fitted.items():
                    pp = model.predict(np.array([features(v, p, name.endswith('-masked')) for p in rows]))
                    for i, prediction in enumerate(predictions): prediction[name] = pp[i]
                per_reader = {}
                for name in predictions[0]:
                    scored = []
                    for i, (prediction, truth) in enumerate(zip(predictions, truths)):
                        a = losses(prediction[name], truth['y']); b = checker.scores(prediction[name], truth['y'])
                        for metric in a:
                            if not np.allclose(a[metric], b[metric], atol=1e-12): raise ValueError('score reconstruction mismatch')
                        scored.append({metric: float(value.mean()) for metric, value in a.items()})
                    per_reader[name] = scored
                    reader_worlds.setdefault(name, []).append({metric: float(np.mean([r[metric] for r in scored])) for metric in scored[0]})
                rival, mechanism, _, _ = registry.PRIMARY[family]
                gain_worlds.append(reader_worlds[rival][-1]['log_loss']-reader_worlds[mechanism][-1]['log_loss'])
                for name in fitted:
                    # Masked rivals have a different information privilege; their
                    # matched exact comparator is the masked reader.
                    ceiling = 'masked' if name.endswith('-masked') else 'law-marginal'
                    headroom_worlds.setdefault(name, []).append(reader_worlds[name][-1]['log_loss']-reader_worlds[ceiling][-1]['log_loss'])
                for goal in (0, 1) if family in ('R1', 'R2') else ():
                    selected = [i for i, t in enumerate(truths) if (t['goal'] if family == 'R1' else t['state'][0]) == goal]
                    for name in per_reader:
                        key = f'goal-{goal}:{name}'
                        if selected:
                            strata_worlds.setdefault(key, []).append(float(np.mean([per_reader[name][i]['log_loss'] for i in selected])))
                for case, (prediction, detail) in enumerate(zip(predictions, details)):
                    score_stream.write(dict(world=world, case=case, probabilities=prediction,
                        scores={name: per_reader[name][case] for name in prediction}))
                    detail_stream.write(dict(world=world, case=case, detail=detail))
                second=[secondary(v,t,d) for t,d in zip(truths,details)]
                for key in second[0]:secondary_worlds.setdefault(key,[]).append(float(np.mean([row[key] for row in second])))
                pulse(phase='family-batch', family=family, batch=batch, variant=v['id'], world=world+1, worlds=plan['worlds'])
        finally:
            for stream in (public_stream, truth_stream, score_stream, detail_stream, ledger_stream): stream.close()
        primary = interval(gain_worlds, [v['id'], batch, 'primary'])
        summary = dict(variant=v, primary=primary, world_gains=gain_worlds,
            readers={name: {metric: interval([r[metric] for r in rows], [v['id'], batch, name, metric])
                       for metric in rows[0]} for name, rows in reader_worlds.items()},
            headroom={name: dict(interval(values, [v['id'], batch, 'headroom', name]),
                          validation_stopped=fitmeta[name]['early_stopped']) for name, values in headroom_worlds.items()},
            strata={name: interval(values, [v['id'], batch, name]) for name, values in strata_worlds.items()},
            secondary={name:interval(values,[v['id'],batch,'secondary',name]) for name,values in secondary_worlds.items()},
            matched_headroom_groups=headroom_groups(headroom_worlds,fitmeta,v,batch),
            laws=min(16, plan['worlds']), independent_worlds=plan['worlds'], law_counts={str(i): sum(w % 16 == i for w in range(plan['worlds'])) for i in range(16)},
            validity=dict(independent_posterior_max_error=max_check, ledger_forecast_max_error=max_roundtrip),
            structural_zero=(family == 'R1' and v['settings']['strength'] == .5) or
                            (family == 'R4' and v['settings']['cue_reliability'] == .5))
        write(folder/'SUMMARY.json', summary); summaries.append(summary)
        index[v['id']] = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
    write(root/'RAW_INDEX.json', index)
    write(root/'SUMMARY.json', dict(family=family, batch=batch, variants=summaries,
        status='independent-check-passed; complete replay still required', fits=fitmeta,
        result_kind='method', scope='finite constructed generator; conditional on frozen training'))
