"""Finite V22 dispatch, retained data, fixed learning curves and paired summaries."""
import gzip
import json
import time
from pathlib import Path
import numpy as np
from ..v18_3.io import write, canonical, file_digest
from . import credibility, homeostasis, discrete, attention, hierarchy
from .common import Direct, score, bootstrap

CARDS = ('A1', 'B1', 'C2', 'A2', 'A3', 'B2', 'C1')


def sample(card, namespace, world, index):
    if card in ('A2', 'B2', 'C2'):
        return discrete.sample(card, namespace, world, index)
    return {'A1': credibility, 'B1': homeostasis, 'A3': attention, 'C1': hierarchy}[card].sample(namespace, world, index)


def features(card, public):
    if card in ('A2', 'B2', 'C2'):
        return discrete.features(card, public)
    return {'A1': credibility, 'B1': homeostasis, 'A3': attention, 'C1': hierarchy}[card].features(public)


def evaluate(card, public, truth):
    if card in ('A2', 'B2', 'C2'):
        return discrete.evaluate(card, public, truth)
    return {'A1': credibility, 'B1': homeostasis, 'A3': attention, 'C1': hierarchy}[card].evaluate(public, truth)


def masks(card, public, y):
    out = np.ones(len(y), float)
    if card == 'C1':
        out[len(public['evidence']):] = 0
    return out


def dataset(card, namespace, count, pulse):
    x, y, mask, records = [], [], [], []
    for i in range(count):
        # Separate world identities, never generator truth in feature encoding.
        public, truth = sample(card, namespace, i % 64, i)
        x.append(features(card, public))
        y.append(truth['y'])
        mask.append(masks(card, public, truth['y']))
        records.append(dict(index=i, world=i % 64, public=public, truth=truth))
        if i % 256 == 0:
            pulse(phase=namespace+'-generation', item=i)
    return np.array(x, np.float32), np.array(y, int), np.array(mask), records


def retain_records(path, records):
    # Stable gzip header makes a full scientific replay byte-comparable.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as out:
            for record in records:
                out.write(canonical(record)+b'\n')


def safe_mean(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()) if np.isfinite(values).all() else None,
                infinite_fraction=float(np.isinf(values).mean()))


def run(root, design, pulse=lambda **kw: None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    card = design['card']
    start = time.process_time()
    budgets = design.get('budgets', [2048, 8192])
    worlds = design.get('worlds', list(range(8)))
    cases = design.get('cases', 256)
    namespace = design.get('namespace', 'development')
    prefix = '' if namespace == 'development' else namespace+'-'
    x, y, mask, records = dataset(card, prefix+'training', max(budgets), pulse)
    vx, vy, vm, validation = dataset(card, prefix+'validation', design.get('validation', 2048), pulse)
    retain_records(root/'training.jsonl.gz', records)
    retain_records(root/'validation.jsonl.gz', validation)
    np.savez_compressed(root/'training.npz', x=x, y=y, mask=mask)
    np.savez_compressed(root/'validation.npz', x=vx, y=vy, mask=vm)
    classes = 9 if card == 'A1' else 3 if card == 'B1' else 2
    fits = []
    fitted = []
    for budget in budgets:
        for fit in design.get('fits', [0, 1]):
            model = Direct(x[:budget], y[:budget], vx, vy, mask[:budget], vm, classes,
                           fit=fit, max_epochs=design.get('epochs', 120), pulse=pulse)
            name = f'direct-{budget}-{fit}'
            model.save(root/(name+'.npz'))
            # Timing is retained separately; scientific replay compares pure metadata.
            meta = dict(model.metadata)
            cpu = meta.pop('cpu_seconds')
            write(root/(name+'.json'), meta)
            fits.append(dict(reader=name, budget=budget, **meta))
            fitted.append((name, model))
            pulse(phase='fit-complete', reader=name, fit_cpu_seconds=cpu)
    summaries = []
    for world in worlds:
        inputs, labels, active, meta, public_records = [], [], [], [], []
        predictions = {}
        for index in range(cases):
            public, truth = sample(card, namespace, world, index)
            readers, details = evaluate(card, public, truth)
            inputs.append(features(card, public))
            labels.append(truth['y'])
            active.append(masks(card, public, truth['y']))
            meta.append(details)
            public_records.append(dict(world=world, index=index, public=public, truth=truth))
            for name, p in readers.items():
                predictions.setdefault(name, []).append(p)
            if index % 32 == 0:
                pulse(phase='science', world=world, item=index)
        xx, yy, mm = np.array(inputs, np.float32), np.array(labels), np.array(active)
        for name, model in fitted:
            predictions[name] = model.predict(xx)
        target = root/f'world-{world}'
        target.mkdir()
        retain_records(target/'records.jsonl.gz', public_records)
        retain_records(target/'details.jsonl.gz', meta)
        if card == 'A3':
            from .legacy_attention import physical, e56_accounting
            write(target/'G5_PHYSICAL.json', physical(namespace, world))
            write(target/'E56_ACCOUNTING.json', [e56_accounting(r['public'], r['truth']) for r in public_records])
        np.savez_compressed(target/'inputs.npz', x=xx, y=yy, mask=mm)
        for name, prob in predictions.items():
            p = np.array(prob)
            scores = score(p, yy, mm)
            np.savez_compressed(target/(name+'.npz'), p=p, **scores)
            summaries.append(dict(world=world, reader=name, metrics={k: safe_mean(v) for k, v in scores.items()}))
    headroom = []
    for budget in budgets:
        delta = []
        for world in worlds:
            exact = next(row['metrics']['log_loss']['mean'] for row in summaries if row['world'] == world and row['reader'] == 'law-marginal')
            direct = [row['metrics']['log_loss']['mean'] for row in summaries if row['world'] == world and row['reader'].startswith(f'direct-{budget}-')]
            delta.append(float(np.mean(direct))-exact)
        interval = bootstrap(delta)
        stopped = all(f['early_stopped'] for f in fits if f['budget'] == budget)
        disposition = ('HEADROOM' if interval['lower'] > .02 else 'PARITY' if interval['upper'] <= .02 else 'UNRESOLVED') if stopped else 'UNRESOLVED'
        headroom.append(dict(budget=budget, interval=interval, direct_validation_stopped=stopped, disposition=disposition,
                             scope='pooled diagnostic only; admission requires matching target/stratum in CONTRASTS.json'))
    summary = dict(card=card, namespace=namespace, worlds=worlds, cases_per_world=cases,
                   fits=fits, rows=summaries, headroom=headroom,
                   scientific_state='executed; independent check and complete replay required',
                   scope='finite constructed mechanism; miniature architecture untested; no confirmation')
    write(root/'SUMMARY.json', summary)
    from .contrasts import summarize
    summarize(root)
    manifest = {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'SCIENTIFIC_MANIFEST.json', manifest)
    return dict(card=card, cpu_seconds=time.process_time()-start, files=len(manifest), worlds=len(worlds))
