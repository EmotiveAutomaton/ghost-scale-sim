"""Predeclared readout groups; head-specific and shift results never hidden in means."""
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import bootstrap
from ..v18_3.io import read, write

GROUPS = {
    'A1': ('k', 'expertise', 'source_goal', 'competence', 'condition'),
    'B1': ('d', 'k', 'shift', 'condition'),
    'C2': ('k', 'share_lib', 'share_pref', 'coupled', 'disclosed', 'unrelated'),
    'A2': ('signature', 'swapped', 'source_goal'),
    'A3': ('coupled', 'prefix', 'benign'),
    'B2': ('condition', 'devalued', 'route'),
    'C1': ('branching', 'coverage'),
}
TARGETS = {'A1': ['property'], 'B1': ['future-choice'], 'C2': ['operation', 'preference'],
           'A2': ['content-property', 'top-goal', 'technique'], 'A3': ['purpose', 'value', 'technique'],
           'B2': ['acquisition-route', 'future-choice']}


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(row) for row in f]


def summarize(root):
    root = Path(root)
    summary = read(root/'SUMMARY.json')
    card = summary['card']
    grouped = defaultdict(list)
    mechanism = []
    for world in summary['worlds']:
        path = root/f'world-{world}'
        details = load(path/'details.jsonl.gz')
        with np.load(path/'inputs.npz') as a:
            y, mask = a['y'], a['mask']
        keys = [tuple((name, row[name]) for name in GROUPS[card]) for row in details]
        groups = defaultdict(list)
        for i, key in enumerate(keys):
            groups[key].append(i)
        for file in sorted(path.glob('*.npz')):
            if file.name == 'inputs.npz':
                continue
            with np.load(file) as a:
                p = a['p']
            q = np.take_along_axis(p, y[..., None], -1)[..., 0]
            with np.errstate(divide='ignore'):
                loss = -np.log(q)
            brier = (p*p).sum(-1)-2*q+1
            for key, ids in groups.items():
                if card == 'C1':
                    levels = np.array(details[ids[0]]['levels'])
                    heads = [(f'depth-{depth}', np.flatnonzero(levels == depth)) for depth in range(5)]
                else:
                    heads = [(name, [h]) for h, name in enumerate(TARGETS[card])]
                for target, hh in heads:
                    for metric, values in [('log_loss', loss), ('brier', brier)]:
                        values = values[np.ix_(ids, hh)]
                        mean = float(values.mean())
                        grouped[(key, file.stem, target, metric)].append((world, mean))
        # Extra mechanism endpoints retain their whole-world summaries separately.
        for key, ids in groups.items():
            row = dict(world=world, group=dict(key), cases=len(ids))
            subset = [details[i] for i in ids]
            if card == 'A1':
                row.update(belief_shift=float(np.mean([r['belief_shift'] for r in subset])),
                           signed_error=float(np.mean([r['signed_error'] for r in subset])),
                           detected_steering=float(np.mean([r['steering_probability'] >= .9 for r in subset])),
                           shift_curve=np.mean([r['shift_curve'] for r in subset], axis=0).tolist())
            elif card == 'B1':
                row['recovery'] = {reader: {m: float(np.mean([r['interval_recovery'][reader][m] for r in subset]))
                                            for m in ('coverage', 'width')} for reader in ('law-marginal', 'effort-weighted', 'cost-ignorant')}
                row['distance'] = float(np.mean([r['distance'] for r in subset]))
                row['choice_consistency'] = float(np.mean([r['choice_consistency'] for r in subset]))
            elif card == 'A2':
                for metric in ('content_uptake', 'technique_production', 'goal_probability'):
                    row[metric] = {name: float(np.mean([r[metric][name] for r in subset])) for name in subset[0][metric]}
            elif card == 'A3':
                row['policies'] = {}
                for policy in subset[0]['trajectories']:
                    endpoints = [r['trajectories'][policy][-1] for r in subset]
                    row['policies'][policy] = dict(memory=np.mean([r['memory'] for r in endpoints], axis=0).tolist(),
                        recognition=np.mean([r['posterior'] for r in endpoints], axis=0).tolist(),
                        flag_rate=float(np.mean([r['fired'] for r in endpoints])),
                        production_accuracy=float(np.mean([r['production'][policy] == r['technique_truth'] for r in subset])))
            elif card == 'C1':
                row['focus_ablations'] = {name: np.mean([r['ablations'][name] for r in subset], axis=0).tolist()
                                           for name in ('adjacent', 'distant')}
            mechanism.append(row)
    estimates = []
    for (key, reader, target, metric), values in sorted(grouped.items()):
        estimates.append(dict(group=dict(key), reader=reader, target=target, metric=metric,
                              world_values=[dict(world=w, value=v if np.isfinite(v) else None) for w, v in values],
                              interval=bootstrap([v for _, v in values])))
    headroom = []
    for key, _, target, metric in sorted(k for k in grouped if k[1] == 'law-marginal' and k[3] == 'log_loss'):
        exact = dict(grouped[(key, 'law-marginal', target, metric)])
        for budget in sorted({f['budget'] for f in summary['fits']}):
            fits = [f for f in summary['fits'] if f['budget'] == budget]
            tables = [dict(grouped.get((key, f['reader'], target, metric), [])) for f in fits]
            worlds = sorted(set(exact).intersection(*(set(t) for t in tables)))
            differences = [np.mean([t[w] for t in tables])-exact[w] for w in worlds]
            if not differences:
                continue
            interval = bootstrap(differences)
            eligible = len(worlds) == len(summary['worlds']) and len(worlds) >= 8 and all(f['early_stopped'] for f in fits)
            state = 'UNRESOLVED'
            if eligible and interval['finite']:
                state = 'HEADROOM' if interval['lower'] > .02 else 'PARITY' if interval['upper'] <= .02 else 'UNRESOLVED'
            headroom.append(dict(group=dict(key), target=target, budget=budget, interval=interval,
                                 all_worlds_and_validation_stops=eligible, disposition=state))
    result = dict(card=card, estimates=estimates, mechanism=mechanism, matched_headroom=headroom,
                  margin_nats=.02, probability_margin=.02,
                  inference='descriptive whole-world intervals; no multiplicity-adjusted confirmation',
                  criteria='do not equate a significant pooled score with any card prediction; evaluate its target and fixed strata')
    write(root/'CONTRASTS.json', result)
    return result
