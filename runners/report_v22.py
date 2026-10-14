"""Retained-data V22 review. No fitting, simulation, or producer posterior calls.

Scores are marginal by target. Fits are averaged within coefficient worlds;
contrasts and percentile intervals then resample eight paired worlds. This is
descriptive analysis of the fixed first pass, not a new acceptance criterion.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def interval(values):
    x = np.asarray(values, float)
    finite = bool(np.isfinite(x).all())
    out = dict(worlds=len(x), mean=float(x.mean()) if finite and len(x) else None,
               world_values=[float(v) if np.isfinite(v) else None for v in x],
               nonfinite_worlds=int((~np.isfinite(x)).sum()))
    if finite and len(x):
        draws = np.random.default_rng(220014).integers(len(x), size=(4096, len(x)))
        lo, hi = np.quantile(x[draws].mean(1), [.025, .975])
        out.update(lower=float(lo), upper=float(hi))
    return out


def binding(packet, campaign):
    plan, done, valid, replay = [read(packet / name) for name in
                                ('PLAN.json', 'COMPLETE.json', 'VALIDITY.json', 'REPLAY.json')]
    for key, name in [('plan_sha256', 'PLAN.json'), ('validity_sha256', 'VALIDITY.json'),
                      ('replay_sha256', 'REPLAY.json'), ('summary_sha256', 'original/SUMMARY.json')]:
        assert done[key] == digest(packet / name), (packet.name, key)
    assert plan['acceptance_sha256'] == digest(campaign / 'ACCEPTANCE.json')
    capsule = campaign / 'capsules' / plan['capsule']
    assert plan['archive_sha256'] == digest(capsule / 'SOURCE.zip')
    for name, h in plan['sources'].items():
        assert digest(capsule / 'source' / name) == h, name
    assert valid['passed'] and replay['passed'] and replay['complete_scientific_replay']
    for side in ('original', 'replay'):
        manifest = packet / side / 'SCIENTIFIC_MANIFEST.json'
        assert digest(manifest) == valid['manifest_sha256'] == replay[side + '_manifest_sha256']
        for name, h in read(manifest).items():
            assert digest(packet / side / name) == h, (side, name)
    queue = read(campaign / 'QUEUE.json')['jobs']
    assert next(j for j in queue if j['id'] == packet.name)['plan_sha256'] == digest(packet / 'PLAN.json')
    return dict(job=packet.name, card=plan['card'], environment=plan['environment'],
                source_archive_sha256=plan['archive_sha256'], source_files=len(plan['sources']),
                acceptance_sha256=plan['acceptance_sha256'],
                receipts={n: digest(packet / n) for n in ('PLAN.json', 'COMPLETE.json', 'VALIDITY.json', 'REPLAY.json')},
                summary_sha256=done['summary_sha256'],
                contrasts_sha256=digest(packet / 'original/CONTRASTS.json'),
                validity=valid, replay=replay, binding_verified=True,
                independent_receipts_reused=True, repeated_science=False)


class Study:
    def __init__(self, packet):
        self.packet = packet
        self.summary = read(packet / 'original/SUMMARY.json')
        self.card = self.summary['card']
        self.worlds = []
        self.out = {}
        for world in self.summary['worlds']:
            p = packet / 'original' / f'world-{world}'
            details, obs = records(p / 'details.jsonl.gz'), records(p / 'records.jsonl.gz')
            with np.load(p / 'inputs.npz') as a:
                y, mask = a['y'], a['mask']
            probs = {}
            for file in p.glob('*.npz'):
                if file.name != 'inputs.npz':
                    with np.load(file) as a:
                        probs[file.stem] = a['p']
            scores = {}
            for reader, prob in probs.items():
                q = np.take_along_axis(prob, y[..., None], -1)[..., 0]
                with np.errstate(divide='ignore'):
                    loss = -np.log(q)
                scores[reader] = dict(log_loss=loss, brier=(prob**2).sum(-1)-2*q+1,
                                      accuracy=(prob.argmax(-1) == y).astype(float),
                                      confident_error=((prob.max(-1) >= .9) & (prob.argmax(-1) != y)).astype(float))
            # Average fitted scores, never count the two fits as two worlds.
            for budget in (2048, 8192):
                scores[f'direct-{budget}'] = {m: (scores[f'direct-{budget}-0'][m] +
                                                  scores[f'direct-{budget}-1'][m])/2 for m in scores['law-marginal']}
            self.worlds.append(dict(world=world, path=p, details=details, records=obs,
                                    y=y, mask=mask, probabilities=probs, scores=scores))

    def ids(self, w, selection):
        return [i for i, d in enumerate(w['details']) if all(d[k] == v for k, v in selection.items())]

    def measure(self, name, fn):
        self.out[name] = interval([float(fn(w)) for w in self.worlds])
        return self.out[name]

    def mean(self, w, reader, selection, head, metric='log_loss'):
        ids = self.ids(w, selection)
        return w['scores'][reader][metric][np.ix_(ids, np.atleast_1d(head))].mean()

    def compare(self, name, selection, head, rival, reference='law-marginal'):
        for metric in ('log_loss', 'brier', 'accuracy', 'confident_error'):
            self.measure(name + '/' + metric, lambda w, m=metric:
                         self.mean(w, rival, selection, head, m)-self.mean(w, reference, selection, head, m))

    def score(self, name, selection, head):
        for reader in self.worlds[0]['scores']:
            if reader.endswith(('-0', '-1')):
                continue
            for metric in ('log_loss', 'brier', 'accuracy', 'confident_error'):
                self.measure(name + '/' + reader + '/' + metric,
                             lambda w, r=reader, m=metric: self.mean(w, r, selection, head, m))


def mechanisms(s):
    if s.card == 'A1':
        for k, expertise, goal, competence in itertools.product((0, 2, 8, 32), (.4, 1.2), (0, 1), (.35, 1.2)):
            sel = dict(condition='native', k=k, expertise=expertise, source_goal=goal, competence=competence)
            prefix = f'curves/k{k}/noise{expertise}/goal{goal}/source-noise{competence}'
            for j in range(13):
                s.measure(prefix + f'/D{j*.5:g}', lambda w, j=j, sel=sel:
                          np.mean([w['details'][i]['shift_curve'][j] for i in s.ids(w, sel)]))
        for condition, k, goal in itertools.product(('native', 'unrelated', 'duplicated'), (0, 2, 8, 32), (0, 1)):
            sel = dict(condition=condition, k=k, source_goal=goal)
            s.measure(f'detection/{condition}/k{k}/goal{goal}', lambda w, sel=sel:
                      np.mean([w['details'][i]['steering_probability'] >= .9 for i in s.ids(w, sel)]))
        for condition in ('native', 'unrelated', 'duplicated'):
            s.score(condition, dict(condition=condition), 0)
            for rival in ('fixed-0.2', 'fixed-0.54', 'fixed-0.9', 'label-only', 'duplicated-product', 'rigid-same-source'):
                s.compare(condition + '/gain-over-' + rival, dict(condition=condition), 0, rival)
        for k in (0, 2, 8, 32):
            for rival in ('fixed-0.2', 'fixed-0.54', 'fixed-0.9'):
                s.compare(f'native/k{k}/gain-over-{rival}', dict(condition='native', k=k), 0, rival)
    elif s.card == 'B1':
        for condition, k in itertools.product(('native', 'zero-cost', 'independent-effort', 'alias'), (2, 8, 32)):
            sel = dict(condition=condition, k=k)
            for reader, metric in itertools.product(('law-marginal', 'effort-weighted', 'cost-ignorant'), ('coverage', 'width')):
                s.measure(f'recovery/{condition}/k{k}/{reader}/{metric}', lambda w, sel=sel, r=reader, m=metric:
                          np.mean([w['details'][i]['interval_recovery'][r][m] for i in s.ids(w, sel)]))
            for metric in ('coverage', 'width'):
                s.measure(f'recovery/{condition}/k{k}/effort-minus-exact/{metric}', lambda w, sel=sel, m=metric:
                          np.mean([w['details'][i]['interval_recovery']['effort-weighted'][m]-
                                   w['details'][i]['interval_recovery']['law-marginal'][m] for i in s.ids(w, sel)]))
            s.compare(f'{condition}/k{k}/effort-minus-exact', sel, 0, 'effort-weighted')
        for condition, shift in itertools.product(('native', 'zero-cost', 'independent-effort', 'alias'), (False, True)):
            s.score(f'{condition}/shift{shift}', dict(condition=condition, shift=shift), 0)
        for condition in ('native', 'zero-cost', 'independent-effort', 'alias'):
            for reader in s.worlds[0]['scores']:
                if reader.endswith(('-0', '-1')):
                    continue
                for metric in ('log_loss', 'brier', 'accuracy', 'confident_error'):
                    s.measure(f'shift-change/{condition}/{reader}/{metric}', lambda w, r=reader, m=metric, c=condition:
                              s.mean(w, r, dict(condition=c, shift=True), 0, m)-s.mean(w, r, dict(condition=c, shift=False), 0, m))
        for d in (2, 3, 4):
            s.measure(f'distance-consistency-correlation/d{d}', lambda w, d=d: np.corrcoef(
                [[w['details'][i][m] for i in s.ids(w, dict(condition='native', d=d))]
                 for m in ('distance', 'choice_consistency')])[0, 1])
    elif s.card == 'C2':
        for coupled, disclosed, unrelated, k in itertools.product((False, True), (False, True), (False, True), (0, 2, 8, 32)):
            sel = dict(coupled=coupled, disclosed=disclosed, unrelated=unrelated, k=k)
            label = f'coupled{coupled}/disclosed{disclosed}/unrelated{unrelated}/k{k}'
            for head, target in enumerate(('operation', 'preference')):
                s.score(label + '/' + target, sel, head)
                s.compare(label + '/' + target + '/without-records-minus-exact', sel, head, 'without-records')
                for factor in ('share_lib', 'share_pref'):
                    s.measure(label + '/' + target + '/' + factor + '-benefit', lambda w, sel=sel, h=head, f=factor:
                              s.mean(w, 'law-marginal', dict(sel, **{f: False}), h)-s.mean(w, 'law-marginal', dict(sel, **{f: True}), h))
    elif s.card == 'A2':
        for signature, swapped, goal in itertools.product((False, True), (False, True), (0, 1)):
            sel = dict(signature=signature, swapped=swapped, source_goal=goal)
            label = f'signature{signature}/swapped{swapped}/goal{goal}'
            for head, target in enumerate(('content', 'goal', 'technique')):
                s.score(label + '/' + target, sel, head)
            for reader in ('law-marginal', 'missing-process', 'wrong-signature-map', 'oracle-law'):
                for metric in ('content_uptake', 'technique_production', 'goal_probability'):
                    s.measure(label + '/' + reader + '/' + metric, lambda w, sel=sel, r=reader, m=metric:
                              np.mean([w['details'][i][m][r] for i in s.ids(w, sel)]))
                # Signed uptake cancels across opposite claims. Preserve it above;
                # absolute and claim-directed change answer amount absorbed.
                s.measure(label + '/' + reader + '/absolute-content-uptake', lambda w, sel=sel, r=reader:
                          np.mean([abs(w['details'][i]['content_uptake'][r]) for i in s.ids(w, sel)]))
            s.measure(label + '/missing-minus-exact-absolute-uptake', lambda w, sel=sel:
                      np.mean([abs(w['details'][i]['content_uptake']['missing-process'])-
                               abs(w['details'][i]['content_uptake']['law-marginal']) for i in s.ids(w, sel)]))
    elif s.card == 'B2':
        for condition, devalued in itertools.product(('contingent', 'degraded', 'alias'), (False, True)):
            sel = dict(condition=condition, devalued=devalued)
            label = f'{condition}/devalued{devalued}'
            for h, target in enumerate(('route', 'choice')):
                s.score(label + '/' + target, sel, h)
            for rival in ('model-based-only', 'association-only', 'prior-mixture', 'direct-8192'):
                s.compare(label + '/choice/gain-over-' + rival, sel, 1, rival)
            for route in (0, 1):
                s.measure(label + f'/route{route}/observed-choice', lambda w, sel=dict(sel, route=route):
                          w['y'][s.ids(w, sel), 1].mean())
        s.measure('alias/max-route-deviation-from-prior', lambda w: np.max(np.abs(
                  w['probabilities']['law-marginal'][s.ids(w, dict(condition='alias')), 0, 1]-.5)))
    elif s.card == 'A3':
        policies = ('no-gate', 'early-guard', 'procedure-only', 'withdraw-.7', 'withdraw-.9')
        for coupled, prefix, benign in itertools.product((False, True), (4, 8, 16), (False, True)):
            sel = dict(coupled=coupled, prefix=prefix, benign=benign)
            label = f'coupled{coupled}/prefix{prefix}/benign{benign}'
            def endpoint(w, policy, metric):
                vals = []
                for i in s.ids(w, sel):
                    d = w['details'][i]; end = d['trajectories'][policy][-1]
                    if metric == 'technique':
                        v = float(d['production'][policy] == d['technique_truth'])
                    elif metric == 'recognition':
                        v = float((end['posterior'][0] > .5) == w['y'][i, 0])
                    elif metric == 'flag':
                        v = float(end['fired'])
                    else:
                        v = abs(end['memory'][0 if metric == 'purpose' else 1]-.5)
                    vals.append(v)
                return np.mean(vals)
            cached = {p: {m: [endpoint(w, p, m) for w in s.worlds]
                          for m in ('purpose', 'value', 'technique', 'recognition', 'flag')} for p in policies}
            for policy in policies:
                for metric, values in cached[policy].items():
                    s.out[label + '/' + policy + '/' + metric] = interval(values)
            # Fixed random policy mixtures: weight on withdrawal, rest on no gate.
            # All weights retained. Never choose a best threshold from outcomes.
            for policy, weight in itertools.product(('withdraw-.7', 'withdraw-.9'), (0., .25, .5, .75, 1.)):
                for metric in cached[policy]:
                    delta = weight*np.array(cached[policy][metric])+(1-weight)*np.array(cached['no-gate'][metric])-np.array(cached['early-guard'][metric])
                    s.out[label + f'/mixture/{policy}/weight{weight}/{metric}-minus-guard'] = interval(delta)
        s.out['physical'] = physical_summary(s)
        for metric in ('process_uptake', 'goal_uptake', 'value_uptake'):
            for policy in ('no-gate', 'early-guard'):
                s.measure(f'E56/{policy}/{metric}', lambda w, m=metric, p=policy:
                          np.mean([r[p][m] for r in read(w['path'] / 'E56_ACCOUNTING.json')]))
    elif s.card == 'C1':
        for branch, coverage in itertools.product((2, 3), (.25, .5, .75)):
            sel = dict(branching=branch, coverage=coverage)
            label = f'branch{branch}/coverage{coverage}'
            for depth in range(5):
                heads = list(range(sum(branch**d for d in range(depth)), sum(branch**d for d in range(depth+1))))
                s.score(label + f'/depth{depth}', sel, heads)
                for rival in ('bottom-up', 'top-down', 'direct-8192'):
                    s.compare(label + f'/depth{depth}/gain-over-{rival}', sel, heads, rival)
            for removal in ('adjacent', 'distant'):
                for metric in ('log_loss', 'accuracy'):
                    def changed(w, removal=removal, metric=metric, sel=sel):
                        values = []
                        for i in s.ids(w, sel):
                            d = w['details'][i]; h = d['focus']; y = w['y'][i, h]
                            p = np.array(d['ablations'][removal]); q = w['probabilities']['law-marginal'][i, h]
                            values.append(float(-np.log(p[y])+np.log(q[y])) if metric == 'log_loss' else float(p.argmax()==y)-float(q.argmax()==y))
                        return np.mean(values)
                    s.measure(label + '/' + removal + '-removal/' + metric, changed)


def physical_summary(s):
    grouped = {}
    for w in s.worlds:
        obj = read(w['path'] / 'G5_PHYSICAL.json')
        for row in obj['baseline']['rows'] + obj['withdrawal']:
            method = row.get('method', 'withdraw-' + str(row.get('threshold')))
            key = row['condition'] + '/' + method
            entry = grouped.setdefault(key, dict(uptake=[], execution_success=[], neutral_target_success=[], chosen_target_success=[], cost=[], fired=[]))
            entry['uptake'].append(row['unwanted_goal_uptake'] if 'threshold' in row else row['scores']['unwanted_goal_uptake'])
            entry['execution_success'].append(np.mean([o['execution']['successfully_stopped'] for o in row['outputs']]))
            for j, target in enumerate(('neutral_target_success', 'chosen_target_success')):
                o = row['outputs'][j]
                entry[target].append(float(o['execution']['state'] == o['target'] and o['execution']['successfully_stopped']))
            entry['cost'].append(np.mean([o['charged_total'] for o in row['outputs']]))
            entry['fired'].append(float(row.get('fired', False)))
    return {key: {m: interval(v) for m, v in row.items()} for key, row in grouped.items()}


def export_tables(s, destination):
    c = read(s.packet / 'original/CONTRASTS.json')
    with (destination / 'STRATA.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['group', 'reader', 'target', 'metric', 'worlds', 'mean', 'lower', 'upper', 'nonfinite_worlds'])
        for row in c['estimates']:
            vals = [np.inf if x['value'] is None else x['value'] for x in row['world_values']]
            out = interval(vals)
            writer.writerow([json.dumps(row['group'], sort_keys=True, separators=(',', ':')), row['reader'], row['target'], row['metric'],
                             out['worlds'], out['mean'], out.get('lower'), out.get('upper'), out['nonfinite_worlds']])
    write(destination / 'HEADROOM.json', c['matched_headroom'])
    write(destination / 'FITS.json', [dict(reader=r['reader'], budget=r['budget'], epochs=r['epochs'],
                                        early_stopped=r['early_stopped'], metadata_sha256=digest(s.packet/'original'/(r['reader']+'.json')),
                                        weights_sha256=digest(s.packet/'original'/(r['reader']+'.npz'))) for r in s.summary['fits']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cost', type=Path, required=True)
    p.add_argument('--cards', nargs='+')
    args = p.parse_args()
    start = time.process_time()
    try:
        for packet in sorted((args.campaign / 'packets').iterdir()):
            if args.cards and read(packet / 'PLAN.json')['card'] not in args.cards:
                continue
            evidence = binding(packet, args.campaign)
            s = Study(packet)
            mechanisms(s)
            target = args.output / s.card.lower()
            target.mkdir(parents=True, exist_ok=True)
            write(target / 'PROVENANCE.json', evidence)
            write(target / 'ANALYSIS.json', dict(card=s.card, estimates=s.out,
                  inference='4096 percentile resamples of paired coefficient worlds; fits averaged within worlds; descriptive, no multiplicity correction',
                  differences='named rival minus reference; log loss/Brier lower is better, accuracy higher is better; nonfinite worlds never silently dropped',
                  script_sha256=digest(__file__)))
            export_tables(s, target)
            print(s.card, len(s.out), evidence['validity']['independently_scored_forecasts'], flush=True)
    finally:
        write(args.cost, dict(cpu_seconds=time.process_time()-start, kind='retained-data analysis; no repeated science', source_sha256=digest(__file__)))


if __name__ == '__main__':
    main()
