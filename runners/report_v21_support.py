"""Regroup an immutable C03 block; reconstruct masks and supplied-law inputs.

No fitting, admission, or new scientific sampling occurs here. Fits stay nested
inside worlds. Infinite log losses remain explicit rather than being capped.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import time
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from ghostscale.validation.soundingline.v18_4.priority import below_normal
from ghostscale.validation.soundingline.v21 import world as W
from ghostscale.validation.soundingline.v21.checker import scalar_mixture


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def interval(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        return dict(mean=None, lower=None, upper=None, state='infinite; no finite interval')
    # Resample whole worlds after averaging the two fits within each world.
    rng = np.random.default_rng(21)
    draws = np.array([np.mean(values[rng.choice(len(values), len(values), replace=True)])
                      for _ in range(4096)])
    return dict(mean=float(np.mean(values)), lower=float(np.quantile(draws, .025)),
                upper=float(np.quantile(draws, .975)), state='descriptive paired world bootstrap')


def report(packet):
    plan = read(packet / 'PLAN.json')
    d = plan['design']
    assert d['handler'] == 'support'
    source = Path(d['forecast_root'])
    complete = read(packet / 'COMPLETE.json')
    original = read(packet / 'SUMMARY.json')
    for name, expected in complete['files'].items():
        assert file_digest(packet / name) == expected, name
    assert original['population'], 'no admitted support observations'
    prerequisite = read(source / 'SUMMARY.json')
    assert prerequisite['screen_admitted']
    assert all(f['early_stopped'] for f in prerequisite['fit_records'])
    worlds = d['lineages']
    fits = d['fit_seeds']
    cached = {}
    max_reference_error = 0.
    # Independent sequential law/maker grouping checks all supplied posteriors.
    for world in worlds:
        with np.load(source / 'reader' / f'{world}_points.npz') as z:
            x = z['evidence']
        with np.load(source / 'evaluator' / f'{world}_points.npz') as z:
            truth, supplied = z['truth'], np.exp(z['supplied_logp'])
        rows = W.enumerate_world(world, d['shift'], d['regime'])
        reference = scalar_mixture([rows], x, d['tier'], W.CONTEXTS,
                                   d['history_tier'], d['observed'], d['history_observed'])
        error = float(np.max(np.abs(reference - supplied)))
        max_reference_error = max(max_reference_error, error)
        assert error < 1e-10
        assert np.array_equal(reference > 0, supplied > 0)
        cached[world] = x, truth, reference > 0
    observations = {}
    cases = []
    for fit in fits:
        with np.load(source / 'training' / f'{fit}_points.npz') as z:
            train_x, train_y = z['evidence'], z['labels']
        table = {}
        for x, y in zip(train_x, train_y):
            table.setdefault(tuple(x[:13]), set()).add(int(y))
        for world in worlds:
            x, truth, exact = cached[world]
            with np.load(packet / 'raw' / f'{fit}-{world}-none_points.npz') as z:
                raw = z['logp']
            physical = np.ones_like(exact)
            physical[x[:, 5] == 0] = W.LABELS[:, 0] == 0
            learned = np.array([[label in table.get(tuple(row[:13]), range(128))
                                 for label in range(128)] for row in x])
            masks = {'none': np.ones_like(exact), 'physical': physical,
                     'learned': learned, 'supplied-exact': exact}
            for rate in (0, .01, .05, .2):
                rng = np.random.default_rng(W.seed('v21-support-omission', rate, fit))
                mask = exact & ~(rng.random(128) < rate)
                empty = ~mask.any(1)
                mask[empty] = exact[empty]
                masks[f'omission-{rate}'] = mask
            for arm, mask in masks.items():
                name = f'{fit}-{world}-{arm}_points.npz'
                with np.load(packet / 'raw' / name) as z:
                    arrays = {k: z[k] for k in z.files}
                assert np.array_equal(truth, arrays['truth'])
                assert np.array_equal(np.isfinite(arrays['logp']), mask)
                # Independently normalize the retained unchanged predictor in probability space.
                p = np.exp(raw) * mask
                p /= p.sum(axis=1, keepdims=True)
                assert np.allclose(p, np.exp(arrays['logp']), atol=1e-12, rtol=1e-10)
                means = {k: np.mean(v, axis=0) for k, v in arrays.items()
                         if k not in ('logp', 'truth')}
                means['infinite_log_loss_fraction'] = float(np.isinf(arrays['log_loss']).mean())
                exclusion = float((~mask[np.arange(len(truth)), truth]).mean())
                summary_arm = 'corrupted-exact' if arm.startswith('omission-') else arm
                match = [r for r in original['population'] if r['fit'] == fit and
                         r['lineage'] == world and r['support'] == summary_arm and
                         (summary_arm != 'corrupted-exact' or r['omission'] == float(arm.split('-')[1]))]
                assert len(match) == 1
                for key, value in safe(means).items():
                    old = match[0]['metrics'][key]
                    assert old == value if value is None else np.allclose(old, value, atol=1e-12), key
                assert exclusion == match[0]['truth_exclusion']
                means['truth_exclusion'] = exclusion
                means['incompatible_mass'] = float((p * ~exact).sum(axis=1).mean())
                observations[fit, world, arm] = means
                if fit == fits[0] and world == worlds[0] and arm in ('learned', 'omission-0.2', 'supplied-exact'):
                    excluded = np.flatnonzero(~mask[np.arange(len(truth)), truth])
                    i = int(excluded[0]) if len(excluded) else 0
                    cases.append(dict(scope='evaluator-only illustrative case', arm=arm, fit=fit,
                                      world=world, row=i, evidence=x[i].tolist(), target=int(truth[i]),
                                      retained_candidates=np.flatnonzero(mask[i]).tolist(),
                                      truth_retained=bool(mask[i, truth[i]]),
                                      probability_of_truth=float(p[i, truth[i]])))
    arms = list(masks)
    metrics = list(means)
    world_rows = []
    for world in worlds:
        for arm in arms:
            values = {key: np.mean([observations[f, world, arm][key] for f in fits], axis=0)
                      for key in metrics}
            world_rows.append(dict(world=world, support=arm, metrics=safe(values)))
    totals = {}
    for arm in arms:
        totals[arm] = safe({key: np.mean([observations[f, w, arm][key] for w in worlds for f in fits], axis=0)
                           for key in metrics})
    contrasts = []
    for a, b in [('supplied-exact', 'none'), ('learned', 'none'),
                 ('physical', 'none'), ('omission-0.01', 'supplied-exact'),
                 ('omission-0.05', 'supplied-exact'), ('omission-0.2', 'supplied-exact')]:
        for key in ('log_loss', 'brier', 'correct', 'truth_exclusion'):
            values = [np.mean([observations[f, w, a][key] - observations[f, w, b][key]
                               for f in fits]) for w in worlds]
            contrasts.append(dict(arm=a, reference=b, metric=key, **interval(values)))
    return dict(packet=packet.name, worlds=len(worlds), fits_per_world=len(fits),
                cases_per_world=d['test_cases'], forecast_rows=len(observations)*d['test_cases'],
                paired_world_rows=world_rows, arm_means=totals, contrasts=contrasts,
                supplied_reference_max_abs_error=max_reference_error,
                all_summary_metrics_reconstructed=True, all_masks_reconstructed=True,
                illustrative_cases=cases,
                limitations=['Native original-law artifact-only current evidence; four complete past episodes.',
                             'One retained converged direct predictor per fit; no method-by-support comparison.',
                             'No shifted C03 block accepted here; no native-to-shift contrast available.',
                             'Upstream law-marginal family full replay is separate; no new ceiling claim.',
                             'Eight development worlds, two shared fits nested within each; no confirmation.',
                             'Candidate-identity pruning is a finite stress test, not a model of deployment noise.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    below_normal()
    started = time.process_time()
    state = 'failed'
    try:
        write(args.out, report(args.root))
        state = 'complete'
    finally:
        write(args.campaign / 'attempts' / ('support-regroup-' + str(os.getpid()) + '.json'),
              dict(job=args.root.name, role='independent_regroup', state=state,
                   cpu_seconds=time.process_time()-started, at=now()))
