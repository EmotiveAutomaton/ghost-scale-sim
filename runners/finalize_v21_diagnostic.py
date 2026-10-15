"""Read-only reconstruction of the single, partitioned V21 final diagnostic.

Never dispatches science, changes a gate, or accepts a partial world collection.
Writes a separate immutable review product; publication and closure are separate.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import time
import uuid
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from ghostscale.validation.soundingline.v21 import metrics as M
from runners.review_v21 import verify

READERS = ('converged-direct', 'naive-bayes', 'naive-bayes-without-evidence')


def decision(interval, converged):
    if not converged or interval['lower'] is None or interval['upper'] is None:
        return 'UNRESOLVED'
    if interval['lower'] > .02:
        return 'EXCESS READER ERROR'
    return 'PARITY' if interval['upper'] <= .02 else 'UNRESOLVED'


def pooled_worlds(rows, worlds, reader, metric='headroom_nats'):
    """One equally weighted observation per world; the two fits stay nested."""
    selected = [r for r in rows if r['reader'] == reader]
    expected = {(w, fit) for w in worlds for fit in (0, 1)}
    keys = [(r['lineage'], r['fit']) for r in selected]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing, duplicate or foreign world/fit')
    return [float(np.mean([np.inf if r[metric] is None else r[metric]
                          for r in selected if r['lineage'] == w])) for w in worlds]


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return dict(mean=None, lower=None, upper=None, worlds=len(values),
                    nonfinite_worlds=int((~np.isfinite(values)).sum()),
                    disposition='Uncapped nonfinite values retained; finite interval unavailable.')
    return M.bootstrap(values)


def same_arrays(left, right):
    with np.load(left) as a, np.load(right) as b:
        if set(a.files) != set(b.files):
            raise ValueError('training/model array names differ')
        for key in a.files:
            if not np.array_equal(a[key], b[key], equal_nan=True):
                raise ValueError('training/model identity differs: ' + key)


def expected_loss(p, logp):
    terms = np.zeros_like(p)
    np.multiply(p, logp, out=terms, where=p > 0)
    return -terms.sum(axis=1)


def reconstruct(campaign, manifest_path):
    manifest = read(manifest_path)
    worlds = manifest['design']['lineages']
    parts = manifest['partitions']
    actual = [w for part in parts for w in part['lineages']]
    if sorted(actual) != sorted(worlds) or len(set(actual)) != len(actual):
        raise ValueError('partitions must cover each frozen world exactly once')
    if worlds != list(range(71900000, 71900032)):
        raise ValueError('unexpected frozen final population')
    reference = campaign / 'packets/v21-headroom-00-l64-n8192'
    baseplan = read(reference / 'PLAN.json')
    reference_complete = read(reference / 'COMPLETE.json')
    if file_digest(reference / 'PLAN.json') != reference_complete['plan_sha256']:
        raise ValueError('development reference plan changed')
    rows, ceilings, receipts, fits, decompositions = [], [], [], [], []
    for part in parts:
        root = campaign / 'packets' / part['id']
        plan = read(root / 'PLAN.json')
        expected = dict(manifest['design'], lineages=part['lineages'], cpu_cap_seconds=part['cpu_cap_seconds'])
        if plan['design'] != expected or plan['sources'] != baseplan['sources']:
            raise ValueError('partition design/source differs from freeze')
        if plan['archive_sha256'] != manifest['source_archive_sha256'] or plan['environment'] != manifest['environment']:
            raise ValueError('partition archive/environment differs')
        if plan['acceptance_sha256'] != manifest['acceptance_sha256']:
            raise ValueError('partition acceptance differs')
        receipts.append(verify(root))
        summary = read(root / 'SUMMARY.json')
        if not all(summary['controls'].values()) or summary['ceiling_status'] != 'EXACT':
            raise ValueError('invalid numerical instrument')
        fits.extend(summary['fit_records'])
        for fit in (0, 1):
            for suffix in ('_points.npz', '-direct-model_points.npz'):
                rel = 'training/' + str(fit) + suffix
                if file_digest(reference / rel) != reference_complete['files'][rel]:
                    raise ValueError('development training reference changed')
                same_arrays(root / rel, reference / rel)
        by_key = {(r.get('fit'), r['lineage'], r['reader']): r for r in summary['population']}
        if len(by_key) != len(summary['population']):
            raise ValueError('duplicate reported population entry')
        for world in part['lineages']:
            with np.load(root / 'evaluator' / f'{world}_points.npz') as e:
                supplied, marginal, truth = e['supplied_logp'], e['marginal_logp'], e['truth']
                p, pm = np.exp(supplied), np.exp(marginal)
                irreducible = expected_loss(p, supplied)
                law = expected_loss(p, marginal) - irreducible
                brier_irreducible = 1 - (p*p).sum(axis=1)
                brier_law = ((p-pm)**2).sum(axis=1)
                for key, value in [('irreducible_nats', irreducible), ('law_uncertainty_nats', law),
                                   ('irreducible_brier', brier_irreducible), ('law_uncertainty_brier', brier_law)]:
                    np.testing.assert_allclose(e[key], value, rtol=1e-10, atol=1e-10)
                bayes = np.maximum(pm @ M.LABELS, 1 - pm @ M.LABELS)
                np.testing.assert_allclose(e['bit_bayes_accuracy'], bayes, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(e['supported_targets'], (pm > 0).sum(axis=1))
                with np.load(root / 'raw' / f'{world}-law-marginal_points.npz') as raw_marginal:
                    np.testing.assert_array_equal(truth, raw_marginal['truth'])
                law_empirical = float(-marginal[np.arange(len(truth)), truth].mean())
                ceilings.append(dict(lineage=world, irreducible_nats=float(irreducible.mean()),
                    law_uncertainty_nats=float(law.mean()), irreducible_brier=float(brier_irreducible.mean()),
                    law_uncertainty_brier=float(brier_law.mean()), bit_bayes_accuracy=bayes.mean(axis=0).tolist(),
                    supported_targets=float(((pm > 0).sum(axis=1)).mean())))
                for fit in (0, 1):
                    for reader in READERS:
                        stem = f'{fit}-{world}-{reader}'
                        with np.load(root / 'raw' / (stem + '_points.npz')) as a:
                            arrays = {key: a[key] for key in a.files}
                        np.testing.assert_array_equal(arrays['truth'], truth)
                        scores = M.mean_scores(M.scores(arrays['logp'], truth))
                        reported = by_key[(fit, world, reader)]
                        for key, value in scores.items():
                            if value is None:
                                if reported['metrics'][key] is not None: raise ValueError('lost infinite score')
                            else: np.testing.assert_allclose(value, reported['metrics'][key], atol=1e-10, rtol=1e-10)
                        headroom = float(arrays['log_loss'].mean()) - law_empirical
                        if np.isfinite(headroom):
                            np.testing.assert_allclose(headroom, reported['headroom_nats'], atol=1e-10, rtol=1e-10)
                        elif reported['headroom_nats'] is not None:
                            raise ValueError('nonfinite headroom serialization differs')
                        rows.append(dict(lineage=world, fit=fit, reader=reader,
                                         headroom_nats=headroom if np.isfinite(headroom) else None, **scores))
                        q = np.exp(arrays['logp'])
                        excess = expected_loss(p, arrays['logp']) - expected_loss(p, marginal)
                        excess_brier = ((q-p)**2).sum(axis=1) - brier_law
                        with np.load(root / 'evaluator' / (stem + '-budget_points.npz')) as b:
                            np.testing.assert_allclose(b['reader_excess_nats'], excess, atol=1e-10, rtol=1e-10)
                            np.testing.assert_allclose(b['reader_excess_brier'], excess_brier, atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(irreducible + law + excess, expected_loss(p, arrays['logp']), atol=1e-10, rtol=1e-10)
                        np.testing.assert_allclose(brier_irreducible + brier_law + excess_brier, 1+(q*q).sum(axis=1)-2*(p*q).sum(axis=1), atol=1e-10, rtol=1e-10)
                        mean_excess = float(excess.mean())
                        decompositions.append(dict(lineage=world, fit=fit, reader=reader,
                            reader_excess_nats=mean_excess if np.isfinite(mean_excess) else None,
                            nonfinite_excess_fraction=float((~np.isfinite(excess)).mean()),
                            reader_excess_brier=float(excess_brier.mean())))
    converged = all(f['early_stopped'] for f in fits)
    comparisons = {reader: {metric: interval(pooled_worlds(rows, worlds, reader, metric))
        for metric in ('headroom_nats', 'log_loss', 'brier', 'correct', 'top8', 'zero_mass', 'below_floor', 'confident_error')}
        for reader in READERS}
    primary = comparisons['converged-direct']['headroom_nats']
    return dict(at=now(), passed=True, manifest_sha256=file_digest(manifest_path), worlds=32,
        cases=32*manifest['design']['test_cases'], independent_fits=2, replicated_fits=len(fits),
        all_training_and_weights_match_development=True, all_fits_validation_stopped=converged,
        decision=decision(primary, converged), primary=primary, readers=comparisons,
        world_rows=rows, ceiling_world_rows=ceilings, decomposition_world_rows=decompositions,
        score_receipts=receipts, source_family_replays=manifest['reused_full_family_replays'],
        scope='One frozen diagnostic; existing family replays reused by hash; no maker-superiority or human-mechanism confirmation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time(); state = 'failed'
    try:
        if args.out.exists(): raise ValueError('review output already exists')
        write(args.out, reconstruct(args.campaign, args.manifest)); state = 'complete'
    finally:
        write(args.campaign / 'attempts' / ('final-diagnostic-review-' + uuid.uuid4().hex + '.json'),
            dict(at=now(), job='v21-f01-final', role='independent_review', state=state, cpu_seconds=time.process_time()-started))
