"""Bounded numerical review through the existing scientific queue.

Reads only immutable complete packets. Reused checks bind both their receipt and
the unchanged completion/files; this consumer never accepts a scientific claim.
"""
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest
from ghostscale.validation.soundingline.v20.runtime import fingerprint
from runners.review_v21 import verify
from runners.report_v21_support import report as support_report, safe, interval


def checked_packet(binding, environment, cache, pulse):
    packet = Path(binding['root'])
    def sha(path):
        path = Path(path)
        if path not in cache:
            cache[path] = file_digest(path)
        return cache[path]
    assert sha(packet / 'COMPLETE.json') == binding['complete_sha256'], 'complete binding'
    assert sha(packet / 'PLAN.json') == binding['plan_sha256'], 'plan binding'
    plan = read(packet / 'PLAN.json')
    complete = read(packet / 'COMPLETE.json')
    assert complete['plan_sha256'] == binding['plan_sha256'], 'completion plan'
    assert plan['environment'] == environment, 'environment'
    assert plan['acceptance_sha256'] == binding['acceptance_sha256'], 'acceptance'
    assert sha(packet / 'SOURCE.zip') == plan['archive_sha256'], 'archive'
    for name, expected in plan['sources'].items():
        assert sha(Path(binding['source']) / name) == expected, name
    for name, expected in complete['files'].items():
        assert sha(packet / name) == expected, name
    pulse(phase='review-bound-packet', packet=packet.name)
    if binding.get('score_receipt'):
        prior = Path(binding['score_receipt'])
        assert sha(prior) == binding['score_receipt_sha256'], 'score receipt'
        receipt = read(prior)
        assert receipt['passed'] and receipt['complete_sha256'] == binding['complete_sha256']
        return dict(receipt, reused_score_receipt_sha256=sha(prior))
    return verify(packet)


def regroup_rows(packet):
    """Reconstruct population means from saved per-case score arrays."""
    design = read(packet / 'PLAN.json')['design']
    summary = read(packet / 'SUMMARY.json')
    handler = design['handler']
    rows = []
    if handler not in ('learned', 'adaptation', 'headroom', 'ceiling'):
        return None
    for row in summary.get('population', []):
        parts = [str(row[k]) for k in ('fit', 'k', 'lineage') if k in row]
        name = '-'.join(parts + [row['reader']]) + '_points.npz'
        with np.load(packet / 'raw' / name) as z:
            metrics = {k: np.mean(z[k], axis=0) for k in row['metrics']
                       if k != 'infinite_log_loss_fraction'}
            metrics['infinite_log_loss_fraction'] = float(np.isinf(z['log_loss']).mean())
        for key, value in safe(metrics).items():
            old = row['metrics'][key]
            assert old == value if value is None else np.allclose(old, value, atol=1e-12), (name, key)
        rows.append(dict(row, metrics=safe(metrics)))
    worlds = sorted({r['lineage'] for r in rows})
    arms = sorted({(r['reader'], r.get('k', -1)) for r in rows})
    world_rows = []
    for world in worlds:
        for reader, k in arms:
            subset = [r for r in rows if r['lineage'] == world and r['reader'] == reader
                      and r.get('k', -1) == k]
            values = {}
            for metric in subset[0]['metrics']:
                observations = [r['metrics'][metric] for r in subset]
                values[metric] = None if any(v is None for v in observations) else safe(np.mean(observations, axis=0))
            world_rows.append(dict(world=world, reader=reader, k=k, metrics=values,
                                   fits_nested=len(subset)))
    contrasts = []
    reference = 'converged-direct' if handler == 'learned' else 'direct-finetuned' if handler == 'adaptation' else None
    if reference:
        for reader, k in arms:
            if reader == reference:
                continue
            for metric in ('log_loss', 'brier', 'correct', 'confident_error'):
                paired = []
                for world in worlds:
                    a = next(r for r in world_rows if r['world'] == world and r['reader'] == reader and r['k'] == k)
                    b = next(r for r in world_rows if r['world'] == world and r['reader'] == reference and r['k'] == k)
                    paired.append(a['metrics'][metric] - b['metrics'][metric])
                contrasts.append(dict(reader=reader, reference=reference, k=k, metric=metric,
                                      **interval(paired)))
    return dict(packet=packet.name, design=design, state=summary.get('state'),
                world_rows=world_rows, contrasts=contrasts, all_summary_metrics_reconstructed=True,
                scope='Descriptive development; two fits nested within each world, no acceptance or confirmation.')


def run(root, design, pulse):
    environment = fingerprint()
    cache = {}
    receipts = []
    for binding in design['packets']:
        packet = Path(binding['root'])
        receipt = checked_packet(binding, environment, cache, pulse)
        write(root / 'scores' / (packet.name + '.json'), receipt)
        receipts.append(receipt)
        regrouped = regroup_rows(packet)
        if regrouped is not None:
            # Paths stay private in PLAN; result products contain scientific design only.
            for key in ('ceiling_root', 'headroom_root', 'forecast_root', 'source_root'):
                regrouped['design'].pop(key, None)
            write(root / 'regroup' / (packet.name + '.json'), regrouped)
        d = read(packet / 'PLAN.json')['design']
        if d['handler'] == 'support' and read(packet / 'SUMMARY.json').get('population'):
            pulse(phase='independent-support-regroup', packet=packet.name)
            write(root / 'support' / (packet.name + '.json'), support_report(packet))
    for bound in design['replay_receipts']:
        assert file_digest(Path(bound['path'])) == bound['sha256'], 'full replay receipt changed'
    # Bind all five complete family replays again; they are already checked targets.
    target_hashes = {r['packet']: r['complete_sha256'] for r in receipts}
    for pair in design['family_replays']:
        assert target_hashes[pair['original']] == pair['original_complete_sha256']
        assert target_hashes[pair['replay']] == pair['replay_complete_sha256']
        assert pair['passed']
    pulse(phase='review-complete')
    return dict(card='G21-F04', controls=dict(complete_bindings=True, scores=True, summary_regroup=True),
                packets=len(receipts), reused_score_receipts=sum('reused_score_receipt_sha256' in r for r in receipts),
                forecasts=sum(r['forecast_rows'] for r in receipts),
                retained_files=sum(r['files'] for r in receipts),
                full_family_replays_reused=len(design['family_replays']),
                scientific_acceptance=False,
                remaining=['Reader native-to-shift contrasts, strongest-rival interpretation and full write-through.',
                           'B07 native modular forecasts and duplicate-removal adaptation arm were not retained.',
                           'F03 opening evidence tiers are not the full registered intervention fixture set.'])
