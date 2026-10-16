"""Portable review of complete R/N families, including partitioned logical batches.

Does not sample, fit, select a new variant, alter a card or acknowledge events.
Unavailable predictions remain unavailable in the independently checked summaries.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from pathlib import Path
import time
from ghostscale.validation.soundingline.v23 import family, runtime, registry
from ghostscale.validation.soundingline.v23.common import read, write, digest, canonical
from runners.review_v23_calibration import check


def replay_owners(root, plan):
    """A logical parent assembles; each child must have its own sequential replay."""
    if plan.get('world_indices') is not None:
        raise ValueError('partial worlds are not a reviewable logical batch')
    parts = [root]
    if plan.get('scheduler_only'):
        from ghostscale.validation.soundingline.v23.partition import part_paths
        parts = part_paths(root, plan)
    receipts, covered = [], []
    for index, part in enumerate(parts):
        child, _ = family.verify_packet(part)
        if part != root:
            worlds = child.get('world_indices')
            from ghostscale.validation.soundingline.v23.partition import verify_part_binding
            verify_part_binding(root,plan,part,child,index)
            covered.extend(worlds)
        original, replay = (read(part/(label+'-owner.json')) for label in ('original', 'replay'))
        if original['pid'] == replay['pid'] or original['at'] >= replay['at']:
            raise ValueError('distinct sequential replay processes required')
        receipts.append(dict(packet=part.name, original_pid=original['pid'], replay_pid=replay['pid'],
            original_owner_sha256=digest(part/'original-owner.json'),
            replay_owner_sha256=digest(part/'replay-owner.json'),
            complete_sha256=digest(part/'COMPLETE.json')))
    if plan.get('scheduler_only') and sorted(covered) != list(range(plan['worlds'])):
        raise ValueError('duplicate or incomplete partition worlds')
    return receipts


def validate_raw_identities(root, plan, summary):
    total = 0
    if plan['family']=='N1':
        from ghostscale.validation.soundingline.v23.common import rng
        import numpy as np
        for item in summary['variants']:
            identity=item['variant']['id'];phase=['discovery',plan['batch']]
            for world in range(item['worlds']):
                folder=root/'original'/identity/f'world-{world:04d}'
                if read(folder/'IDENTITY.json')['identity']!=['N1',identity,phase,world]:raise ValueError('population identity differs')
                with np.load(folder/'truth_points.npz') as truth:
                    expected=int(rng(identity,phase,'law-block',world//16).permutation(32)[world%16])
                    if int(truth['law'])!=expected or len(truth['y'])!=item['cases_per_world']:raise ValueError('population law or maker coverage differs')
                    total+=len(truth['y'])
        return total
    for item in summary['variants']:
        identity = item['variant']['id']; seen = set(); folder = root/'original'/identity
        for pred, truth in zip(family.rows(folder/'prediction_points.jsonl.gz'),
                               family.rows(folder/'truth_points.jsonl.gz'), strict=True):
            key = pred['world'], pred['case']
            if key in seen or key[0] not in range(plan['worlds']) or key[1] not in range(plan['cases']):
                raise ValueError('duplicate or invalid raw case')
            seen.add(key)
            if truth['identity'] != [plan['family'], identity, plan['batch'], *key]:
                raise ValueError('case identity changed')
            if truth['truth']['law'] != key[0] % 16:
                raise ValueError('unbalanced law menu')
        if len(seen) != plan['worlds']*plan['cases']:
            raise ValueError('case coverage mismatch')
        total += len(seen)
    return total


def fit_evidence(root, plan):
    fit_root = Path(plan['fit_directory']) if plan.get('fit_directory') else root/'original/fits'
    for relative, expected in plan.get('fit_hashes', {}).items():
        if digest(fit_root/relative) != expected:
            raise ValueError('frozen fit changed')
    return dict(summary=read(fit_root/'FITS.json'), summary_sha256=digest(fit_root/'FITS.json'))


def review(campaign, name, packets, output):
    started = time.process_time(); runtime.below_normal()
    campaign, output = Path(campaign).resolve(), Path(output)
    packets = [Path(p).resolve() for p in packets]
    output.mkdir(parents=True, exist_ok=False)
    try:
        cal = campaign/'packets/v23-calibration-3'
        _, cal_summary = family.verify_packet(cal)
        validation = check(read(cal/'original/VALIDATION.json'))
        accepted = read(campaign/'CALIBRATION_ACCEPTANCE.json')
        if (not validation['qualifies'] or cal_summary['state'] != 'CALIBRATION_PASSED' or
                accepted['acceptance_sha256'] != digest(campaign/'ACCEPTANCE.json') or
                accepted['complete_sha256'] != digest(cal/'COMPLETE.json')):
            raise ValueError('calibration unresolved or changed')
        policy = read(cal/'original/CHOSEN_BEFORE_VALIDATION.json')
        receipts, trajectory, diagnostics, raw_index = [], [], {}, {}
        total = 0
        for number, root in enumerate(packets, 1):
            plan, summary = family.verify_packet(root)
            if plan['family'] != name or plan['batch'] != number:
                raise ValueError('wrong packet sequence')
            if plan['sources']['ghostscale/validation/soundingline/v23/adaptive.py'] != accepted['adaptive_source_sha256']:
                raise ValueError('calibrated policy source changed')
            if digest(plan['source_archive']) != plan['source_archive_sha256']:
                raise ValueError('source archive changed')
            source = Path(plan['source_archive']).with_suffix('')
            for relative, expected in plan['sources'].items():
                if digest(source/relative) != expected:
                    raise ValueError('source capsule changed')
            owners = replay_owners(root, plan)
            fits = fit_evidence(root, plan)
            count = validate_raw_identities(root, plan, summary)
            rebuilt = family.reconstruct(root)
            card = read(root/'MACHINE_CARD.json')
            if (card['complete_sha256'] != digest(root/'COMPLETE.json') or
                    card['numerical_review_sha256'] != digest(root/'NUMERICAL_REVIEW.json') or
                    card['calibration_complete_sha256'] != accepted['complete_sha256'] or
                    card['predecessor_cards'] != [digest(p/'MACHINE_CARD.json') for p in packets[:number-1]]):
                raise ValueError('machine card bindings changed')
            if canonical(rebuilt) != canonical(read(root/'NUMERICAL_REVIEW.json')):
                raise ValueError('independent numerical review changed')
            decision = family.decide(name, packets[:number], policy)
            if any(canonical(card[k]) != canonical(value) for k, value in decision.items()):
                raise ValueError('calibrated machine decision differs')
            for item in summary['variants']:
                diagnostics.setdefault(item['variant']['id'], []).append(dict(batch=number, summary=item))
            trajectory.append({k: card.get(k) for k in ('batch', 'state', 'q', 'best_id', 'active_ids')})
            raw_index[root.name] = read(root/'REPLAY.json')['files']
            receipts.append(dict(packet=root.name, cases=count, owners=owners,
                complete_sha256=digest(root/'COMPLETE.json'), replay_sha256=digest(root/'REPLAY.json'),
                numerical_review_sha256=digest(root/'NUMERICAL_REVIEW.json'),
                machine_card_sha256=digest(root/'MACHINE_CARD.json'), fits_sha256=fits['summary_sha256'],
                source_archive_sha256=plan['source_archive_sha256']))
            total += count
            print('verified', root.name, flush=True)
        last = read(packets[-1]/'MACHINE_CARD.json')
        write(output/'DIAGNOSTICS.json', diagnostics)
        write(output/'CURVE.json', last['curve'])
        write(output/'TRAJECTORY.json', trajectory)
        write(output/'RAW_HASH_INDEX.json', raw_index)
        write(output/'REVIEW.json', dict(state='verified_discovery', family=name, decision=last['state'],
            best_observed=last['best_id'], nominated=last['best_id'] if last['state']=='NOMINATED' else None,
            confirmation=False, variants=len(diagnostics), cases=total, receipts=receipts,
            fits=fits, calibration_rechecked=validation, full_fresh_process_replay=True,
            result_kind='constructed method',
            limitations='Adaptive discovery, not confirmation. Unavailable predictions and target strata remain explicit. Engineering validity does not establish human inference.'))
        return output
    finally:
        write(campaign/'attempts'/('review-'+output.name+'.json'), dict(cpu_seconds=time.process_time()-started,
            state='review_complete' if (output/'REVIEW.json').exists() else 'review_failed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--family', choices=list(registry.PRIMARY), required=True)
    parser.add_argument('--packets', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(review(args.campaign, args.family, args.packets, args.output))
