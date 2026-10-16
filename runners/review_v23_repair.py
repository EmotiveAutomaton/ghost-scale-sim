"""Read-only scientific reconstruction and portable repair-family review products.

Never samples worlds, retrains fits, changes a machine card, or acknowledges an
event. The only outputs are new review artifacts and a separately charged cost.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import math
import time
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v23 import family, runtime, registry
from ghostscale.validation.soundingline.v23.common import read, write, digest, interval, canonical
from runners.review_v23_calibration import check


def review(campaign, name, packets, output):
    started = time.process_time(); runtime.below_normal()
    campaign, output = Path(campaign).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        cal = campaign/'packets/v23-calibration-3'
        _, cal_summary = family.verify_packet(cal)
        validation = check(read(cal/'original/VALIDATION.json'))
        if not validation['qualifies'] or cal_summary['state'] != 'CALIBRATION_PASSED':
            raise ValueError('calibration unresolved')
        acceptance = read(campaign/'CALIBRATION_ACCEPTANCE.json')
        if acceptance['acceptance_sha256'] != digest(campaign/'ACCEPTANCE.json'):
            raise ValueError('acceptance changed')
        if acceptance['complete_sha256'] != digest(cal/'COMPLETE.json'):
            raise ValueError('replacement calibration binding changed')
        policy = read(cal/'original/CHOSEN_BEFORE_VALIDATION.json')
        accumulated, raw_index, receipts, trajectory = {}, {}, [], []
        total_cases = 0
        for number, root in enumerate(packets, 1):
            root = Path(root); plan, summary = family.verify_packet(root)
            if plan['family'] != name or plan['batch'] != number:
                raise ValueError('wrong packet sequence')
            if plan['sources']['ghostscale/validation/soundingline/v23/adaptive.py'] != acceptance['adaptive_source_sha256']:
                raise ValueError('calibrated policy source changed')
            original_owner, replay_owner = (read(root/(label+'-owner.json')) for label in ('original', 'replay'))
            if original_owner['pid'] == replay_owner['pid'] or original_owner['at'] >= replay_owner['at']:
                raise ValueError('distinct sequential replay processes required')
            source = Path(plan['source_archive']).with_suffix('')
            if digest(plan['source_archive']) != plan['source_archive_sha256']:
                raise ValueError('source archive changed')
            for relative, expected in plan['sources'].items():
                if digest(source/relative) != expected: raise ValueError('source capsule changed')
            if number > 1:
                for relative, expected in plan['fit_hashes'].items():
                    if digest(Path(plan['fit_directory'])/relative) != expected: raise ValueError('frozen fit changed')
            indexes = read(root/'original/RAW_INDEX.json')
            for identity, files in indexes.items():
                for relative, expected in files.items():
                    if digest(root/'original'/identity/relative) != expected: raise ValueError('raw index mismatch')
            raw_index[root.name] = read(root/'REPLAY.json')['files']
            max_error = 0.
            for item in summary['variants']:
                v = item['variant']; folder = root/'original'/v['id']
                target = accumulated.setdefault(v['id'], dict(variant=v, worlds=[], readers={}, strata={}, contrasts={}))
                predictions = list(family.rows(folder/'prediction_points.jsonl.gz'))
                truths = list(family.rows(folder/'truth_points.jsonl.gz'))
                if len(predictions) != plan['worlds']*plan['cases'] or len(truths) != len(predictions):
                    raise ValueError('case coverage mismatch')
                for world in range(plan['worlds']):
                    lo, hi = world*plan['cases'], (world+1)*plan['cases']
                    pp, tt = predictions[lo:hi], truths[lo:hi]
                    for case, (p, t) in enumerate(zip(pp, tt, strict=True)):
                        if (p['world'], p['case']) != (world, case) or t['identity'] != [name, v['id'], number, world, case]:
                            raise ValueError('case identity changed')
                        if t['truth']['law'] != world % 16: raise ValueError('unbalanced law menu')
                    y = np.array([t['truth']['y'] for t in tt], dtype=int)
                    goal = np.array([t['truth']['goal'] if name == 'R1' else t['truth']['state'][0] for t in tt]) if name in ('R1', 'R2') else None
                    per_reader = {}
                    for reader in pp[0]['probabilities']:
                        p = np.array([row['probabilities'][reader] for row in pp])
                        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0):
                            raise ValueError('invalid saved probabilities')
                        chosen = np.take_along_axis(p, y[..., None], axis=-1)[..., 0]
                        with np.errstate(divide='ignore'):
                            loss = -np.log(chosen).mean(-1)
                        onehot = np.eye(p.shape[-1])[y]
                        scored = dict(log_loss=loss, half_brier=((p-onehot)**2).sum(-1).mean(-1)/2,
                                      accuracy=(p.argmax(-1) == y).mean(-1))
                        for metric, values in scored.items():
                            saved = np.array([row['scores'][reader][metric] for row in pp])
                            error = float(np.max(np.abs(values-saved))); max_error = max(max_error, error)
                            if error > 1e-12: raise ValueError('independent proper score mismatch')
                            target['readers'].setdefault(reader+':'+metric, []).append(float(values.mean()))
                            for g in (0, 1) if goal is not None else ():
                                if np.any(goal == g):
                                    target['strata'].setdefault(f'goal-{g}:{reader}:{metric}', []).append(float(values[goal == g].mean()))
                        per_reader[reader] = loss
                    rival, mechanism = registry.PRIMARY[name][:2]
                    gain = float(np.mean(per_reader[rival]-per_reader[mechanism]))
                    if not math.isclose(gain, item['world_gains'][world], abs_tol=1e-12, rel_tol=0):
                        raise ValueError('independent primary world mean mismatch')
                    target['worlds'].append(dict(batch=number, world=world, law=world % 16, gain=gain))
                    contrasts = {'primary': (rival, mechanism)}
                    if name == 'R1':
                        contrasts.update(skepticism=('missing-process', 'masked'),
                            false_skepticism_masked=('masked', 'missing-process'),
                            false_skepticism_visible=('law-marginal', 'missing-process'))
                    for label, (a, b) in contrasts.items():
                        values = per_reader[a]-per_reader[b]
                        target['contrasts'].setdefault(label, []).append(float(values.mean()))
                        for g in (0, 1) if goal is not None else ():
                            if np.any(goal == g):
                                target['contrasts'].setdefault(label+f':goal-{g}', []).append(float(values[goal == g].mean()))
                    for reader in summary['fits']:
                        reference = 'masked' if reader.endswith('-masked') else 'law-marginal'
                        target['contrasts'].setdefault('headroom:'+reader, []).append(float(np.mean(per_reader[reader]-per_reader[reference])))
                rebuilt = interval([row['gain'] for row in target['worlds'] if row['batch'] == number], [v['id'], number, 'primary'])
                for key in ('mean', 'se', 'lower', 'upper'):
                    if not math.isclose(rebuilt[key], item['primary'][key], abs_tol=1e-12, rel_tol=0):
                        raise ValueError('bootstrap reconstruction differs')
                total_cases += len(predictions)
            card = read(root/'MACHINE_CARD.json')
            decision = family.decide(name, packets[:number], policy)
            if any(canonical(card[k]) != canonical(value) for k, value in decision.items()):
                raise ValueError('calibrated machine decision differs')
            trajectory.append({k: card.get(k) for k in ('batch', 'state', 'q', 'best_id', 'active_ids')})
            receipts.append(dict(packet=root.name, complete_sha256=digest(root/'COMPLETE.json'),
                replay_sha256=digest(root/'REPLAY.json'), raw_index_sha256=digest(root/'original/RAW_INDEX.json'),
                machine_card_sha256=digest(root/'MACHINE_CARD.json'), source_archive_sha256=plan['source_archive_sha256'],
                original_pid=original_owner['pid'], replay_pid=replay_owner['pid'], independent_score_max_error=max_error,
                replay_file_count=len(raw_index[root.name]), cpu_seconds=read(root/'STATUS.json')['cpu_seconds']))
            print('verified', root.name, flush=True)
        diagnostic = {}
        for identity, item in accumulated.items():
            entry = dict(variant=item['variant'], worlds=item['worlds'])
            for group in ('readers', 'strata', 'contrasts'):
                entry[group] = {key: interval(values, [name, 'review', identity, group, key]) for key, values in item[group].items()}
            diagnostic[identity] = entry
        last = read(Path(packets[-1])/'MACHINE_CARD.json')
        write(output/'DIAGNOSTICS.json', diagnostic)
        write(output/'CURVE.json', last['curve'])
        write(output/'TRAJECTORY.json', trajectory)
        write(output/'RAW_HASH_INDEX.json', raw_index)
        write(output/'REVIEW.json', dict(state='verified_discovery', family=name, decision=last['state'],
            nominated=last['best_id'] if last['state']=='NOMINATED' else None, best_observed=last['best_id'],
            confirmation=False, variants=len(accumulated), cases=total_cases, receipts=receipts,
            calibration_rechecked=validation, fits=read(Path(packets[0])/'original/fits/FITS.json'),
            result_kind='constructed method', full_fresh_process_replay=True,
            limitations='Adaptive discovery; raw and shrunk estimates do not establish confirmation. Stratum intervals average nonempty whole worlds. No human inference.'))
        return output
    finally:
        write(campaign/'attempts'/('review-'+output.name+'.json'), dict(cpu_seconds=time.process_time()-started,
            state='review_complete' if (output/'REVIEW.json').exists() else 'review_failed'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--family', choices=['R1', 'R2', 'R3', 'R4'], required=True); p.add_argument('--packets', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    print(review(a.campaign, a.family, a.packets, a.output))
