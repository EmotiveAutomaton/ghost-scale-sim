"""Four owner-approved retained-path checks; never resumes the closed queue."""
import argparse
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v27.common import read, write, digest
from ghostscale.validation.soundingline.v27 import contract
from runners import run_k3_redesign as base
from runners.run_v27 import frozen_source
from runners.watch_v23_events import owner, event

REPO = Path(__file__).resolve().parents[1]
JOBS = (('R-low', 1024, False, 600.), ('I-low', 1024, True, 600.),
        ('R-high', 16384, False, 3600.), ('I-high', 16384, True, 3600.))
CAP = 4*3600
EXTRAS = (*base.EXTRAS, 'runners/diagnose_k3_redesign.py')


def now():
    return datetime.now(timezone.utc)


def within_budget(prior, current, reservation=0.):
    """Independent original phase limits, original total, new four-hour cap."""
    if any(not np.isfinite(x) or x < 0 for x in [*prior.values(), *current.values(), reservation]):
        raise ValueError('invalid accounting value')
    return (sum(current.values())+reservation <= CAP
        and sum(prior.values())+sum(current.values())+reservation <= 48*3600
        and prior['A']+current['A']+reservation <= base.POOLS['A']
        and all(prior[k]+current[k] <= base.POOLS[k] for k in ('B', 'C', 'D')))


def prepare(parent, campaign, protocol):
    parent, c = parent.resolve(), campaign.resolve()
    if c.exists():
        raise FileExistsError('diagnostic state already exists; no reset')
    closeout = read(parent/'CLOSEOUT.json')
    if closeout['state'] != 'closed' or not closeout['owned_scientific_services_stopped']:
        raise ValueError('parent must be closed')
    old = read(parent/'ACCEPTANCE.json')
    prior = base.charges(parent)
    # The first review recorded CPU in FINAL_VALIDITY rather than accounting/.
    # Preserve it, and explicitly allow for the two otherwise unmeasured review
    # hosts. These 240 seconds are an allowance, not a CPU measurement.
    final = read(REPO/'results/k3-redesign-20261110/FINAL_VALIDITY.json')
    prior['D'] += final['phase_d_cpu_seconds_measured_through_report']+240.
    if not within_budget(prior, dict.fromkeys(base.POOLS, 0.), CAP):
        raise ValueError('diagnostic no longer fits original total/phase allocation')
    c.mkdir(parents=True)
    for name in ('jobs', 'accounting', 'events', 'notifications', 'controller'):
        (c/name).mkdir()
    a = old | dict(id='K3-RD1-DIAG1', opened=now().isoformat(), parent_opened=old['opened'],
        diagnostic_cpu_seconds=CAP, parent_acceptance_sha256=digest(parent/'ACCEPTANCE.json'),
        authorization='Owner explicitly approved four retained-path diagnostic jobs on 2026-11-11.',
        scientific_draws=0, new_worlds=0, production_authorized=False)
    write(c/'ACCEPTANCE.json', a)
    write(c/'PRIOR_CHARGES.json', dict(phases=prior, parent=str(parent),
        parent_files={p.name:digest(p) for p in (parent/'accounting').glob('*.json')},
        extra_measured_phase_d_seconds=final['phase_d_cpu_seconds_measured_through_report'],
        review_host_allowance_seconds=240., parent_closeout_sha256=digest(parent/'CLOSEOUT.json')))
    shutil.copyfile(parent/'ENVIRONMENT.json', c/'ENVIRONMENT.json')
    shutil.copyfile(parent/'SOURCE.json', c/'SOURCE.json')
    authority = dict(id='K3-RD1-DIAG1', at=now().isoformat(), protocol_sha256=digest(protocol),
        diagnostic_cpu_seconds=CAP, cumulative_cpu_seconds=48*3600, maximum_compute_reservations=8400,
        protected_setup_analysis_report_seconds=6000, parent_closeout_sha256=digest(parent/'CLOSEOUT.json'),
        prior_charges_sha256=digest(c/'PRIOR_CHARGES.json'), scientific_draws=0, new_worlds=0,
        no_automatic_repair=True, no_gate_or_ladder_change=True, production_authorized=False)
    write(c/'DIAGNOSTIC_AUTHORIZATION.json', authority)
    for identity, count, independent, cap in JOBS:
        original = parent/'jobs'/f'cal-f05-n{count:06d}-r00'
        if read(original/'COMPLETE.json')['state'] != 'COMPLETE':
            raise ValueError('retained original must be complete')
        base.plan_job(c, identity, 'A', 'paths', cap, fixture=5, lane='old', particles=count,
            replicate=0, independent=independent, diagnostic_authorization_sha256=digest(c/'DIAGNOSTIC_AUTHORIZATION.json'),
            original_root=str(original), original_complete_sha256=digest(original/'COMPLETE.json'),
            original_result_sha256=digest(original/'RESULT.json'))
    controller_source = frozen_source(c, EXTRAS)
    write(c/'CONTROLLER_SOURCE.json', dict(root=str(controller_source),
        manifest_sha256=digest(controller_source/'SOURCE_MANIFEST.json')))
    write(c/'accounting'/'preparation.json', dict(phase='A', state='STOPPED', cpu_seconds=time.process_time()))
    return controller_source


def compare_arrays(original, candidate, independent):
    if len(original) != len(candidate):
        raise ValueError('incomplete hypothesis roster')
    errors = []
    for a, b in zip(original, candidate):
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('invalid prefix likelihood arrays')
        if not independent and not np.array_equal(x, y):
            raise ArithmeticError('original numerical replay differs')
        errors.append(float(np.max(abs(x-y))))
    return dict(passed=max(errors, default=float('inf')) <= 1e-9,
                maximum_error=max(errors, default=None), per_hypothesis_maximum_error=errors)


def finish_owned(c, job, error=None):
    # Validate the receipt before base.finish closes its native handle. A missing
    # receipt is a retained execution failure, never a second close of that handle.
    if error is None and job['child'].poll() == 0:
        try:
            receipt = read(job['root']/'WORKER_COMPLETE.json')
            if (digest(job['root']/'RESULT.json') != receipt['result_sha256']
                    or digest(job['root']/'PLAN.json') != receipt['plan_sha256']):
                error = 'worker receipt binding mismatch'
        except (OSError, ValueError, KeyError) as exc:
            error = 'missing or malformed worker receipt: '+str(exc)
    return base.finish(c, job, error)


def verify_job(root):
    plan = read(root/'PLAN.json')
    original = Path(plan['original_root'])
    if digest(original/'COMPLETE.json') != plan['original_complete_sha256'] or digest(original/'RESULT.json') != plan['original_result_sha256']:
        raise ValueError('retained original changed')
    old, new = read(original/'RESULT.json'), read(root/'RESULT.json')
    if digest(root/'data/fixture_points.json') != old['raw_sha256'] or new['raw_sha256'] != old['raw_sha256']:
        raise ValueError('retained world was not reproduced exactly')
    old_logs, new_logs = [], []
    for expected, actual in zip(old['hypotheses'], new['hypotheses']):
        if expected['file'] != actual['file']:
            raise ValueError('hypothesis ordering changed')
        left, right = original/'data'/expected['file'], root/'data'/actual['file']
        if digest(left) != expected['sha256'] or digest(right) != actual['sha256']:
            raise ValueError('hypothesis binding mismatch')
        x, y = read(left), read(right)
        if (x['point'], x['law']) != (y['point'], y['law']):
            raise ValueError('hypothesis identity changed')
        old_logs.append(x['prefix_log_likelihood'])
        new_logs.append(y['prefix_log_likelihood'])
    if len(old_logs) != 45 or len(new['hypotheses']) != 45:
        raise ValueError('incomplete full-reader roster')
    check = compare_arrays(old_logs, new_logs, plan['independent'])
    byte_replay = None
    if not plan['independent']:
        byte_replay = digest(root/'RESULT.json') == plan['original_result_sha256']
        check['passed'] &= byte_replay
    check.update(job=root.name, particles=plan['particles'], independent=plan['independent'],
        byte_replay=byte_replay, raw_world_identical=True, hypotheses=45, records=512,
        result_sha256=digest(root/'RESULT.json'), original_result_sha256=plan['original_result_sha256'],
        scientific_draws=0, production_authorized=False)
    write(root/'VERIFICATION.json', check)
    return check


def saved_diagnostics(parent):
    rows = []
    for count in (1024, 4096, 16384):
        ess, roots, zero = [], [], []
        for replicate in range(4):
            root = parent/'jobs'/f'cal-f05-n{count:06d}-r{replicate:02d}'
            result = read(root/'RESULT.json')
            for row in result['hypotheses']:
                path = root/'data'/row['file']
                if digest(path) != row['sha256']:
                    raise ValueError('retained diagnostic changed')
                diagnostic = read(path)['diagnostics']
                if diagnostic.get('exact_static'):
                    continue
                ess.append(min(diagnostic['ancestor_effective_sample_size'])/count)
                roots.append(diagnostic['surviving_initial_ancestors'][-1])
                zero.append(max(diagnostic['zero_support_parents']))
        rows.append(dict(particles=count, dynamic_hypothesis_paths=len(ess),
            minimum_normalized_ancestor_ess=min(ess), median_normalized_minimum_ess=float(np.median(ess)),
            final_initial_ancestors_minimum=min(roots), final_initial_ancestors_median=float(np.median(roots)),
            maximum_zero_support_parents=max(zero)))
    return dict(rows=rows, interpretation='Retained diagnostics are descriptive; ancestry retention alone does not prove the cause of numerical error.')


def run(c):
    runtime.below_normal()
    active, checked, limit, samples, previous = {}, {}, 4, [], None
    outcome = None
    a = read(c/'ACCEPTANCE.json');authorization = read(c/'DIAGNOSTIC_AUTHORIZATION.json')
    prior = read(c/'PRIOR_CHARGES.json')['phases']
    parent = Path(read(c/'PRIOR_CHARGES.json')['parent'])
    if digest(c/'PRIOR_CHARGES.json') != authorization['prior_charges_sha256']:
        raise ValueError('prior charges changed')
    with owner(c/'controller'):
        try:
            if (c/'QUEUE_COMPLETE.json').exists() or any((p/'STARTED.json').exists() for p in (c/'jobs').iterdir()):
                raise ValueError('diagnostic cannot be silently restarted')
            while outcome is None:
                current = now()
                write(c/'accounting'/'controller.json', dict(phase='D', state='RUNNING', cpu_seconds=time.process_time()), immutable=False)
                fraction, previous = base.host_sample(previous)
                if fraction is not None:
                    samples.append((time.monotonic(), fraction));samples=samples[-180:]
                    limit=contract.load_limit(samples, limit)
                for identity, job in list(active.items()):
                    used=runtime.child_cpu(job['handle']);error=None
                    if used>job['plan']['cpu_cap_seconds'] or time.monotonic()-job['opened']>min(64800,2*job['plan']['cpu_cap_seconds']+300):
                        error='diagnostic job resource cap'
                    if current>=datetime.fromisoformat(a['drain']) or (c/'STOP').exists():
                        error='original drain or explicit stop'
                    if runtime.memory()['available']<2*1024**3:
                        error='host memory protection'
                    if error or job['child'].poll() is not None:
                        state=finish_owned(c,job,error);del active[identity]
                        if state!='COMPLETE':
                            outcome='FAILED_EXECUTION: '+identity
                        else:
                            check=verify_job(job['root']);checked[identity]=check
                            if not check['passed']:
                                outcome='NUMERICAL_RECONSTRUCTION_DISCREPANCY: '+identity
                    else:
                        base.account(c,job['root'],'RUNNING',used,job['plan']['cpu_cap_seconds'])
                if outcome:
                    break
                if len(checked)==4:
                    outcome='DIAGNOSTIC_PASSED_CALIBRATION_REMAINS_FAILED';break
                totals=base.charges(c)
                for identity,_,_,_ in JOBS:
                    root=c/'jobs'/identity
                    if (root/'STARTED.json').exists():continue
                    if len(active)>=limit:break
                    plan=read(root/'PLAN.json')
                    if plan['diagnostic_authorization_sha256']!=digest(c/'DIAGNOSTIC_AUTHORIZATION.json'):
                        raise ValueError('diagnostic authority binding changed')
                    if (current>=datetime.fromisoformat(a['new_starts_close'])
                            or current+timedelta(seconds=plan['cpu_cap_seconds']*2+300)>=datetime.fromisoformat(a['drain'])):
                        outcome='RESOURCE_DEFERRED: original last-start boundary';break
                    if not within_budget(prior,totals,plan['cpu_cap_seconds']):
                        outcome='RESOURCE_DEFERRED: cumulative or diagnostic cap';break
                    active[identity]=base.start(c,root);totals['A']+=plan['cpu_cap_seconds']
                write(c/'QUEUE_STATUS.json',dict(state='RUNNING',gear=2,pid=os.getpid(),heartbeat=current.isoformat(),
                    active=list(active),completed=list(checked),pending=4-len(checked)-len(active),worker_limit=limit,
                    charges=base.charges(c),prior_charges=prior,scientific_draws=0,production_authorized=False),immutable=False)
                time.sleep(5)
        except BaseException as exc:
            outcome='DIAGNOSTIC_FAILED: '+str(exc)
            write(c/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
        finally:
            for job in active.values():
                finish_owned(c,job,'finite diagnostic stop: '+str(outcome))
            try:
                write(c/'RETAINED_DIAGNOSTICS.json',saved_diagnostics(parent))
            except BaseException as exc:
                write(c/'RETAINED_DIAGNOSTICS_FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc()))
                outcome=str(outcome)+'; RETAINED_DIAGNOSTICS_FAILED: '+str(exc)
            write(c/'accounting'/'controller.json',dict(phase='D',state='STOPPED',cpu_seconds=time.process_time()),immutable=False)
            write(c/'QUEUE_COMPLETE.json',dict(at=now().isoformat(),outcome=outcome,checks=checked,
                charges=base.charges(c),prior_charges=prior,scientific_draws=0,new_worlds=0,
                production_authorized=False,original_calibration_remains_failed=True))
            write(c/'QUEUE_STATUS.json',dict(state='DRAINED',gear=2,outcome=outcome,active=[],
                completed=list(checked),pending=0,heartbeat=now().isoformat(),charges=base.charges(c),
                scientific_draws=0,production_authorized=False),immutable=False)
            event(c,'diagnostic-finite-end','queue_depleted',dict(outcome=outcome,receipt_sha256=digest(c/'QUEUE_COMPLETE.json')))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--prepare-from',type=Path);p.add_argument('--protocol',type=Path)
    args=p.parse_args()
    if args.prepare_from:
        if not args.protocol:p.error('--protocol is required with --prepare-from')
        print(prepare(args.prepare_from,args.campaign,args.protocol),flush=True)
    else:
        run(args.campaign.resolve())
