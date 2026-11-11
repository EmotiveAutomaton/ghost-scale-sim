"""K3-RD1 finite validation, immutable job plans, native CPU and Gear 2 control.

No production mode exists. A failed frozen gate ends this allocation's queue.
Calibration chooses one of the approved adjacent population pairs, prospectively
freezes all 30 validation fixtures, then runs them one fixture at a time (up to
four independent replicas concurrently). Native handles enforce hard resource
limits even during a compiled numerical kernel.
"""
import argparse
import ctypes
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v27 import contract, registry
from ghostscale.validation.soundingline.v27.common import read, write, digest
from ghostscale.validation.soundingline.k3_redesign import experiment
from runners.run_v27 import frozen_source, env_for
from runners.watch_v23_events import owner, event

REPO = Path(__file__).resolve().parents[1]
POOLS = dict(A=12*3600, B=24*3600, C=8*3600, D=4*3600)
EXTRAS = ('runners/run_k3_redesign.py', 'runners/watch_k3_redesign.py', 'runners/review_shutdown.py')


def now():
    return datetime.now(timezone.utc)


def environment():
    import numba, llvmlite
    value = runtime.environment()
    value['numba_version'] = numba.__version__
    value['llvmlite_version'] = llvmlite.__version__
    files = {}
    for module in (numba, llvmlite):
        package = Path(module.__file__).parent
        for path in sorted(package.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.pyd', '.dll'):
                files[path.relative_to(package.parent).as_posix()] = digest(path)
    value['compiler_library_files'] = files
    return value


def host_sample(previous):
    idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError('host CPU measurement failed')
    current = (idle.value, kernel.value+user.value)
    fraction = None if previous is None or current[1] <= previous[1] else 1-(current[0]-previous[0])/(current[1]-previous[1])
    return fraction, current


def authority(proposal_sha256, opened):
    return dict(id='K3-RD1', opened=opened, proposal_sha256=proposal_sha256,
        authorization='Owner approved the separate redesign and Gear 2 on 2026-11-10.',
        scientific_draws=0, production_authorized=False, total_cpu_seconds=48*3600,
        pools_seconds=POOLS, new_starts_close=contract.NEW_STARTS_CLOSE, drain=contract.DRAIN,
        workers=4, overload_workers=2, numerical_threads=1, priority='below-normal',
        gpu=False, per_job_cpu_ceiling=12*3600, per_job_wall_ceiling=18*3600,
        health_hours=6, automatic_healings=0)


def prepare(campaign, proposal):
    c = campaign.resolve()
    if (c/'ACCEPTANCE.json').exists():
        raise FileExistsError('existing acceptance cannot be reset')
    c.mkdir(parents=True, exist_ok=True)
    for name in ('jobs', 'accounting', 'events', 'notifications', 'controller'):
        (c/name).mkdir()
    write(c/'ACCEPTANCE.json', authority(digest(proposal), now().isoformat()))
    write(c/'ENVIRONMENT.json', environment())
    source = frozen_source(c, EXTRAS)
    write(c/'SOURCE.json', dict(root=str(source), manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
                              archive_sha256=digest(source/'SOURCE.zip')))
    write(c/'CALIBRATION_PROTOCOL.json', dict(phase='A', scientific_draws=0,
        fixture_order=[5, 0, 1, 8, 11, 14], counts=list(experiment.COUNTS), replicates=4,
        limits=experiment.LIMITS, selection='Lowest adjacent pair passing every all-pair calibration check; otherwise stop.',
        source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
        known_answer_controls_required=True, no_production=True))
    plan_job(c, 'controls', 'A', 'controls', 900.)
    return source


def plan_job(c, identity, phase, kind, cap, **fields):
    if not 0 < cap <= 43200 or phase not in POOLS or Path(identity).name != identity:
        raise ValueError('invalid job identity/resource cap')
    a = read(c/'ACCEPTANCE.json')
    s = read(c/'SOURCE.json')
    p = dict(id=identity, phase=phase, kind=kind, cpu_cap_seconds=cap, wall_cap_seconds=64800,
        source=s, acceptance_sha256=digest(c/'ACCEPTANCE.json'),
        environment_sha256=digest(c/'ENVIRONMENT.json'), scientific_draws=0,
        interpreter=str(Path(sys.executable).resolve()), **fields)
    if a['production_authorized'] or a['pools_seconds'] != POOLS:
        raise ValueError('unexpected authority')
    root = c/'jobs'/identity
    root.mkdir(exist_ok=True)
    write(root/'PLAN.json', p)
    return root


def verify(c, plan):
    if digest(c/'ACCEPTANCE.json') != plan['acceptance_sha256']:
        raise ValueError('acceptance changed')
    if digest(c/'ENVIRONMENT.json') != plan['environment_sha256']:
        raise ValueError('environment binding changed')
    if environment() != read(c/'ENVIRONMENT.json'):
        raise ValueError('scientific environment changed')
    source = Path(plan['source']['root'])
    if digest(source/'SOURCE_MANIFEST.json') != plan['source']['manifest_sha256'] or digest(source/'SOURCE.zip') != plan['source']['archive_sha256']:
        raise ValueError('source capsule changed')
    for rel, expected in read(source/'SOURCE_MANIFEST.json').items():
        if digest(source/rel) != expected:
            raise ValueError('source byte changed: '+rel)
    if plan['phase'] == 'B':
        if digest(c/'VALIDATION_FREEZE.json') != plan['validation_freeze_sha256']:
            raise ValueError('validation freeze changed')


def worker(c, root):
    runtime.below_normal()
    runtime.child_handshake(root, 'worker')
    plan = read(root/'PLAN.json')
    verify(c, plan)
    last = [0.]
    def pulse(**progress):
        used = time.process_time()
        if used > plan['cpu_cap_seconds'] or now() >= datetime.fromisoformat(contract.DRAIN) or (c/'STOP').exists():
            raise TimeoutError('job CPU/drain/stop boundary')
        if time.monotonic()-last[0] > 5:
            write(root/'PROGRESS.json', dict(pid=os.getpid(), heartbeat=now().isoformat(),
                cpu_seconds=used, **progress), immutable=False)
            last[0] = time.monotonic()
    try:
        pulse(phase='starting')
        if plan['kind'] == 'controls':
            result = experiment.controls(pulse)
            write(root/'RESULT.json', result)
            if not result['passed']:
                raise ArithmeticError('known-answer control failure')
        elif plan['kind'] == 'paths':
            result = experiment.paths(plan['fixture'], plan['lane'], plan['particles'], plan['replicate'],
                                      root/'data', pulse, plan.get('independent', False))
            write(root/'RESULT.json', result)
        elif plan['kind'] == 'cost':
            from ghostscale.validation.soundingline.k3_redesign.cost import run
            write(root/'RESULT.json', run(plan, root, pulse))
        else:
            raise ValueError('uncommissioned worker kind')
        write(root/'WORKER_COMPLETE.json', dict(result_sha256=digest(root/'RESULT.json'),
            plan_sha256=digest(root/'PLAN.json'), scientific_draws=0))
    except BaseException as exc:
        write(root/'FAILURE.json', dict(error_type=type(exc).__name__, error=str(exc),
                                      traceback=traceback.format_exc()))
        raise


def charges(c):
    result = dict.fromkeys(POOLS, 0.)
    for path in (c/'accounting').glob('*.json'):
        row = read(path)
        if row['phase'] not in result:
            raise ValueError('unrecognized accounting phase')
        result[row['phase']] += max(float(row.get('cpu_seconds', 0.)),
                                   float(row.get('reserved_cpu_seconds', 0.)))
    return result


def eligible(plan, totals, current):
    cap = plan['cpu_cap_seconds']
    return (current < datetime.fromisoformat(contract.NEW_STARTS_CLOSE)
        and current+timedelta(seconds=cap*2+300) < datetime.fromisoformat(contract.DRAIN)
        and 0 < cap <= 43200 and totals[plan['phase']]+cap <= POOLS[plan['phase']])


def account(c, root, state, used, reserve=0., **fields):
    plan = read(root/'PLAN.json')
    write(c/'accounting'/(root.name+'.json'), dict(phase=plan['phase'], state=state,
        cpu_seconds=used, reserved_cpu_seconds=reserve, **fields), immutable=False)


def start(c, root):
    plan = read(root/'PLAN.json')
    if (root/'STARTED.json').exists():
        raise ValueError('started job may not be silently repeated')
    write(root/'STARTED.json', dict(at=now().isoformat(), plan_sha256=digest(root/'PLAN.json')))
    account(c, root, 'RUNNING', 0., plan['cpu_cap_seconds'])
    env = env_for(plan['source']['root'])
    env['NUMBA_NUM_THREADS'] = '1'
    env['NUMBA_CACHE_DIR'] = str(c/'numba-cache')
    log = (root/'worker.log').open('xb')
    child = subprocess.Popen([plan['interpreter'], '-B', '-m', 'runners.run_k3_redesign',
        '--campaign', str(c), '--worker', str(root)], cwd=plan['source']['root'], env=env,
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        handle = runtime.attach_child(child, root, 'worker')
    except BaseException:
        # Only the just-created, owned process is affected. Preserve evidence.
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
        log.close()
        account(c, root, 'FAILED', plan['cpu_cap_seconds'], error='launch handshake failed; full cap retained as uncertain CPU')
        write(root/'COMPLETE.json', dict(state='FAILED', error='launch handshake failed', scientific_draws=0))
        raise
    return dict(root=root, child=child, handle=handle, log=log, opened=time.monotonic(), plan=plan)


def finish(c, job, error=None):
    child, handle, root = job['child'], job['handle'], job['root']
    if child.poll() is None and error:
        ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(handle), 1)
        ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 10000)
        child.wait(timeout=10)
    used = runtime.child_cpu(handle)
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
    job['log'].close()
    if error is None and child.returncode != 0:
        error = f'worker exit {child.returncode}'
    if error is None:
        receipt = read(root/'WORKER_COMPLETE.json')
        if digest(root/'RESULT.json') != receipt['result_sha256'] or digest(root/'PLAN.json') != receipt['plan_sha256']:
            error = 'worker receipt binding mismatch'
    state = 'FAILED' if error else 'COMPLETE'
    account(c, root, state, used, wall_seconds=time.monotonic()-job['opened'], error=error)
    write(root/'COMPLETE.json', dict(state=state, at=now().isoformat(), cpu_seconds=used,
        result_sha256=digest(root/'RESULT.json') if (root/'RESULT.json').exists() else None,
        plan_sha256=digest(root/'PLAN.json'), error=error, scientific_draws=0))
    if error:
        event(c, root.name+'-failed', 'batch_failed', dict(job=root.name, error=error))
    return state


def calibration_jobs(c):
    jobs = []
    for index in read(c/'CALIBRATION_PROTOCOL.json')['fixture_order']:
        for count in experiment.COUNTS:
            for rep in range(4):
                identity = f'cal-f{index:02d}-n{count:06d}-r{rep:02d}'
                root = c/'jobs'/identity
                if not (root/'PLAN.json').exists():
                    # Conservative pilot reservation; caps remain hard. No
                    # extrapolation silently grants more than the phase share.
                    cap = 180.+count*.15
                    plan_job(c, identity, 'A', 'paths', cap, fixture=index, lane='old', particles=count, replicate=rep)
                jobs.append(root)
    for label in ('replay', 'independent'):
        identity = 'cal-f05-n001024-r00-'+label
        root = c/'jobs'/identity
        if not (root/'PLAN.json').exists():
            plan_job(c, identity, 'A', 'paths', 600., fixture=5, lane='old', particles=1024,
                     replicate=0, independent=label == 'independent')
        jobs.append(root)
    return jobs


def select_resolution(c):
    import numpy as np
    original = read(c/'jobs'/'cal-f05-n001024-r00'/'RESULT.json')
    replay = read(c/'jobs'/'cal-f05-n001024-r00-replay'/'RESULT.json')
    independent = read(c/'jobs'/'cal-f05-n001024-r00-independent'/'RESULT.json')
    if original != replay:
        raise ArithmeticError('full calibration byte replay failed')
    errors = []
    for h in range(45):
        relative = Path('data')/f'hypothesis-{h:03d}.json'
        x = read(c/'jobs'/'cal-f05-n001024-r00'/relative)['prefix_log_likelihood']
        y = read(c/'jobs'/'cal-f05-n001024-r00-independent'/relative)['prefix_log_likelihood']
        errors.append(float(np.max(np.abs(np.array(x)-y))))
    write(c/'FULL_PATH_RECONSTRUCTION.json', dict(passed=max(errors) <= 1e-9,
        byte_replay=True, per_hypothesis_maximum_error=errors, maximum_error=max(errors),
        records=512, hypotheses=45, scientific_draws=0))
    if max(errors) > 1e-9:
        raise ArithmeticError('full independent reference path failed')
    assessments = []
    for pair in ((1024, 4096), (4096, 16384)):
        fixtures = []
        for index in experiment.CALIBRATION:
            rows = [read(c/'jobs'/f'cal-f{index:02d}-n{n:06d}-r{r:02d}'/'RESULT.json') for n in pair for r in range(4)]
            fixtures.append(dict(index=index, **experiment.all_pairs(rows)))
        assessments.append(dict(counts=list(pair), passed=all(x['passed'] for x in fixtures), fixtures=fixtures))
    write(c/'CALIBRATION_ASSESSMENT.json', dict(assessments=assessments, scientific_draws=0))
    passing = [row for row in assessments if row['passed']]
    if not passing:
        return None
    return passing[0]['counts']


def freeze_validation(c, counts):
    roster = [dict(index=i, lane=lane) for lane in ('old', 'fresh') for i in range(15)]
    freeze = dict(counts=counts, replicates=8, roster=roster, limits=experiment.LIMITS,
        source=read(c/'SOURCE.json'), calibration_sha256=digest(c/'CALIBRATION_ASSESSMENT.json'),
        at=now().isoformat(), fresh_seed_namespace='excluded-k3-rd1-validation-20261110',
        scientific_draws=0, stop='First failed frozen gate or resource boundary; no automatic repair.')
    write(c/'VALIDATION_FREEZE.json', freeze)
    calibration_costs = [read(c/'accounting'/(p.name+'.json'))['cpu_seconds']
                         for p in (c/'jobs').glob('cal-*') if read(p/'PLAN.json')['particles'] == counts[-1]]
    cap = min(43200., max(calibration_costs)*2+60.)
    for row in roster:
        for count in counts:
            for rep in range(8):
                identity = f"val-{row['lane']}-f{row['index']:02d}-n{count:06d}-r{rep:02d}"
                plan_job(c, identity, 'B', 'paths', cap, fixture=row['index'], lane=row['lane'],
                    particles=count, replicate=rep, validation_freeze_sha256=digest(c/'VALIDATION_FREEZE.json'))
    return roster


def coverage(c, outcome):
    passed = (c/'VALIDATION_RESULT.json').exists() and read(c/'VALIDATION_RESULT.json')['passed']
    rows = []
    for card in registry.cards():
        if card['family'] != 'K3':
            continue
        state = 'validated' if card['id'] == 'K3' and passed else 'unvalidated'
        reason = 'Numerical primary only; production is not admitted.' if state == 'validated' else outcome
        if card.get('changes', {}).get('stock_noise'):
            state, reason = 'unsupported', 'No noisy-stock observation likelihood is implemented or admitted by this primary validation.'
        rows.append(dict(card=card['id'], state=state, reason=reason, production_authorized=False))
    write(c/'COVERAGE.json', dict(cards=rows, scientific_draws=0))


def controller(c):
    runtime.below_normal()
    active, samples, limit = {}, [], 4
    cpu_previous = None
    outcome, phase, stage = None, 'A', 'controls'
    verified_fixtures = []
    with owner(c/'controller'):
        try:
            if (c/'QUEUE_COMPLETE.json').exists():
                raise ValueError('finite queue already ended')
            if any((p/'STARTED.json').exists() and not (p/'COMPLETE.json').exists() for p in (c/'jobs').iterdir()):
                raise ValueError('interrupted owned job requires explicit recovery receipt, never automatic replay')
            while outcome is None:
                current = now()
                # Pool accounting includes controller and monitor CPU.
                write(c/'accounting'/'controller.json', dict(phase='A', state='RUNNING',
                    cpu_seconds=time.process_time(), reserved_cpu_seconds=0.), immutable=False)
                sample, cpu_previous = host_sample(cpu_previous)
                if sample is not None:
                    samples.append((time.monotonic(), sample))
                    samples = samples[-180:]
                    limit = contract.load_limit(samples, limit)
                for identity, job in list(active.items()):
                    used = runtime.child_cpu(job['handle'])
                    error = None
                    if used > job['plan']['cpu_cap_seconds'] or time.monotonic()-job['opened'] > 64800:
                        error = 'hard job CPU/wall limit'
                    if current >= datetime.fromisoformat(contract.DRAIN) or (c/'STOP').exists():
                        error = 'immutable drain or stop requested'
                    if runtime.memory()['available'] < 2*1024**3:
                        error = 'host memory protection'
                    if error or job['child'].poll() is not None:
                        state = finish(c, job, error)
                        del active[identity]
                        if state == 'FAILED':
                            outcome = 'FAILED: '+identity
                    else:
                        account(c, job['root'], 'RUNNING', used, job['plan']['cpu_cap_seconds'])
                if outcome:
                    break
                if stage == 'controls':
                    wanted = [c/'jobs'/'controls']
                elif stage == 'calibration':
                    wanted = calibration_jobs(c)
                elif stage == 'validation':
                    freeze = read(c/'VALIDATION_FREEZE.json')
                    row = freeze['roster'][len(verified_fixtures)]
                    wanted = [c/'jobs'/f"val-{row['lane']}-f{row['index']:02d}-n{n:06d}-r{r:02d}"
                              for n in freeze['counts'] for r in range(8)]
                else:
                    wanted = [c/'jobs'/identity for identity in read(c/'COST_FREEZE.json')['jobs']]
                if all((p/'COMPLETE.json').exists() for p in wanted):
                    if any(read(p/'COMPLETE.json')['state'] != 'COMPLETE' for p in wanted):
                        outcome = 'FAILED: retained terminal job'
                    elif stage == 'controls':
                        stage = 'calibration'
                        event(c, 'controls-passed', 'phase_boundary', dict(result_sha256=digest(wanted[0]/'RESULT.json')))
                    elif stage == 'calibration':
                        counts = select_resolution(c)
                        if counts is None:
                            outcome = 'CALIBRATION_FAILED: neither approved adjacent population pair passed'
                        else:
                            freeze_validation(c, counts)
                            stage, phase = 'validation', 'B'
                            event(c, 'validation-frozen', 'phase_boundary', dict(freeze_sha256=digest(c/'VALIDATION_FREEZE.json')))
                    elif stage == 'validation':
                        assessment = experiment.all_pairs([read(p/'RESULT.json') for p in wanted])
                        record = dict(**row, **assessment)
                        write(c/f"validation-{row['lane']}-{row['index']:02d}.json", record)
                        verified_fixtures.append(record)
                        if not assessment['passed']:
                            outcome = 'VALIDATION_FAILED: frozen all-pair numerical gate'
                        if outcome or len(verified_fixtures) == 30:
                            write(c/'VALIDATION_RESULT.json', dict(passed=not outcome, fixtures=verified_fixtures, scientific_draws=0))
                        if not outcome and len(verified_fixtures) == 30:
                            from ghostscale.validation.soundingline.k3_redesign.cost import prepare as cost_prepare
                            cost_prepare(c, plan_job)
                            stage, phase = 'cost', 'C'
                            event(c, 'validation-passed', 'phase_boundary', dict(result_sha256=digest(c/'VALIDATION_RESULT.json')))
                    else:
                        from ghostscale.validation.soundingline.k3_redesign.cost import verify_cost
                        write(c/'COST_RESULT.json', verify_cost(c))
                        outcome = 'VALIDATION_AND_COST_COMPLETE'
                    continue
                remaining = [p for p in wanted if not (p/'STARTED.json').exists()]
                totals = charges(c)
                for root in remaining:
                    if len(active) >= limit:
                        break
                    plan = read(root/'PLAN.json')
                    if eligible(plan, totals, current):
                        active[root.name] = start(c, root)
                        totals[plan['phase']] += plan['cpu_cap_seconds']
                pending_count = len([p for p in wanted if not (p/'COMPLETE.json').exists()])-len(active)
                write(c/'QUEUE_STATUS.json', dict(state='RUNNING', phase=phase, stage=stage, pid=os.getpid(),
                    heartbeat=current.isoformat(), active=list(active), pending=pending_count,
                    worker_limit=limit, gear=2, charges=charges(c), scientific_draws=0), immutable=False)
                if not active and remaining:
                    outcome = 'RESOURCE_DEFERRED: phase reservation or immutable start/drain boundary'
                time.sleep(5)
        except BaseException as exc:
            outcome = 'CONTROLLER_FAILED: '+str(exc)
            write(c/'CONTROLLER_FAILURE.json', dict(error=str(exc), traceback=traceback.format_exc()))
        finally:
            for job in active.values():
                finish(c, job, 'finite queue stopped: '+str(outcome))
            write(c/'accounting'/'controller.json', dict(phase='A', state='STOPPED', cpu_seconds=time.process_time(), reserved_cpu_seconds=0.), immutable=False)
            if outcome:
                coverage(c, outcome)
                write(c/'QUEUE_COMPLETE.json', dict(at=now().isoformat(), outcome=outcome,
                    charges=charges(c), scientific_draws=0, review_required=True, production_authorized=False))
                write(c/'QUEUE_STATUS.json', dict(state='DRAINED', phase=phase, stage=stage,
                    outcome=outcome, active=[], pending=0, heartbeat=now().isoformat(),
                    charges=charges(c), gear=2, scientific_draws=0), immutable=False)
                event(c, 'queue-finite-end', 'queue_depleted', dict(outcome=outcome, receipt_sha256=digest(c/'QUEUE_COMPLETE.json')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--prepare', type=Path)
    p.add_argument('--worker', type=Path)
    a = p.parse_args()
    if a.prepare:
        print(prepare(a.campaign, a.prepare), flush=True)
    elif a.worker:
        worker(a.campaign.resolve(), a.worker.resolve())
    else:
        controller(a.campaign.resolve())


if __name__ == '__main__':
    main()
