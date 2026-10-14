"""One finite V22 lane, below the shared Gear-2 ceiling; never restart partial jobs."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from ghostscale.validation.soundingline.v18_3.native import ProcessClock
from ghostscale.validation.soundingline.v18_4.priority import below_normal
from ghostscale.validation.soundingline.v20.runtime import available_memory
from ghostscale.validation.soundingline.v22.runtime import spent
from runners.watch_v18_4_events import alive


def priority_clear(path, current=None):
    current = current or datetime.now(timezone.utc)
    if (path/'CLOSEOUT.json').exists():
        return True
    status = read(path/'QUEUE-STATUS.json')
    fresh = (current-datetime.fromisoformat(status['heartbeat'])).total_seconds() < 90
    return fresh and alive(status['pid']) and not status['active'] and status['pending'] == 0 and (path/'HOLD').exists()


def event(campaign, identity, kind, **extra):
    path = campaign/'events'/(identity+'.json')
    if not path.exists():
        write(path, dict(id=identity, kind=kind, at=now(), **extra))


def supervise(campaign, python):
    below_normal()
    with local_owner(campaign/'supervisor'):
        acceptance = read(campaign/'ACCEPTANCE.json')
        digest = file_digest(campaign/'ACCEPTANCE.json')
        cpu_start = time.process_time()
        service_base = read(campaign/'SERVICE_COST.json')['cpu_seconds'] if (campaign/'SERVICE_COST.json').exists() else 0.
        for job in read(campaign/'QUEUE.json')['jobs']:
            root = Path(job['root'])
            if (root/'STATUS.json').exists() and not (root/'COMPLETE.json').exists() and not (root/'DISPOSITION.json').exists():
                raise ValueError('interrupted identity needs explicit recovery disposition')
        active = None
        while True:
            if file_digest(campaign/'ACCEPTANCE.json') != digest:
                raise ValueError('acceptance changed')
            current = datetime.now(timezone.utc)
            if active:
                child, job, log, started, clock = active
                root = Path(job['root'])
                status = read(root/'STATUS.json') if (root/'STATUS.json').exists() else None
                if status and clock is None and status['parent_pid'] in (child.pid, os.getpid()):
                    clock = ProcessClock(status['pid'])
                    active = child, job, log, started, clock
                if child.poll() is None:
                    exceeded = clock and clock.seconds() > job['cpu_cap_seconds']+5
                    expired = time.monotonic()-started > job['wall_cap_seconds']
                    if exceeded or expired:
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                    # An unexpected V21 start is an ownership intervention, never
                    # permission to kill or alter V21. Stop this bounded own worker.
                    v21 = read(Path(acceptance['priority_campaign'])/'QUEUE-STATUS.json')
                    if v21['active'] and len(v21['active']) >= 3:
                        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                else:
                    log.close()
                    cost_path = campaign/'attempts'/(root.name+'.json')
                    cost = read(cost_path) if cost_path.exists() else dict(job=root.name, cpu_seconds=0)
                    if clock:
                        cost['native_cpu_seconds'] = clock.seconds()
                        clock.close()
                    else:
                        cost['uncertainty_cpu_seconds'] = min(job['cpu_cap_seconds'], time.monotonic()-started)
                    done = child.returncode == 0 and (root/'COMPLETE.json').exists()
                    cost['state'] = 'complete' if done else 'failed'
                    write(cost_path, cost, immutable=False)
                    if not done:
                        write(root/'DISPOSITION.json', dict(state='INSTRUMENT FAILED', exit_code=child.returncode,
                                                           reason='execution/verification/resource failure; retain partial output'))
                    event(campaign, job['id'], 'batch_complete' if done else 'batch_failed', job=job['id'])
                    active = None
            jobs = read(campaign/'QUEUE.json')['jobs']
            resolved = [j for j in jobs if (Path(j['root'])/'COMPLETE.json').exists() or (Path(j['root'])/'DISPOSITION.json').exists()]
            waiting = [j for j in jobs if j not in resolved and (active is None or j['id'] != active[1]['id'])]
            stopping = (campaign/'STOP').exists() or current >= datetime.fromisoformat(acceptance['deadline'])
            if not active and not waiting:
                event(campaign, 'finite-queue-resolved', 'final_review_due')
                state = 'science_complete; awaiting final review'
            elif stopping and not active:
                for job in waiting:
                    write(Path(job['root'])/'DISPOSITION.json', dict(state='UNEXECUTED', reason='finite allocation stopped/expired'))
                event(campaign, 'allocation-ended', 'final_review_due')
                state = 'science_stopped; awaiting final review'
            else:
                state = 'running' if active else 'waiting-for-priority-or-review'
            service = service_base+time.process_time()-cpu_start
            write(campaign/'SERVICE_COST.json', dict(cpu_seconds=service, pid=os.getpid(), at=now()), immutable=False)
            write(campaign/'QUEUE-STATUS.json', dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(),
                  state=state, gear=2, workers=1, active=[] if active is None else [dict(id=active[1]['id'], child_pid=active[0].pid)],
                  completed=sum((Path(j['root'])/'COMPLETE.json').exists() for j in jobs),
                  disposed=sum((Path(j['root'])/'DISPOSITION.json').exists() for j in jobs), pending=len(waiting)), immutable=False)
            if state.startswith('science_'):
                return
            if not active and not stopping and not (campaign/'HOLD').exists() and priority_clear(Path(acceptance['priority_campaign'])):
                # Do not overlap the V21 freeze/confirmation/report interval.
                before = datetime.fromisoformat(acceptance['priority_blackout_start'])
                after = datetime.fromisoformat(acceptance['priority_blackout_end'])
                job = waiting[0] if waiting else None
                if job and (current.timestamp()+job['wall_cap_seconds'] < before.timestamp() or current >= after):
                    if spent(campaign)+max(service, acceptance['setup_service_reserve_seconds'])+job['cpu_cap_seconds'] <= acceptance['execution_cpu_ceiling_seconds']:
                        if available_memory() >= acceptance['free_memory_bytes']+acceptance['worker_memory_bytes']:
                            root = Path(job['root'])
                            if file_digest(root/'PLAN.json') != job['plan_sha256']:
                                raise ValueError('plan differs')
                            log = (root/'worker.log').open('ab', buffering=0)
                            child = subprocess.Popen([str(python), '-B', '-m', 'runners.run_v22', '--root', str(root), '--campaign', str(campaign)],
                                      cwd=job['source'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)|getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
                            active = child, job, log, time.monotonic(), None
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    try:
        supervise(args.campaign, args.python)
    except BaseException as exc:
        event(args.campaign, 'supervisor-failed-'+str(os.getpid()), 'supervisor_failed', error=str(exc))
        raise
