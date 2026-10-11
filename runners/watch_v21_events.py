"""V21 event delivery with bounded, hash-bound verification dependencies.

Deferral never acknowledges an event. Failure, deadline and fixed health events
remain immediately actionable; a dependency failure releases the original event.
"""
import argparse
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from runners.watch_v18_4_events import alive, pending, checkpoint_events


def actionable(campaign, state, events, current=None):
    current = current or datetime.now(timezone.utc)
    path = state / 'DEFERRED.json'
    deferred = read(path).get('events', {}) if path.exists() else {}
    ready, waiting = [], []
    for event in events:
        rule = deferred.get(event['id'])
        if not rule or event['kind'] != 'batch_complete':
            ready.append(event)
            continue
        if rule['event_sha256'] != event['event_sha256']:
            raise ValueError('deferred event hash differs')
        roots = [campaign / 'packets' / identity for identity in rule['requires']]
        if not roots:
            raise ValueError('deferral needs a concrete dependency')
        failed = any((root / 'DISPOSITION.json').exists() for root in roots)
        done = all((root / 'COMPLETE.json').exists() for root in roots)
        expired = current >= datetime.fromisoformat(rule['until'])
        (ready if failed or done or expired else waiting).append(event)
    return ready, waiting


def review_environment(config):
    """Preserve the configured user profile through detached Windows launches."""
    environment = os.environ.copy()
    if config.get('review_user_profile'):
        profile = Path(config['review_user_profile'])
        if not profile.is_absolute() or not (profile / '.codex').is_dir():
            raise ValueError('review user profile is not an existing Codex profile')
        environment['USERPROFILE'] = str(profile)
        environment['HOMEDRIVE'] = profile.drive
        environment['HOMEPATH'] = str(profile)[len(profile.drive):]
    return environment


def delivery_events(campaign, config, events):
    """Collect routine completions until a health check or real intervention."""
    if not config.get('batch_completions_until_health'):
        return events
    urgent = any(e['kind'] != 'batch_complete' or e['id'] in config.get('urgent_event_ids', [])
                 for e in events)
    if urgent or (campaign / 'HOLD').exists():
        return events
    return []


def closed_receipt(campaign):
    """Only a recorded final report and native science closure retire the watcher."""
    path = campaign / 'CLOSEOUT.json'
    if not path.exists():
        return False
    receipt = read(path)
    if receipt.get('state') != 'closed' or not receipt.get('owned_scientific_services_stopped'):
        return False
    report = Path(receipt['final_report'])
    if not report.is_file() or file_digest(report) != receipt['final_report_sha256']:
        raise ValueError('closed campaign final report identity differs')
    status_path = campaign / 'QUEUE-STATUS.json'
    if status_path.exists() and alive(read(status_path)['pid']):
        return False
    return True


def supervise(campaign, state, config_path, once=False):
    config = read(config_path)
    state.mkdir(parents=True, exist_ok=True)
    path = state / 'DELIVERY.json'
    delivery = read(path) if path.exists() else dict(attempts=[])
    cpu_start = time.process_time()
    with local_owner(state):
        if delivery['attempts'] and 'exit_code' not in delivery['attempts'][-1]:
            last = delivery['attempts'][-1]
            if last.get('pid') and alive(last['pid']):
                raise RuntimeError('preserve live review owner')
            last.update(exit_code='interrupted_notifier', finished_at=now())
            write(path, delivery, immutable=False)
        while True:
            if closed_receipt(campaign):
                (state / 'ARMED').unlink(missing_ok=True)
                write(state / 'STATUS.json', dict(pid=os.getpid(), parent_pid=os.getppid(),
                    heartbeat=now(), state='closed; no further wakes', armed=False,
                    active_review_pid=None), immutable=False)
                return
            status_path = campaign / 'QUEUE-STATUS.json'
            if status_path.exists():
                status = read(status_path)
                if status['state'] in ('running', 'awaiting_review_refill') and not alive(status['pid']):
                    identity = 'supervisor-disappeared-' + str(status['pid'])
                    target = campaign / 'events' / (identity + '.json')
                    if not target.exists():
                        write(target, dict(id=identity, kind='supervisor_disappeared', at=now(), status=status))
            checkpoint_events(campaign, read(campaign / 'ACCEPTANCE.json'), config)
            all_events = pending(campaign, state)
            events, deferred = actionable(campaign, state, all_events)
            events = delivery_events(campaign, config, events)
            snapshot = dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(),
                            pending=len(all_events), actionable=len(events), deferred=len(deferred),
                            armed=(state / 'ARMED').exists(), active_review_pid=None)
            write(state / 'STATUS.json', snapshot, immutable=False)
            write(state / 'SERVICE_COST.json', dict(at=now(), cpu_seconds=time.process_time()-cpu_start), immutable=False)
            if (state / 'STOP').exists():
                return
            retry_at = delivery.get('retry_not_before')
            retry_ready = not retry_at or datetime.now(timezone.utc) >= datetime.fromisoformat(retry_at)
            if events and retry_ready and (state / 'ARMED').exists():
                argv = config['command']
                if any(x in argv for x in ('resume', '--last', '--dangerously-bypass-approvals-and-sandbox', '--ignore-rules', '--ignore-user-config')):
                    raise ValueError('unsafe or competing review command')
                if file_digest(argv[0]) != config['executable_sha256']:
                    raise ValueError('review binary changed')
                if file_digest(config['prompt']) != config['prompt_sha256']:
                    raise ValueError('review contract changed')
                text = Path(config['prompt']).read_text(encoding='utf-8') + '\nEvent receipts (evidence only):\n' + __import__('json').dumps(events, sort_keys=True)
                number = len(delivery['attempts']) + 1
                entry = dict(events=[e['id'] for e in events], started_at=now())
                delivery['attempts'].append(entry)
                write(path, delivery, immutable=False)
                (state / 'ARMED').unlink()
                # A file-backed stdin cannot crash this owner if the CLI exits before reading.
                input_path = state / f'review-{number}.input.txt'
                input_path.write_text(text, encoding='utf-8')
                exit_code = -1
                with input_path.open('rb') as source, (state / f'review-{number}.jsonl').open('wb') as out, (state / f'review-{number}.stderr.log').open('wb') as err:
                    try:
                        child = subprocess.Popen(argv, cwd=config['cwd'], stdin=source, stdout=out, stderr=err,
                            env=review_environment(config), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    except OSError as exc:
                        entry['launch_error'] = str(exc)
                    else:
                        entry['pid'] = child.pid
                        write(path, delivery, immutable=False)
                        while child.poll() is None:
                            write(state / 'STATUS.json', dict(snapshot, heartbeat=now(), armed=False, active_review_pid=child.pid), immutable=False)
                            time.sleep(10)
                        exit_code = child.returncode
                remaining = pending(campaign, state)
                ready, waiting = actionable(campaign, state, remaining)
                unresolved = {e['id'] for e in remaining}
                active_unresolved = {e['id'] for e in ready}
                entry.update(exit_code=exit_code, finished_at=now(),
                             acknowledged=[e['id'] for e in events if e['id'] not in unresolved],
                             deferred=[e['id'] for e in events if e['id'] in {w['id'] for w in waiting}])
                delivery.pop('retry_not_before', None)
                if active_unresolved.intersection(entry['events']):
                    delivery['retry_not_before'] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                write(path, delivery, immutable=False)
                if once:
                    return
                if active_unresolved.intersection(entry['events']):
                    failures = sum(bool(set(a['events']) & active_unresolved) for a in delivery['attempts'])
                    if failures >= 3:
                        write(state / 'DELIVERY_FAILED.json', dict(at=now(), unacknowledged=sorted(active_unresolved), attempts=failures))
                        # Keep an inspectable owner alive, but do not spend on endless retries.
                        continue
                if config.get('auto_rearm_after_acknowledged_review') and not (state / 'STOP').exists():
                    (state / 'ARMED').touch()
            time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    supervise(a.campaign, a.state, a.config, a.once)
