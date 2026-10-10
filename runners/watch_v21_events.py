"""V21 event delivery with bounded, hash-bound verification dependencies.

Deferral never acknowledges an event. Failure, deadline and fixed health events
remain immediately actionable; a dependency failure releases the original event.
"""
import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
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
            snapshot = dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(),
                            pending=len(all_events), actionable=len(events), deferred=len(deferred),
                            armed=(state / 'ARMED').exists(), active_review_pid=None)
            write(state / 'STATUS.json', snapshot, immutable=False)
            write(state / 'SERVICE_COST.json', dict(at=now(), cpu_seconds=time.process_time()-cpu_start), immutable=False)
            if (state / 'STOP').exists():
                return
            if events and (state / 'ARMED').exists():
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
                with (state / f'review-{number}.jsonl').open('wb') as out, (state / f'review-{number}.stderr.log').open('wb') as err:
                    child = subprocess.Popen(argv, cwd=config['cwd'], stdin=subprocess.PIPE, stdout=out, stderr=err,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    entry['pid'] = child.pid
                    write(path, delivery, immutable=False)
                    child.stdin.write(text.encode('utf-8'))
                    child.stdin.close()
                    while child.poll() is None:
                        write(state / 'STATUS.json', dict(snapshot, heartbeat=now(), armed=False, active_review_pid=child.pid), immutable=False)
                        time.sleep(10)
                remaining = pending(campaign, state)
                ready, waiting = actionable(campaign, state, remaining)
                unresolved = {e['id'] for e in remaining}
                active_unresolved = {e['id'] for e in ready}
                entry.update(exit_code=child.returncode, finished_at=now(),
                             acknowledged=[e['id'] for e in events if e['id'] not in unresolved],
                             deferred=[e['id'] for e in events if e['id'] in {w['id'] for w in waiting}])
                write(path, delivery, immutable=False)
                if once:
                    return
                if active_unresolved.intersection(entry['events']):
                    failures = sum(bool(set(a['events']) & active_unresolved) for a in delivery['attempts'])
                    if failures >= 3:
                        write(state / 'DELIVERY_FAILED.json', dict(at=now(), unacknowledged=sorted(active_unresolved), attempts=failures))
                        return
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
