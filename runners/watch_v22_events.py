"""V22 event reviews with an exclusive handoff from the existing V21 notifier.

The older notifier remains alive. Its arm token is held only during an actual V22
review, with a durable escrow receipt and a quiescence check before delivery.
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
from runners.watch_v21_events import review_environment, closed_receipt


def peer_busy(peer):
    if (peer.parent/'CLOSEOUT.json').exists():
        return False
    status_path = peer/'STATUS.json'
    if not status_path.exists():
        return True
    status = read(status_path)
    current = datetime.now(timezone.utc)
    if (current-datetime.fromisoformat(status['heartbeat'])).total_seconds() > 45:
        return True
    if status.get('active_review_pid') and alive(status['active_review_pid']):
        return True
    delivery_path = peer/'DELIVERY.json'
    if delivery_path.exists():
        attempts = read(delivery_path)['attempts']
        if attempts and 'exit_code' not in attempts[-1]:
            return True
    return False


def release_peer(state, peer):
    receipt = state/'PEER_ARM_ESCROW.json'
    if receipt.exists():
        stored = read(receipt)
        if stored['was_armed'] and not (peer/'STOP').exists() and not (peer.parent/'CLOSEOUT.json').exists():
            (peer/'ARMED').touch()
        write(state/'PEER_ARM_RELEASED.json', dict(at=now(), escrow=stored), immutable=False)
        receipt.unlink()


def supervise(campaign, state, config_path, once=False):
    config = read(config_path)
    peer = Path(config['peer_state'])
    state.mkdir(parents=True, exist_ok=True)
    path = state/'DELIVERY.json'
    delivery = read(path) if path.exists() else dict(attempts=[])
    cpu_start = time.process_time()
    with local_owner(state):
        if delivery['attempts'] and 'exit_code' not in delivery['attempts'][-1]:
            last = delivery['attempts'][-1]
            if last.get('pid') and alive(last['pid']):
                raise RuntimeError('preserve live review owner')
            last.update(exit_code='interrupted_notifier', finished_at=now())
            write(path, delivery, immutable=False)
        release_peer(state, peer)
        while True:
            if closed_receipt(campaign):
                (state/'ARMED').unlink(missing_ok=True)
                write(state/'STATUS.json', dict(pid=os.getpid(), heartbeat=now(), state='closed', armed=False,
                      active_review_pid=None), immutable=False)
                return
            checkpoint_events(campaign, read(campaign/'ACCEPTANCE.json'), config)
            events = pending(campaign, state)
            status_path = campaign/'QUEUE-STATUS.json'
            if status_path.exists():
                status = read(status_path)
                if status['state'] in ('running', 'waiting-for-priority-or-review') and not alive(status['pid']):
                    name = 'supervisor-disappeared-'+str(status['pid'])
                    if not (campaign/'events'/(name+'.json')).exists():
                        write(campaign/'events'/(name+'.json'), dict(id=name, kind='supervisor_disappeared', at=now()))
            snapshot = dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(), armed=(state/'ARMED').exists(),
                            pending=len(events), active_review_pid=None)
            write(state/'STATUS.json', snapshot, immutable=False)
            write(state/'SERVICE_COST.json', dict(cpu_seconds=time.process_time()-cpu_start, at=now()), immutable=False)
            if (state/'STOP').exists():
                return
            urgent = any(e['kind'] != 'batch_complete' for e in events)
            # Routine completed batches collect until the fixed check or depletion.
            if events and urgent and (state/'ARMED').exists() and not peer_busy(peer):
                if (peer.parent/'CLOSEOUT.json').exists():
                    peer_was_armed = False
                else:
                    peer_was_armed = (peer/'ARMED').exists()
                    if not peer_was_armed:
                        time.sleep(10)
                        continue  # active operator suppression is authoritative
                    write(state/'PEER_ARM_ESCROW.json', dict(at=now(), was_armed=True))
                    (peer/'ARMED').unlink(missing_ok=True)
                    time.sleep(12)  # allow a launch already in flight to announce itself
                    if peer_busy(peer):
                        release_peer(state, peer)
                        continue
                try:
                    argv = config['command']
                    if any(x in argv for x in ('resume', '--last', '--dangerously-bypass-approvals-and-sandbox', '--ignore-rules', '--ignore-user-config')):
                        raise ValueError('unsafe/competing reviewer command')
                    if file_digest(argv[0]) != config['executable_sha256'] or file_digest(config['prompt']) != config['prompt_sha256']:
                        raise ValueError('review contract changed')
                    number = len(delivery['attempts'])+1
                    prompt = Path(config['prompt']).read_text(encoding='utf-8')+'\nSupplied event evidence:\n'+__import__('json').dumps(events, sort_keys=True)
                    entry = dict(started_at=now(), events=[e['id'] for e in events])
                    delivery['attempts'].append(entry)
                    write(path, delivery, immutable=False)
                    (state/'ARMED').unlink()
                    inp = state/f'review-{number}.input.txt'
                    inp.write_text(prompt, encoding='utf-8')
                    with inp.open('rb') as source, (state/f'review-{number}.jsonl').open('wb') as out, (state/f'review-{number}.stderr.log').open('wb') as err:
                        child = subprocess.Popen(argv, cwd=config['cwd'], stdin=source, stdout=out, stderr=err,
                                     env=review_environment(config), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        entry['pid'] = child.pid
                        write(path, delivery, immutable=False)
                        while child.poll() is None:
                            write(state/'STATUS.json', dict(snapshot, heartbeat=now(), armed=False, active_review_pid=child.pid), immutable=False)
                            time.sleep(10)
                    unresolved = {e['id'] for e in pending(campaign, state)}
                    entry.update(exit_code=child.returncode, finished_at=now(), acknowledged=[e['id'] for e in events if e['id'] not in unresolved])
                    write(path, delivery, immutable=False)
                    if once:
                        return
                    failures = sum(bool(set(a['events']) & unresolved) for a in delivery['attempts'])
                    if failures >= 3:
                        write(state/'DELIVERY_FAILED.json', dict(at=now(), unacknowledged=sorted(unresolved), attempts=failures))
                    elif not (state/'STOP').exists():
                        (state/'ARMED').touch()
                        if unresolved.intersection(entry['events']):
                            # Cheap native backoff, no extra model loop.
                            for _ in range(30):
                                if (state/'STOP').exists():
                                    break
                                time.sleep(10)
                finally:
                    release_peer(state, peer)
            time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    supervise(args.campaign, args.state, args.config, args.once)
