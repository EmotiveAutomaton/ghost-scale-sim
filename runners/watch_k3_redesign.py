"""One finite K3-RD1 native watcher: six-hour health plus actionable events.

Native Codex reviews create durable local reports. They do not inject messages
into an existing interactive conversation. Completed queues receive one final
review and no further recurring health events.
"""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v27.common import read, write
from runners.watch_v23_events import (owner, event, pending, deliver, closed,
    require_native_review_host, ReviewOwnerBusyError, deferred_events, alive)


def scheduled(c, current):
    a = read(c/'ACCEPTANCE.json')
    if (c/'QUEUE_COMPLETE.json').exists():
        return
    anchor = datetime.fromisoformat(a['opened'])
    drain = datetime.fromisoformat(a['drain'])
    number = 1
    point = anchor+timedelta(hours=6)
    while point <= current and point < drain:
        event(c, f'k3-rd1-health-{number:03d}', 'six_hour_health', dict(due=point.isoformat()))
        number += 1
        point += timedelta(hours=6)
    plan = c/'WAKE_PLAN.json'
    if plan.exists():
        wake = read(plan)
        if current >= datetime.fromisoformat(wake['due']):
            event(c, wake['id'], 'planned_review', dict(due=wake['due'], reason=wake['reason']))
    status = read(c/'QUEUE_STATUS.json') if (c/'QUEUE_STATUS.json').exists() else {}
    if status.get('state') == 'RUNNING' and not alive(status['pid']):
        event(c, 'controller-disappeared', 'service_failed', dict(pid=status['pid']))
    if current >= datetime.fromisoformat(a['new_starts_close']):
        event(c, 'last-start-boundary', 'final_deadline', dict(due=a['new_starts_close']))
    if current >= drain:
        (c/'STOP').touch(exist_ok=True)
        event(c, 'drain-boundary', 'final_deadline', dict(due=a['drain']))


def run(c, state, config):
    require_native_review_host()
    cfg = read(config)
    cost_path = c/'accounting'/f'monitor-{os.getpid()}.json'
    with owner(state):
        try:
            while not closed(c) and not (state/'STOP').exists():
                current = datetime.now(timezone.utc)
                scheduled(c, current)
                events = pending(c, state)
                deferred = deferred_events(c, state, events, current)
                actionable = [e for e in events if e['id'] not in deferred]
                write(state/'STATUS.json', dict(state='monitoring', pid=os.getpid(),
                    heartbeat=current.isoformat(), armed=(state/'ARMED').exists(),
                    pending_events=len(actionable), interactive_chat_delivery=False), immutable=False)
                write(cost_path, dict(phase='A', state='RUNNING', cpu_seconds=time.process_time()), immutable=False)
                if actionable and (state/'ARMED').exists():
                    try:
                        deliver(c, state, cfg, actionable)
                    except ReviewOwnerBusyError:
                        pass
                time.sleep(30)
        finally:
            (state/'ARMED').unlink(missing_ok=True)
            write(state/'STATUS.json', dict(state='stopped', pid=os.getpid(),
                at=datetime.now(timezone.utc).isoformat(), armed=False), immutable=False)
            write(cost_path, dict(phase='A', state='STOPPED', cpu_seconds=time.process_time()), immutable=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('campaign', 'state', 'config'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(a.campaign.resolve(), a.state.resolve(), a.config.resolve())
