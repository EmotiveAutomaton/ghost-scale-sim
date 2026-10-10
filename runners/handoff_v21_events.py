"""Wait for a pinned notifier and its current child to exit, then replace it.

No process is killed. The old notifier consumes a local STOP only after its
current child exits; the marker is archived before the successor starts.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_3.native import ProcessClock
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from runners.watch_v18_4_events import alive


def run(plan_path):
    plan = read(plan_path)
    root = plan_path.parent
    state = Path(plan['state'])
    cpu = time.process_time()
    with local_owner(root / 'handoff-owner'):
        clock = ProcessClock(plan['old_pid'])
        if clock.created != plan['old_created']:
            raise ValueError('old notifier creation identity differs')
        write(root / 'HANDOFF_STATUS.json', dict(at=now(), pid=os.getpid(), phase='waiting for current reviewer and old notifier',
                                               old_pid=plan['old_pid'], reviewer_pid=plan['reviewer_pid']), immutable=False)
        while alive(plan['old_pid']):
            time.sleep(1)
        if alive(plan['reviewer_pid']):
            raise RuntimeError('review owner still alive; refusing competing notifier')
        old_cpu = clock.seconds()
        clock.close()
        if file_digest(state / 'STOP') != plan['stop_sha256']:
            raise ValueError('handoff stop marker changed')
        (state / 'STOP').rename(root / 'RETIRED_NOTIFIER_STOP')
        for name, expected in plan['source_files'].items():
            if file_digest(Path(plan['source']) / name) != expected:
                raise ValueError('successor source changed')
        with Path(plan['log']).open('ab', buffering=0) as log:
            child = subprocess.Popen(plan['command'], cwd=plan['source'], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        # Do not arm until the successor proves it acquired the existing owner lock.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('successor notifier exited before ownership')
            status = read(state / 'STATUS.json')
            if status['pid'] not in (plan['old_pid'],) and status.get('parent_pid') in (child.pid, os.getpid()):
                (state / 'ARMED').touch()
                write(root / 'HANDOFF_STATUS.json', dict(at=now(), pid=os.getpid(), phase='successor owns notifications; armed',
                                                       old_pid=plan['old_pid'], old_native_cpu_seconds=old_cpu,
                                                       successor=status, wrapper_pid=child.pid,
                                                       helper_cpu_seconds=time.process_time()-cpu), immutable=False)
                return
            time.sleep(.25)
        raise TimeoutError('successor did not prove native ownership')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
