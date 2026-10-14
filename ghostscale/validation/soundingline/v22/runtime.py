"""Immutable V22 identities, complete verification/replay and bounded accounting."""
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from ..v16.runtime import local_owner
from ..v18_3.io import read, write, file_digest, now
from ..v18_4.priority import below_normal
from ..v20.runtime import memory, available_memory
from .studies import run as science
from .checker import verify, compare_replay

REPO = Path(__file__).resolve().parents[4]


def environment():
    import platform
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def spent(campaign):
    return sum(max(r.get('cpu_seconds', 0), r.get('native_cpu_seconds', 0), r.get('uncertainty_cpu_seconds', 0))
               for r in (read(p) for p in (campaign/'attempts').glob('*.json')))


def run(root, campaign):
    root, campaign = Path(root), Path(campaign)
    with local_owner(root):
        if (root/'STATUS.json').exists() or (root/'COMPLETE.json').exists():
            raise ValueError('executed identity cannot be overwritten')
        plan = read(root/'PLAN.json')
        acceptance = read(campaign/'ACCEPTANCE.json')
        if file_digest(campaign/'ACCEPTANCE.json') != plan['acceptance_sha256']:
            raise ValueError('acceptance changed')
        if plan['environment'] != environment():
            raise ValueError('environment changed')
        for name, digest in plan['sources'].items():
            if file_digest(REPO/name) != digest:
                raise ValueError('source changed: '+name)
        if file_digest(campaign/'capsules'/plan['capsule']/'SOURCE.zip') != plan['archive_sha256']:
            raise ValueError('source archive changed')
        below_normal()
        start, cpu = time.monotonic(), time.process_time()
        state = dict(phase='starting', state='running')
        lock = threading.RLock()
        stopped = threading.Event()
        fault = []
        last_artifact = [0., 0]
        def pulse(**updates):
            with lock:
                state.update(updates)
                used = time.process_time()-cpu
                mem = memory()
                write(root/'STATUS.json', dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(),
                                               cpu_seconds=used, **mem, **state), immutable=False)
                write(campaign/'attempts'/(root.name+'.json'), dict(job=root.name, state=state['state'],
                      cpu_seconds=used, wall_seconds=time.monotonic()-start, peak_bytes=mem['peak_bytes']), immutable=False)
                if state['state'] != 'running':
                    return
                if fault:
                    raise RuntimeError(fault[0])
                if used > plan['cpu_cap_seconds']:
                    raise TimeoutError('packet CPU cap')
                if (campaign/'STOP').exists() or datetime.now(timezone.utc) >= datetime.fromisoformat(acceptance['deadline']):
                    raise TimeoutError('campaign stopped or expired')
                if mem['working_bytes'] > acceptance['worker_memory_bytes'] or available_memory() < acceptance['free_memory_bytes']:
                    raise MemoryError('memory reserve')
                if shutil.disk_usage(campaign).free < acceptance['free_disk_bytes']:
                    raise OSError('disk reserve')
                if time.monotonic()-last_artifact[0] > 60:
                    last_artifact[:] = [time.monotonic(), sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))]
                if last_artifact[1] > plan['artifact_cap_bytes']:
                    raise OSError('packet artifact cap')
        def beat():
            while not stopped.wait(30):
                try:
                    pulse()
                except Exception as exc:
                    fault.append(str(exc))
                    return
        pulse()
        timer = threading.Thread(target=beat, daemon=True)
        timer.start()
        try:
            science(root/'original', plan['design'], pulse)
            pulse(phase='independent-check')
            verify(root/'original', root/'VALIDITY.json')
            pulse(phase='full-scientific-replay')
            science(root/'replay', plan['design'], pulse)
            pulse(phase='replay-comparison')
            compare_replay(root/'original', root/'replay', root/'REPLAY.json')
            write(root/'COMPLETE.json', dict(at=now(), plan_sha256=file_digest(root/'PLAN.json'),
                  validity_sha256=file_digest(root/'VALIDITY.json'), replay_sha256=file_digest(root/'REPLAY.json'),
                  summary_sha256=file_digest(root/'original/SUMMARY.json'),
                  state='numerically verified; scientific interpretation and public write-through pending'))
            pulse(state='complete')
        except BaseException as exc:
            pulse(state='failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            stopped.set()
            timer.join()
