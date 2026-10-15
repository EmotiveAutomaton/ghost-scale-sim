"""Source-bound, single-owner V23 execution with durable accounting."""
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from .common import digest, read, write

REPO = Path(__file__).resolve().parents[4]


def environment():
    import numpy, scipy
    result = dict(python=sys.version, executable_sha256=digest(sys.executable),
                  base_executable_sha256=digest(sys._base_executable),
                  numpy=numpy.__version__, scipy=scipy.__version__)
    files = {}
    for module in (numpy, scipy):
        package = Path(module.__file__).parent
        for directory in (package, package.with_name(package.name+'.libs')):
            if directory.exists():
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and path.suffix in ('.py', '.pyd', '.dll'):
                        files[path.relative_to(package.parent).as_posix()] = digest(path)
    result['scientific_library_files'] = files
    return result


def now():
    return datetime.now(timezone.utc).isoformat()


def below_normal():
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
            raise OSError('unable to set below-normal scientific priority')


def memory():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)]+[(name, ctypes.c_ulonglong)
            for name in ('total', 'available', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]
    if os.name != 'nt': return {'available': 2**63}
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('memory observation failed')
    return dict(available=int(status.available), available_commit=int(status.available_page))


def cost(campaign):
    return sum(max(row.get('cpu_seconds', 0), row.get('native_cpu_seconds', 0),
                   row.get('uncertainty_cpu_seconds', 0))
               for row in (read(p) for p in (Path(campaign)/'attempts').glob('*.json')))


def verify_plan(root, campaign):
    plan, acceptance = read(root/'PLAN.json'), read(campaign/'ACCEPTANCE.json')
    if digest(campaign/'ACCEPTANCE.json') != plan['acceptance_sha256']:
        raise ValueError('acceptance byte pin changed')
    if environment() != read(campaign/'ENVIRONMENT.json'):
        raise ValueError('scientific environment changed')
    if digest(campaign/'ENVIRONMENT.json') != acceptance['environment_sha256']:
        raise ValueError('environment pin changed')
    for relative, expected in plan['sources'].items():
        if digest(REPO/relative) != expected: raise ValueError('source changed: '+relative)
    for path, expected in plan.get('dependency_files', {}).items():
        if digest(Path(path)) != expected: raise ValueError('dependency byte pin changed')
    if digest(Path(plan['source_archive'])) != plan['source_archive_sha256']:
        raise ValueError('source archive changed')
    predecessor = Path(acceptance['predecessor_closeout'])
    if digest(predecessor) != acceptance['predecessor_closeout_sha256']:
        raise ValueError('predecessor closure changed')
    prior = read(predecessor)
    if prior['state'] != 'closed' or not prior['owned_scientific_services_stopped']:
        raise ValueError('V21 remains open')
    if now() < acceptance['opened'] or now() >= acceptance['deadline']:
        raise ValueError('outside immutable V23 window')
    if plan['task'] not in ('calibration', 'pilot'):
        admission = read(campaign/'CALIBRATION_ACCEPTANCE.json')
        summary = Path(admission['summary'])
        completion = Path(admission['complete'])
        if admission['acceptance_sha256'] != digest(campaign/'ACCEPTANCE.json'):
            raise ValueError('calibration admission belongs to another acceptance')
        if digest(summary) != admission['summary_sha256'] or digest(completion) != admission['complete_sha256']:
            raise ValueError('calibration acceptance bindings changed')
        if read(summary)['state'] != 'CALIBRATION_PASSED' or read(completion)['state'] != 'verified':
            raise ValueError('calibration not passed')
        if plan['sources'].get('ghostscale/validation/soundingline/v23/adaptive.py') != admission['adaptive_source_sha256']:
            raise ValueError('stopping policy differs from accepted calibration')
    return plan, acceptance


def compare(original, replay):
    def index(path):
        return {p.relative_to(path).as_posix(): digest(p) for p in path.rglob('*')
                if p.is_file() and not p.name.endswith('-cost.json')}
    a, b = index(original), index(replay)
    if a != b:
        raise ValueError('full scientific replay differs: '+str([k for k in set(a)|set(b) if a.get(k) != b.get(k)][:10]))
    return dict(state='passed', files=a, exclusions='separately recorded CPU timing only')


def child_cpu(child):
    """Read cumulative CPU from this owned handle, including after normal exit."""
    if os.name != 'nt':
        raise RuntimeError('this native runner requires a Windows process-time adapter')
    kernel = ctypes.windll.kernel32
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p]*4
    kernel.GetProcessTimes.restype = ctypes.c_int
    creation, exit_time, kernel_time, user = (ctypes.c_ulonglong() for _ in range(4))
    handle = child if isinstance(child, int) else int(child._handle)
    if not kernel.GetProcessTimes(ctypes.c_void_p(handle),
            ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user)):
        raise OSError('owned child CPU unavailable')
    return (kernel_time.value+user.value)/10_000_000


def child_handshake(root, label):
    write(root/(label+'-owner.json'), dict(pid=os.getpid(), parent_pid=os.getppid(), at=now()))
    deadline = time.monotonic()+30
    while not (root/(label+'-parent-ready.json')).exists():
        if time.monotonic() > deadline: raise TimeoutError('parent accounting handshake missing')
        time.sleep(.05)


def attach_child(child, root, label):
    """Keep an actual interpreter handle, not just Windows' venv redirector."""
    path = root/(label+'-owner.json'); deadline = time.monotonic()+30
    while not path.exists():
        if child.poll() is not None: raise RuntimeError('child exited before accounting handshake')
        if time.monotonic() > deadline: raise TimeoutError('child accounting handshake missing')
        time.sleep(.05)
    owner = read(path)
    if owner['pid'] != child.pid and owner['parent_pid'] != child.pid:
        raise ValueError('unexpected native child ownership')
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000 | 0x100000 | 0x0001, False, owner['pid'])
    if not handle: raise OSError('cannot retain owned scientific process handle')
    write(root/(label+'-parent-ready.json'), dict(parent_pid=os.getpid(), child_pid=owner['pid']))
    return int(handle)


def execute_pass(root, campaign, label):
    """One independent process per full pass; process metadata stays outside science."""
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    if label not in ('original', 'replay'): raise ValueError('unknown pass')
    child_handshake(root, label)
    plan, _ = verify_plan(root, campaign)
    below_normal(); started = time.process_time()
    def pulse(**updates):
        write(root/(label+'-progress.json'), dict(pid=os.getpid(), heartbeat=now(),
            cpu_seconds=time.process_time()-started, **updates), immutable=False)
        if time.process_time()-started > plan['cpu_cap_seconds']/2:
            raise TimeoutError('per-pass CPU cap reached')
    pulse(phase='starting')
    if plan['task'] == 'calibration':
        from .calibration import run as calibrate
        calibrate(root/label, pulse, cpu_cap_seconds=plan['cpu_cap_seconds']/2)
    elif plan['task'] == 'family-batch':
        from .pipeline import run_batch
        run_batch(root/label, plan, pulse)
    elif plan['task'] == 'pilot':
        from .pilot import run as pilot
        pilot(root/label, plan, pulse)
    else:
        raise ValueError('unimplemented task cannot be admitted: '+plan['task'])
    pulse(phase='pass-complete')


def run(root, campaign):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    plan, acceptance = verify_plan(root, campaign)
    with (root/'OWNER.json').open('x', encoding='utf-8') as f:
        f.write('{"pid":'+str(os.getpid())+',"started":"'+now()+'"}\n')
    if (root/'STATUS.json').exists(): raise ValueError('identity was already executed')
    below_normal(); start, cpu = time.monotonic(), time.process_time()
    state = dict(state='running', phase='starting'); lock = threading.RLock()
    ended = threading.Event(); faults = []
    initial_cost = cost(campaign); children_cpu = 0.; active_child = None; actual_handle = None
    def pulse(**updates):
        with lock:
            state.update(updates)
            used = time.process_time()-cpu+children_cpu+(child_cpu(actual_handle) if actual_handle is not None else 0.)
            write(root/'STATUS.json', dict(pid=os.getpid(), parent_pid=os.getppid(), heartbeat=now(),
                cpu_seconds=used, **state), immutable=False)
            write(campaign/'attempts'/(root.name+'.json'), dict(job=root.name, cpu_seconds=used,
                wall_seconds=time.monotonic()-start, **state), immutable=False)
            if state['state'] != 'running': return
            if faults: raise RuntimeError(faults[0])
            if used > plan['cpu_cap_seconds']: raise TimeoutError('job CPU cap reached')
            if time.monotonic()-start > plan.get('wall_cap_seconds',plan['cpu_cap_seconds']*4):
                raise TimeoutError('job elapsed cap reached')
            limit = acceptance['cpu_seconds']-(0 if plan.get('protected') else acceptance['protected_cpu_seconds'])
            if initial_cost+used >= limit: raise TimeoutError('campaign CPU reserve reached')
            if now() >= acceptance['deadline'] or (campaign/'STOP').exists(): raise TimeoutError('campaign stop')
            if memory()['available'] < acceptance['free_memory_bytes']: raise MemoryError('host memory reserve')
            if shutil.disk_usage(campaign).free < acceptance['free_disk_bytes']: raise OSError('host disk reserve')
    def heartbeat():
        while not ended.wait(30):
            try: pulse()
            except Exception as exc:
                faults.append(str(exc)); return
    pulse(); thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        for label in ('original', 'replay'):
            pulse(phase=label)
            argv = [sys.executable, '-B', '-m', 'runners.run_v23', '--root', str(root),
                    '--campaign', str(campaign), '--single-pass', label]
            with (root/(label+'.log')).open('xb') as log:
                active_child = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    actual_handle = attach_child(active_child, root, label)
                    while active_child.poll() is None:
                        pulse(child_pid=active_child.pid)
                        try: active_child.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                    pulse(child_pid=active_child.pid)
                    if active_child.returncode:
                        raise RuntimeError(label+' exited '+str(active_child.returncode))
                finally:
                    if active_child.poll() is None:
                        if actual_handle:
                            ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(actual_handle), 1)
                        active_child.terminate(); active_child.wait(timeout=30)
                    with lock:
                        if actual_handle:
                            children_cpu += child_cpu(actual_handle)
                            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(actual_handle))
                        else:
                            # Unmeasurable pre-handshake failure: charge the startup bound.
                            children_cpu += 30.
                        actual_handle = None; active_child = None
        receipt = compare(root/'original', root/'replay'); write(root/'REPLAY.json', receipt)
        write(root/'COMPLETE.json', dict(state='verified', at=now(), plan_sha256=digest(root/'PLAN.json'),
            replay_sha256=digest(root/'REPLAY.json'), summary_sha256=digest(root/'original/SUMMARY.json')))
        pulse(state='complete', phase='review-needed')
    except BaseException as exc:
        pulse(state='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        ended.set(); thread.join()
