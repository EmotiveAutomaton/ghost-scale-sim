"""Release one hash-bound administrative hold after native memory recovery.

This low-cost helper never dispatches science, restarts an owner, or runs a model.
The existing supervisor retains every admission, CPU and deadline guard.
"""
import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from ghostscale.validation.soundingline.v18_3.io import read, write, file_digest, now
from ghostscale.validation.soundingline.v20.runtime import available_memory


def commit_headroom():
    """Read system commit capacity; physical free RAM alone cannot admit an allocation."""
    class PerformanceInformation(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
                'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
                'KernelNonpaged', 'PageSize')
        ] + [(name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]
    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = (ctypes.POINTER(PerformanceInformation), wintypes.DWORD)
    api.GetPerformanceInfo.restype = wintypes.BOOL
    if not api.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not info.PageSize or info.CommitLimit < info.CommitTotal:
        raise ValueError('invalid native commit counters')
    return dict(committed_bytes=info.CommitTotal * info.PageSize,
                commit_limit_bytes=info.CommitLimit * info.PageSize,
                commit_free_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize)


def native_identity(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        a, b, c, d = (wintypes.FILETIME() for _ in range(4))
        exit_code = wintypes.DWORD()
        if not kernel.GetProcessTimes(handle, ctypes.byref(a), ctypes.byref(b), ctypes.byref(c), ctypes.byref(d)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=pid, creation=(a.dwHighDateTime << 32) | a.dwLowDateTime,
                    alive=exit_code.value == 259)
    finally:
        kernel.CloseHandle(handle)


def matches(actual, expected):
    return bool(actual and actual['alive'] and actual['pid'] == expected['pid']
                and actual['creation'] == expected['creation'])


def release_ready(free, required, owner_matches, reviewer_alive, active, other_review_alive=False,
                  *, commit_free=None, commit_required=None):
    return (commit_free is not None and commit_required is not None
            and commit_free >= commit_required and free >= required and owner_matches
            and not reviewer_alive and not active and not other_review_alive)


def supervise(campaign, contract):
    config = read(contract)
    hold = campaign / 'HOLD'
    status_dir = contract.parent
    acceptance = read(campaign / 'ACCEPTANCE.json')
    required = acceptance['free_memory_reserve_bytes'] + acceptance['worker_memory_ceiling_bytes'] + 2**29
    # Before releasing an idle queue, cover the reserve and every permitted worker.
    commit_required = (acceptance['free_memory_reserve_bytes']
                       + acceptance['workers'] * acceptance['worker_memory_ceiling_bytes'] + 2**29)
    started = time.process_time()
    identity = native_identity(os.getpid())
    stable = 0
    while True:
        stamp = dict(at=now(), pid=os.getpid(), parent_pid=os.getppid(), identity=identity,
                     cpu_seconds=time.process_time() - started, required_free_bytes=required,
                     required_commit_free_bytes=commit_required)
        write(status_dir / 'SERVICE_COST.json', stamp, immutable=False)
        if (campaign / 'STOP').exists() or (campaign / 'SUPERVISOR_STOP').exists() or datetime.now(timezone.utc) >= datetime.fromisoformat(acceptance['report_start']):
            write(status_dir / 'STATUS.json', dict(stamp, state='stopped at owned boundary'), immutable=False)
            return
        if not hold.exists() or file_digest(hold) != config['hold_sha256']:
            write(status_dir / 'STATUS.json', dict(stamp, state='hold changed; no mutation'), immutable=False)
            return
        owner = native_identity(config['supervisor']['pid'])
        if not matches(owner, config['supervisor']):
            write(status_dir / 'STATUS.json', dict(stamp, state='owner changed; no mutation'), immutable=False)
            return
        reviewer = native_identity(config['reviewer']['pid'])
        notice = read(campaign / 'notifications/STATUS.json')
        other_pid = notice.get('active_review_pid')
        other_review = native_identity(other_pid) if other_pid else None
        queue = read(campaign / 'QUEUE-STATUS.json')
        free = available_memory()
        commit = commit_headroom()
        ready = release_ready(free, required, True, matches(reviewer, config['reviewer']), queue['active'],
                              bool(other_review and other_review['alive']),
                              commit_free=commit['commit_free_bytes'], commit_required=commit_required)
        stable = stable + 1 if ready else 0
        write(status_dir / 'STATUS.json', dict(stamp, state='waiting for owner exit and stable memory',
              free_memory_bytes=free, **commit, stable_observations=stable), immutable=False)
        if stable >= 2:
            # Recheck the exact hold immediately before removing this administrative marker.
            if file_digest(hold) != config['hold_sha256']:
                raise ValueError('hold changed at release')
            hold.unlink()
            write(status_dir / 'STATUS.json', dict(stamp, state='released; existing supervisor owns dispatch',
                  free_memory_bytes=free, **commit, stable_observations=stable), immutable=False)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args()
    if args.probe:
        write(args.probe, dict(pid=os.getpid(), parent_pid=os.getppid(), identity=native_identity(os.getpid()),
                             free_memory_bytes=available_memory(), **commit_headroom(), at=now()))
    else:
        if not args.campaign or not args.contract:
            parser.error('campaign and contract required')
        supervise(args.campaign, args.contract)
