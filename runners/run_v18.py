"""Finite V18 worker, module-form launch and bounded native supervision."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from ghostscale.validation.soundingline.v16.records import read, write, now
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18.runtime import freeze, run


def native_identity(pid):
    """Windows PID plus creation time; a reused numeric PID is a different owner."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        created, ended, system, user = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *map(ctypes.byref, (created, ended, system, user))):
            raise OSError("cannot establish native process identity")
        if ended.dwLowDateTime or ended.dwHighDateTime:
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def watch(root):
    """The existing native queue pattern, restricted to V18's finite worker.

    The owning exec receives termination directly. No model polling, scheduled
    model turn, notification subprocess or unrelated campaign launch occurs.
    """
    supervisor = root / "supervisor"
    with local_owner(supervisor):
        previous = read(supervisor / "EVENTS.json") if (supervisor / "EVENTS.json").exists() else []
        if (root / "RUN_COMPLETE.json").exists():
            return 0
        if previous and previous[-1]["event"] == "launch":
            old = previous[-1]
            identity = old.get("native_creation_time")
            if identity is None:
                raise RuntimeError("uncertain prior ownership needs review")
            while native_identity(old["pid"]) == identity:
                time.sleep(1)  # Native wait only; no model wake.
            previous.append({"event": "completion" if (root/"RUN_COMPLETE.json").exists() else "disappearance",
                             "pid": old["pid"], "at": now()})
            write(supervisor/"EVENTS.json", previous, immutable=False)
            if (root/"RUN_COMPLETE.json").exists(): return 0
        used = sum(e["event"] == "launch" for e in previous)
        for number in range(used, 3):
            with (supervisor / f"worker-{number}.log").open("ab", buffering=0) as output:
                child = subprocess.Popen([sys.executable, "-B", "-m", "runners.run_v18", "run", "--root", str(root)],
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
                previous.append({"event": "launch", "pid": child.pid, "native_creation_time": native_identity(child.pid),
                                 "supervisor_pid": os.getpid(), "at": now()})
                write(supervisor / "EVENTS.json", previous, immutable=False)
                code = child.wait()
            previous.append({"event": "completion" if code == 0 else "failure", "pid": child.pid,
                             "exit_code": code, "at": now()})
            write(supervisor / "EVENTS.json", previous, immutable=False)
            if code == 0:
                return 0
            # An explicit apparatus/source failure requires review, not repetition.
            if (root / "STATUS.json").exists() and read(root / "STATUS.json")["state"] == "failed":
                return code
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--namespace", default="v18-selective-acquisition-science-1")
    parser.add_argument("--constructors", type=int, default=64)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--started-at")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--stop-after-blocks", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.started_at:
            parser.error("freeze needs the immutable --started-at")
        freeze(args.root, namespace=args.namespace, constructors=args.constructors, histories=args.histories,
               started_at=args.started_at, admission=read(args.admission) if args.admission else None,
               development=args.development)
        print("V18 plan frozen", flush=True)
    elif args.command == "watch":
        return watch(args.root.resolve())
    else:
        result = run(args.root, stop_after_blocks=args.stop_after_blocks)
        print(result.get("execution_state", result.get("state")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
