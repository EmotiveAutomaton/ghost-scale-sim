"""Harmless hidden-launch check: no research, acceptance, queue or source writes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'
import argparse
import ctypes
from pathlib import Path
import sys
from ghostscale.validation.soundingline.v23.runtime import below_normal, now, child_handshake
from ghostscale.validation.soundingline.v23.common import write

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--accounting-handshake', action='store_true')
    args = parser.parse_args(); below_normal()
    if args.accounting_handshake:
        child_handshake(args.out.parent, 'fixture')
        sum(i*i for i in range(2000000))
    kernel = ctypes.windll.kernel32
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel.GetPriorityClass.restype = ctypes.c_ulong
    kernel.GetConsoleWindow.restype = ctypes.c_void_p
    console = kernel.GetConsoleWindow(); priority = kernel.GetPriorityClass(kernel.GetCurrentProcess())
    assert not console, 'background fixture has a console window'
    assert priority == 0x4000
    write(args.out, dict(at=now(), passed=True, pid=os.getpid(), parent_pid=os.getppid(),
        executable=sys.executable, console_window=int(console or 0), priority=priority,
        threads={key: os.environ[key] for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}))
