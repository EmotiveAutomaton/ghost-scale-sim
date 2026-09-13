"""Consoleless Windows task entry point; invoke this file with pythonw.exe.

Example: pythonw.exe launch_background.py --cwd SOURCE --log LOG --
         ABSOLUTE_PYTHON_EXE -B -m runners.watch_v17_continuation ...

The scheduler retains a waiting parent and receives the child's exit code.
No campaign modules are imported here; scientific source and ownership stay with
the supplied command. Keep a pinned copy outside a live scientific checkout.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or not Path(command[0]).is_absolute():
        parser.error("an absolute executable path is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("ab", buffering=0) as output:
        try:
            if sys.platform != "win32":
                raise RuntimeError("This entry point is for Windows pythonw.exe")
            child = subprocess.Popen(
                command,
                cwd=args.cwd.resolve(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
                shell=False,
            )
            return child.wait()
        except Exception:
            output.write(traceback.format_exc().encode("utf-8"))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
