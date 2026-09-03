"""Watchdog for the V15 168-hour window.

Relaunches a dead runner, gated on *progress* rather than on liveness alone: a runner that is alive
and making no progress is as dead as one that exited, and V13 lost time to exactly that. Progress
is the number of checkpoint lines plus the number of resolved verdicts plus the number of executed
coverage blocks; if none of them moves for ``STALL_POLLS`` polls, the runner is killed and
relaunched.

Relaunch is module form (``python -X faulthandler -m runners.run_v15``) for the reason in
``run_v15_wrapped.ps1``: a script-path command line is what the sibling project's orphan sweeper
matches, and matching it is what cost V14 its first window.

The watchdog stops when the window closes. It never extends a deadline and never starts a second
runner: it checks for a live one first.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import runtime_contract as RC  # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402
from ghostscale.validation.soundingline.v15.atomicio import write_json_atomic  # noqa: E402

PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)
STATUS = v15_dir() / "RUNNER_STATUS.json"
WATCHDOG = v15_dir() / "WATCHDOG_STATUS.json"
CHECKPOINTS = v15_dir() / "CHECKPOINTS.jsonl"
COVERAGE_DIR = v15_dir("coverage")
POLL_S = 120
STALL_POLLS = 8                 # about sixteen minutes with no progress
MAX_RELAUNCHES = 80


def progress() -> int:
    n = 0
    if CHECKPOINTS.exists():
        try:
            n += sum(1 for _ in CHECKPOINTS.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    for lane in ("discovery", "transfer", "attack", "confirmation"):
        try:
            n += len(list(verdict_dir(lane).glob("*.json")))
        except OSError:
            pass
    blocks = COVERAGE_DIR / "blocks.jsonl"
    if blocks.exists():
        try:
            with blocks.open("r", encoding="utf-8", errors="ignore") as f:
                n += sum(1 for _ in f)
        except OSError:
            pass
    return n


def tree_cpu(pid: int | None) -> float:
    """CPU seconds of the runner and every descendant. A healthy runner deep inside one long card
    moves no checkpoint line, no verdict file and no coverage block -- ``pool.map`` blocks the
    parent -- but its workers burn CPU. Counting that CPU as progress is what stops the watchdog
    killing a healthy runner 48 minutes into a 90-minute card, which it did at hour 0.26 of this
    window and which cost the morning."""
    if not pid:
        return 0.0
    try:
        import psutil
        p = psutil.Process(int(pid))
        total = sum(t.user + t.system for t in [p.cpu_times()])
        for ch in p.children(recursive=True):
            try:
                t = ch.cpu_times()
                total += t.user + t.system
            except Exception:                                     # noqa: BLE001, S110
                pass
        return float(total)
    except Exception:                                             # noqa: BLE001
        return 0.0


def status_heartbeat() -> str:
    """The runner's own heartbeat timestamp; it moves during wait loops that touch no file the
    file-progress counter reads."""
    if not STATUS.exists():
        return ""
    try:
        return str(json.loads(STATUS.read_text(encoding="utf-8")).get("heartbeat", ""))
    except (json.JSONDecodeError, OSError):
        return ""


def runner_pid() -> int | None:
    if not STATUS.exists():
        return None
    try:
        return int(json.loads(STATUS.read_text(encoding="utf-8")).get("pid"))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.is_running() and "python" in (p.name() or "").lower()
    except Exception:                                             # noqa: BLE001
        return False


def kill(pid: int | None) -> None:
    if not pid:
        return
    try:
        import psutil
        p = psutil.Process(int(pid))
        for ch in p.children(recursive=True):
            try:
                ch.kill()
            except Exception:                                     # noqa: BLE001, S110
                pass
        p.kill()
    except Exception:                                             # noqa: BLE001, S110
        pass


def launch(stage: str = "all") -> subprocess.Popen:
    env = dict(os.environ)
    env.update({"PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": ""})
    logs = v15_dir("logs")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (logs / f"watchdog_run_{stamp}.out.log").open("w", encoding="utf-8")
    err = (logs / f"watchdog_run_{stamp}.err.log").open("w", encoding="utf-8")
    # MODULE FORM. A `runners/run_v15.py` command line is what the sibling project's orphan
    # sweeper matches, and it killed V14's runner seven times.
    return subprocess.Popen([str(PY), "-X", "faulthandler", "-m", "runners.run_v15",
                             "--stage", stage],
                            cwd=str(REPO), env=env, stdout=out, stderr=err)


def note(**kw) -> None:
    write_json_atomic(WATCHDOG, {"pid": os.getpid(),
                                 "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "elapsed_hours": round(RC.elapsed_hours(), 3),
                                 "phase": RC.phase(), **kw})


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--poll", type=int, default=POLL_S)
    a = ap.parse_args()
    relaunches, last, stalled = 0, progress(), 0
    last_cpu, last_beat, last_seen_pid = 0.0, "", None
    note(started=time.strftime("%Y-%m-%dT%H:%M:%S"), progress=last, relaunches=0)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} watchdog started; progress {last}")
    # Start the runner immediately if none is alive, rather than idling a whole poll first. The
    # watchdog checks for a live runner before launching, so it never starts a second one.
    if not alive(runner_pid()):
        print(f"{time.strftime('%H:%M:%S')} no live runner; launching (module form)")
        launch(a.stage)
        time.sleep(20)
    while True:
        if RC.window_closed() and RC.phase() == "report":
            note(stopped="window closed", relaunches=relaunches)
            print("window closed; watchdog exiting")
            return 0
        time.sleep(a.poll)
        pid = runner_pid()
        now = progress()
        if pid != last_seen_pid:                # a new runner starts its CPU counter from zero
            last_cpu, last_seen_pid = 0.0, pid
        cpu = tree_cpu(pid)
        beat = status_heartbeat()
        # Progress is any of: a new checkpoint/verdict/coverage line, the runner's process tree
        # burning CPU (a long card), or the runner's own heartbeat moving (a wait loop). A hung
        # runner moves none of the three.
        moved = (now > last) or (cpu > last_cpu + 5.0) or (beat != last_beat and beat)
        last_cpu, last_beat = max(cpu, last_cpu), beat
        if moved:
            last, stalled = max(now, last), 0
        else:
            stalled += 1
        live = alive(pid)
        note(progress=now, stalled_polls=stalled, runner_pid=pid, runner_alive=live,
             relaunches=relaunches)
        if live and stalled < STALL_POLLS:
            continue
        if relaunches >= MAX_RELAUNCHES:
            note(stopped="relaunch budget exhausted", relaunches=relaunches)
            print("relaunch budget exhausted")
            return 1
        if live:
            print(f"{time.strftime('%H:%M:%S')} runner alive but no progress for "
                  f"{stalled} polls; killing pid {pid}")
            kill(pid)
        relaunches += 1
        print(f"{time.strftime('%H:%M:%S')} relaunch {relaunches} (module form)")
        launch(a.stage)
        stalled = 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
