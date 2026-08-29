"""Watchdog for the V14 run: keeps the multi-day queue healthy without touching it.

Every five minutes it reads the runner's heartbeat. If the run process has died, it
relaunches ``--stage all`` (the
checkpoints and the completion ledger make that a resume, not a restart). Two guards keep a
deterministic crash from looping: at most three relaunches total, and a relaunch is only
allowed if the completion ledger grew since the previous one. When the guards trip, the
watchdog stops and leaves ``results/v14/WATCHDOG_HALTED.json`` as a loud marker. A stale
heartbeat on a living process is only logged: the heaviest cards report progress in
ten-percent chunks, and a second concurrent run would race the checkpoints.

The watchdog exits on its own when the report stage finishes (the runner status reads
``idle``). It writes its own journal to ``results/v14/logs/watchdog.log``. This file is not
part of the locked program: it never reads or writes scientific state.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATUS = REPO / "results" / "v14" / "RUNNER_STATUS.json"
COMPLETION = REPO / "results" / "v14" / "COMPLETION.json"
LOG = REPO / "results" / "v14" / "logs" / "watchdog.log"
HALT = REPO / "results" / "v14" / "WATCHDOG_HALTED.json"
PY = REPO / ".venv" / "Scripts" / "python.exe"

POLL_S = 300
STALE_S = 1800
MAX_RELAUNCHES = 6      # progress-gated: each relaunch must follow ledger growth, so the cap only bounds a slow-recurring death


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")


def heartbeat_age_s() -> float | None:
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
        hb = datetime.fromisoformat(s["heartbeat"])
        return (datetime.now() - hb).total_seconds(), s.get("stage"), s.get("pid")
    except Exception:                                        # noqa: BLE001
        return None


def pid_alive(pid) -> bool:
    if not pid:
        return False
    r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def ledger_entries() -> int:
    try:
        return len(json.loads(COMPLETION.read_text(encoding="utf-8")).get("entries", {}))
    except Exception:                                        # noqa: BLE001
        return -1


def relaunch() -> int:
    out = (REPO / "results" / "v14" / "logs" / f"run_all_resume_{time.strftime('%m%d_%H%M%S')}.log").open("w", encoding="utf-8")
    p = subprocess.Popen([str(PY), str(REPO / "runners" / "run_v14.py"), "--stage", "all"],
                         cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    return p.pid


def main() -> None:
    relaunches = 0
    entries_at_last_relaunch = ledger_entries()
    log(f"watchdog started; ledger entries {entries_at_last_relaunch}")
    while True:
        time.sleep(POLL_S)
        got = heartbeat_age_s()
        if got is None:
            log("no readable status; waiting")
            continue
        age, stage, pid = got
        if stage == "idle":
            log("runner reports idle (program finished); watchdog exiting")
            return
        alive = pid_alive(pid)
        if alive:
            if age >= STALE_S:
                log(f"heartbeat stale {age / 60:.0f} min but the process is alive; the heaviest cards update "
                    "their heartbeat only at ten-percent chunks, so this is logged, never acted on - a second "
                    "concurrent run would race the checkpoints")
            continue
        reason = "process dead"
        now_entries = ledger_entries()
        if relaunches >= MAX_RELAUNCHES or (relaunches > 0 and now_entries <= entries_at_last_relaunch):
            HALT.write_text(json.dumps({"halted": datetime.now().isoformat(timespec="seconds"), "reason": reason,
                                        "relaunches": relaunches, "ledger_entries": now_entries,
                                        "note": "no progress between relaunches or the cap reached; a deterministic failure needs a person"},
                                       indent=2), encoding="utf-8", newline="\n")
            log(f"HALTED: {reason}; relaunches {relaunches}, entries {now_entries}")
            return
        pid2 = relaunch()
        relaunches += 1
        entries_at_last_relaunch = now_entries
        log(f"{reason}; relaunched as pid {pid2} (relaunch {relaunches}/{MAX_RELAUNCHES}, entries {now_entries})")
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
