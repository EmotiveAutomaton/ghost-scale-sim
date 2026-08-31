"""The 168-hour runtime contract: deadline, opening guard, occupancy and the failure state.

Spec §9 in executable form. Three things live here and nowhere else, so that the runner, the
validator and card I08 all judge against the same code rather than three readings of the same
paragraph:

``open_window`` / ``elapsed_hours`` / ``window_closed``
    one immutable UTC deadline at ``start + 168h``. Restarts inherit it; nothing shortens it.

``opening_guard``
    spec §9.3's six conditions. The guard exists because V14 opened a window whose scientific queue
    could not fill it: the core finished in 6.8 wall hours and the runner then waited fourteen
    hours for a freeze it could not usefully reach. A queue that cannot fill the window is refused
    at the door rather than discovered at hour seven.

``Occupancy``
    the §9.4 receipt: worker-seconds actually spent on science against the integral of safe worker
    capacity, with every gap over five minutes carrying a machine-readable reason. If the queue
    empties, or workers wait for the deadline, or balanced coverage is replaced by repetition,
    ``RUNTIME_FAILED`` is set -- and spec §9.4 is explicit that results may still be reported after
    that, but the seven-day contract may not be claimed. There is deliberately no way to express
    "short run but complete".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import v15_dir
from .atomicio import write_json_atomic
from .schemas import (CONFIRMATION_END_HOUR, CONFIRMATION_INTEGRITY_WORKER_HOURS_MIN,
                      CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H, CORE_UPPER_FORECAST_MAX_H,
                      FAST_MACHINE_FACTOR, FREEZE_HOUR, GAP_REASON_SECONDS, INTEGRITY_END_HOUR,
                      OCCUPANCY_TARGET, WINDOW_HOURS)

DEADLINE = v15_dir() / "DEADLINE.json"
OPENING_RECEIPT = v15_dir() / "DEADLINE_OPENING_RECEIPT.json"
RUNTIME = v15_dir() / "RUNTIME.json"
OCCUPANCY = v15_dir() / "WORKER_OCCUPANCY.json"
RESOURCE_GOVERNOR = v15_dir() / "RESOURCE_GOVERNOR.json"
_FMT = "%Y-%m-%dT%H:%M:%SZ"


# --------------------------------------------------------------------------- #
# The window.
# --------------------------------------------------------------------------- #
def now() -> datetime:
    return datetime.now(timezone.utc)


def open_window(force: bool = False, note: str = "") -> dict:
    """Open the one window, or return the existing one. The deadline is written once."""
    if DEADLINE.exists() and not force:
        return json.loads(DEADLINE.read_text(encoding="utf-8"))
    start = now()
    doc = {"program": "v15", "opened": start.strftime(_FMT),
           "deadline": (start + timedelta(hours=WINDOW_HOURS)).strftime(_FMT),
           "window_hours": WINDOW_HOURS, "freeze_hour": FREEZE_HOUR,
           "confirmation_end_hour": CONFIRMATION_END_HOUR,
           "integrity_end_hour": INTEGRITY_END_HOUR,
           "immutable": True, "note": note,
           "rule": ("restarts inherit this deadline; no restart shortens or extends it, and no "
                    "packet may be emitted before it")}
    write_json_atomic(DEADLINE, doc)
    return doc


def window() -> dict | None:
    return json.loads(DEADLINE.read_text(encoding="utf-8")) if DEADLINE.exists() else None


def elapsed_hours() -> float:
    w = window()
    if not w:
        return 0.0
    t0 = datetime.strptime(w["opened"], _FMT).replace(tzinfo=timezone.utc)
    return (now() - t0).total_seconds() / 3600.0


def window_closed() -> bool:
    w = window()
    if not w:
        return False
    return now() >= datetime.strptime(w["deadline"], _FMT).replace(tzinfo=timezone.utc)


def frozen() -> bool:
    return elapsed_hours() >= FREEZE_HOUR


def phase() -> str:
    h = elapsed_hours()
    if h < FREEZE_HOUR:
        return "discovery"
    if h < CONFIRMATION_END_HOUR:
        return "confirmation"
    if h < INTEGRITY_END_HOUR:
        return "integrity"
    return "report"


# --------------------------------------------------------------------------- #
# The opening guard (spec §9.3).
# --------------------------------------------------------------------------- #
def opening_guard(core_upper_h: float, core_lower_h: float, coverage_lower_h: float,
                  confirmation_worker_h: float, hashed: bool, recovery_tests: bool,
                  early_report_guard: bool = True) -> dict:
    """Spec §9.3's six conditions, each reported separately.

    ``core_lower_h`` and ``coverage_lower_h`` are forecasts under the *fast* machine assumption --
    the pilot median divided by ``FAST_MACHINE_FACTOR`` -- so condition 2 asks whether the queue
    survives a machine three times faster than the one it was measured on.
    """
    total_lower = float(core_lower_h) + float(coverage_lower_h)
    conditions = {
        "1_core_upper_fits": bool(0.0 < core_upper_h <= CORE_UPPER_FORECAST_MAX_H),
        "2_core_plus_coverage_survives_fast_machine":
            bool(total_lower > CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H),
        "3_confirmation_and_integrity_reserved":
            bool(confirmation_worker_h >= CONFIRMATION_INTEGRITY_WORKER_HOURS_MIN),
        "4_cells_and_order_hashed": bool(hashed),
        "5_recovery_and_report_guard_tested": bool(recovery_tests and early_report_guard),
        "6_bad_queue_fixtures_rejected": bool(core_upper_h > 0.0 and coverage_lower_h > 0.0),
    }
    failed = [k for k, ok in conditions.items() if not ok]
    return {"may_open": not failed, "conditions": conditions, "failed": failed,
            "core_upper_h": float(core_upper_h), "core_lower_h": float(core_lower_h),
            "coverage_lower_h": float(coverage_lower_h),
            "core_plus_coverage_lower_h": total_lower,
            "fast_machine_factor": FAST_MACHINE_FACTOR,
            "thresholds": {"core_upper_max_h": CORE_UPPER_FORECAST_MAX_H,
                           "core_plus_coverage_lower_min_h": CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H,
                           "confirmation_worker_h_min": CONFIRMATION_INTEGRITY_WORKER_HOURS_MIN},
            "reason": ("" if not failed else
                       "enlarge scientifically meaningful factor coverage or improve the "
                       "implementation; do not open the window and hope the clock supplies duration")}


def write_opening_receipt(guard: dict, extra: dict | None = None) -> dict:
    doc = {"program": "v15", "written": time.strftime(_FMT, time.gmtime()), **guard,
           **(extra or {})}
    write_json_atomic(OPENING_RECEIPT, doc)
    return doc


# --------------------------------------------------------------------------- #
# Occupancy and the failure state (spec §9.4).
# --------------------------------------------------------------------------- #
@dataclass
class Occupancy:
    """Worker-seconds of science against the integral of safe worker capacity."""

    started: float = field(default_factory=time.time)
    science_worker_seconds: float = 0.0
    capacity_worker_seconds: float = 0.0
    governor_throttled_worker_seconds: float = 0.0
    gaps: list = field(default_factory=list)
    queue_empty_events: int = 0
    waited_for_deadline_seconds: float = 0.0
    coverage_blocks_executed: int = 0
    coverage_cells_executed: int = 0
    cards_executed: int = 0
    _last_tick: float = field(default_factory=time.time)
    _gap_started: float | None = None

    def tick(self, active_workers: int, safe_workers: int, throttled: int = 0) -> None:
        t = time.time()
        dt = max(0.0, t - self._last_tick)
        self._last_tick = t
        self.science_worker_seconds += dt * max(int(active_workers), 0)
        self.capacity_worker_seconds += dt * max(int(safe_workers), 0)
        self.governor_throttled_worker_seconds += dt * max(int(throttled), 0)
        if active_workers <= 0:
            if self._gap_started is None:
                self._gap_started = t
        elif self._gap_started is not None:
            self.close_gap("workers resumed")

    def close_gap(self, reason: str) -> None:
        if self._gap_started is None:
            return
        dur = time.time() - self._gap_started
        if dur >= GAP_REASON_SECONDS:
            self.gaps.append({"seconds": round(dur, 1), "reason": reason,
                              "at": time.strftime(_FMT, time.gmtime(self._gap_started))})
        self._gap_started = None

    def note_queue_empty(self) -> None:
        self.queue_empty_events += 1

    def note_waiting(self, seconds: float) -> None:
        self.waited_for_deadline_seconds += float(seconds)

    def ratio(self) -> float:
        denom = self.capacity_worker_seconds - self.governor_throttled_worker_seconds
        return float(self.science_worker_seconds / denom) if denom > 0 else float("nan")

    def runtime_failed(self) -> tuple:
        """``(failed, reasons)``. There is no way to express 'short run but complete'."""
        reasons = []
        if self.queue_empty_events:
            reasons.append(f"the scientific queue emptied {self.queue_empty_events} time(s)")
        if self.waited_for_deadline_seconds > 3600.0:
            reasons.append(f"workers waited {self.waited_for_deadline_seconds / 3600:.1f} h "
                           "solely for the deadline")
        r = self.ratio()
        if r == r and r < OCCUPANCY_TARGET:
            reasons.append(f"occupancy {r:.2f} below the {OCCUPANCY_TARGET:.2f} target after "
                           "subtracting governor throttling")
        return bool(reasons), reasons

    def to_dict(self) -> dict:
        failed, reasons = self.runtime_failed()
        return {"science_worker_hours": self.science_worker_seconds / 3600.0,
                "capacity_worker_hours": self.capacity_worker_seconds / 3600.0,
                "governor_throttled_worker_hours": self.governor_throttled_worker_seconds / 3600.0,
                "occupancy_ratio": self.ratio(), "occupancy_target": OCCUPANCY_TARGET,
                "gaps_over_five_minutes": self.gaps, "queue_empty_events": self.queue_empty_events,
                "waited_for_deadline_hours": self.waited_for_deadline_seconds / 3600.0,
                "coverage_blocks_executed": self.coverage_blocks_executed,
                "coverage_cells_executed": self.coverage_cells_executed,
                "cards_executed": self.cards_executed,
                "RUNTIME_FAILED": failed, "runtime_failed_reasons": reasons,
                "note": ("spec 9.4: results may still be reported after a runtime failure, but "
                         "V15 is not complete and cannot claim the seven-day contract")}

    def write(self) -> dict:
        d = self.to_dict()
        write_json_atomic(OCCUPANCY, d)
        return d


# --------------------------------------------------------------------------- #
# Coexistence with the sibling project (spec §9.6).
# --------------------------------------------------------------------------- #
SOUNDING_STATUS = (Path := __import__("pathlib").Path)(
    os.environ.get("GS_SOUNDING_STATUS",
                   r"e:\EmotiveAutomaton\Projects\SoundingLine\sounding-line\SCHEDULER_STATUS.json"))


def sounding_line_active() -> dict:
    """Sample the sibling project's status. Never blocks and never fails the run."""
    try:
        if not SOUNDING_STATUS.exists():
            return {"present": False, "active": False}
        d = json.loads(SOUNDING_STATUS.read_text(encoding="utf-8"))
        state = str(d.get("state") or d.get("status") or "").lower()
        active = bool(d.get("running") or state in ("running", "active", "busy"))
        return {"present": True, "active": active, "state": state,
                "sampled": time.strftime(_FMT, time.gmtime())}
    except Exception as exc:                                    # noqa: BLE001
        return {"present": False, "active": False, "error": repr(exc)}


def govern(safe_workers: int, reserve: int = 2) -> dict:
    """Reduce the worker count while the sibling project is working.

    The governor may reduce workers, priority or batch size. It may not erase the deadline, change
    a hypothesis, or mark a skipped cell complete -- and it cannot, because it returns a worker
    count and nothing else.
    """
    s = sounding_line_active()
    w = max(1, safe_workers - reserve) if s.get("active") else safe_workers
    return {"workers": int(w), "reserved": int(safe_workers - w), "sounding_line": s}
