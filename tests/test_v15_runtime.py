"""The 168-hour runtime contract, tested before the clock is allowed to start (spec §9).

Every test here corresponds to a way V14 or V13 lost time:

* the opening guard exists because V14 opened a window whose queue could not fill it;
* the module-form launch exists because the sibling project's orphan sweeper killed V14's runner
  seven times;
* the early-report guard exists because a checkpoint, a dashboard or a bridge file is exactly how
  an early packet gets created without anyone deciding to create one;
* ``RUNTIME_FAILED`` has no softened form because "short run but complete" is the sentence spec
  §9.4 forbids.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostscale.validation.soundingline.v15 import runtime_contract as RC

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# The opening guard.
# --------------------------------------------------------------------------- #
def test_the_guard_refuses_a_v14_sized_queue():
    g = RC.opening_guard(core_upper_h=7, core_lower_h=6, coverage_lower_h=0,
                         confirmation_worker_h=30, hashed=True, recovery_tests=True)
    assert not g["may_open"]
    assert "2_core_plus_coverage_survives_fast_machine" in g["failed"]


def test_the_guard_refuses_an_empty_queue():
    g = RC.opening_guard(core_upper_h=0, core_lower_h=0, coverage_lower_h=0,
                         confirmation_worker_h=0, hashed=True, recovery_tests=True)
    assert not g["may_open"]


def test_the_guard_refuses_a_queue_a_fast_machine_would_empty():
    g = RC.opening_guard(core_upper_h=40, core_lower_h=12, coverage_lower_h=90,
                         confirmation_worker_h=30, hashed=True, recovery_tests=True)
    assert not g["may_open"]


def test_the_guard_admits_a_healthy_queue():
    g = RC.opening_guard(core_upper_h=90, core_lower_h=30, coverage_lower_h=400,
                         confirmation_worker_h=30, hashed=True, recovery_tests=True)
    assert g["may_open"], g["failed"]


def test_the_guard_refuses_when_confirmation_is_not_reserved():
    g = RC.opening_guard(core_upper_h=90, core_lower_h=30, coverage_lower_h=400,
                         confirmation_worker_h=1, hashed=True, recovery_tests=True)
    assert not g["may_open"]
    assert "3_confirmation_and_integrity_reserved" in g["failed"]


def test_the_guard_refuses_an_unhashed_queue():
    g = RC.opening_guard(core_upper_h=90, core_lower_h=30, coverage_lower_h=400,
                         confirmation_worker_h=30, hashed=False, recovery_tests=True)
    assert not g["may_open"]


# --------------------------------------------------------------------------- #
# Occupancy and the failure state.
# --------------------------------------------------------------------------- #
def test_an_emptied_queue_is_a_runtime_failure():
    occ = RC.Occupancy()
    occ.note_queue_empty()
    failed, reasons = occ.runtime_failed()
    assert failed and reasons


def test_waiting_for_the_deadline_is_a_runtime_failure():
    occ = RC.Occupancy()
    occ.note_waiting(7200)
    failed, reasons = occ.runtime_failed()
    assert failed and any("waited" in r for r in reasons)


def test_there_is_no_softened_form_of_a_runtime_failure():
    """``RUNTIME_FAILED`` is a boolean with reasons. There is deliberately no 'short run' field
    that a report could use to call the contract complete anyway."""
    occ = RC.Occupancy()
    occ.note_queue_empty()
    d = occ.to_dict()
    assert d["RUNTIME_FAILED"] is True
    assert "short_run" not in d
    assert "cannot claim the seven-day contract" in d["note"]


def test_occupancy_subtracts_governor_throttling():
    occ = RC.Occupancy()
    occ._last_tick -= 10.0
    occ.tick(active_workers=4, safe_workers=8, throttled=4)
    r = occ.ratio()
    assert 0.9 < r <= 1.01, r


# --------------------------------------------------------------------------- #
# The window.
# --------------------------------------------------------------------------- #
def test_the_window_constants_are_the_spec_s():
    assert RC.WINDOW_HOURS == 168.0
    assert RC.FREEZE_HOUR == 150.0
    assert RC.CONFIRMATION_END_HOUR == 166.0
    assert RC.INTEGRITY_END_HOUR == 168.0


def test_phase_moves_through_the_declared_stages():
    assert RC.phase() in ("discovery", "confirmation", "integrity", "report")


# --------------------------------------------------------------------------- #
# The launcher, and the orphan sweeper.
# --------------------------------------------------------------------------- #
def test_the_launcher_uses_module_form():
    """A `runners/run_v15.py` command line is what the sibling project's orphan sweeper matches."""
    ps1 = REPO / "runners" / "run_v15_wrapped.ps1"
    assert ps1.exists()
    text = ps1.read_text(encoding="utf-8")
    assert '"-m", "runners.run_v15"' in text or '"-m"' in text and "runners.run_v15" in text
    assert "runners\\run_v15.py" not in text and "runners/run_v15.py" not in text


def test_the_watchdog_relaunches_in_module_form():
    wd = (REPO / "runners" / "watchdog_v15.py").read_text(encoding="utf-8")
    assert '"-m", "runners.run_v15"' in wd


# --------------------------------------------------------------------------- #
# The early-report guard.
# --------------------------------------------------------------------------- #
def test_the_report_refuses_before_the_deadline(tmp_path, monkeypatch):
    import runners.report_v15 as REP
    monkeypatch.setattr(REP.RC, "window_closed", lambda: False)
    monkeypatch.setattr(REP.RC, "elapsed_hours", lambda: 3.0)
    out = REP.run(force=False, draft=False)
    assert out.get("refused") is True
    assert out.get("written") is None


def test_a_draft_never_lands_in_the_docs_tree(monkeypatch):
    import runners.report_v15 as REP
    monkeypatch.setattr(REP.RC, "window_closed", lambda: False)
    monkeypatch.setattr(REP.RC, "elapsed_hours", lambda: 3.0)
    out = REP.run(force=False, draft=True)
    assert out.get("draft") is True
    assert "docs" not in str(out.get("written"))


def test_checkpoints_carry_no_narrative():
    """Spec §9.1: machine-readable checkpoints may be written for recovery and must not contain
    narrative conclusions."""
    p = REPO / "results" / "v15" / "CHECKPOINTS.jsonl"
    if not p.exists():
        pytest.skip("no checkpoints yet")
    banned = ("we found", "suggests", "shows that", "demonstrates", "conclude")
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        low = line.lower()
        for b in banned:
            assert b not in low, f"a checkpoint contains narrative: {line[:160]}"


# --------------------------------------------------------------------------- #
# Resume and the stale checkpoint.
# --------------------------------------------------------------------------- #
def test_a_stale_checkpoint_is_recomputed_rather_than_reused():
    from ghostscale.validation.soundingline.v15 import common as C
    C.save_ckpt("smoke", "TEST", 0, 0, "hash-a", {"x": 1}, {})
    assert C.load_ckpt("smoke", "TEST", 0, 0, "hash-a") == {"x": 1}
    assert C.load_ckpt("smoke", "TEST", 0, 0, "hash-b") is None


def test_the_governor_can_only_change_the_worker_count():
    g = RC.govern(8, reserve=2)
    assert set(g) == {"workers", "reserved", "sounding_line"}
    assert 1 <= g["workers"] <= 8
