"""V15 gate discipline. A gate records; this suite fails.

The rule this file exists to enforce is the one V14 paid for twice: **a gate bar is never a
criterion bar**. V14 put a criterion's magnitude into twenty-six live gates, and the consequence
was that three small *real* effects were recorded as INSTRUMENT_FAILED at scale -- a gate demanding
the criterion's magnitude fails exactly when the finding is a modest true positive. V15's trunk M
then reproduced the same error in a second form, putting the criterion's *direction* into a
positive gate, so that a card discovering a negative answer would be filed as a broken instrument.

Both forms are checked here: no live or no-oracle gate may carry a nonzero expected value, and no
committed verdict may be INSTRUMENT_FAILED because of a gate whose name matches its own criterion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V15 = REPO / "results" / "validation" / "soundingline" / "v15"
LANES = ("", "transfer", "attacks", "confirmation")


def _verdicts():
    out = []
    for lane in LANES:
        d = V15 / lane if lane else V15
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                out.append((p, json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def test_manifest_matches_the_spec_counts():
    from ghostscale.validation.soundingline.v15 import manifest as M
    cards = M.build_cards()
    assert len(M.mandatory(cards)) == 112
    assert len(M.attacks(cards)) == 24
    counts = {}
    for c in M.mandatory(cards):
        counts[c.trunk] = counts.get(c.trunk, 0) + 1
    assert counts == {"I": 8, "C": 14, "M": 12, "E": 12, "G": 10, "V": 10, "S": 10, "R": 8,
                      "F": 10, "H": 8, "P": 8, "B": 2}


def test_every_causal_card_declares_a_hidden_event():
    from ghostscale.validation.soundingline.v15 import manifest as M
    from ghostscale.validation.soundingline.v15.schemas import ENDPOINTS
    for c in M.mandatory(M.build_cards()):
        if not c.causal:
            continue
        assert c.endpoints, f"{c.id} is causal and declares no endpoint"
        for e in c.endpoints:
            assert e in ENDPOINTS, f"{c.id} declares an unknown endpoint {e!r}"


def test_every_card_states_the_basis_of_its_effect_size():
    from ghostscale.validation.soundingline.v15 import manifest as M
    for c in M.build_cards():
        assert c.sesoi_basis, f"{c.id} has a smallest effect of interest with no stated basis"


def test_battery_cannot_carry_a_magnitude():
    """``battery`` has no parameter through which a criterion bar could reach a gate."""
    import inspect

    from ghostscale.validation.soundingline.v15.cards import battery
    src = inspect.getsource(battery)
    assert "min_change=0.0" in src
    sig = inspect.signature(battery)
    for name in ("live", "prediction"):
        assert name in sig.parameters
    # the only place a magnitude may appear is the criterion helper
    assert "min_change=float" not in src


def test_no_committed_gate_carries_a_nonzero_bar():
    bad = []
    for p, v in _verdicts():
        for g in (v.get("gates") or {}).get("gates", []):
            if g.get("kind") in ("live", "no_oracle") and abs(float(g.get("expected") or 0.0)) > 1e-12:
                bad.append(f"{p.name}:{g.get('name')} expected={g.get('expected')}")
    assert not bad, ("a live or no-oracle gate is carrying a magnitude; that belongs in the "
                     f"criterion: {bad[:8]}")


def test_state_and_criterion_status_are_separate_columns():
    for p, v in _verdicts():
        assert "state" in v, f"{p.name} has no state"
        assert "criterion_status" in v, f"{p.name} has no criterion status"
        if v["state"] == "LANDED":
            assert v["criterion_status"] != "UNEVALUATED", \
                f"{p.name} landed without evaluating a criterion"


def test_no_verdict_uses_the_forbidden_done_state():
    for p, v in _verdicts():
        assert v.get("state") != "DONE", f"{p.name} uses DONE, which is not a state"


def test_committed_verdicts_have_no_failed_gates():
    bad = []
    for p, v in _verdicts():
        g = v.get("gates") or {}
        if g.get("failed_names"):
            bad.append(f"{p.name}: {g['failed_names']}")
        if g.get("unexpected_passes"):
            bad.append(f"{p.name}: unexpected passes {g['unexpected_passes']}")
    assert not bad, bad[:8]


def test_a_simulator_discovery_cleared_the_causal_distance_audit():
    bad = []
    for p, v in _verdicts():
        cd = v.get("causal_distance") or {}
        if v.get("claim_class") == "SIMULATOR_DISCOVERY" and cd \
                and not cd.get("promotable_as_discovery", True):
            bad.append(f"{p.name}: {cd.get('limiting_distance')}")
    assert not bad, ("a card claims SIMULATOR_DISCOVERY while its limiting causal distance is a "
                     f"readout or a planted signature: {bad}")


def test_causal_distance_fixtures_classify_as_declared():
    from ghostscale.validation.soundingline.v15 import causal_distance as CD
    a = CD.audit_fixtures()
    assert a["all_ok"], a["rows"]
