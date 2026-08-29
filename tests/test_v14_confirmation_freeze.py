"""The confirmation freeze must bind on every entry path (GHOST_SCALE_AGENT_HARDENING.md, H1).

The reproduced fault: the runner recorded the frozen candidate ids and their discovery hashes,
then recomputed the promoted set on resume and ran *that*, leaving the recorded freeze
untouched. A card promoted after the freeze therefore executed on the confirmation lineage
without any amendment.

These tests exercise the packet logic against temporary fixtures. No manifest is loaded, no
verdict is written, no worker pool is created and no confirmation world is sampled: every test
here operates on dictionaries and tmp_path files.

Also covered: the atomic publication helper the runners now use for working state (H4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RC = pytest.importorskip("runners.run_v14_confirmation")
from ghostscale.validation.soundingline.v14.atomicio import write_json_atomic  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: a two-card manifest and a matching frozen packet, all on tmp_path.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _never_touch_real_results(tmp_path, monkeypatch):
    """Redirect every real-results writer this module can reach onto tmp_path.

    Writing to ``results/v14/`` from a test would inject fixture data into the scientific
    record: an earlier draft of these tests did exactly that, leaving a fabricated amendment
    (with placeholder lock hashes) in ``results/v14/AMENDMENTS.json``. Autouse, so a test added
    later cannot forget it.
    """
    monkeypatch.setattr(RC, "LEDGER", tmp_path / "CONFIRMATION.json")
    recorded = []
    monkeypatch.setattr(RC.M, "add_amendment", lambda *a, **k: recorded.append(a))
    return recorded



#: The fields ``Card`` requires positionally. Filled with placeholders so the fixture builds a
#: real Card without pulling in the 152-card manifest.
_REQUIRED = {"construction": "constructed", "target": "t", "estimand": "e",
             "null_expectation": "chance", "alternative_expectation": "above chance",
             "strongest_rival": "a simpler reader"}


def _doc():
    return {"cards": [
        {"id": "C04", "trunk": "C", "status": "LANDED", "question": "self privilege",
         "lanes": ["discovery", "confirmation"], "claim_ceiling": "CONSTRUCTED_MECHANISM",
         "wave": 2, **_REQUIRED},
        {"id": "O02", "trunk": "O", "status": "LANDED", "question": "factored cost reader",
         "lanes": ["discovery", "confirmation"], "claim_ceiling": "METHOD", "wave": 2, **_REQUIRED},
        {"id": "G01", "trunk": "G", "status": "LANDED", "question": "stance",
         "lanes": ["discovery", "confirmation"], "claim_ceiling": "METHOD", "wave": 3, **_REQUIRED},
    ]}


@pytest.fixture()
def frozen(tmp_path, monkeypatch):
    """A packet frozen over C04 and O02 while G01 is *also* currently promotable.

    G01 is the whole point: it stands in for a card the healing pass promotes after the freeze.
    """
    disc = tmp_path / "discovery"
    disc.mkdir()
    for cid in ("C04", "O02", "G01"):
        (disc / f"{cid}.json").write_text(json.dumps({"card": cid, "state": "LANDED"}), encoding="utf-8")

    monkeypatch.setattr(RC, "_discovery_path", lambda cid: disc / f"{cid}.json")
    monkeypatch.setattr(RC, "_lock_identity",
                        lambda: {"structural_lock_sha256": "struct-aaa", "scientific_lock_sha256": "sci-bbb"})

    doc = _doc()
    packet = RC.make_packet(doc, ["C04", "O02"])
    return doc, packet, disc


# --------------------------------------------------------------------------- #
# Acceptance: unchanged packet resumes; a proper subset is accepted.
# --------------------------------------------------------------------------- #
def test_unchanged_packet_verifies_and_resumes(frozen):
    doc, packet, _ = frozen
    assert RC.verify_packet(doc, packet) == []
    assert RC.resolve_ids(packet, None) == ["C04", "O02"]


def test_proper_subset_is_accepted(frozen):
    _, packet, _ = frozen
    assert RC.resolve_ids(packet, ["O02"]) == ["O02"]
    assert RC.resolve_ids(packet, ["O02", "C04"]) == ["O02", "C04"]   # caller order preserved


def test_repeated_ids_collapse(frozen):
    _, packet, _ = frozen
    assert RC.resolve_ids(packet, ["C04", "C04"]) == ["C04"]


# --------------------------------------------------------------------------- #
# The reproduced fault: a card promoted after the freeze must not execute.
# --------------------------------------------------------------------------- #
def test_resume_runs_the_frozen_packet_not_todays_promotion(frozen):
    doc, packet, _ = frozen
    assert RC.resolve_ids(packet, None) == ["C04", "O02"], "an automatic resume must not pick up G01"


def test_card_outside_the_packet_is_refused(frozen):
    _, packet, _ = frozen
    with pytest.raises(RC.FreezeViolation) as e:
        RC.resolve_ids(packet, ["G01"])
    assert "G01" in str(e.value)


def test_mixed_subset_is_refused_whole(frozen):
    """One extra id poisons the request; the legal half must not run anyway."""
    _, packet, _ = frozen
    with pytest.raises(RC.FreezeViolation):
        RC.resolve_ids(packet, ["C04", "G01"])


def test_unknown_id_is_refused(frozen):
    _, packet, _ = frozen
    with pytest.raises(RC.FreezeViolation):
        RC.resolve_ids(packet, ["ZZ99"])


def test_empty_subset_is_deliberate_not_expansion(frozen):
    """``--only`` with no names must not fall through to today's promotion result."""
    _, packet, _ = frozen
    with pytest.raises(RC.FreezeViolation):
        RC.resolve_ids(packet, [])


# --------------------------------------------------------------------------- #
# Identity verification.
# --------------------------------------------------------------------------- #
def test_mutated_discovery_bytes_fail(frozen):
    doc, packet, disc = frozen
    (disc / "C04.json").write_text(json.dumps({"card": "C04", "state": "LANDED", "tampered": True}), encoding="utf-8")
    problems = RC.verify_packet(doc, packet)
    assert any("C04" in p and "bytes changed" in p for p in problems), problems


def test_missing_discovery_verdict_fails(frozen):
    doc, packet, disc = frozen
    (disc / "O02.json").unlink()
    assert any("O02" in p and "missing" in p for p in RC.verify_packet(doc, packet))


def test_changed_criteria_identity_fails(frozen):
    """A re-specified card definition is an amendment, not a resume."""
    doc, packet, _ = frozen
    doc["cards"][0]["question"] = "a different question"
    assert any("C04" in p and "changed since the freeze" in p for p in RC.verify_packet(doc, packet))


def test_working_status_alone_does_not_break_the_packet(frozen):
    """Status moves as the program runs; it is not scientific content."""
    doc, packet, _ = frozen
    for d in doc["cards"]:
        d["status"] = "RUNNING"
    assert RC.verify_packet(doc, packet) == []


def test_changed_structural_lock_fails(frozen, monkeypatch):
    doc, packet, _ = frozen
    monkeypatch.setattr(RC, "_lock_identity",
                        lambda: {"structural_lock_sha256": "struct-CHANGED", "scientific_lock_sha256": "sci-bbb"})
    assert any("structural lock changed" in p for p in RC.verify_packet(doc, packet))


def test_changed_scientific_lock_fails(frozen, monkeypatch):
    doc, packet, _ = frozen
    monkeypatch.setattr(RC, "_lock_identity",
                        lambda: {"structural_lock_sha256": "struct-aaa", "scientific_lock_sha256": "sci-CHANGED"})
    assert any("scientific lock changed" in p for p in RC.verify_packet(doc, packet))


def test_missing_legacy_field_is_unverifiable_never_a_match(frozen):
    """A packet written by an older schema must not be silently blessed as verified."""
    doc, packet, _ = frozen
    legacy = {k: v for k, v in packet.items() if k not in ("card_identity", "structural_lock_sha256")}
    problems = RC.verify_packet(doc, legacy)
    assert any("unverifiable" in p for p in problems), problems


def test_resolved_cards_are_still_verified(frozen):
    """Verification does not skip a card merely because it is already marked resolved."""
    doc, packet, disc = frozen
    (disc / "C04.json").write_text(json.dumps({"tampered": True}), encoding="utf-8")
    assert any("C04" in p for p in RC.verify_packet(doc, packet))


def test_empty_packet_is_refused(frozen):
    doc, _, _ = frozen
    assert RC.verify_packet(doc, {"promoted": [], "schema": 1}) != []


# --------------------------------------------------------------------------- #
# A rejection must not touch packet or result files.
# --------------------------------------------------------------------------- #
def test_rejection_leaves_the_ledger_byte_identical(tmp_path, frozen, monkeypatch):
    """A refusal is inert: no packet change, no ledger write, no experimental work."""
    doc, packet, _ = frozen
    ledger_path = tmp_path / "CONFIRMATION.json"
    write_json_atomic(ledger_path, {"cards": {}, "frozen": packet})
    before = ledger_path.read_bytes()
    monkeypatch.setattr(RC, "LEDGER", ledger_path)

    def explode(*a, **k):                                    # any worker use is a test failure
        raise AssertionError("a refused confirmation must not create experimental work")

    monkeypatch.setattr(RC, "promoted", explode)
    with pytest.raises(RC.FreezeViolation):
        RC.run(doc, pool=None, wait_for_freeze=False, only=["G01"], amend=False)
    assert ledger_path.read_bytes() == before


def test_first_freeze_is_written_before_any_work(tmp_path, frozen, monkeypatch):
    """The packet must be on disk before the first confirmation world is touched."""
    doc, _, _ = frozen
    ledger_path = tmp_path / "CONFIRMATION.json"
    monkeypatch.setattr(RC, "LEDGER", ledger_path)
    monkeypatch.setattr(RC, "promoted", lambda d: ["C04", "O02"])
    monkeypatch.setattr(RC, "select_candidates", lambda d: {"promoted": ["C04", "O02"], "eligible": ["C04", "O02"], "by_flight": {}, "cap": 4, "rule": "test"})

    seen = {}

    def fake_run_card(*a, **k):
        seen["frozen_on_disk"] = json.loads(ledger_path.read_text(encoding="utf-8"))["frozen"]["promoted"]
        raise RuntimeError("stop after the first card")

    import runners.run_v14 as R
    monkeypatch.setattr(R, "run_card", fake_run_card)
    monkeypatch.setattr(R, "record_runtime", lambda *a, **k: None)
    monkeypatch.setattr(R, "tier_for", lambda d: ("T2", {"confirmation_worlds": 1, "repeats": 1}))

    RC.run(doc, pool=None, wait_for_freeze=False, only=None)
    assert seen["frozen_on_disk"] == ["C04", "O02"]


# --------------------------------------------------------------------------- #
# Atomic publication (H4) for the writers outside the structural lock.
# --------------------------------------------------------------------------- #
def test_atomic_write_replaces_whole_file(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": 1})
    write_json_atomic(p, {"a": 2, "b": [1, 2, 3]})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2, "b": [1, 2, 3]}


def test_unserialisable_payload_leaves_the_old_file_intact(tmp_path):
    """Serialise before truncating: a bad payload must not destroy good state."""
    p = tmp_path / "x.json"
    write_json_atomic(p, {"good": True})
    before = p.read_bytes()

    class Boom:
        def __repr__(self):
            raise ValueError("nope")

    with pytest.raises(Exception):
        write_json_atomic(p, {"bad": Boom()}, default=lambda o: (_ for _ in ()).throw(TypeError("x")))
    assert p.read_bytes() == before


def test_no_temp_files_survive(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": 1})
    assert [q.name for q in tmp_path.iterdir()] == ["x.json"]


def test_atomic_write_is_lf(tmp_path):
    """Committed files are LF; a CRLF working copy breaks the ledger hash across clones."""
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": 1, "b": 2})
    assert b"\r\n" not in p.read_bytes()


# --------------------------------------------------------------------------- #
# Recorded amendment: the healing plan's confirmation step is permitted, never silent.
# The curator kept HEALING_PLAN.md over a strict refusal (2026-08-28).
# --------------------------------------------------------------------------- #
def test_amendment_adds_the_card_and_preserves_the_original(frozen):
    doc, packet, _ = frozen
    new, rec = RC.amend_packet(doc, packet, ["G01"], "healing pass promoted G01")

    assert new["promoted"] == ["C04", "O02", "G01"]
    assert rec["added"] == ["G01"]
    assert rec["original_packet"]["promoted"] == ["C04", "O02"], "the original must survive verbatim"
    assert rec["reason"] == "healing pass promoted G01"
    assert new["amendment_count"] == 1
    from ghostscale.validation.soundingline.v14 import common as C
    assert new["amended_from"] == C.obj_sha(packet), "the replacement names the packet it replaced"


def test_amended_packet_lets_the_added_card_run(frozen):
    doc, packet, _ = frozen
    new, _ = RC.amend_packet(doc, packet, ["G01"], "r")
    assert RC.verify_packet(doc, new) == []
    assert RC.resolve_ids(new, ["G01"]) == ["G01"]
    assert RC.resolve_ids(new, None) == ["C04", "O02", "G01"]


def test_amendment_keeps_the_original_hashes_of_frozen_cards(frozen):
    """A card that drifted since the freeze must not be re-blessed by amending in a new card."""
    doc, packet, disc = frozen
    (disc / "C04.json").write_text(json.dumps({"tampered": True}), encoding="utf-8")
    new, _ = RC.amend_packet(doc, packet, ["G01"], "r")
    assert new["discovery_hashes"]["C04"] == packet["discovery_hashes"]["C04"]
    assert any("C04" in p and "bytes changed" in p for p in RC.verify_packet(doc, new))


def test_amendment_chain_is_cumulative(frozen):
    doc, packet, _ = frozen
    a, _ = RC.amend_packet(doc, packet, ["G01"], "first")
    b, rec = RC.amend_packet(doc, a, [], "second")
    assert b["amendment_count"] == 2
    assert rec["original_packet"]["promoted"] == ["C04", "O02", "G01"]


def test_added_card_gets_an_untouched_lineage(frozen):
    """Adding a card must not reuse or perturb another card's confirmation worlds."""
    from ghostscale.validation.soundingline.v14 import common as C
    seeds = {cid: C.seed(f"confirmation|{cid}|w1000|r0|") for cid in ("C04", "O02", "G01")}
    assert len(set(seeds.values())) == 3, seeds


def test_no_amend_still_refuses(frozen, tmp_path, monkeypatch):
    """--no-amend keeps the strict behaviour and writes nothing."""
    doc, packet, _ = frozen
    ledger_path = tmp_path / "CONFIRMATION.json"
    write_json_atomic(ledger_path, {"cards": {}, "frozen": packet})
    before = ledger_path.read_bytes()
    monkeypatch.setattr(RC, "LEDGER", ledger_path)
    with pytest.raises(RC.FreezeViolation):
        RC.run(doc, pool=None, wait_for_freeze=False, only=["G01"], amend=False)
    assert ledger_path.read_bytes() == before


def test_amendment_is_recorded_before_any_work(tmp_path, frozen, monkeypatch):
    """The amendment must be durable on disk before the added card is executed."""
    doc, packet, _ = frozen
    ledger_path = tmp_path / "CONFIRMATION.json"
    write_json_atomic(ledger_path, {"cards": {}, "frozen": packet})
    monkeypatch.setattr(RC, "LEDGER", ledger_path)
    monkeypatch.setattr(RC.M, "add_amendment", lambda *a, **k: None)

    seen = {}

    def fake_run_card(*a, **k):
        d = json.loads(ledger_path.read_text(encoding="utf-8"))
        seen["promoted"] = d["frozen"]["promoted"]
        seen["amendments"] = len(d.get("amendments", []))
        raise RuntimeError("stop")

    import runners.run_v14 as R
    monkeypatch.setattr(R, "run_card", fake_run_card)
    monkeypatch.setattr(R, "record_runtime", lambda *a, **k: None)
    monkeypatch.setattr(R, "tier_for", lambda d: ("T2", {"confirmation_worlds": 1, "repeats": 1}))

    RC.run(doc, pool=None, wait_for_freeze=False, only=["G01"], amend=True)
    assert seen["promoted"] == ["C04", "O02", "G01"]
    assert seen["amendments"] == 1


def test_atomic_write_is_byte_neutral_with_platform_newlines(tmp_path):
    """``newline=None`` must reproduce the previous ``write_text`` bytes exactly.

    The structural and workload locks hash the *working copy* of EXPECTED_CELLS.json and
    EXPECTED_CELLS_TEMPLATE.json, which is CRLF on Windows. Making the atomic writers emit LF
    would silently move a hash the locks already recorded, so the atomicity fix is required to
    be byte-neutral at those call sites.
    """
    doc = {"a": 1, "b": {"c": [1, 2, 3]}, "d": "x"}
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps(doc, indent=2), encoding="utf-8")       # the previous call, verbatim
    write_json_atomic(new, doc, newline=None)
    assert new.read_bytes() == old.read_bytes()


def test_manifest_writers_are_byte_neutral():
    """Guard the real call sites, not just the helper."""
    import inspect
    from ghostscale.validation.soundingline.v14 import manifest as M
    for fn in (M.save_manifest, M.write_manifest, M.write_cells_template,
               M.instantiate_cells, M.write_coverage):
        src = inspect.getsource(fn)
        assert "newline=None" in src, f"{fn.__name__} must preserve the file's existing byte format"


# --------------------------------------------------------------------------- #
# Supersession: a corrected discovery verdict leaves the packet through a record, then re-enters
# through an ordinary amendment carrying its new hash.
# --------------------------------------------------------------------------- #
def test_supersede_removes_the_card_and_records_it(frozen):
    doc, packet, disc = frozen
    ledger = {"cards": {"C04": {"state": "LANDED"}, "O02": {"state": "LANDED"}}, "frozen": packet}
    record = RC.supersede(ledger, ["C04"], "instrument corrected; discovery verdict superseded")
    assert ledger["frozen"]["promoted"] == ["O02"]
    assert "C04" not in ledger["frozen"]["discovery_hashes"] and "C04" not in ledger["frozen"]["card_identity"]
    assert "C04" not in ledger["cards"] and ledger["superseded"]["C04"] == {"state": "LANDED"}
    assert record["removed"] == ["C04"] and record["original_packet"]["promoted"] == ["C04", "O02"]
    assert ledger["amendments"][-1] is record and ledger["frozen"]["amendment_count"] == 1


def test_superseded_card_re_enters_with_its_new_hash(frozen):
    doc, packet, disc = frozen
    ledger = {"cards": {}, "frozen": packet}
    RC.supersede(ledger, ["C04"], "corrected")
    (disc / "C04.json").write_text(json.dumps({"card": "C04", "state": "LANDED", "v": 2}), encoding="utf-8")
    assert RC.verify_packet(doc, ledger["frozen"]) == []          # the drifted card is no longer verified: it left the packet
    new_packet, rec = RC.amend_packet(doc, ledger["frozen"], ["C04"], "healing")
    assert rec["added"] == ["C04"]
    assert new_packet["discovery_hashes"]["C04"] == RC.C.file_sha(disc / "C04.json")
    assert new_packet["discovery_hashes"]["O02"] == packet["discovery_hashes"]["O02"]


def test_supersede_refuses_a_card_outside_the_packet(frozen):
    doc, packet, disc = frozen
    ledger = {"cards": {}, "frozen": packet}
    with pytest.raises(RC.FreezeViolation):
        RC.supersede(ledger, ["G01"], "not in the packet")
    assert ledger["frozen"] == packet and "amendments" not in ledger
