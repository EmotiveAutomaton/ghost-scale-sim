import copy
import pytest
from ghostscale.validation.soundingline.v16.archive_design import CONDITIONS, CASE_IDS
from ghostscale.validation.soundingline.v16.archive_cases import classify
from ghostscale.validation.soundingline.v16.archive_search import select, novelty, summarize
from ghostscale.validation.soundingline.v16.archive_reference import verify_case, verify_summary
from ghostscale.validation.soundingline.v16.archive_study import candidate
from ghostscale.validation.soundingline.v16.archive_interruption import control
from ghostscale.validation.soundingline.v16.selection import EXTENSION, METHODS
from ghostscale.validation.soundingline.v16.selection_gates import fixture
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import read


def test_archive_known_selection_ambiguity_and_corrected_error_cases(tmp_path):
    full, partial = fixture(full=True), fixture(full=False)
    private = {"maker": {"decoration": {"choice": 1}}}
    with ReaderProcess(tmp_path, extensions=[EXTENSION]) as reader:
        predictions = {method: reader.request(EXTENSION, full, method=method) for method in METHODS}
        reduced = reader.request(EXTENSION, partial, method="selection-aware")
    cases = classify(full, private, predictions, reduced)
    assert cases["direct-equivalence"]
    assert cases["artifact-core-ambiguity"]
    assert not cases["process-narrows-core"]
    assert cases["selection-changes-raw-prediction"]
    assert cases["naive-confident-error"]
    assert cases["rejected-work-corrects-account"]


def test_archive_semantics_reject_uuid_novelty_and_false_equivalence(tmp_path):
    cases = {key: False for key in CASE_IDS}
    cases["direct-equivalence"] = True
    assert novelty(cases, set()) == ["direct-equivalence"]
    assert not novelty(cases, {"direct-equivalence"})
    with pytest.raises(ValueError, match="unregistered"):
        novelty({**cases, "new-random-identifier": True}, set())
    public = fixture()
    with ReaderProcess(tmp_path, extensions=[EXTENSION]) as reader:
        predictions = {method: reader.request(EXTENSION, public, method=method) for method in METHODS}
    predictions["direct-table"]["future_raw_style"] = [0.0, 1.0]
    cases = classify(public, {"maker": {"decoration": {"choice": 1}}}, predictions, predictions["selection-aware"])
    assert not cases["direct-equivalence"]


def test_fixed_and_adaptive_search_have_frozen_bounded_edits():
    history = []
    for step in range(12):
        condition = select("fixed", 0, history)
        history.append({"step": step, "condition": condition["id"], "new_cases": []})
    assert {row["condition"] for row in history} == {row["id"] for row in CONDITIONS}
    history = []
    for step in range(12):
        condition = select("adaptive", 0, history)
        assert condition in CONDITIONS and condition["count"] == 4
        if step >= 3 and step % 4 != 3:
            anchor = next(row for row in CONDITIONS if row["id"] == history[0]["condition"])
            assert sum(condition[key] != anchor[key] for key in ["retention", "process", "rejected"]) == 1
        history.append({"step": step, "condition": condition["id"], "new_cases": ["direct-equivalence"] if step == 0 else []})
    assert select("adaptive", 0, history) == select("adaptive", 0, history)


def test_archive_candidate_independent_physics_and_exact_budget(tmp_path):
    base = tmp_path/"candidate"
    with ReaderProcess(tmp_path/"reader", extensions=[EXTENSION]) as reader:
        result = candidate(base, CONDITIONS[-1], 0, "archive-known-control/fixture", {"packet_hash": "fixture"}, reader, 1)
        assert candidate(base, CONDITIONS[-1], 0, "archive-known-control/fixture", {"packet_hash": "fixture"}, reader, 1) == result
    assert result["independent_check"]["physical_primitives"] == 113
    row = read(base/"units"/(result["case"]["unit_id"]+"_points.json"))
    changed = copy.deepcopy(result["case"])
    changed["cases"]["artifact-core-ambiguity"] = not changed["cases"]["artifact-core-ambiguity"]
    with pytest.raises(ValueError, match="interpretation"):
        verify_case(base, row, changed)
    changed = copy.deepcopy(row)
    changed["arms"]["selection-aware"]["outcomes"]["observed_production_primitives"] = 15
    with pytest.raises(ValueError):
        verify_case(base, changed, result["case"])


def test_archive_followup_cannot_replace_failed_first_witness():
    search = [{"method": method, "replicate": 0, "step": 0, "condition": CONDITIONS[0]["id"],
        "new_cases": ["direct-equivalence"], "candidate_receipt": method+"-first", "physical_primitives": 113,
        "separate_verification_primitives": 10} for method in ["fixed", "adaptive"]]
    followup = [{"condition": condition["id"], "cases": {key: condition != CONDITIONS[0] for key in CASE_IDS}}
                for condition in CONDITIONS for _ in range(4)]
    report = summarize(search, followup, 3)
    assert verify_summary(search, followup, report, 3)["instrument_state"] == "valid"
    assert report["methods"]["fixed"]["mean_distinct_validated_cases"] == 0
    changed = copy.deepcopy(report)
    changed["methods"]["fixed"]["replicates"][0]["witnesses"]["direct-equivalence"]["condition"] = CONDITIONS[1]["id"]
    with pytest.raises(ValueError, match="replaced"):
        verify_summary(search, followup, changed, 3)


def test_actual_archive_interrupt_resume_preserves_all_saved_work(tmp_path):
    result = control(tmp_path, "known-archive-source")
    assert result["instrument_state"] == "valid", result["checks"]
    assert control(tmp_path, "known-archive-source") == result
