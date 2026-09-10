import copy
import pytest
from ghostscale.validation.soundingline.v16.records import write, read, file_digest
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.operational_queue import (
    validate_plan, freeze_plan, queue_state, record_failure, repair_state, dispatchable, verify_closeout, PROOF_FIELDS)


def fixture(tmp_path):
    fixture_campaign(tmp_path)
    return tmp_path


def jobs():
    return {"jobs": [
        {"job_id": "scout", "stage": "discovery", "dependencies": [], "unit_cap": 64,
         "admission": "scout-admission.json", "completion": "scout-completion.json", "contribution": "resolve rival"},
        {"job_id": "confirmation", "stage": "confirmation", "dependencies": ["scout"], "unit_cap": 256,
         "admission": "confirmation-admission.json", "completion": "confirmation-completion.json", "contribution": "fresh frozen claim"}]}


def test_queue_rejects_unbounded_duplicate_missing_and_cyclic_jobs():
    original = jobs()
    for bad in [{"jobs": []}, {"jobs": original["jobs"]*2}]:
        with pytest.raises(ValueError):
            validate_plan(bad)
    for key,value in [("unit_cap",0),("dependencies",["missing"]),("admission","../elsewhere.json")]:
        bad = copy.deepcopy(original)
        bad["jobs"][0][key] = value
        with pytest.raises(ValueError):
            validate_plan(bad)
    bad = copy.deepcopy(original)
    bad["jobs"][0]["dependencies"] = ["confirmation"]
    with pytest.raises(ValueError, match="cyclic"):
        validate_plan(bad)


def test_queue_requires_real_dependencies_and_empty_queue_is_not_closeout(tmp_path):
    root = fixture(tmp_path)
    plan = jobs()
    freeze_plan(root, plan)
    frozen = (root/"operations/QUEUE_PLAN.json").read_bytes()
    write(root/"confirmation-admission.json", {"instrument_state": "valid"})
    assert queue_state(root, plan)["eligible"] == []
    write(root/"scout-admission.json", {"instrument_state": "valid"})
    assert queue_state(root, plan)["eligible"] == ["scout"]
    write(root/"scout-completion.json", {"execution_state": "completed", "instrument_state": "valid"})
    assert queue_state(root, plan)["eligible"] == ["confirmation"]
    write(root/"confirmation-completion.json", {"execution_state": "completed", "instrument_state": "valid"})
    result = queue_state(root, plan)
    assert result["eligible"] == [] and result["campaign_complete"] is False
    assert (root/"operations/QUEUE_PLAN.json").read_bytes() == frozen
    changed = copy.deepcopy(plan)
    changed["jobs"][1]["unit_cap"] += 1
    with pytest.raises(ValueError):
        freeze_plan(root, changed)


def test_completed_receipt_cannot_hide_failed_dependency(tmp_path):
    root = fixture(tmp_path)
    plan = jobs()
    write(root/"scout-completion.json", {"execution_state": "completed", "instrument_state": "failed"})
    write(root/"confirmation-completion.json", {"execution_state": "completed", "instrument_state": "valid"})
    result = queue_state(root, plan)
    assert result["jobs"][1]["state"] == "invalid dependency completion"
    assert not result["eligible"]


def test_three_same_root_failures_quarantine_without_resetting_clock(tmp_path):
    root = fixture(tmp_path)
    clock = (root/"CAMPAIGN.json").read_bytes()
    for expected in ["failed", "failed", "quarantined"]:
        result = record_failure(root, "scout", "worker exited", root_cause="same worker defect", family="fixture")
        assert result["execution_state"] == expected
    assert queue_state(root, jobs())["jobs"][0]["state"] == "quarantined"
    assert (root/"CAMPAIGN.json").read_bytes() == clock
    assert len(list((root/"operations/failures").glob("*.json"))) == 3


def test_existing_repair_is_counted_and_second_distinct_defect_closes_family(tmp_path):
    root = fixture(tmp_path)
    write(root/"repairs/first/PLAN.json", {"repair_id": "first", "family": "R", "root_cause": "first defect"})
    assert repair_state(root, "R")["remaining_repairs"] == 0
    record_failure(root, "inquiry", "retained original", root_cause="first defect", family="R", kind="instrument")
    assert repair_state(root, "R")["state"] == "repair exhausted"
    record_failure(root, "inquiry", "newly broken", root_cause="second defect", family="R", kind="instrument")
    assert repair_state(root, "R")["state"] == "closed"


def test_measured_forecast_respects_original_remaining_deadline(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    write(root/"timing.json", {"observed_seconds": 1})
    forecast = {"kind": "measured forecast", "measured_seconds": 1, "conservative_seconds": 100,
        "closeout_reserve_seconds": 20, "source_path": "timing.json", "source_sha256": file_digest(root/"timing.json")}
    monkeypatch.setattr("ghostscale.validation.soundingline.v16.operational_queue.remaining_seconds", lambda root: 119)
    assert not dispatchable(root, {"job_id": "bounded"}, forecast)["eligible_within_deadline"]
    forecast["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="source changed"):
        dispatchable(root, {"job_id": "bounded"}, forecast)


def test_closeout_requires_all_four_actual_proof_kinds_and_bound_bytes(tmp_path):
    root = fixture(tmp_path)
    write(root/"COMMISSION_MANIFEST.json", {"cards": [{"card_id": "A"}]})
    write(root/"card.json", {"execution_state": "completed", "instrument_state": "valid"})
    inputs = {"cards": [{"card_id": "A", "execution_state": "completed", "instrument_state": "valid",
        "scientific_criterion": "null", "evidence": [{"path": "card.json", "sha256": file_digest(root/"card.json")}]}],
        "expansions": [{"card_id": "A", "state": "exhausted", "reason": "frozen boundary ladder exhausted"}], "proofs": {}}
    assert not verify_closeout(root, inputs)["campaign_closed"]
    for name,field in PROOF_FIELDS.items():
        write(root/(name+".json"), {"execution_state": "completed", "instrument_state": "valid", field: True})
        inputs["proofs"][name] = {"path": name+".json", "sha256": file_digest(root/(name+".json"))}
    assert verify_closeout(root, inputs)["campaign_closed"]
    # A valid integrity receipt cannot stand in for scientific world replay.
    write(root/"integrity.json", {"execution_state": "completed", "instrument_state": "valid", "whole_unit_replay": False})
    inputs["proofs"]["scientific_replay"] = {"path": "integrity.json", "sha256": file_digest(root/"integrity.json")}
    assert not verify_closeout(root, inputs)["campaign_closed"]
    inputs["cards"][0]["evidence"][0]["sha256"] = "changed"
    with pytest.raises(ValueError):
        verify_closeout(root, inputs)
