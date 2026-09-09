from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v16.gates import run_gates, fixture_observation
from ghostscale.validation.soundingline.v16.records import canonical, write, digest
from ghostscale.validation.soundingline.v16.inference import read_public
from ghostscale.validation.soundingline.v16.vertical import run_case
from ghostscale.validation.soundingline.v16.reaggregate import regenerate


def test_known_answer_gates_detect_their_broken_instruments():
    receipt = run_gates()
    assert receipt["instrument_state"] == "valid", receipt


def test_real_case_is_reaggregated_and_missing_raw_fails(tmp_path):
    record = run_case(tmp_path, packet_hash="fixture-code", index=0)
    reduced = regenerate(tmp_path)
    assert reduced["n"] == 1
    assert reduced["future_log_score_sum"] == record["outcomes"]["future_log_score"]
    (tmp_path / "private" / (record["unit_id"] + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        regenerate(tmp_path)


def test_empty_archive_cannot_claim_completion(tmp_path):
    with pytest.raises(ValueError, match="empty raw"):
        regenerate(tmp_path)


def test_reader_rejects_evaluator_fields_and_impossible_artifact():
    obs = fixture_observation()
    obs["true_library"] = [[0, 1]]
    with pytest.raises(ValueError, match="contract"):
        read_public(canonical(obs))
    with pytest.raises(ValueError, match="zero evidence"):
        read_public(canonical(fixture_observation(15)))


def test_immutable_evidence_cannot_be_replaced(tmp_path):
    path = tmp_path / "unit.json"
    write(path, {"a": 1})
    write(path, {"a": 1})
    with pytest.raises(ValueError, match="immutable"):
        write(path, {"a": 2})
