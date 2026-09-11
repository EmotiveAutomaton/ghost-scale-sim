import pytest
from runners import finalize_v16_aggregate_supplement as producer
from ghostscale.validation.soundingline.v16.records import write, read, file_digest


def test_aggregate_join_preserves_parent_inputs_and_binds_supplement(tmp_path, monkeypatch):
    write(tmp_path/"baseline_points.json", {"files": {"original": "original-hash"}})
    baseline = {"path": "baseline_points.json", "sha256": file_digest(tmp_path/"baseline_points.json")}
    write(tmp_path/"parent.json", {"execution_state": "completed", "instrument_state": "valid", "full_aggregate_regeneration": True,
        "raw_input_baseline": baseline, "phases": {"unchanged": "arithmetic"}})
    parent = {"path": "parent.json", "sha256": file_digest(tmp_path/"parent.json")}
    ref = {"path": "supplement.json", "sha256": "known-reference"}
    # The independent control verifier has its own executed and corruption tests.
    monkeypatch.setattr(producer, "verify", lambda root, reference: {"records": [{"path": "controls/one_points.json", "sha256": "new-hash"}],
        "independent_audit": {"path": "audit.json", "sha256": "audit-hash"}, "condition_controls": 1})
    result = producer.join(tmp_path, tmp_path/"joined", parent, ref)
    assert result["phases"] == {"unchanged": "arithmetic"}
    files = read(tmp_path/result["raw_input_baseline"]["path"])["files"]
    assert files == {"original": "original-hash", "results/v16/controls/one_points.json": "new-hash"}
    assert result["parent_aggregate_proof"] == parent and result["supplemental_commission_controls"] == ref
    with pytest.raises(ValueError, match="preserve"):
        producer.join(tmp_path, tmp_path/"joined", parent, ref)
    write(tmp_path/"parent.json", {**read(tmp_path/"parent.json"), "full_aggregate_regeneration": False}, immutable=False)
    parent["sha256"] = file_digest(tmp_path/"parent.json")
    with pytest.raises(ValueError, match="completed parent arithmetic"):
        producer.join(tmp_path, tmp_path/"invalid", parent, ref)
