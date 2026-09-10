from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.final_disposition import final_control_join, checked_summary
from ghostscale.validation.soundingline.v16.records import write, file_digest


def test_final_control_join_rejects_wrong_source_and_incomplete_coverage():
    packet = {"packet_hash": "actual-source", "identity": {"design": {"cards": [
        {"card_id": "K03", "n_per_condition": 1024, "design": {"conditions": [{"id": "one"}, {"id": "two"}]}}]}}}
    science = {"execution_state": "completed", "instrument_state": "valid", "packet_hash": "actual-source",
        "completed_condition_records": 2048, "cards": [{"card_id": "K03"}]}
    control = {"execution_state": "completed", "instrument_state": "valid", "source_packet_hash": "actual-source",
        "source_condition_records": 2048, "condition_controls": 2, "cards": [{"card_id": "K03"}]}
    assert final_control_join(science, control, packet)["controlled_conditions"] == 2
    for field, value in [("source_packet_hash", "old-scout"), ("condition_controls", 1),
                         ("source_condition_records", 1024), ("execution_state", "running"), ("cards", [])]:
        bad = deepcopy(control)
        bad[field] = value
        with pytest.raises(ValueError):
            final_control_join(science, bad, packet)


def test_final_source_rejects_a_changed_completed_aggregate(tmp_path):
    summary = {"n_maker_packets": 64, "conditions": {"one": {"contrasts": [{"n_makers": 64}]}}}
    write(tmp_path/"AGGREGATE.json", summary)
    write(tmp_path/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid",
        "aggregate_sha256": file_digest(tmp_path/"AGGREGATE.json")})
    write(tmp_path/"INDEPENDENT_AUDIT.json", {"instrument_state": "valid"})
    assert checked_summary(tmp_path, "AGGREGATE.json", "INDEPENDENT_AUDIT.json", expected_n=64, expected_conditions=["one"]) == summary
    summary["n_maker_packets"] = 32
    write(tmp_path/"AGGREGATE.json", summary, immutable=False)
    with pytest.raises(ValueError, match="summary changed"):
        checked_summary(tmp_path, "AGGREGATE.json", "INDEPENDENT_AUDIT.json", expected_n=64, expected_conditions=["one"])
