import copy
import pytest
from ghostscale.validation.soundingline.v16.consumer_frames import requests, PREFIX
from ghostscale.validation.soundingline.v16.attack_access import alias, comparison
from ghostscale.validation.soundingline.v16.records import write


def test_consumer_frames_preserve_both_paid_phases_without_private_data(tmp_path):
    row = {"unit_id": "unit", "card_id": "V03"}
    original = {"phase": 1, "answer": None, "task_id": "opaque"}
    answered = {**original, "phase": 2, "answer": {"artifact": 1}}
    write(tmp_path/"public/unit.json", original)
    write(tmp_path/"predictions/unit-phase-1.json", {"arms": {"all": {"query": "both"}}})
    write(tmp_path/"predictions/unit.json", {"public_inputs": {"all": answered},
                                            "arms": {"all": {"probability": 0.5}}})
    write(tmp_path/"private/unit.json", {"true_cause": "never-extracted", "seed": 9})
    frames = list(requests(tmp_path, row))
    assert len(frames) == 2
    assert [frame["public"]["phase"] for frame in frames] == [1, 2]
    assert all(frame["kind"] == PREFIX+"preference_probe:public_reader" for frame in frames)
    assert all(frame["options"] == {"policy": "all"} for frame in frames)
    assert all("true_cause" not in frame["public"] for frame in frames)
    assert frames[0]["public"]["answer"] is None


def test_access_aliases_preserve_scientific_information_and_broken_output_is_detected():
    observed = {"task_id": "one", "lineage_id": "two", "nested": [
        {"task_id": "three", "program": [0, 1], "goal": 3}], "budget": 2}
    changed = alias(observed)
    assert changed["task_id"] != observed["task_id"]
    assert changed["lineage_id"] != observed["lineage_id"]
    assert changed["nested"][0]["program"] == [0, 1] and changed["budget"] == 2
    comparison({"probabilities": [0.5, 0.5]}, {"probabilities": [0.5, 0.5]})
    with pytest.raises(ValueError, match="access/cache"):
        comparison({"probabilities": [1, 0]}, {"probabilities": [0, 1]})
