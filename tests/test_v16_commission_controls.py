from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.commission_controls import specification, CONTROLS
from ghostscale.validation.soundingline.v16.commission_control_audit import SavedReader, verify_records
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
from ghostscale.validation.soundingline.v16.records import write, read, file_digest


@pytest.mark.parametrize("attack,card", [("X06", "O04"), ("X03", "V03")])
def test_missing_commissioned_controls_execute_and_retained_witness_replays(tmp_path, attack, card):
    producer, families = CONTROLS[attack]
    with ReaderProcess(tmp_path/"reader", extensions=EXTENSIONS) as reader:
        result = producer(reader, families[card])
    assert result["instrument_state"] == "valid"
    saved = SavedReader(result["requests"])
    assert producer(saved, families[card]) == result
    assert saved.index == len(saved.frames)
    corrupted = deepcopy(result["requests"])
    corrupted[0]["public"]["changed"] = True
    with pytest.raises(ValueError, match="request changed"):
        producer(SavedReader(corrupted), families[card])
    with pytest.raises(ValueError, match="omitted a request"):
        producer(SavedReader([]), families[card])


def source_fixture(root):
    write(root/"COMMISSION_MANIFEST.json", {"cards": [{"card_id": "O04", "dependency_ids": ["X05", "X06"]}]})
    write(root/"study-products-1/NATIVE_CARDS.json", {"cards": [{"card_id": "O04", "final_source": "boundary-expansion-1/O04"}]})
    unit = root/"boundary-expansion-1/O04/units/known_points.json"
    write(unit, {"unit_id": "known", "card_id": "O04", "condition": "known", "seed_components": {"index": 0}})
    condition = root/"boundary-controls-1/O04/condition.json"
    write(condition, {"source_unit_id": "known", "source_unit_sha256": file_digest(unit)})
    control = root/"boundary-controls-1/O04/COMPLETION.json"
    write(control, {"instrument_state": "valid", "required_adversaries": ["X05"], "conditions": [
        {"condition": "known", "receipt": condition.relative_to(root).as_posix(), "sha256": file_digest(condition)}]})
    return unit, control


def test_supplement_selection_is_commission_bound_and_retains_original_condition(tmp_path):
    unit, control = source_fixture(tmp_path)
    spec = specification(tmp_path)
    assert spec["gaps"][0]["missing_adversaries"] == ["X06"]
    assert spec["gaps"][0]["conditions"][0]["unit"]["sha256"] == file_digest(unit)
    with pytest.raises(ValueError, match="omitted a required condition"):
        verify_records(tmp_path, spec, [])
    write(unit, {**read(unit), "changed": True}, immutable=False)
    with pytest.raises(ValueError, match="changed"):
        specification(tmp_path)


def test_supplement_refuses_changed_or_duplicate_condition_inventory(tmp_path):
    _, control = source_fixture(tmp_path)
    original = read(control)
    write(control, {**original, "conditions": original["conditions"]*2}, immutable=False)
    with pytest.raises(ValueError, match="duplicate"):
        specification(tmp_path)
    write(control, original, immutable=False)
    spec = specification(tmp_path)
    spec["gaps"][0]["conditions"] = []
    with pytest.raises(ValueError, match="omitted an original condition"):
        verify_records(tmp_path, spec, [])
