import pytest
from ghostscale.validation.soundingline.v16.records import read, write, file_digest
from ghostscale.validation.soundingline.v16.runtime import freeze
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.expansion_adapter import design, execute_unit
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.study import raw_manifest
from ghostscale.validation.soundingline.v16.packet_control_runner import sources, source_inventory, execute
from ghostscale.validation.soundingline.v16.packet_control_interruption import control


def known_source(root):
    fixture_campaign(root)
    entry = {"card_id": "K01", "n_per_condition": 1, "constructors": 1,
             "namespace": "known-packet-control-source", "design": design("K01")}
    packet = freeze(root, "constructor-expansion-1", sources(root), {"cards": [entry]})
    base = root/"constructor-expansion-1/K01"
    with ReaderProcess(base/"reader-work", extensions=EXTENSIONS) as reader:
        for condition in entry["design"]["conditions"]:
            execute_unit("K01", base, condition, 0, namespace=entry["namespace"], packet=packet,
                         reader=reader, constructors=1, scope="fixture")
    write(base/"RAW_MANIFEST.json", raw_manifest(base))
    write(base/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid",
          "raw_manifest_sha256": file_digest(base/"RAW_MANIFEST.json")})
    write(base.parent/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid"})
    return entry, packet


def test_packet_controls_actual_interruption_preserves_partial_evidence(tmp_path):
    source = tmp_path/"source"
    known_source(source)
    receipt = control(tmp_path/"runtime", source)
    assert receipt["instrument_state"] == "valid", receipt["checks"]
    assert receipt["completed_condition_controls"] == 3


def test_packet_controls_sampling_tampering_and_missing_resume_fail(tmp_path):
    source = tmp_path/"source"
    entry, packet = known_source(source)
    base = source/"constructor-expansion-1/K01"
    inventory, selected = source_inventory(base, entry, packet["packet_hash"])
    assert len(inventory["unit_hashes"]) == 3 and len(selected) == 3
    path = next((base/"units").glob("*_points.json"))
    changed = read(path)
    changed["seed_components"]["index"] = 100
    write(path, changed, immutable=False)
    with pytest.raises(ValueError, match="saved manifest"):
        source_inventory(base, entry, packet["packet_hash"])
    destination = tmp_path/"no-packet"
    fixture_campaign(destination)
    with pytest.raises(ValueError, match="existing frozen packet"):
        execute(destination, lambda **kwargs: None, resume=True, source_root=source)
