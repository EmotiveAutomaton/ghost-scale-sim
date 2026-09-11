import json
from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import digest
from ghostscale.validation.soundingline.v16.commission_controls import CONTROLS
from ghostscale.validation.soundingline.v16.commission_control_audit_v2 import SavedReader
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS


@pytest.mark.parametrize("attack,card", [("X06", "O04"), ("X03", "V03")])
def test_witness_round_trip_preserves_content_and_detects_changed_physics(tmp_path, attack, card):
    producer, families = CONTROLS[attack]
    with ReaderProcess(tmp_path/"reader", extensions=EXTENSIONS) as reader:
        original = producer(reader, families[card])
    retained = json.loads(json.dumps(original))
    replay = producer(SavedReader(retained["requests"]), families[card])
    assert replay["instrument_state"] == "valid"
    assert digest(replay) == digest(retained)
    changed = deepcopy(retained)
    changed["witness"]["fabricated_physical_fact"] = True
    assert digest(replay) != digest(changed)
    changed = deepcopy(retained)
    first = next(iter(changed["checks"]))
    changed["checks"][first] = False
    assert digest(replay) != digest(changed)
