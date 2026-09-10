import copy
import pytest
from ghostscale.validation.soundingline.v16.collision_cases import CASES, Recorder, ambiguous
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.recoded_interface import OPERATIONS


@pytest.mark.parametrize("family", sorted(CASES))
def test_actual_collision_and_separating_evidence_controls(tmp_path, family):
    with ReaderProcess(tmp_path, extensions=[name for name in OPERATIONS if ":" in name]) as reader:
        recorded = Recorder(reader)
        checks, witness = CASES[family](recorded)
        assert checks and all(checks.values()), {"family": family, "checks": checks}
        assert len(recorded.frames) >= 2 and witness


def test_collision_controls_reject_forced_certainty_and_ignored_separating_evidence(tmp_path):
    assert ambiguous([0.5, 0.5])
    assert not ambiguous([1.0, 0.0])
    assert not ambiguous([0.2, 0.2])
    with ReaderProcess(tmp_path) as reader:
        recorded = Recorder(reader)
        checks, _ = CASES["reading"](recorded)
        assert all(checks.values())
    class IgnoresEvidence:
        def request(self, kind, public, **options):
            return copy.deepcopy(recorded.frames[0]["result"])
    broken, _ = CASES["reading"](Recorder(IgnoresEvidence()))
    assert not broken["real_prefix_changes_repertoire_evidence"]


@pytest.mark.parametrize("family", ["reading", "inquiry", "opportunity", "self", "trajectory"])
def test_actual_standalone_collision_controls(tmp_path, family):
    from ghostscale.validation.soundingline.v16.transfer_stable import install_consumer, Consumer
    from ghostscale.validation.soundingline.v16.attack_collisions import StandaloneAdapter, check_case
    install_consumer(tmp_path)
    with Consumer(tmp_path) as consumer:
        adapter = StandaloneAdapter(consumer)
        result = check_case(adapter, family)
        assert result["instrument_state"] == "valid", result["checks"]
        assert len(adapter.sent) == len(result["requests"])
        assert all(record["response"]["ok"] for record in adapter.sent)
        assert all("witness" not in record["public"] for record in adapter.sent)
