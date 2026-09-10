import copy
import pytest
from ghostscale.validation.soundingline.v16.context_cases import CASES
from ghostscale.validation.soundingline.v16.collision_cases import Recorder
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.recoded_interface import OPERATIONS


@pytest.mark.parametrize("family", sorted(CASES))
def test_actual_false_context_correction_and_trust_boundary(tmp_path, family):
    with ReaderProcess(tmp_path, extensions=[name for name in OPERATIONS if ":" in name]) as reader:
        recorded = Recorder(reader)
        checks, witness = CASES[family](recorded)
        assert checks and all(checks.values()), {"family": family, "checks": checks}
        assert len(recorded.frames) >= 2 and witness


def test_false_context_controls_reject_ignored_evidence_and_rewritten_goal(tmp_path):
    with ReaderProcess(tmp_path) as reader:
        recorded = Recorder(reader)
        checks, _ = CASES["self"](recorded)
        assert all(checks.values())
    class Broken:
        def request(self, kind, public, **options):
            return copy.deepcopy(recorded.frames[0]["result"])
    checks, _ = CASES["self"](Recorder(Broken()))
    assert not checks["fresh_external_artifacts_correct_old_goal_probability"]
    assert not checks["corrected_verified_memory_recovers_old_goal"]
    class RewrittenGoal:
        def request(self, kind, public, **options):
            result = copy.deepcopy(recorded.frames[0]["result"])
            result["adopted_goal"] = public["memory"]["original_goal"]
            return result
    checks, _ = CASES["self"](Recorder(RewrittenGoal()))
    assert not checks["declared_new_goal_is_not_rewritten_by_false_old_memory"]


@pytest.mark.parametrize("family", ["inquiry", "opportunity", "self", "trajectory"])
def test_actual_standalone_false_context_controls(tmp_path, family):
    from ghostscale.validation.soundingline.v16.transfer_stable import install_consumer, Consumer
    from ghostscale.validation.soundingline.v16.attack_collisions import StandaloneAdapter
    install_consumer(tmp_path)
    with Consumer(tmp_path) as consumer:
        adapter = StandaloneAdapter(consumer)
        recorded = Recorder(adapter)
        checks, witness = CASES[family](recorded)
        assert all(checks.values()), checks
        assert len(adapter.sent) == len(recorded.frames)
        assert all("witness" not in record["public"] for record in adapter.sent)
