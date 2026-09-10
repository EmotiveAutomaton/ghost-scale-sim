import pytest
from ghostscale.validation.soundingline.v16.misspecification_cases import CASES, mismatch
from ghostscale.validation.soundingline.v16.collision_cases import Recorder
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.recoded_interface import OPERATIONS


@pytest.mark.parametrize("family", sorted(CASES))
def test_actual_omitted_generator_and_procedure_controls(tmp_path, family):
    with ReaderProcess(tmp_path, extensions=[name for name in OPERATIONS if ":" in name]) as reader:
        recorded = Recorder(reader)
        checks, witness = CASES[family](recorded)
        assert checks and all(checks.values()), {"family": family, "checks": checks}
        assert recorded.frames and witness


def test_misspecification_control_rejects_manufactured_uniform_prediction():
    assert mismatch({"model_mismatch": True, "reason": "outside support"})
    assert mismatch({"explicit_rejection": "ValueError: outside catalog"})
    assert not mismatch({"model_mismatch": True, "future_probabilities": [0.5, 0.5]})
    assert not mismatch({"future_probabilities": [0.5, 0.5]})
    class UniformFallback:
        def request(self, kind, public, **options):
            return {"model_mismatch": True, "future_probabilities": [1/16]*16}
    checks, _ = CASES["reading"](Recorder(UniformFallback()))
    assert not checks["all_declared_readers_expose_missing_support"]


@pytest.mark.parametrize("family", ["reading", "opportunity"])
def test_actual_standalone_omitted_generator_controls(tmp_path, family):
    from ghostscale.validation.soundingline.v16.transfer_stable import install_consumer, Consumer
    from ghostscale.validation.soundingline.v16.attack_collisions import StandaloneAdapter
    install_consumer(tmp_path)
    with Consumer(tmp_path) as consumer:
        adapter = StandaloneAdapter(consumer)
        checks, witness = CASES[family](Recorder(adapter))
        assert all(checks.values()), checks
        assert adapter.sent and witness
