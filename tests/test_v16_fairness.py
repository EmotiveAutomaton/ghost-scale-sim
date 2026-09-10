from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16 import fairness
from ghostscale.validation.soundingline.v16.craft import construct_public
from ghostscale.validation.soundingline.v16.reconstruction import reader as reading
from ghostscale.validation.soundingline.v16.reading_cost_meter import public_reader as meter
from ghostscale.validation.soundingline.v16.collision_cases import reading_case, Recorder


def craft_input(budget=32):
    record = {"attempts": [[0, 1]]*4, "targets": [3]*4}
    return {"schema_version": "v16.craft.public.1", "task_id": "known", "access_tier": "training-with-feedback",
        "targets": [7], "search_primitive_budget": budget,
        "arm_training": {"personal": deepcopy(record), "primitive": deepcopy(record), "pooled": deepcopy(record)}}


def reading_input():
    class Adapter:
        def request(self, kind, public, **options):
            return reading(canonical(public), **options)
    log = Recorder(Adapter())
    reading_case(log)
    return log.frames[0]["public"]


def test_fairness_known_macro_budget_and_initial_learning_costs():
    public = craft_input()
    result = construct_public(canonical(public))
    fairness.check_craft(public, result)
    assert not result["personal"]["submissions"][0]["search_timeout"]
    assert result["primitive"]["submissions"][0]["search_timeout"]
    assert result["personal"]["costs"]["training_primitives"] == 8
    assert result["personal"]["costs"]["library_definition"] == 2
    curves = fairness.amortization(public, result)
    first, last = [x for x in curves if x["arm"] == "personal"][::2]
    assert first["training_primitives_once"] == last["training_primitives_once"] == 8
    assert first["training_plus_search_primitives_per_packet"]-last["training_plus_search_primitives_per_packet"] == 7.75


@pytest.mark.parametrize("breakage", ["macro", "training", "definition", "extra-information"])
def test_fairness_intentionally_broken_cost_and_access_controls(breakage):
    public = craft_input()
    result = construct_public(canonical(public))
    if breakage == "macro":
        result["personal"]["submissions"][0]["attempted_programs"][1]["cost"] = 1
    elif breakage == "training":
        result["personal"]["costs"]["training_primitives"] = 0
    elif breakage == "definition":
        result["personal"]["costs"]["library_definition"] = 0
    else:
        public["arm_training"]["primitive"]["attempts"] = [[2, 3]]*4
    with pytest.raises(ValueError):
        fairness.information("K01", [{"public": public, "kind": "craft"}])
        fairness.check_craft(public, result)


@pytest.mark.parametrize("strategy,multiplier", [("maker", 4), ("direct-table", 8), ("generic", 4), ("primitive", 4)])
def test_fairness_actual_reading_invocations_and_unchanged_results(strategy, multiplier):
    public = reading_input()
    before = reading(canonical(public), strategy)
    measured = meter(canonical(public), strategy)
    assert measured["result"] == before == reading(canonical(public), strategy)
    assert measured["invocations"]["observation_mass"] == multiplier+64
    frame = {"kind": "reading", "public": public, "options": {"strategy": strategy}, "result": before}
    checked = fairness.old_frame_costs(frame)
    assert checked["actual_likelihood_call_invocations"] == measured["invocations"]["observation_mass"]
    if strategy == "direct-table":
        assert before["costs"]["likelihood_evaluations"] != measured["invocations"]["observation_mass"]


def test_fairness_shared_observation_check_rejects_hidden_extra_context():
    frames = [{"kind": "reading", "public": reading_input()} for _ in range(4)]
    fairness.information("K02", frames)
    frames[2]["public"]["hidden_answer"] = 1
    with pytest.raises(ValueError, match="unequal"):
        fairness.information("K02", frames)


def test_fairness_real_process_profile_and_meter_preserve_known_answer(tmp_path):
    from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
    from ghostscale.validation.soundingline.v16.resource_accounting import MeasuredReader
    from ghostscale.validation.soundingline.v16.attack_fairness import METER
    public = reading_input()
    with ReaderProcess(tmp_path/"reader", extensions=[METER]) as reader:
        measured = MeasuredReader(reader)
        actual = measured.request(METER, public, strategy="direct-table")
        assert actual["result"] == reading(canonical(public), "direct-table")
        assert actual["invocations"]["observation_mass"] == 72
        sample = measured.samples[0]
        assert sample["wall_seconds"] > 0
        assert sample["resource_state"] == "measured"
        assert sample["reader_cpu_seconds"] >= 0 and sample["resident_bytes"] > 0
