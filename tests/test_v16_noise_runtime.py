import copy
import pytest
from ghostscale.validation.soundingline.v16.noise_cases import noise_case, PREFIX
from ghostscale.validation.soundingline.v16.collision_cases import Recorder
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.completion_guard import assess, from_progress, bound_receipt, PROOFS
from ghostscale.validation.soundingline.v16.runtime_interruption import interruption_control
from ghostscale.validation.soundingline.v16.records import write


@pytest.mark.parametrize("direction", ["decline", "rise"])
def test_executed_noise_fools_progress_but_has_no_expected_learning_value(tmp_path, direction):
    with ReaderProcess(tmp_path, extensions=[PREFIX+name for name in ["learn_public", "choose", "construct_public"]]) as reader:
        checks, witness = noise_case(Recorder(reader), direction)
    assert all(checks.values()), checks
    assert witness["primitive_cost"] == 202


def test_noise_control_rejects_progress_mislabeled_as_value_learning(tmp_path):
    with ReaderProcess(tmp_path, extensions=[PREFIX+name for name in ["learn_public", "choose", "construct_public"]]) as reader:
        class Broken:
            def request(self, kind, public, **options):
                if options.get("policy") == "value-learning":
                    options["policy"] = "absolute-progress"
                return reader.request(kind, public, **options)
        checks, _ = noise_case(Recorder(Broken()))
    assert not checks["value_learning_declines_zero_gain"]
    assert not checks["actual_policies_match_independent_scalar_reference"]


def valid_account():
    cards = [{"card_id": "A", "execution_state": "completed", "instrument_state": "valid",
              "evidence_verified": True, "scientific_criterion": "null"}]
    expansions = [{"card_id": "A", "state": "exhausted", "reason": "registered ladder examined"}]
    return cards, expansions, {name: {"verified": True} for name in PROOFS}


def test_completion_guard_separates_valid_null_missing_work_and_missing_proofs():
    cards, expansions, proofs = valid_account()
    assert assess(["A"], cards, expansions, proofs)["campaign_clean"]
    assert not assess([], [], [], proofs)["campaign_closed"]
    assert not assess(["A", "B"], cards, expansions, proofs)["campaign_closed"]
    for state in ["running", "checkpointed", "planned"]:
        changed = copy.deepcopy(cards)
        changed[0]["execution_state"] = state
        assert not assess(["A"], changed, expansions, proofs)["campaign_closed"]
    for missing in PROOFS:
        assert not assess(["A"], cards, expansions, {name: value for name, value in proofs.items() if name != missing})["campaign_closed"]
    assert not assess(["A"], cards, [], proofs)["campaign_closed"]
    assert not from_progress({"cards": [{"card_id": "A", "implementation_state": "completed"}]})["campaign_closed"]


def test_completion_guard_keeps_invalid_or_blocked_work_qualified():
    cards, expansions, proofs = valid_account()
    cards[0].update(execution_state="blocked", instrument_state="failed", reason="instrument invalid", scientific_criterion="not_tested", unresolved_dependencies=["ruler"])
    result = assess(["A"], cards, expansions, proofs)
    assert result["campaign_closed"] and not result["campaign_clean"]
    cards[0]["execution_state"] = "completed"
    assert not assess(["A"], cards, expansions, proofs)["campaign_closed"]


def test_closeout_proofs_bind_actual_saved_bytes_and_archive_location(tmp_path):
    expected = write(tmp_path/"proof.json", {"verified": True})
    assert bound_receipt(tmp_path, "proof.json", expected) == {"verified": True}
    (tmp_path/"proof.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        bound_receipt(tmp_path, "proof.json", expected)
    with pytest.raises(ValueError, match="outside"):
        bound_receipt(tmp_path, "../proof.json", expected)


def test_actual_interruption_control_preserves_clock_units_and_scope(tmp_path):
    result = interruption_control(tmp_path, {"known-consumer": "fixture-source"})
    assert result["instrument_state"] == "valid", result["checks"]
    assert interruption_control(tmp_path, {"known-consumer": "fixture-source"}) == result
