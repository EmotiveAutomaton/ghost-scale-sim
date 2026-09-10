"""Known-answer admission for the bounded numerical inquiry amendment."""
import copy
import pytest
from ghostscale.validation.soundingline.v16 import inquiry_stable as primary
from ghostscale.validation.soundingline.v16 import inquiry_stable_reference as reference
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.audit_inquiry import close


def public_case():
    return {"schema_version": "v16.inquiry.1", "task_id": "known-answer",
            "beliefs": [primary.PRIOR, primary.PRIOR], "offers": [0, 1],
            "histories": [[], []], "familiarity": [4, 0], "pending_commands": [[], []],
            "feedback_batches": [1, 1], "future_weights": [0.5, 0.5],
            "opportunity_cost": 0.01, "remaining_interactions": 8, "tie_draw": 0.25,
            "committed_domain": None,
            "offer_probabilities": [[0.1, 0.2, 0.6, 0.1], [0.4, 0.3, 0.2, 0.1]]}


def test_amended_learning_noise_and_broken_controls_are_source_bound():
    from ghostscale.validation.soundingline.v16.inquiry_stable_gates import run
    result = run()
    assert result["instrument_state"] == "valid", result


def test_stable_action_ties_and_resolved_differences():
    assert primary.stable_max_index([0.25, 0.25, 0.25]) == 0
    assert primary.stable_max_index([0.25-1e-16, 0.25]) == 0
    assert primary.stable_max_index([0.25, 0.25+1e-12]) == 1
    for values in ([], [float("nan")], [0, float("inf")]):
        with pytest.raises(ValueError):
            primary.stable_max_index(values)


def test_corrected_policy_matches_independent_scalar_known_answers():
    public = public_case()
    for batch in (1, 2):
        public["feedback_batches"] = [batch, batch]
        for policy in ("decline", "uniform", "surprise", "signed-progress", "absolute-progress",
                       "eig", "recognition", "value-learning"):
            close(primary.choose(canonical(public), policy), reference.chosen(public, policy))
    belief = primary.PRIOR
    for command, cell in enumerate((2, 0, 3)):
        belief = primary.update(belief, command, cell)
        public["beliefs"][0] = belief
        for policy in ("value-learning", "eig", "recognition"):
            close(primary.choose(canonical(public), policy), reference.chosen(public, policy))
        close(primary.construct_public(canonical({"beliefs": public["beliefs"]})),
              reference.constructed({"beliefs": public["beliefs"]}))
        for artifact in primary.GOALS:
            maker = {"belief": belief, "artifact": artifact}
            close(primary.read_maker_public(canonical(maker)), reference.reading(maker))


def test_known_mapping_noise_mixture_has_no_construction_value_from_queries():
    # Only one bijection has support. Every possible answer leaves the same best
    # construction action, so information about the noise mixture has zero value.
    public = public_case()
    public["opportunity_cost"] = 0.0
    failures = []
    for mapping in range(24):
        for noise in (0.01, 0.1, 0.5):
            belief = [0.0]*25
            belief[mapping], belief[24] = 1-noise, noise
            public["beliefs"] = [belief, belief]
            for batch in (1, 2):
                public["feedback_batches"] = [batch, batch]
                result = primary.choose(canonical(public), "value-learning")
                close(result, reference.chosen(public, "value-learning"))
                if result["domain"] is not None:
                    failures.append((mapping, noise, batch, result["scores"]))
    assert not failures, failures


def test_amended_episodes_independent_math_physics_and_resources(tmp_path):
    from ghostscale.validation.soundingline.v16.inquiry_stable_study import execute_unit, summarize
    from ghostscale.validation.soundingline.v16.inquiry_designs import DESIGNS
    from ghostscale.validation.soundingline.v16.audit_inquiry_stable import audit
    from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
    from ghostscale.validation.soundingline.v16.resource_accounting import MeasuredReader
    extensions = [primary.__name__+":"+name for name in
                  ("choose", "learn_public", "construct_public", "read_maker_public")]
    packet = {"packet_hash": "fixture", "identity": {"commission_hash": "fixture"}}
    with ReaderProcess(tmp_path/"reader", extensions=extensions) as reader:
        for card in ("R03", "R05"):
            root = tmp_path/card
            rows = []
            for condition in DESIGNS[card]["conditions"]:
                row = execute_unit(root, card, condition, 0, namespace="fixture-stable-episodes",
                                   packet=packet, reader=MeasuredReader(reader), scope="fixture")
                rows.append(row)
                assert execute_unit(root, card, condition, 0, namespace="fixture-stable-episodes",
                                    packet=packet, reader=MeasuredReader(reader), scope="fixture") == row
                if card == "R03":
                    assert row["arms"]["decline"]["outcomes"]["queries"] == 0
                else:
                    for metric in ("success", "future_log_score"):
                        assert row["arms"]["enact"]["outcomes"][metric] == row["arms"]["observe"]["outcomes"][metric]
                        if condition["practice_examples"] == 0:
                            assert row["arms"]["enact"]["outcomes"][metric] == row["arms"]["unrelated"]["outcomes"][metric]
            receipt = audit(root, summarize(card, rows))
            assert receipt["instrument_state"] == "valid"
            assert receipt["independent_scalar_reader_requests"] > 0
