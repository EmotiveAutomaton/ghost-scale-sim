from copy import deepcopy
from ghostscale.validation.soundingline.v16.catalogue_cases_v2 import labels, clean_failure


def test_enactment_does_not_silently_establish_historical_truth():
    arm = {"outcomes": {"success": 1, "future_log_score": -1},
           "reconstruction": {"program": [0, 1], "search_timeout": False}}
    row = {"card_id": "P01", "arms": {"maker": arm, "without-query": deepcopy(arm)}}
    private = {"true_production_record": {"program": [1, 0]}, "equivalence_classes": [[0, 1], [1, 0]]}
    result = labels(row, {}, private, {})
    assert set(result) == {"enactable-ambiguous"}
    private["true_production_record"]["program"] = [0, 1]
    result = labels(row, {}, private, {})
    assert "enactable-history-match" in result
    assert result["enactable-history-match"]["compatible_routes"] == 2
    row["arms"]["maker"]["outcomes"]["success"] = 0
    assert "accurate-unenactable" not in labels(row, {}, private, {})
    row["arms"]["maker"]["reconstruction"]["search_timeout"] = True
    assert "accurate-unenactable" in labels(row, {}, private, {})


def test_one_realized_loss_is_not_a_population_null_and_recognition_is_separate():
    row = {"card_id": "M01", "arms": {
        "craft": {"outcomes": {"identity_accuracy": 1, "future_core_log_score": -2}},
        "direct-table": {"outcomes": {"identity_accuracy": 0, "future_core_log_score": -1}}}}
    assert "recognition-without-process-gain" in labels(row, {}, {}, {})
    spec = {"arm": "craft", "rival": "direct-table", "target": "future_core_log_score"}
    failure = clean_failure(row, [{"estimand": spec}])
    assert failure["paired_difference"] == -1
    assert "not a population null" in failure["qualification"]
    row["arms"]["craft"]["outcomes"]["identity_accuracy"] = 0
    assert labels(row, {}, {}, {}) == {}


def test_self_explanation_and_selection_controls_require_actual_loss_and_truth():
    row = {"card_id": "S05", "arms": {name: {"outcomes": {"original_goal_success": value}}
        for name, value in [("self-model", 1), ("naive-self", 0), ("direct-completion", 0)]}}
    assert set(labels(row, {}, {}, {})) == {"self-correction-help", "self-explanation-harm"}
    for arm in row["arms"].values():
        arm["outcomes"]["original_goal_success"] = 1
    assert labels(row, {}, {}, {}) == {}
    prediction = {"arms": {"selection-aware": {"acquired_style": [.8, .2]},
                            "release-naive": {"acquired_style": [.1, .9]}}}
    row = {"card_id": "M02", "arms": {}}
    truth = {"maker": {"decoration": {"choice": 0}}}
    assert "selection-looking-personal" in labels(row, {"retention": "selective"}, truth, prediction)
    assert labels(row, {"retention": "all"}, truth, prediction) == {}
    truth["maker"]["decoration"]["choice"] = 1
    assert labels(row, {"retention": "selective"}, truth, prediction) == {}


def test_query_count_is_not_competence_and_zero_gain_is_not_proof_of_noise():
    row = {"card_id": "R03", "arms": {name: {"outcomes": {"queries": 3, "gain": 0}}
        for name in ["value-learning", "absolute-progress", "surprise", "recognition"]}}
    result = labels(row, {}, {}, {})
    assert set(result) == {"inquiry-no-gain"}
    assert "does not identify irreducible noise" in result["inquiry-no-gain"]["qualification"]
    row["arms"]["value-learning"]["outcomes"]["gain"] = .25
    assert "inquiry-competence" in labels(row, {}, {}, {})


def test_paid_reading_query_and_changed_generator_are_different_interfaces():
    row = {"card_id": "P02", "arms": {
        "maker": {"outcomes": {"success": 1, "future_log_score": -1},
                  "reconstruction": {"program": [0, 1], "search_timeout": False}},
        "without-query": {"outcomes": {"future_log_score": -2}}}}
    private = {"true_production_record": {"program": [1, 0]}, "equivalence_classes": [[0, 1], [1, 0]]}
    result = labels(row, {}, private, {})
    assert result["evidence-gain-history-ambiguous"]["paired_future_log_score_gain"] == 1
    assert labels({"card_id": "P04", "arms": {"mixture": {}, "direct-mixture": {}}}, {}, {}, {}) == {}
