from copy import deepcopy
from ghostscale.validation.soundingline.v16.catalogue_audit import predicates


def test_prediction_gain_does_not_erase_historical_ambiguity_or_imply_enactment():
    row = {"card_id": "P02", "arms": {"maker": {"outcomes": {"success": 1, "future_log_score": -.1},
        "reconstruction": {"program": [1], "search_timeout": False}}, "without-query": {"outcomes": {"future_log_score": -1.}}}}
    truth = {"true_production_record": {"program": [1]}, "equivalence_classes": [0, 1]}
    result = predicates(row, {}, truth, {})
    assert set(result) == {"enactable-history-match", "enactable-ambiguous", "evidence-gain-history-ambiguous"}
    changed = deepcopy(row)
    changed["arms"]["maker"]["outcomes"]["success"] = 0
    changed["arms"]["maker"]["reconstruction"]["search_timeout"] = True
    result = predicates(changed, {}, truth, {})
    assert "accurate-unenactable" in result and "enactable-history-match" not in result


def test_identity_success_requires_separate_prediction_comparison_and_correct_interface():
    row = {"card_id": "M01", "arms": {"craft": {"outcomes": {"identity_accuracy": 1, "future_core_log_score": -1.}},
        "direct-table": {"outcomes": {"future_core_log_score": -.5}}}}
    assert "recognition-without-process-gain" in predicates(row, {}, {}, {})
    row["arms"]["craft"]["outcomes"]["future_core_log_score"] = -.1
    assert not predicates(row, {}, {}, {})
    assert not predicates({"card_id": "P04", "arms": {}}, {}, {}, {})
