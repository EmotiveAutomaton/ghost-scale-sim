"""Bounded descriptive case predicates; these are not population estimators.

The historical-program lens is an observed match, not identification of a unique
history. Every label states its comparator. Reader exports never contain truth.
"""
import math

LENSES = {
    "enactable-history-match": "Enactable reconstruction matching the observed production program",
    "enactable-ambiguous": "Enactable reconstruction with unresolved production history",
    "accurate-unenactable": "Accurate historical account unavailable for enactment under the reader's constraints",
    "recognition-without-process-gain": "Correct maker recognition without improved process prediction",
    "evidence-gain-history-ambiguous": "Evidence improves prediction while historical alternatives remain",
    "self-correction-help": "Self-correction helps continued pursuit of the original goal",
    "self-explanation-harm": "Plausible self-explanation harms continued pursuit of the original goal",
    "inquiry-competence": "Inquiry acquires executable competence",
    "inquiry-no-gain": "Inquiry pays for queries without realized competence gain",
    "selection-looking-personal": "Selection makes a personal routine account misleading",
}


def labels(row, public, private, predictions):
    card, arms = row["card_id"], row["arms"]
    result = {}
    def add(name, evidence):
        result[name] = evidence
    if card in {"P01", "P04"}:
        maker = arms["maker"]
        success = maker["outcomes"]["success"] == 1
        program = maker["reconstruction"]["program"]
        truth = private["true_production_record"]["program"]
        routes = len(private["equivalence_classes"])
        if success and program == truth:
            add("enactable-history-match", {"program_matches_observed_truth": True,
                "compatible_routes": routes, "qualification": "Observed program match; unique historical identification is not established"})
        if success and routes > 1:
            add("enactable-ambiguous", {"compatible_routes": routes, "program_matches_observed_truth": program == truth,
                "qualification": "Artifact-compatible routes remain; enactment alone does not identify the route"})
        # A timed-out submitted true program is the only directly jointly measured
        # instance of this lens. Correct repertoire classification is insufficient.
        if not success and program == truth and maker["reconstruction"]["search_timeout"]:
            add("accurate-unenactable", {"program_matches_observed_truth": True, "search_timeout": True})
        gain = maker["outcomes"]["future_log_score"] - arms["without-query"]["outcomes"]["future_log_score"]
        if card == "P04" and gain > 1e-10 and routes > 1:
            add("evidence-gain-history-ambiguous", {"paired_future_log_score_gain": gain,
                "compatible_routes": routes, "rival": "same maker reader with purchased diagnostic evidence removed",
                "qualification": "A single realized continuation; not an expected or population gain"})
    if card == "M01":
        candidate = arms["craft"]["outcomes"]
        rival = arms["direct-table"]["outcomes"]
        gain = candidate["future_core_log_score"]-rival["future_core_log_score"]
        if candidate["identity_accuracy"] == 1 and gain <= 1e-10:
            add("recognition-without-process-gain", {"identity_accuracy": 1,
                "future_core_log_score_gain_over_direct_table": gain,
                "qualification": "Correct in this case; no claim that identity was uniquely identifiable or above chance across makers"})
    if card == "S05":
        informed = arms["self-model"]["outcomes"]["original_goal_success"]
        direct = arms["direct-completion"]["outcomes"]["original_goal_success"]
        naive = arms["naive-self"]["outcomes"]["original_goal_success"]
        if informed > direct:
            add("self-correction-help", {"original_goal_success_gain_over_direct_completion": informed-direct})
        if naive < informed:
            add("self-explanation-harm", {"naive_original_goal_success_loss": informed-naive,
                "qualification": "Intentionally misspecified self-explanation, compared with source-aware inference"})
    if card in {"R02", "R03"}:
        candidate = arms["value-learning"]["outcomes"]
        if candidate["queries"] > 0 and candidate["gain"] > 0:
            add("inquiry-competence", {"queries": candidate["queries"], "realized_heldout_success_gain": candidate["gain"],
                "qualification": "Practice changes a production policy; maker recognition is a separate target"})
        for name in ["absolute-progress", "surprise", "recognition", "value-learning"]:
            outcome = arms[name]["outcomes"]
            if outcome["queries"] > 0 and outcome["gain"] <= 0:
                add("inquiry-no-gain", {"reader": name, "queries": outcome["queries"], "realized_heldout_success_gain": outcome["gain"],
                    "qualification": "A realized failure to gain competence; this alone does not identify irreducible noise"})
                break
    if card == "M02":
        aware, naive = (predictions["arms"][name] for name in ["selection-aware", "release-naive"])
        true_style = private["maker"]["decoration"]["choice"]
        if public["retention"] != "all" and naive["acquired_style"][true_style] < .2 and aware["acquired_style"][true_style] > .5:
            add("selection-looking-personal", {"naive_true_routine_probability": naive["acquired_style"][true_style],
                "selection_aware_true_routine_probability": aware["acquired_style"][true_style],
                "qualification": "Selection-aware and release-naive readers see the same retained evidence"})
    return result


def clean_failure(row, contrasts):
    """First registered comparison with no favorable realized paired difference."""
    for contrast in contrasts:
        spec = contrast["estimand"]
        left = row["arms"][spec["arm"]]["outcomes"][spec["target"]]
        right = row["arms"][spec["rival"]]["outcomes"][spec["target"]]
        if not all(math.isfinite(value) for value in [left, right]):
            raise ValueError("catalogue has a nonfinite registered observation")
        if left-right <= 0:
            return {"estimand": spec, "arm_value": left, "rival_value": right, "paired_difference": left-right,
                "qualification": "One descriptive failure or tie; not a population null or failed confirmation"}
    return None
