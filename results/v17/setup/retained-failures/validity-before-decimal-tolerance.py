"""Known-answer checks for the first literal A consumer."""
from .contracts import Costs, BudgetExhausted, forecast_scores, public_evidence
from .programs import encode, execute, expand
from .craft import make_case, solve


def checks():
    checks = {}
    positive = execute(expand(encode([3, 7])))
    checks["real_trace_accepted"] = positive["legal"] and positive["artifact"] == 136
    checks["invalid_action_rejected"] = not execute([32])["legal"]
    checks["tool_restriction_real"] = not execute([3], forbidden=[3])["legal"]
    checks["intervention_both_directions"] = execute([3])["artifact"] != execute([19], initial=8)["artifact"]
    scores = forecast_scores({"a": .5, "b": .5}, ["a", "b"], "a")
    checks["uninformative_forecast"] = scores["brier_score"] == .5
    checks["support_varies"] = forecast_scores({"a": .2, "b": .3, "c": .5}, ["a", "b", "c"], "b")["brier_score"] == .78
    try:
        forecast_scores({"a": .8, "b": .8}, ["a", "b"], "a")
        checks["bad_normalization_rejected"] = False
    except ValueError:
        checks["bad_normalization_rejected"] = True
    fixture = {"artifact": 8, "public_context": {"world": "graphic"}, "earlier_artifacts": [4],
               "recorded_process": [{"action": 3}], "true_goal": 8, "private": {"sentinel": "HIDDEN"}}
    checks["evidence_tier_boundary"] = public_evidence(fixture, "artifact") == {"artifact": 8, "context": {"world": "graphic"}}
    costs = Costs(search_budget=2)
    costs.charge("hypothetical_execution", 2)
    try:
        costs.charge("argument_binding")
        checks["budget_enforced"] = False
    except BudgetExhausted:
        checks["budget_enforced"] = costs.receipt()["search_total"] == 2
    case = make_case("v17-known-answer", 0, 0, "changed_constraint")
    result = solve(case["public"], "primitive_search", 2048)
    checks["literal_consumer_known_target"] = result["task_success"]
    checks["actual_operations_match"] = result["attempted_primitives"] == result["costs"]["hypothetical_execution"]
    return {"schema": "v17.gates.1", "gates": checks, "passed": all(checks.values()),
            "scope": "first graphic A slice; CLI resume and corrupt-chunk checks also required before screen",
            "not_checked": ["Stitch", "assembly", "recipient inference", "adaptive inference", "observer bridge"]}
