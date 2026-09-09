"""Known-answer admission for acquisition, matched computation and paired estimands."""
from .craft import construct, prepare_unit, construct_public, DESIGN
from .learning import learn, expand
from .records import canonical
from .estimands import Estimand, paired_summary


def run():
    records = []
    def gate(name, checks):
        records.append({"id": name, "evidence_scope": "fixture", "checks": checks,
                        "instrument_state": "valid" if all(checks.values()) else "failed"})
    learned = learn([(0, 1)]*4, [3]*4)
    personal = construct(7, learned.library, primitive_budget=32)
    primitive = construct(7, (), primitive_budget=32)
    null = learn([(0, 1)], [3])
    gate("craft-cost", {
        "known_new_composition_solved": personal["program"] == [0, 1, 2],
        "matched_primitive_budget": personal["search_primitives"] <= 32 and primitive["search_primitives"] <= 32,
        "positive_not_primitive_solved": not personal["search_timeout"] and primitive["search_timeout"],
        "null_without_repetition": construct(7, null.library, primitive_budget=32) == primitive,
        "broken_macro_cost_detected": sum(item["cost"] for item in personal["attempted_programs"]) != personal["considered"],
        "macro_boundary_invariant": expand(("m0", 2), learned.library) == (0, 1, 2),
        "ambiguous_route_boundary": expand(("m0",), learned.library) != (1, 0)})
    example = prepare_unit(DESIGN["conditions"][1], 0, namespace="fixture-craft", constructors=8, evidence_scope="fixture")
    public = canonical(example["public"])
    before = construct_public(public)
    example["private"]["direction"] = 999
    gate("craft-access", {"private_mutation_invariant": before == construct_public(public),
                         "training_changes_capability": learned.library != null.library,
                         "heldout_targets_unseen": example["private"]["heldout_target_check"],
                         "same_reader_bytes_null": before == construct_public(public)})
    estimator = Estimand("fixture-success", "success", "a", "b", "success_fraction", 0.05, "a minus b")
    rows = [{"unit_id": str(i), "constructor_id": str(i//4), "evidence_scope": "fixture",
             "arms": {"a": {"outcomes": {"success": 1.0}}, "b": {"outcomes": {"success": 0.0}}}}
            for i in range(16)]
    positive = paired_summary(rows, estimator, replicates=99)
    same_arm_rejected = False
    try:
        Estimand("broken", "success", "a", "a", "success_fraction", 0.05, "a minus a")
    except ValueError:
        same_arm_rejected = True
    for row in rows:
        row["arms"]["b"]["outcomes"]["success"] = 1.0
    null_result = paired_summary(rows, estimator, replicates=99)
    gate("estimand-identity", {"positive_difference": positive["mean"] == 1,
                              "null_difference": null_result["mean"] == 0,
                              "same_arm_break_rejected": same_arm_rejected,
                              "same_observed_success_cannot_identify_history": null_result["interval_95"] == [0, 0]})
    return {"gates": records, "evidence_scope": "fixture",
            "instrument_state": "valid" if all(r["instrument_state"] == "valid" for r in records) else "failed"}
