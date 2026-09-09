"""Hand-checkable causal controls, including non-nested beliefs and ambiguity."""
from .opportunity import make, decide, intervene, enact, response_probability, public_reader
from .records import canonical


def run():
    rows = []
    def gate(name, checks):
        rows.append({"id":name,"checks":checks,"evidence_scope":"fixture",
                     "instrument_state":"valid" if all(checks.values()) else "failed"})
    causes = ["physical","knowledge","consideration","purpose"]
    baseline = [enact(make(cause),"baseline",lapse_event=False)["intended_option"] for cause in causes]
    vectors = [tuple(enact(make(cause),probe,lapse_event=False)["artifact"]
                     for probe in ["reminder","demonstration","tool","retarget"]) for cause in causes]
    gate("opportunity-realization",{
        "same_initial_choice":baseline == [0,0,0,0],
        "diagnostic_responses_separate":len(set(vectors)) == 4,
        "known_reminder_response":vectors[2][0] == 1 and vectors[0][0] == 0,
        "annotation_only_break_detected":enact(make("consideration"),"baseline",lapse_event=False)["artifact"] != vectors[2][0],
        "ambiguous_initial_boundary":len(set(baseline)) == 1,
        "null_reminder_for_goal":enact(make("purpose"),"reminder",lapse_event=False)["artifact"] == 0})
    false = make("false-affordance")
    corrected = intervene(false,"demonstration")
    gate("nonnested-capabilities",{
        "impossible_action_is_believed_and_considered":1 in false.beliefs and 1 in false.considered and 1 not in false.actual,
        "false_belief_execution_fails":not enact(false,"baseline",lapse_event=False)["legal"],
        "actual_demonstration_corrects_belief":1 not in corrected.beliefs,
        "annotation_only_correction_detected":decide(false)[0] != decide(corrected)[0],
        "realized_null_noop":enact(make("physical"),"baseline",lapse_event=False)["artifact"] == 0,
        "artifact_boundary":enact(false,"baseline",lapse_event=False)["artifact"] == enact(make("knowledge"),"baseline",lapse_event=False)["artifact"]})
    public = {"schema_version":"v16.opportunity.1","task_id":"opaque","access_tier":"paid-probe",
              "final_artifact":0,"observations":[{"probe":"reminder","artifact":1}],
              "allowed_causes":causes+["search"],"target_probe":"demonstration","query_cost":1,
              "reader_search_budget":128}
    inferred = public_reader(canonical(public))
    ignored = public_reader(canonical(public),"generic")
    fixed = public_reader(canonical(public),"fixed-menu")
    altered = dict(public,task_id="different-file-seed")
    gate("opportunity-inference",{
        "normalizes":abs(sum(inferred["cause_posterior"].values())-1) < 1e-10,
        "data_ignoring_break_detected":inferred["cause_posterior"] != ignored["cause_posterior"],
        "wrong_cause_support_is_distinct_arm":fixed["cause_posterior"] != inferred["cause_posterior"],
        "identifier_invariant":inferred == public_reader(canonical(altered)),
        "noise_keeps_remaining_ambiguity":sum(value > 0 for value in inferred["cause_posterior"].values()) >= 2,
        "successful_action_rules_out_inability":inferred["cause_posterior"]["physical"] == 0,
        "known_lapse_probability":abs(response_probability("consideration",0.1,0.08,"reminder")-0.92) < 1e-10})
    return {"gates":rows,"evidence_scope":"fixture",
            "instrument_state":"valid" if all(row["instrument_state"] == "valid" for row in rows) else "failed"}
