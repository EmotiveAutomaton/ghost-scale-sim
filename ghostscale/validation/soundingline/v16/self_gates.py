"""Self-control admission, before any unknown-outcome comparison."""
from .self_monitor import generate,public_reader,evaluate
from .records import canonical


def run():
    gates = []
    def gate(name,checks):
        gates.append({"id":name,"checks":checks,"evidence_scope":"fixture",
                      "instrument_state":"valid" if all(checks.values()) else "failed"})
    public,private,_ = generate("self-gate",{"id":"intact","memory":"intact"},0)
    full = public_reader(canonical(public))
    memory = public_reader(canonical(public),strategy="memory-only")
    partial = {**public,"memory":{**public["memory"],"controller":None},"artifacts":[1-private["controller"]]}
    informed = public_reader(canonical(partial))
    ignored = public_reader(canonical(partial),strategy="memory-only")
    gate("self-information",{
        "intact_control_null":full["controller_probabilities"] == memory["controller_probabilities"],
        "partial_information_changes_posterior":informed["controller_probabilities"] != ignored["controller_probabilities"],
        "data_ignoring_break_detected":informed["controller_probabilities"] != ignored["controller_probabilities"],
        "partial_memory_retains_ambiguity":all(0 < p < 1 for p in informed["controller_probabilities"]),
        "private_state_not_in_reader":public_reader(canonical(partial)) == informed})
    direct = public_reader(canonical(partial),strategy="bayes-error")
    gate("ordinary-error-rival",{
        "same_evidence_same_cost":informed["costs"] == direct["costs"],
        "binary_sufficient_error_boundary":informed["repair"] == direct["repair"],
        "probability_equivalence":max(abs(a-b) for a,b in zip(informed["controller_probabilities"],direct["controller_probabilities"])) < 1e-10,
        "ignored_evidence_is_detectably_different":informed["controller_probabilities"] != ignored["controller_probabilities"]})
    private = {**private,"controller":0,"actual_original_goal":0,"execution_error":0.1}
    reset = {**informed,"adopted_goal":1,"repair":True,"original_goal_probabilities":[1.0,0.0]}
    useful = evaluate(private,reset,reset_coin=0.9,future_coin=0.9)
    harmful = evaluate({**private,"controller":1},reset,reset_coin=0.01,future_coin=0.9)
    untouched = evaluate(private,{**reset,"repair":False},reset_coin=0.9,future_coin=0.9)
    gate("executed-repair",{
        "correct_repair_changes_control":useful["net_repair"] == 1 and useful["controller_after"] == 1,
        "harmful_reset_degrades_execution":harmful["net_repair"] == -1 and harmful["adopted_goal_success"] == 0,
        "annotation_only_repair_detected":untouched["controller_after"] != useful["controller_after"],
        "no_repair_null":untouched["net_repair"] == 0,
        "goal_outcomes_separate":useful["adopted_goal_success"] == 1 and useful["original_goal_success"] == 0 and useful["original_goal_recovery"] == 1})
    source,hidden,_ = generate("source-gate",{"id":"false-memory","memory":"misleading"},0)
    actual_flip = source["memory"]["original_goal"] == 1-hidden["actual_original_goal"]
    source["artifacts"] = []
    source["memory"] = {"original_goal":0,"controller":None}
    source["memory_reliability"] = 0.8
    weighted = public_reader(canonical(source))
    source["memory_reliability"] = 0.5
    null = public_reader(canonical(source))
    source["memory_reliability"] = 1.0
    broken = public_reader(canonical(source))
    gate("source-reliability",{
        "source_corruption_is_actual":actual_flip,
        "known_reliability_posterior":abs(weighted["original_goal_probabilities"][0]-0.8)<1e-10,
        "uninformative_source_null":abs(null["original_goal_probabilities"][0]-0.5)<1e-10,
        "blind_trust_break_detected":abs(broken["original_goal_probabilities"][0]-0.8)>0.1,
        "uncertain_source_boundary":all(0<p<1 for p in weighted["original_goal_probabilities"])})
    return {"gates":gates,"evidence_scope":"fixture",
            "instrument_state":"valid" if all(g["instrument_state"] == "valid" for g in gates) else "failed"}
