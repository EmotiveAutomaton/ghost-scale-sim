"""X05 actual false-context interventions, corrections and trust boundaries.

Known-answer attacks measure susceptibility as well as correction. A trusted
false observation need not be detectable from the observation itself. Replacement
after a verified source correction is distinguished from learning from an
additional execution; neither is silently represented as a second observation.
"""
import copy
from itertools import permutations
from .collision_cases import PREFIX, ambiguous, different
from .recoding import close


def self_case(log, trajectory=False):
    base = {"schema_version": "v16.self.1", "task_id": "false-memory-control", "access_tier": "uncertain-memory",
        "memory": {"original_goal": 1, "controller": None}, "artifacts": [], "current_goal": 0,
        "retargeted": True, "memory_reliability": 0.8, "artifact_reliability": 1.0,
        "monitoring_cost": 1.0, "repair_cost": 0.2, "target_request": "original-goal-and-next-action"}
    def call(value, history=()):
        public = {"schema_version": "v16.self-trajectory.1", "task_id": "false-memory-control", "base": value, "history": list(history)} if trajectory else value
        return log.call("trajectory" if trajectory else "self", public, strategy="self-model")
    wrong = call(base)
    observed = copy.deepcopy(base)
    observed["artifacts"] = [0]*8
    data = call(observed)
    corrected = copy.deepcopy(observed)
    corrected["memory"] = {"original_goal": 0, "controller": None}
    corrected["memory_reliability"] = 1.0
    source = call(corrected)
    checks = {"wrong_memory_moves_old_goal_belief": wrong["original_goal_probabilities"][1] > 0.5,
        "fresh_external_artifacts_correct_old_goal_probability": data["original_goal_probabilities"][0] > wrong["original_goal_probabilities"][0],
        "prediction_changes_with_actual_external_evidence": data["future_probabilities"][0] > wrong["future_probabilities"][0],
        "declared_new_goal_is_not_rewritten_by_false_old_memory": all(item["adopted_goal"] == 0 for item in [wrong, data, source]),
        "corrected_verified_memory_recovers_old_goal": close(source["original_goal_probabilities"], [1, 0]),
        "noisy_artifacts_do_not_force_unique_old_goal": ambiguous(data["original_goal_probabilities"])}
    if trajectory:
        intervention = call(base, [{"reset_requested": True, "reset_goal": 0, "observed_artifact": 0}])
        checks["own_reset_output_does_not_verify_forgotten_goal"] = close(intervention["original_goal_probabilities"], wrong["original_goal_probabilities"])
        checks["own_reset_can_change_current_prediction"] = different(intervention["future_probabilities"], wrong["future_probabilities"])
    return checks, {"actual_original_goal": 0, "false_memory": 1, "new_declared_goal": 0,
        "new_evidence": "eight actual independent noisy-controller emissions of zero, with positive probability under the declared model",
        "correction": "separately verified old-goal memory replaces the false source; original attack receipt retained",
        "limit": "uncertain memory can be countered; a falsely declared perfect source is not automatically recognized as false"}


def inquiry_case(log):
    maps = list(permutations(range(4)))
    wrong_index = maps.index((1, 0, 2, 3))
    belief = [0.2/24]*25
    belief[wrong_index] = 0.8
    public = {"beliefs": [belief, belief], "examples": [], "forget_domains": []}
    before = log.call(PREFIX+"inquiry_stable:learn_public", public)
    choice = log.call(PREFIX+"inquiry_stable:construct_public", {"beliefs": before["beliefs"]})
    corrected = copy.deepcopy(public)
    corrected["examples"] = [{"domain": 0, "command": 0, "cell": 0}]
    after = log.call(PREFIX+"inquiry_stable:learn_public", corrected)
    new_choice = log.call(PREFIX+"inquiry_stable:construct_public", {"beliefs": after["beliefs"]})
    old_maker = log.call(PREFIX+"inquiry_stable:read_maker_public", {"belief": before["beliefs"][0], "artifact": 3})
    new_maker = log.call(PREFIX+"inquiry_stable:read_maker_public", {"belief": after["beliefs"][0], "artifact": 3})
    return {"incorrect_uncertain_prior_is_retained_without_feedback": close(before["beliefs"][0], belief),
        "actual_command_outcome_rules_out_wrong_mapping": after["beliefs"][0][wrong_index] == 0,
        "one_example_leaves_other_map_entries_uncertain": ambiguous(after["beliefs"][0]),
        "untouched_domain_does_not_gain_private_knowledge": close(after["beliefs"][1], before["beliefs"][1]),
        "permitted_feedback_changes_construction": choice["programs"] != new_choice["programs"],
        "new_maker_prediction_changes_separately": different(old_maker["future_probabilities"], new_maker["future_probabilities"])}, {
        "actual_mapping": [0, 1, 2, 3], "misleading_prior_mapping": list(maps[wrong_index]),
        "actual_feedback": corrected["examples"], "scope": "incorrect uncertain prior corrected by real feedback; not a model of corrupted labels",
        "goal_scope": "construction target and Bayesian belief are separate inputs; no goal-adoption channel exists in this reader"}


def opportunity_case(log):
    from .opportunity import make, enact, CAUSES
    maker = make("false-affordance")
    base = {"schema_version": "v16.opportunity.1", "task_id": "false-affordance-control", "access_tier": "artifact-only",
        "final_artifact": 0, "observations": [], "allowed_causes": list(CAUSES), "target_probe": "higher-reward",
        "query_cost": 0, "reader_search_budget": 1024}
    before = log.call("opportunity", base, model="latent-menu")
    executions = [enact(maker, probe, lapse_event=False) for probe in ["baseline", "higher-reward", "reminder", "tool"]]
    observed = {**base, "observations": [{"probe": probe, "artifact": output["artifact"]} for probe, output in zip(
        ["baseline", "higher-reward", "reminder", "tool"], executions)], "query_cost": 3, "access_tier": "paid-opportunity-tests"}
    after = log.call("opportunity", observed, model="latent-menu")
    restricted = log.call("opportunity", observed, model="accurate-belief")
    return {"actual_belief_exceeds_physical_feasibility": 1 in maker.beliefs and 1 not in maker.actual,
        "false_affordance_attempt_fails_physically": executions[0]["artifact"] == 0,
        "real_tool_changes_executable_outcome": executions[-1]["artifact"] == 1,
        "actual_interventions_change_cause_account": different(before["cause_posterior"], after["cause_posterior"]),
        "actual_interventions_change_future_prediction": different(before["future_probabilities"], after["future_probabilities"]),
        "cause_ambiguity_survives": ambiguous(after["cause_posterior"]),
        "accurate_belief_rival_cannot_recover_excluded_false_belief": "false-affordance" not in restricted["cause_posterior"]}, {
        "executions": executions, "actual_cause": "false-affordance",
        "limit": "an excluded cause cannot be recovered by inference; interventions still need not separate every physical/belief rival",
        "goal_scope": "maker purpose is distinct from reader cause belief; this reader has no adopted-goal output"}


def recognition_case(log):
    from .recognition_gates import fixture
    public = fixture()
    true_references = copy.deepcopy(public["references"])
    public["references"] = list(reversed(public["references"]))
    wrong = log.call(PREFIX+"recognition:public_reader", public, method="craft")
    corrected = copy.deepcopy(public)
    for group in range(2):
        corrected["references"][group].extend(true_references[group]*4)
    after = log.call(PREFIX+"recognition:public_reader", corrected, method="craft")
    return {"misassigned_reference_context_misidentifies_source": wrong["identity"][1] > 0.9,
        "later_actual_labeled_examples_correct_identity": after["identity"][0] > 0.9,
        "surface_context_cannot_rewrite_hidden_order": close(wrong["historical_core"], [0.5, 0.5]) and close(after["historical_core"], [0.5, 0.5]),
        "identity_correction_does_not_invent_core_prediction": close(after["future_core"], [0.5, 0.5])}, {
        "misleading_context": "three incorrectly assigned style references per source",
        "new_evidence": "twelve further genuinely labeled style examples per source, feasible repeated emissions under the public law",
        "limit": "reference labels are trusted observations; the reader cannot establish their provenance from style alone",
        "goal_scope": "identity confidence and order prediction are distinct; there is no adopted-goal output"}


def audience_case(log):
    from .audience_gates import fixture
    from .graphic_reference import interpret
    public = fixture("audience-only")
    wrong = log.call(PREFIX+"audience:public_reader", public, policy="audience-only")
    corrected = {**public, "audience": 0}
    after = log.call(PREFIX+"audience:public_reader", corrected, policy="audience-only")
    return {"false_audience_context_changes_real_action": wrong["new_composition"] != after["new_composition"],
        "corrected_context_produces_actual_audience_zero_work": interpret(after["new_composition"])["artifact"] == sum(2**i for i in [0, 1, 3, 4, 5]),
        "context_does_not_rewrite_old_core_history": close(wrong["historical_core"], after["historical_core"]) and close(wrong["historical_vector"], after["historical_vector"]),
        "both_submissions_remain_executable": all(interpret(item["new_composition"])["legal"] for item in [wrong, after])}, {
        "actual_audience": 0, "false_context": 1, "correction": "verified audience cue replaces previous false cue",
        "limit": "the finite reader treats supplied audience context as exact; correction requires new source information, not a claim of automatic deception detection"}


def dependency_case(log):
    from .assembly import World
    from .assembly_reference import interpret
    world = World().public()
    actual = interpret(world, [0, 6, 1])["state"]
    false_training = [{"date": d, "program": [0, 1, 9], "target": [0, 0, -1], "feedback": True} for d in range(8)]
    public = {"schema_version": "v16.dependency-monitor.1", "task_id": "false-routine-memory", "world": world,
        "training": false_training, "goal": [0, 0, -1], "visible_parts": [0, 1], "phase": 2,
        "preparatory_program": [], "inspected_state": None, "search_budget": 512, "old_routine_reused": True}
    wrong = log.call(PREFIX+"dependency_monitor:public_reader", public, policy="self-model")
    corrected = {**public, "inspected_state": actual}
    after = log.call(PREFIX+"dependency_monitor:public_reader", corrected, policy="inspect-now")
    execution = interpret(world, after["repair"]["program"], initial=actual)
    return {"false_training_memory_hides_actual_orientation_conflict": not wrong["detected_conflict"] and wrong["assumed_state"] != actual,
        "real_orientation_inspection_corrects_assumed_state": after["assumed_state"] == actual and after["detected_conflict"],
        "corrected_repair_executes_from_true_state": execution["successfully_stopped"] and execution["state"] == public["goal"],
        "goal_itself_is_not_replaced": public["goal"] == corrected["goal"]}, {
        "actual_old_program": [0, 6, 1], "misremembered_old_program": [0, 1], "inspection": actual,
        "inspection_charge": "the fixture supplies a purchased inspection; scientific receipts separately charge the request before its result",
        "limit": "faithful routine memory is an explicit S03 assumption; false memory defeats simulation until external inspection corrects it"}


def multi_actor_case(log):
    from .multi_actor_gates import fixture
    public = fixture(shared=True)
    false = copy.deepcopy(public)
    false["brief_view"]["new_brief"] = public["brief"]
    rejected = None
    try:
        wrong = log.call(PREFIX+"multi_actor:public_reader", false, policy="all")
    except RuntimeError as error:
        if str(error) != "ValueError: incorrect brief intervention":
            raise
        rejected = str(error)
        log.frames.append({"kind": PREFIX+"multi_actor:public_reader", "public": copy.deepcopy(false),
                           "options": {"policy": "all"}, "request_error": rejected})
        wrong = {}
    after = log.call(PREFIX+"multi_actor:public_reader", public, policy="all")
    detected = rejected is not None or wrong.get("model_mismatch", False)
    changed = detected or different(wrong.get("shared_brief"), after["shared_brief"])
    return {"actual_changed_brief_separates_shared_instruction": close(after["shared_brief"], [0, 1]),
        "incorrect_probe_context_is_detected_or_changes_account": changed,
        "correct_context_does_not_invent_revision_ownership": after["revision_relation"][1] > 0 and after["revision_relation"][2] > 0}, {
        "misleading_context": "the changed-brief result is falsely labeled as an unchanged-brief observation",
        "correction": "verified probe context replaces the mislabeled field; original false receipt preserved",
        "false_record_detected_as_mismatch": detected, "explicit_request_rejection": rejected,
        "limit": "trusted context is part of the model; source correction is not endogenous inference of source honesty"}


def tradeoffs_case(log):
    from .tradeoffs_gates import fixture
    public = fixture()
    false = copy.deepcopy(public)
    for event in false["history"]:
        event["context"]["price"] = 0.8
    wrong = log.call(PREFIX+"tradeoffs:public_reader", false, policy="chronological")
    after = log.call(PREFIX+"tradeoffs:public_reader", public, policy="chronological")
    return {"false_cost_context_changes_inferred_profile": different(wrong["profile_posterior"], after["profile_posterior"]),
        "corrected_cost_context_changes_future_prediction": different(wrong["probabilities"], after["probabilities"]),
        "finite_profile_uncertainty_retained": ambiguous(after["profile_posterior"]),
        "physical_observations_and_future_task_unchanged": [event["state"] for event in false["history"]] == [event["state"] for event in public["history"]] and false["future_context"] == public["future_context"]}, {
        "actual_historical_price": 0.05, "misreported_price": 0.8,
        "correction": "verified dated price records replace erroneous context; observed maker actions unchanged",
        "limit": "profile recovery is conditional on the supplied cost law and context; plausible false costs can bias it without a mismatch flag",
        "goal_scope": "future task remains fixed; the reader infers a maker profile and has no own-goal adoption channel"}


def preference_case(log):
    from .preference_probe_gates import fixture
    from .assembly import World
    from .assembly_reference import interpret
    public = fixture()
    actual_state = interpret(World().public(), [2, 9], initial=[1, 0, -1])["state"]
    false = {**public, "phase": 2, "answer": {"query": "all", "state": [0, 0, 0]}}
    wrong = log.call(PREFIX+"preference_probe:public_reader", false, policy="all")
    corrected = {**public, "phase": 2, "answer": {"query": "all", "state": actual_state}}
    after = log.call(PREFIX+"preference_probe:public_reader", corrected, policy="all")
    return {"misreported_probe_changes_cause_account": different(wrong["cause_posterior"], after["cause_posterior"]),
        "corrected_probe_changes_future_prediction": different(wrong["probabilities"], after["probabilities"]),
        "recovered_probe_keeps_rivals_indistinguishable": after["cause_posterior"][1] == after["cause_posterior"][2] and after["cause_entropy"] > 0,
        "old_artifact_and_historical_programs_do_not_change": wrong["reproduction_program"] == after["reproduction_program"] and wrong["historical_programs"] == after["historical_programs"]}, {
        "actual_probe_program": [2, 9], "actual_start": [1, 0, -1], "actual_state": actual_state,
        "misreported_state": [0, 0, 0], "correction": "external verification corrects the paid probe answer; this is not an extra independent observation",
        "limit": "a plausible false answer can mislead the trusted-input model; no independent sensor-authenticity model is implemented"}


CASES = {"self": self_case, "trajectory": lambda log: self_case(log, True), "inquiry": inquiry_case,
    "opportunity": opportunity_case, "recognition": recognition_case, "audience": audience_case,
    "dependency": dependency_case, "multi-actor": multi_actor_case, "tradeoffs": tradeoffs_case,
    "preference-probe": preference_case}
