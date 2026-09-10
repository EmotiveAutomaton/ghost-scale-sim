"""Constructed ambiguity witnesses and genuine separating public observations.

These are known-answer controls, not new maker samples. A posterior over an
acquired repertoire, an exact program, and a useful prediction are distinct.
"""
import copy
from .records import canonical
from .reference import interpret
from .recoding import close

PREFIX = "ghostscale.validation.soundingline.v16."


class Recorder:
    def __init__(self, reader):
        self.reader, self.frames = reader, []

    def call(self, kind, public, **options):
        result = self.reader.request(kind, public, **options)
        self.frames.append({"kind": kind, "public": copy.deepcopy(public), "options": options, "result": result})
        return result


def ambiguous(values):
    values = list(values.values()) if isinstance(values, dict) else list(values)
    return abs(sum(values)-1) < 1e-10 and sum(value > 1e-12 for value in values) > 1


def different(left, right):
    return not close(left, right)


def reading_case(log):
    public = {"schema_version": "v16.public.2", "task_id": "collision", "lineage_id": "fixture",
        "access_tier": "artifact-only", "final_artifact": 3,
        "declared_context": {"training_attempts": 16, "current_observation": {
            "artifact": 3, "target": 3, "feasible": list(range(8)), "prefix": []}},
        "permitted_prior_artifacts": [], "permitted_query_descriptions": [], "query_costs": {},
        "target_request": {"future_target": 12}, "reader_action_budget": 128}
    witnesses = [[0, 1], [1, 0]]
    before = log.call("reading", public, strategy="maker")
    after = []
    for program in witnesses:
        observed = copy.deepcopy(public)
        observed["declared_context"]["current_observation"]["prefix"] = program[:1]
        observed["access_tier"] = "paid-process-prefix"
        observed["permitted_query_descriptions"] = ["process"]
        observed["query_costs"] = {"process": 1}
        after.append(log.call("reading", observed, strategy="maker"))
    direct = log.call("reading", public, strategy="direct-table")
    checks = {"two_executed_histories_same_artifact": all(interpret(route)["artifact"] == 3 for route in witnesses),
        "repertoire_posterior_retains_support": ambiguous(before["history_probabilities"]),
        "real_prefix_changes_repertoire_evidence": different(after[0]["history_probabilities"], after[1]["history_probabilities"]),
        "prefix_does_not_invent_unique_repertoire": all(ambiguous(item["history_probabilities"]) for item in after),
        "same_information_prediction_needs_no_explicit_history": close(before["future_probabilities"], direct["future_probabilities"])}
    return checks, {"histories": witnesses, "artifact": 3,
        "posterior_target": "acquired repertoire class, not primitive program string",
        "separating_observation": "actually executed first primitive of the current history"}


def inquiry_case(log):
    from .inquiry_stable import PRIOR
    maps = [(0, 1, 2, 3), (1, 0, 2, 3)]
    before_public = {"beliefs": [list(PRIOR), list(PRIOR)], "examples": [], "forget_domains": []}
    before = log.call(PREFIX+"inquiry_stable:learn_public", before_public)
    constructed = log.call(PREFIX+"inquiry_stable:construct_public", {"beliefs": before["beliefs"]})
    before_reading = log.call(PREFIX+"inquiry_stable:read_maker_public", {"belief": before["beliefs"][0], "artifact": 3})
    after = []
    for mapping in maps:
        observed = copy.deepcopy(before_public)
        observed["examples"] = [{"domain": 0, "command": 0, "cell": mapping[0]}]
        after.append(log.call(PREFIX+"inquiry_stable:learn_public", observed))
    readings = [log.call(PREFIX+"inquiry_stable:read_maker_public", {"belief": item["beliefs"][0], "artifact": 3}) for item in after]
    checks = {"different_maps_same_executed_two_command_artifact": [sum(1 << mapping[c] for c in (0, 1)) for mapping in maps] == [3, 3],
        "no_feedback_does_not_identify_map": ambiguous(before["beliefs"][0]),
        "one_real_command_output_separates_candidates": different(after[0]["beliefs"][0], after[1]["beliefs"][0]),
        "one_example_does_not_identify_every_command": all(ambiguous(item["beliefs"][0]) for item in after),
        "same_artifact_keeps_two_new_maker_orders": abs(before_reading["future_probabilities"][0]-before_reading["future_probabilities"][1]) < 1e-10,
        "actual_example_changes_unseen_maker_continuation": different(readings[0]["future_probabilities"], readings[1]["future_probabilities"]),
        "executable_choice_is_not_a_claim_of_unique_history": len(constructed["programs"]) == 12}
    return checks, {"maps": maps, "commands": [0, 1], "artifact": 3,
                    "separating_observation": "permitted command0/output example", "posterior_target": "map and irreducible-noise law"}


def opportunity_case(log):
    from .opportunity import make, enact
    causes = ["physical", "knowledge", "consideration", "purpose", "search"]
    baseline = [enact(make(cause), "baseline", lapse_event=False) for cause in causes]
    public = {"schema_version": "v16.opportunity.1", "task_id": "collision", "access_tier": "artifact-only",
        "final_artifact": 0, "observations": [], "allowed_causes": causes, "target_probe": "demonstration",
        "query_cost": 0, "reader_search_budget": 128}
    before = log.call("opportunity", public, model="latent-menu")
    action = enact(make("consideration"), "reminder", lapse_event=False)
    observed = {**public, "access_tier": "paid-reminder", "query_cost": 1,
                "observations": [{"probe": "reminder", "artifact": action["artifact"]}]}
    after = log.call("opportunity", observed, model="latent-menu")
    return {"different_causes_same_initial_artifact": all(work["artifact"] == 0 for work in baseline),
        "initial_causes_remain_ambiguous": ambiguous(before["cause_posterior"]),
        "successful_real_reminder_rules_out_physical_inability": action["artifact"] == 1 and after["cause_posterior"]["physical"] == 0,
        "remaining_noise_and_cause_ambiguity_preserved": ambiguous(after["cause_posterior"]),
        "actual_intervention_changes_account": different(before["cause_posterior"], after["cause_posterior"])}, {
        "causes": causes, "baseline_executions": baseline, "separating_execution": action,
        "posterior_target": "distinct opportunity causes, not intended program"}


def self_case(log, trajectory=False):
    base = {"schema_version": "v16.self.1", "task_id": "collision", "access_tier": "absent-memory",
        "memory": {"original_goal": None, "controller": None}, "artifacts": [0], "current_goal": None,
        "retargeted": False, "memory_reliability": 1.0, "artifact_reliability": 1.0,
        "monitoring_cost": 1.0, "repair_cost": 0.2, "target_request": "known-original-goal"}
    learned = copy.deepcopy(base)
    learned["memory"]["original_goal"] = 0
    learned["access_tier"] = "accurate-goal-memory"
    if trajectory:
        public = {"schema_version": "v16.self-trajectory.1", "task_id": "collision", "base": base, "history": []}
        observed = {**public, "base": learned}
        kind = "trajectory"
    else:
        public, observed, kind = base, learned, "self"
    before = log.call(kind, public, strategy="self-model")
    after = log.call(kind, observed, strategy="self-model")
    rival = log.call(kind, observed, strategy="bayes-error")
    return {"same_artifact_does_not_identify_original_goal": ambiguous(before["original_goal_probabilities"]),
        "accurate_recovered_goal_memory_separates_goals": close(after["original_goal_probabilities"], [1, 0]),
        "goal_memory_does_not_force_known_controller": ambiguous(after["controller_probabilities"]),
        "same_information_error_rival_agrees": close(after["controller_probabilities"], rival["controller_probabilities"]),
        "declared_goal_stays_separate_from_adopted_goal": observed.get("current_goal", observed.get("base", {}).get("current_goal")) is None}, {
        "artifact": 0, "original_goals": [0, 1], "separating_observation": "accurate original-goal memory",
        "scope": "one noisy artifact has positive likelihood under both acquired-goal histories; control need not be known"}


def recognition_case(log):
    from .recognition_gates import fixture
    from .graphic_reference import interpret as physical
    kind = PREFIX+"recognition:public_reader"
    before = log.call(kind, fixture(), method="craft")
    after = log.call(kind, fixture(process=True), method="craft")
    routes = [[0, 1, 2, 4, 5], [1, 0, 2, 4, 5]]
    return {"distinct_orders_execute_same_artifact": physical(routes[0])["artifact"] == physical(routes[1])["artifact"],
        "recognition_does_not_identify_order": before["identity"][0] > 0.9 and close(before["historical_core"], [0.5, 0.5]),
        "unseen_core_prediction_remains_chance": close(before["future_core"], [0.5, 0.5]),
        "observed_first_action_identifies_current_order": close(after["historical_core"], [1, 0]),
        "process_evidence_changes_future_prediction": different(before["future_core"], after["future_core"])}, {
        "programs": routes, "separating_observation": "actual first primitive", "targets": "identity, historical order and future order separately"}


def selection_case(log):
    from .selection_gates import fixture
    kind = PREFIX+"selection:public_reader"
    before = log.call(kind, fixture(), method="selection-aware")
    exposed = fixture(full=True)
    for batch in exposed["history"]:
        for event in batch["released"]:
            event["first_action"] = 1
        for event in batch["full_candidates"]:
            event["observation"]["first_action"] = 1
    after = log.call(kind, exposed, method="selection-aware")
    return {"selected_style_does_not_identify_core": close(before["future_raw_core"], [0.5, 0.5]),
        "same_artifacts_with_actual_order_observations_change_prediction": different(before["future_raw_core"], after["future_raw_core"]),
        "rejected_production_changes_style_account": different(before["acquired_style"], after["acquired_style"]),
        "future_motor_variation_remains": ambiguous(after["future_raw_core"])}, {
        "separating_observation": "all actually produced candidates and their first primitives",
        "physical_witness": "swap the initial place0/place1 order in each fixture program; its artifact is unchanged"}


def audience_case(log):
    from .audience_gates import fixture
    kind = PREFIX+"audience:public_reader"
    before = log.call(kind, fixture("none"), policy="none")
    context = log.call(kind, fixture("audience-only"), policy="audience-only")
    after = log.call(kind, fixture("both"), policy="both")
    return {"old_orders_remain_ambiguous": ambiguous(before["historical_vector"]),
        "audience_alone_is_not_separating_old_order_evidence": close(before["historical_vector"], context["historical_vector"]),
        "rehearsal_process_changes_repertoire_evidence": different(before["historical_core"], after["historical_core"]),
        "new_rehearsal_does_not_identify_every_old_route": ambiguous(after["historical_vector"])}, {
        "separating_observation": "actual rehearsal first primitives; audience is a distinct intervention",
        "remaining_boundary": "old current-route order remains uncertain even after new rehearsal"}


def multi_actor_case(log):
    from .multi_actor_gates import fixture
    kind = PREFIX+"multi_actor:public_reader"
    own, other = fixture(revision="self"), fixture(revision="other")
    before = log.call(kind, own, policy="all")
    twin = log.call(kind, other, policy="all")
    absent = log.call(kind, fixture(revision="none"), policy="all")
    return {"two_real_actor_topologies_same_all_permitted_evidence": own == other,
        "identical_evidence_preserves_prediction": before == twin,
        "no_invented_second_maker": before["revision_relation"][1] > 0 and before["revision_relation"][2] > 0,
        "actual_absence_of_revision_is_separable": close(absent["revision_relation"], [1, 0, 0])}, {
        "remaining_boundary": "self and other are not separated when independently acquired routines coincide",
        "positive_control": "an actually absent revision can be identified; no fictitious ownership observation"}


def tradeoff_case(log):
    from .tradeoffs_gates import fixture
    from .assembly import World, execute
    kind = PREFIX+"tradeoffs:public_reader"
    informative = fixture()
    forced = copy.deepcopy(informative)
    for event in forced["history"]:
        event["context"]["max_steps"] = 1
        event["state"] = list(event["context"]["initial"])
    before = log.call(kind, forced, policy="chronological")
    after = log.call(kind, informative, policy="chronological")
    world = World(tuple(forced["world"]["parents"]), tuple(forced["world"]["defaults"]))
    return {"forced_stop_histories_actually_execute": all(execute(world, [9], initial=event["context"]["initial"])["state"] == event["state"] for event in forced["history"]),
        "forced_history_does_not_identify_profile": close(list(before["profile_posterior"].values()), [0.2]*5),
        "real_options_supply_separating_evidence": different(before["profile_posterior"], after["profile_posterior"]),
        "finite_noisy_history_does_not_force_unique_profile": ambiguous(after["profile_posterior"])}, {
        "collision": "same eight forced STOP choices under all five profile schedules",
        "separating_observation": "actual choices among non-singleton physically executable menus"}


def preference_case(log):
    from .preference_probe_gates import fixture
    from .assembly import World, execute
    kind = PREFIX+"preference_probe:public_reader"
    public = fixture()
    before = log.call(kind, {**public, "phase": 2}, policy="none")
    after = log.call(kind, {**public, "phase": 2, "answer": {"query": "all", "state": [1, 0, 0]}}, policy="all")
    execution = execute(World(), [2, 9], initial=[1, 0, -1])
    return {"shared_history_keeps_all_causes": close(before["cause_posterior"], [1/3]*3),
        "probe_observation_is_physically_realizable": execution["state"] == [1, 0, 0] and execution["successfully_stopped"],
        "probe_changes_cause_evidence": different(before["cause_posterior"], after["cause_posterior"]),
        "still_indistinguishable_causes_keep_equal_weight": abs(after["cause_posterior"][1]-after["cause_posterior"][2]) < 1e-10,
        "probe_does_not_manufacture_unique_cause": ambiguous(after["cause_posterior"])}, {
        "collision": "identical forced-stop histories of profile, capability and audience rivals",
        "remaining_boundary": "the full probe still leaves capability and audience causes behaviorally indistinguishable"}


def mechanism_case(log):
    world = {"family": "W1", "option_training": [[0, 1], [2, 3]]*8}
    public = {"schema_version": "v16.mechanism-reader.1", "task_id": "collision", "world": world,
        "training": {"attempts": 8}, "current": {"goal": 3, "artifact": 3}, "prior_works": [],
        "future_goal": 12, "production_budget": 128, "max_steps": 3, "beta": 1.0,
        "length_cost": 0.3, "reader_budget": 128}
    kind = PREFIX+"mechanism_reader:public_reader"
    before = log.call(kind, public, strategy="mixture")
    after = log.call(kind, {**public, "prior_works": [{"goal": 3, "artifact": 0}]*4}, strategy="mixture")
    return {"known_colliding_primitive_histories": interpret([0, 1])["artifact"] == interpret([1, 0])["artifact"] == 3,
        "initial_mechanism_remains_ambiguous": ambiguous(before["family_posterior"]),
        "legal_prior_works_supply_different_mechanism_evidence": different(before["family_posterior"], after["family_posterior"]),
        "global_collision_count_is_not_unique_history": before["grammatically_compatible_routes"] > 1}, {
        "collision": "legal place0/place1 order swap under a softmax mechanism with positive finite route weights",
        "posterior_target": "production-family and repertoire mixture",
        "scope": "global grammar collision count is separate from posterior-weighted history ambiguity; no unique route claim"}


def assembly_mechanism_case(log):
    from .assembly import World, execute
    from .assembly_reference import interpret as independent
    world = World((-1, 0, 0), (0, 0, 0))
    programs = [[0, 1, 2, 9], [0, 2, 1, 9]]
    works = [execute(world, program) for program in programs]
    observed = execute(world, [0, 6, 1, 9])
    public = {"schema_version": "v16.mechanism-reader.1", "task_id": "assembly-collision",
        "world": {"family": "W2", **world.public()},
        "training": {"attempts": 8, "reliability": 0.9, "topic_probability": 0.8},
        "current": {"goal": [0, 0, 0], "artifact": works[0]["artifact"]}, "prior_works": [],
        "future_goal": [1, 0, -1], "production_budget": 128, "max_steps": 4,
        "beta": 1.0, "length_cost": 0.2, "reader_budget": 128}
    kind = PREFIX+"mechanism_reader:public_reader"
    before = log.call(kind, public, strategy="mixture")
    after = log.call(kind, {**public, "prior_works": [{"goal": [1, 0, -1], "artifact": observed["artifact"]}]*4}, strategy="mixture")
    return {"independent_dependency_physics_agrees": all(work == independent(world.public(), program) for work, program in zip(works, programs)),
        "child_orders_give_identical_stopped_assembly": works[0]["artifact"] == works[1]["artifact"] and all(work["successfully_stopped"] for work in works),
        "finished_assembly_does_not_identify_mechanism": ambiguous(before["family_posterior"]),
        "actual_additional_work_changes_mechanism_evidence": different(before["family_posterior"], after["family_posterior"]),
        "compatible_history_count_stays_distinct_from_reconstruction": before["grammatically_compatible_routes"] >= 2}, {
        "world": world.public(), "programs": programs, "separating_work": observed,
        "scope": "independent assembly law with a separately configured reader; no fitted W1 parameter transfer"}


CASES = {"reading": reading_case, "inquiry": inquiry_case, "opportunity": opportunity_case,
         "self": self_case, "trajectory": lambda log: self_case(log, True), "recognition": recognition_case,
         "selection": selection_case, "audience": audience_case, "multi-actor": multi_actor_case,
         "tradeoffs": tradeoff_case, "preference-probe": preference_case, "mechanism": mechanism_case,
         "assembly-mechanism": assembly_mechanism_case}
