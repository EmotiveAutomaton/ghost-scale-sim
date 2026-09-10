"""X06 known alternative generators, omitted procedures and model-scope failures.

An attack may expose a predictive error without a mismatch flag when the false
model assigns positive probability to the data. Such failures are retained as
limits of the scientific reader, not converted into an immunity certificate.
"""
import copy
from .collision_cases import PREFIX, ambiguous, different
from .recoding import close


def read_or_reject(log, kind, public, **options):
    try:
        return log.call(kind, public, **options)
    except RuntimeError as error:
        if not str(error).startswith("ValueError:"):
            raise
        rejection = {"explicit_rejection": str(error)}
        log.frames.append({"kind": kind, "public": copy.deepcopy(public), "options": options, "request_error": str(error)})
        return rejection


def mismatch(result):
    return (result.get("model_mismatch") is True or "explicit_rejection" in result) and not any(
        key in result for key in ["probabilities", "future_probabilities", "cause_posterior", "identity"])


def extended_board(program):
    """Independent changed-horizon world: four rather than three transitions."""
    if len(program) > 4 or any(type(action) is not int or not 0 <= action < 8 for action in program):
        raise ValueError("outside declared four-step attack world")
    cells = [0]*4
    for action in program:
        cells[action % 4] = int(action < 4)
    return sum(value*2**i for i, value in enumerate(cells))


def reading_case(log):
    public = {"schema_version": "v16.public.2", "task_id": "changed-horizon", "lineage_id": "fixture",
        "access_tier": "artifact-only", "final_artifact": 15,
        "declared_context": {"training_attempts": 16, "current_observation": {
            "artifact": 15, "target": 15, "feasible": list(range(8)), "prefix": []}},
        "permitted_prior_artifacts": [], "permitted_query_descriptions": [], "query_costs": {},
        "target_request": {"future_target": 12}, "reader_action_budget": 128}
    results = [read_or_reject(log, "reading", public, strategy=policy) for policy in ["maker", "direct-table", "generic", "primitive"]]
    return {"alternative_generator_really_produces_four_cell_artifact": extended_board([0, 1, 2, 3]) == 15,
        "all_declared_readers_expose_missing_support": all(mismatch(result) for result in results),
        "no_uniform_predictive_fallback": all("future_probabilities" not in result for result in results)}, {
        "actual_program": [0, 1, 2, 3], "actual_horizon": 4, "assumed_horizon": 3,
        "scope": "physically extended finite world outside the reader's registered production horizon; no inference claimed from unsupported artifact"}


def options_case(log):
    from .options import expand
    training = {"attempts": [[0, 1]]*4, "targets": [3]*4}
    public = {"schema_version": "v16.options.1", "task_id": "omitted-horizon", "targets": [15],
        "own_training": training, "pooled_training": copy.deepcopy(training), "budget": 4096}
    result = log.call(PREFIX+"options_study:public_read", public)
    graph = result["options"]["graph"]
    unsupported = [expand(option, 15, graph["edges"]) for option in graph["options"]]
    return {"extended_world_really_reaches_target": extended_board([0, 1, 2, 3]) == 15,
        "every_three_step_planner_retains_unreachable_target": all(arm["submissions"][0].get("unreachable") for arm in result.values()),
        "unobserved_state_is_not_invented_option_support": bool(unsupported) and all(item["unsupported"] for item in unsupported),
        "missing_graph_coverage_is_explicit": graph["observed_state_fraction"] < 1}, {
        "actual_program": [0, 1, 2, 3], "assumed_max_steps": 3, "changed_max_steps": 4,
        "limitation": "unreachable is relative to the finite planner and current graph, not physical impossibility in all worlds"}


def mechanism_case(log, assembly=False):
    if assembly:
        from .assembly import World
        from .assembly_reference import interpret
        world = World().public()
        executed = interpret(world, [0, 1, 2, 9])
        public = {"schema_version": "v16.mechanism-reader.1", "task_id": "new-assembly-procedure",
            "world": {"family": "W2", **world}, "training": {"attempts": 0, "reliability": 0.9, "topic_probability": 0.8},
            "current": {"goal": [0, 0, 0], "artifact": executed["artifact"]}, "prior_works": [],
            "future_goal": [0, 0, 0], "production_budget": 512, "max_steps": 2,
            "beta": 1.0, "length_cost": 0.3, "reader_budget": 512}
        narrow = read_or_reject(log, PREFIX+"mechanism_reader:public_reader", public, strategy="mixture")
        broader = log.call(PREFIX+"mechanism_reader:public_reader", {**public, "max_steps": 4}, strategy="mixture")
        return {"new_four_step_assembly_executes": executed["successfully_stopped"] and executed["state"] == [0, 0, 0],
            "assumed_short_horizon_exposes_mismatch": mismatch(narrow),
            "explicit_broader_horizon_restores_support": not broader["model_mismatch"],
            "broader_model_remains_a_distribution": abs(sum(broader["future_probabilities"])-1) < 1e-10}, {
            "actual_program": [0, 1, 2, 9], "actual_cost": executed["primitive_cost"],
            "assumed_horizon": 2, "broader_horizon": 4, "scope": "known-answer support discriminator, not fitted W1-to-W2 transfer"}
    from .reference import interpret
    public = {"schema_version": "v16.mechanism-reader.1", "task_id": "different-generator",
        "world": {"family": "W1", "option_training": [[0, 1], [2, 3]]*8}, "training": {"attempts": 8},
        "current": {"goal": 3, "artifact": 4}, "prior_works": [], "future_goal": 12,
        "production_budget": 128, "max_steps": 3, "beta": 1.0, "length_cost": 0.3, "reader_budget": 128}
    narrow = log.call(PREFIX+"mechanism_reader:public_reader", public, strategy="bounded-model")
    broader = log.call(PREFIX+"mechanism_reader:public_reader", public, strategy="mixture")
    return {"idiosyncratic_generator_really_executes_artifact": interpret([2])["artifact"] == 4,
        "assumed_goal_solver_exposes_no_support": mismatch(narrow),
        "broader_production_catalog_explains_possible_artifact": not broader["model_mismatch"],
        "broader_repertoire_not_forced_unique": ambiguous(broader["library_posterior"])}, {
        "actual_generator": "emit primitive2 regardless of requested two-cell goal", "actual_program": [2],
        "scope": "a physically legal artifact can be outside a particular producer model; broader support is not proof of the actual producer's history"}


def opportunity_case(log):
    from dataclasses import replace
    from .opportunity import make, enact, CAUSES
    maker = replace(make("purpose"), purpose=1)
    execution = enact(maker, "baseline", lapse_event=False)
    public = {"schema_version": "v16.opportunity.1", "task_id": "different-opportunity-generator", "access_tier": "artifact-only",
        "final_artifact": execution["artifact"], "observations": [{"probe": "baseline", "artifact": execution["artifact"]}],
        "allowed_causes": list(CAUSES), "target_probe": "reminder", "query_cost": 0, "reader_search_budget": 1024}
    result = log.call("opportunity", public, model="latent-menu")
    return {"new_maker_executes_previously_absent_option": execution["artifact"] == 1,
        "absence_conditioned_cause_catalog_exposes_mismatch": mismatch(result),
        "mismatch_does_not_become_uniform_cause_posterior": "cause_posterior" not in result}, {
        "actual_execution": execution, "new_generator": "capable maker knowingly pursues option1 before any intervention",
        "scope": "the old cause catalog is conditioned on an initially absent option; successful baseline behavior needs a different producer model"}


def recognition_case(log):
    from .recognition_gates import fixture
    from .graphic_reference import interpret
    public = fixture()
    novel = copy.deepcopy(public)
    novel["anonymous"] = [{"artifact": interpret([0, 1, 2, 8, 9])["artifact"], "first_action": None}]
    rejected = read_or_reject(log, PREFIX+"recognition:public_reader", novel, method="craft")
    reversed_purpose = copy.deepcopy(public)
    program = [0, 1, 2, 6, 7]
    reversed_purpose["anonymous"] = [{"artifact": interpret(program)["artifact"], "first_action": None}]*3
    wrong = log.call(PREFIX+"recognition:public_reader", reversed_purpose, method="craft")
    return {"novel_legal_decoration_exposes_catalog_limit": mismatch(rejected),
        "within_catalog_changed_purpose_can_misidentify_known_source": wrong["identity"][1] > 0.9,
        "identity_error_does_not_create_unique_order": close(wrong["historical_core"], [0.5, 0.5])}, {
        "novel_program": [0, 1, 2, 8, 9], "actual_changed_purpose_source": 0, "changed_purpose_program": program,
        "reader_identity_choice": wrong["identity_choice"],
        "scientific_boundary": "in-support misspecification can give confidently wrong identity without a mismatch flag; this is an exposed error, not robustness"}


def selection_case(log):
    from .selection_gates import fixture
    public = fixture()
    false = {**copy.deepcopy(public), "retention": "random"}
    wrong = log.call(PREFIX+"selection:public_reader", false, method="selection-aware")
    full = log.call(PREFIX+"selection:public_reader", fixture(full=True), method="selection-aware")
    return {"omitted_selection_misattributes_released_signature": wrong["acquired_style"][0] > 0.9,
        "actual_rejected_candidates_reverse_raw_production_account": full["acquired_style"][1] > 0.99,
        "raw_and_release_predictions_remain_separate": different(full["future_raw_style"], full["future_release_style"]),
        "omitted_law_is_not_automatically_detectable": "model_mismatch" not in wrong}, {
        "actual_batches": "three batches each with three style1 candidates followed by one selected style0 candidate",
        "omitted_mechanism": "first audience-matching retention presented as random",
        "scientific_boundary": "released artifacts alone can fit a wrong production/selection explanation; rejected work is a real discriminator"}


def multi_actor_case(log):
    from .multi_actor_gates import fixture
    from .graphic_reference import interpret
    public = fixture(policy="artifact-only")
    p = public["world"]["permutation"]
    programs = [p[:2]+[p[2+topic]]+p[4:6] for topic in [0, 1, 0]]
    public["history"] = [interpret(program)["artifact"] for program in programs]
    result = read_or_reject(log, PREFIX+"multi_actor:public_reader", public, policy="artifact-only")
    return {"changed_purpose_history_really_executes": all(interpret(program)["legal"] for program in programs),
        "fixed_purpose_topologies_expose_missing_mechanism": mismatch(result),
        "no_fabricated_uniform_topology": "topology" not in result}, {
        "actual_programs": programs, "new_generator": "one maker changes own topic across three works under an unchanged public brief",
        "scope": "the registered role model keeps own purpose or shared brief fixed across these works; changing purpose is outside its support"}


def tradeoffs_case(log):
    from .tradeoffs_gates import fixture
    from .assembly_reference import interpret
    public = fixture()
    execution = interpret(public["world"], [2, 9], initial=[1, 0, -1])
    public["history"][0]["context"]["max_steps"] = 1
    public["history"][0]["state"] = execution["state"]
    result = log.call(PREFIX+"tradeoffs:public_reader", public, policy="chronological")
    return {"actual_extended_planner_adds_part_then_stops": execution["successfully_stopped"] and execution["state"] == [1, 0, 0],
        "one_step_assumption_cannot_explain_two_primitive_choice": mismatch(result),
        "missing_procedure_not_replaced_by_confident_profile": "profile_posterior" not in result}, {
        "actual_program": [2, 9], "actual_expanded_cost": execution["primitive_cost"], "assumed_max_steps": 1,
        "scope": "unmodeled planning allowance, not a free macro; an unsupported choice cannot justify a persistent-profile conclusion"}


def preference_case(log):
    from .preference_probe_gates import fixture
    from .assembly_reference import interpret
    public = fixture()
    actual_world = {"parents": [-1, -1, 0], "defaults": [0, 0, 0]}
    execution = interpret(actual_world, [3, 9], initial=[1, 0, -1])
    public.update(phase=2, answer={"query": "all", "state": execution["state"]})
    result = log.call(PREFIX+"preference_probe:public_reader", public, policy="all")
    return {"changed_physical_dependency_has_real_expanded_cost": execution["successfully_stopped"] and execution["primitive_cost"] == 2 and execution["state"] == [-1, 0, -1],
        "three_candidate_planners_expose_omitted_procedure": mismatch(result),
        "no_uniform_cause_fallback": "cause_posterior" not in result}, {
        "actual_program": [3, 9], "actual_world": actual_world, "assumed_world": public["world"], "actual_execution": execution,
        "scope": "part1 is independently supported in the changed world; its survival after root removal contradicts the reader's parent law",
        "initial_setup_limit": "a proposed four-step novel orientation program was already in support; that failed calibration and source identity are preserved separately"}


CASES = {"reading": reading_case, "options": options_case, "mechanism": mechanism_case,
    "assembly-mechanism": lambda log: mechanism_case(log, True), "opportunity": opportunity_case,
    "recognition": recognition_case, "selection": selection_case, "multi-actor": multi_actor_case,
    "tradeoffs": tradeoffs_case, "preference-probe": preference_case}
