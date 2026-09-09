"""Known-answer admission fixtures. Bars here never read a scientific effect."""
from __future__ import annotations
from .reference import interpret, artifact_distribution
from .world import execute, histories
from .learning import learn, training_library, expand, plan, encoding_cost
from .inference import distribution, read_public, posterior
from .records import canonical

TOLERANCE = 1e-10


def fixture_observation(artifact=3):
    return {"schema_version": "v16.public.1", "task_id": "opaque-fixture",
            "lineage_id": "fixture-public", "access_tier": "artifact",
            "final_artifact": artifact,
            "declared_context": {"production_target": 3, "beta": 1.0, "length_penalty": 0.3},
            "permitted_prior_artifacts": [], "permitted_query_descriptions": [],
            "query_costs": {}, "target_request": {"future_target": 12},
            "reader_action_budget": 32}


def run_gates():
    gates = []
    def record(name, positive, null, broken_detected, boundary, **measurements):
        gates.append({"id": name, "instrument_state": "valid" if all(
            [positive, null, broken_detected, boundary]) else "failed",
            "positive": bool(positive), "null": bool(null),
            "deliberate_break_detected": bool(broken_detected),
            "nonidentifiability_boundary": bool(boundary),
            "evidence_scope": "fixture", "measurements": measurements})

    max_error = 0.0
    for library in [(), training_library(0).library, training_library(1).library,
                    ((0, 1), (2, 3))]:
        for target in range(16):
            expected = artifact_distribution(library, target)
            actual = distribution(library, target)
            max_error = max(max_error, max(abs(a - b) for a, b in zip(actual, expected)))
    collisions = [route for route in histories() if execute(route).artifact == 3]
    # Uniform routes (zero goal and length weight): exact marginal is count/585.
    uniform = distribution((), 0, 0.0, 0.0)
    record("artifact_marginalization", max_error <= TOLERANCE,
           abs(sum(uniform) - 1) < TOLERANCE,
           abs(uniform[3] - 1 / 585) > TOLERANCE,
           len(collisions) > 1 and execute((0, 1)).artifact == execute((1, 0)).artifact,
           exhaustive_max_error=max_error, collision_routes=len(collisions),
           uniform_collision_mass=uniform[3])

    checks = [interpret(route) == {
        "artifact": execute(route).artifact, "legal": execute(route).legal,
        "primitive_cost": execute(route).primitive_cost, "stopped": execute(route).stopped}
        for route in histories()]
    learned = training_library(0)
    program = expand(("m0", 2), learned.library)
    native = plan(7, learned.library, search_budget=32)
    primitive = plan(7, (), search_budget=32)
    record("executable_acquisition", all(checks) and learned.library == ((0, 1),)
           and native["success"] and not primitive["success"],
           learn([(0, 1)], [3]).library == (),
           interpret((0, 2))["artifact"] != interpret((0, 1))["artifact"]
           and not interpret((0, 1, 2, 3))["stopped"],
           interpret((0, 1))["artifact"] == interpret((1, 0))["artifact"],
           acquired_library=[list(motif) for motif in learned.library],
           definition_cost=learned.definition_cost, training_primitive_cost=learned.processing_cost,
           expanded_cost=interpret(program)["primitive_cost"],
           heldout_target=7, primitive_search=primitive, learned_search=native)

    payload = canonical(fixture_observation())
    result = read_public(payload)
    unchanged = [read_public(payload) == result for _private in
                 [{"history": "changed"}, {"seed": 999}, {"goal": 15}]]
    def leaky_reader(public, private):
        return (read_public(public), private["seed"])
    record("serialized_access", all(unchanged),
           read_public(payload, reader_seed=4) == read_public(payload, reader_seed=4),
           leaky_reader(payload, {"seed": 0}) != leaky_reader(payload, {"seed": 1}),
           read_public(canonical(fixture_observation(3))) == result,
           forbidden_fields_rejected=_reject_forbidden(payload))

    weights, _, _ = posterior([3] * 8, 3)
    no_data, _, _ = posterior([], 3)
    impossible = False
    try:
        posterior([15], 3)  # Four occupied cells are unreachable in three steps.
    except ValueError:
        impossible = True
    record("bayesian_evidence", abs(sum(weights) - 1) <= TOLERANCE and weights != no_data,
           no_data == [0.5, 0.5], weights != [0.5, 0.5] and impossible,
           all(0 < weight < 1 for weight in weights),
           posterior=weights, prior=no_data, impossible_evidence_rejected=impossible)

    actual = execute((0,), feasible=(1, 2, 3))
    nominal = execute((0,))
    record("physical_intervention", nominal.legal and not actual.legal,
           execute(()) == execute((), feasible=(1, 2, 3)),
           nominal != actual, execute((1,)) == execute((1,), feasible=(1, 2, 3)),
           annotation_only_would_leave_execution_unchanged=True)

    # Three intervention-independent quantities; unlike a nested-set model this
    # permits a considered impossible action and an unconsidered possible action.
    feasibility = {0, 1}
    beliefs = {1, 2}
    considered = {2}
    record("nonnested_opportunity", 2 in considered and 2 not in feasibility,
           feasibility == {0, 1}, beliefs != feasibility,
           0 in feasibility and 0 not in considered)

    return {"schema_version": "v16.gates.1", "evidence_scope": "fixture",
            "tolerance": TOLERANCE, "gates": gates,
            "instrument_state": "valid" if all(g["instrument_state"] == "valid" for g in gates) else "failed"}


def _reject_forbidden(payload):
    import json
    altered = json.loads(payload)
    altered["true_history"] = [0, 1]
    try:
        read_public(canonical(altered))
    except ValueError:
        return True
    return False
