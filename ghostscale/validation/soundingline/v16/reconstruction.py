"""Finite learned-library observers and an equally informed direct joint-table rival.

The latent prior is derived from executed training's successful-fragment counts.
Reader APIs receive public JSON bytes; no maker/evaluator object is accepted.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import exp, factorial, log
import json
import random
from .world import execute, distance, histories
from .learning import learn, encoding_cost
from .craft import construct
from .records import seed_for, digest

RELIABILITIES = (0.8, 0.9, 1.0)
TOPIC_PROBABILITIES = (0.75, 0.85, 0.95)
MOTIFS = ((0, 1), (2, 3))
LIBRARIES = ((), (MOTIFS[0],), (MOTIFS[1],), MOTIFS)


@lru_cache(maxsize=16)
def library_prior(count):
    # Exact multinomial integration: success count >=3 admits each length-2 motif.
    # Instruction failures are one-step incomplete traces and cannot yield another motif.
    weights = [0.0] * 4
    for reliability, topic, direction in product(RELIABILITIES, TOPIC_PROBABILITIES, range(2)):
        p0 = reliability * (topic if direction == 0 else 1-topic)
        p1 = reliability * (1-topic if direction == 0 else topic)
        failure = 1-reliability
        for count0 in range(count+1):
            for count1 in range(count-count0+1):
                missed = count-count0-count1
                coefficient = factorial(count)/(factorial(count0)*factorial(count1)*factorial(missed))
                mass = coefficient*p0**count0*p1**count1*failure**missed
                index = (1 if count0 >= 3 else 0) + (2 if count1 >= 3 else 0)
                weights[index] += mass/18
    return tuple(weights)


def acquire_history(namespace, constructor_id, maker_index, count):
    constructor_rng = random.Random(seed_for(namespace, "constructor", constructor_id))
    reliability = constructor_rng.choice(RELIABILITIES)
    topic = constructor_rng.choice(TOPIC_PROBABILITIES)
    rng = random.Random(seed_for(namespace, maker_index, "training"))
    direction = rng.randrange(2)
    attempts, goals = [], []
    for episode in range(count):
        chosen = direction if rng.random() < topic else 1-direction
        motif = MOTIFS[chosen]
        attempt = motif if rng.random() < reliability else motif[:1]
        attempts.append(attempt)
        goals.append(3 if chosen == 0 else 12)
    acquisition = learn(attempts, goals)
    index = (1 if MOTIFS[0] in acquisition.library else 0) + (2 if MOTIFS[1] in acquisition.library else 0)
    return {"constructor": {"reliability": reliability, "topic_probability": topic},
            "direction": direction, "dates": list(range(count)),
            "attempts": [list(trace) for trace in attempts], "goals": goals,
            "feedback": list(acquisition.feedback), "library_index": index,
            "library": [list(motif) for motif in acquisition.library],
            "training_primitive_cost": acquisition.processing_cost,
            "library_definition_cost": acquisition.definition_cost}


@lru_cache(maxsize=128)
def route_model(library_index, target, feasible=tuple(range(8))):
    library = LIBRARIES[library_index]
    routes = tuple(route for route in histories() if all(action in feasible for action in route))
    weights = tuple(exp(-distance(execute(route).artifact, target) - 0.3*encoding_cost(route, library))
                    for route in routes)
    total = sum(weights)
    return routes, tuple(weight/total for weight in weights)


@lru_cache(maxsize=512)
def observation_mass(library_index, target, artifact, prefix=(), feasible=tuple(range(8))):
    routes, weights = route_model(library_index, target, feasible)
    return sum(weight for route, weight in zip(routes, weights)
               if execute(route).artifact == artifact and route[:len(prefix)] == prefix)


def sample_work(rng, library_index, target, feasible=tuple(range(8))):
    routes, weights = route_model(library_index, target, feasible)
    route = rng.choices(routes, weights=weights, k=1)[0]
    return {"program": list(route), "artifact": execute(route).artifact,
            "target": target, "feasible": list(feasible)}


def encode_observation(work, *, reveal_prefix=False):
    return {"artifact": work["artifact"], "target": work["target"],
            "feasible": work["feasible"], "prefix": work["program"][:1] if reveal_prefix else []}


PUBLIC_KEYS = {"schema_version", "task_id", "lineage_id", "access_tier", "final_artifact",
               "declared_context", "permitted_prior_artifacts", "permitted_query_descriptions",
               "query_costs", "target_request", "reader_action_budget"}


def reader(payload: bytes, strategy="maker"):
    public = json.loads(payload)
    if set(public) != PUBLIC_KEYS or public["schema_version"] != "v16.public.2":
        raise ValueError("public reconstruction schema violation")
    context = public["declared_context"]
    count = context["training_attempts"]
    evidence = [context["current_observation"], *public["permitted_prior_artifacts"]]
    if any(set(obs) != {"artifact", "target", "feasible", "prefix"} for obs in evidence):
        raise ValueError("undeclared observation fields")
    prior = library_prior(count)
    weights = list(prior)
    calls = 0
    for observed in evidence:
        for index in range(4):
            weights[index] *= observation_mass(index, observed["target"], observed["artifact"],
                                               tuple(observed["prefix"]), tuple(observed["feasible"]))
            calls += 1
        total = sum(weights)
        if total == 0:
            return {"model_mismatch": True, "reason": "zero likelihood under all represented acquisition histories"}
        weights = [weight/total for weight in weights]
    future_target = public["target_request"]["future_target"]
    future_feasible = tuple(public["target_request"].get("feasible", range(8)))
    if strategy == "generic":
        weights = list(prior)
    elif strategy == "primitive":
        weights = [1.0, 0.0, 0.0, 0.0]
    elif strategy not in {"maker", "direct-table"}:
        raise ValueError("unknown reconstruction strategy")
    if strategy == "direct-table":
        # Directly construct the joint table over observed artifacts and future
        # outcomes; it never produces or uses a recovered individual-library label.
        joint = [0.0]*16
        for index in range(4):
            observation_probability = prior[index]
            for observed in evidence:
                observation_probability *= observation_mass(index, observed["target"], observed["artifact"],
                                                            tuple(observed["prefix"]), tuple(observed["feasible"]))
            for artifact in range(16):
                joint[artifact] += observation_probability*observation_mass(
                    index, future_target, artifact, (), future_feasible)
        normalizer = sum(joint)
        probabilities = [value/normalizer for value in joint]
        library = MOTIFS  # generic construction repertoire, not a recovered label
    else:
        probabilities = [sum(weights[index]*observation_mass(index, future_target, artifact, (), future_feasible)
                             for index in range(4)) for artifact in range(16)]
        library = LIBRARIES[max(range(4), key=lambda index: weights[index])]
    construction = construct(public["final_artifact"], library,
                             primitive_budget=public["reader_action_budget"])
    return {"reader": strategy, "model_mismatch": False, "future_probabilities": probabilities,
            "history_probabilities": None if strategy == "direct-table" else weights,
            "reconstruction": construction,
            "costs": {"likelihood_evaluations": calls + 64,
                      "permitted_evidence_items": len(evidence),
                      "query_cost": sum(public["query_costs"].values()),
                      "finite_model_training_support": 4*585},
            "rival_scope": "same public finite acquisition model; direct joint table" if strategy == "direct-table" else None}


def prepare(namespace, card_id, condition, index, constructors=8):
    constructor_id = f"constructor-{index % constructors:03d}"
    acquisition = acquire_history(namespace, constructor_id, (condition["id"], index),
                                  condition.get("training_attempts", 16))
    library = acquisition["library_index"]
    rng = random.Random(seed_for(namespace, condition["id"], index, "observation"))
    current = sample_work(rng, library, 3)
    prior = [encode_observation(sample_work(rng, library, 3))
             for _ in range(condition.get("prior_works", 0))]
    query = condition.get("query", "none")
    current_observation = encode_observation(current, reveal_prefix=query == "process")
    if query in {"continuation", "tool", "prior-work"}:
        intervention_target = 12 if query == "continuation" else 3
        feasible = tuple(range(1, 8)) if query == "tool" else tuple(range(8))
        prior.append(encode_observation(sample_work(rng, library, intervention_target, feasible)))
    # Public IDs are random opaque IDs, allocated once by the storage layer. The
    # deterministic private unit identity is never substituted into this public slot.
    public = {"schema_version": "v16.public.2", "task_id": None, "lineage_id": "public-acquisition-family",
              "access_tier": f"prior-{condition.get('prior_works',0)}-query-{query}",
              "final_artifact": current["artifact"],
              "declared_context": {"training_attempts": condition.get("training_attempts",16),
                                   "current_observation": current_observation},
              "permitted_prior_artifacts": prior,
              "permitted_query_descriptions": [query] if query != "none" else [],
              "query_costs": {"earlier_works": condition.get("prior_works",0),
                              "diagnostic": 0 if query == "none" else 1},
              "target_request": {"future_target": 12},
              "reader_action_budget": condition.get("search_budget",128)}
    private = {"acquisition_record": acquisition, "true_production_record": current,
               "original_goal": 3, "current_goal": 12,
               "actual_feasibility": list(range(8)), "maker_beliefs": list(range(8)),
               "considered_alternatives": "all finite legal routes",
               "controller_state": library,
               "equivalence_classes": [list(route) for route in histories()
                                       if execute(route).artifact == current["artifact"]]}
    return public, private, constructor_id
