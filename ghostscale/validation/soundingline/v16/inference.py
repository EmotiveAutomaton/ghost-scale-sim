"""Exact finite inverse model; all artifact-producing histories are marginalized."""
from __future__ import annotations
from functools import lru_cache
from math import exp, log
import json
from .world import step, distance
from .learning import training_library, plan


@lru_cache(maxsize=256)
def distribution(library, target, beta=1.0, length_penalty=0.3):
    # DP merges prefixes with identical board and two-step coding state. Counts
    # preserve route multiplicity; the scalar reference enumerates each route.
    states = {(0, -1, 0, 0): 1}
    artifacts = [0.0] * 16
    for length in range(4):
        for (board, last, previous_cost, cost), count in states.items():
            artifacts[board] += count * exp(-beta * distance(board, target)
                                            - length_penalty * cost)
        if length == 3:
            break
        following = {}
        for (board, last, previous_cost, cost), count in states.items():
            for action in range(8):
                new_cost = min(cost + 1, previous_cost + 1) if (last, action) in library else cost + 1
                key = (step(board, action), action, cost, new_cost)
                following[key] = following.get(key, 0) + count
        states = following
    normalizer = sum(artifacts)
    return tuple(weight / normalizer for weight in artifacts)


PUBLIC_FIELDS = {"schema_version", "task_id", "lineage_id", "access_tier", "final_artifact",
                 "declared_context", "permitted_prior_artifacts",
                 "permitted_query_descriptions", "query_costs", "target_request",
                 "reader_action_budget"}


def observe(payload: bytes):
    observation = json.loads(payload)
    if set(observation) != PUBLIC_FIELDS or observation["schema_version"] != "v16.public.1":
        raise ValueError("public observation contract violation")
    for artifact in [observation["final_artifact"], *observation["permitted_prior_artifacts"]]:
        if type(artifact) is not int or not 0 <= artifact < 16:
            raise ValueError("model mismatch: unsupported artifact")
    return observation


def posterior(artifacts, target, *, beta=1.0, length_penalty=0.3):
    libraries = tuple(training_library(index).library for index in range(2))
    weights = [0.5, 0.5]
    likelihood_evaluations = 0
    for artifact in artifacts:
        for index, library in enumerate(libraries):
            weights[index] *= distribution(library, target, beta, length_penalty)[artifact]
            likelihood_evaluations += 1
        normalizer = sum(weights)
        if normalizer == 0:
            raise ValueError("model mismatch: zero evidence likelihood")
        weights = [weight / normalizer for weight in weights]
    return weights, libraries, likelihood_evaluations


def read_public(payload: bytes, *, reader="maker", reader_seed=0):
    obs = observe(payload)
    context = obs["declared_context"]
    target = context["production_target"]
    beta = context.get("beta", 1.0)
    penalty = context.get("length_penalty", 0.3)
    artifacts = obs["permitted_prior_artifacts"] + [obs["final_artifact"]]
    weights, libraries, calls = posterior(artifacts, target, beta=beta, length_penalty=penalty)
    if reader == "generic":
        weights = [0.5, 0.5]
    elif reader == "primitive":
        libraries = ((), ())
        weights = [0.5, 0.5]
    elif reader != "maker":
        raise ValueError("unknown reader")
    future_target = obs["target_request"]["future_target"]
    future = [sum(weights[j] * distribution(library, future_target, beta, penalty)[artifact]
                  for j, library in enumerate(libraries)) for artifact in range(16)]
    selected = min(range(2), key=lambda i: (-weights[i], i))
    reconstruction = plan(obs["final_artifact"], libraries[selected],
                          search_budget=obs["reader_action_budget"])
    return {"reader": reader, "posterior": weights, "future_probabilities": future,
            "reconstruction": reconstruction, "likelihood_evaluations": calls,
            "access_cost": len(obs["permitted_prior_artifacts"]),
            "reader_seed": reader_seed}


def log_score(probabilities, truth):
    if probabilities[truth] <= 0:
        raise ValueError("unbounded log loss must be recorded as model mismatch")
    return log(probabilities[truth])
