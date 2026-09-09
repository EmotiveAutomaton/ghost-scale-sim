"""A retained native case from acquisition through prospective prediction and scoring."""
from __future__ import annotations
import random
from dataclasses import asdict
from pathlib import Path
from .records import canonical, digest, write, read, now, seed_for
from .learning import training_library
from .world import histories, execute, distance
from .learning import encoding_cost
from .inference import read_public, log_score
from .reference import interpret
from .gates import fixture_observation
from math import exp


def draw_route(rng, library, target):
    routes = tuple(histories())
    weights = [exp(-distance(execute(route).artifact, target)
                   - 0.3 * encoding_cost(route, library)) for route in routes]
    return rng.choices(routes, weights=weights, k=1)[0]


def run_case(root: Path, *, packet_hash: str, index=0):
    unit_id = digest(["v16-native-fixture", index])[:24]
    unit_path = root / "units" / f"{unit_id}_points.json"
    if unit_path.exists():
        previous = read(unit_path)
        if previous["packet_hash"] != packet_hash:
            raise ValueError("completed unit has a different packet")
        return previous
    stream = random.Random(seed_for("v16-native-fixture", index, "generation"))
    direction = stream.randrange(2)
    acquisition = training_library(direction)
    route = draw_route(stream, acquisition.library, 3)
    public = fixture_observation(execute(route).artifact)
    public["task_id"] = unit_id
    public["lineage_id"] = "native-fixture-1"
    public["permitted_prior_artifacts"] = [
        execute(draw_route(stream, acquisition.library, 3)).artifact for _ in range(4)]
    public["access_tier"] = "prior-works-4"
    write(root / "public" / f"{unit_id}.json", public)
    prediction = read_public(canonical(public), reader="maker",
                             reader_seed=seed_for("v16-native-fixture", index, "reader"))
    prediction_path = root / "predictions" / f"{unit_id}.json"
    submitted_at = read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash = write(prediction_path, {
        "submitted_at": submitted_at, "observation_hash": digest(public),
        "prediction": prediction})
    # Future is sampled only after prediction has been durably submitted.
    future_stream = random.Random(seed_for("v16-native-fixture", index, "future"))
    continuation = draw_route(future_stream, acquisition.library, 12)
    truth = {"true_production_record": list(route), "acquisition_record": asdict(acquisition),
             "original_goal": 3, "current_goal": 12, "controller_state": direction,
             "actual_feasibility": list(range(8)), "maker_beliefs": list(range(8)),
             "considered_alternatives": "full finite route support",
             "hidden_continuations": list(continuation),
             "future_artifact": execute(continuation).artifact,
             "intervention_outcomes": {},
             "equivalence_classes": [list(candidate) for candidate in histories()
                                     if execute(candidate).artifact == public["final_artifact"]],
             "scoring_contract": {"prediction": "log probability of future artifact",
                                  "execution": "independent primitive interpreter"}}
    truth_hash = write(root / "private" / f"{unit_id}.json", truth)
    execution = interpret(prediction["reconstruction"]["program"])
    record = {"unit_id": unit_id, "card_id": "P01", "condition": "native-fixture",
              "evidence_scope": "fixture", "packet_hash": packet_hash,
              "constructor_id": "fixture", "maker_history_id": unit_id,
              "lineage": "native-fixture-1", "seed_components": {"index": index},
              "observation_hash": digest(public), "prediction_hash": prediction_hash,
              "truth_hash": truth_hash, "prediction": prediction, "truth": truth,
              "public": public, "execution": execution,
              "outcomes": {"success": execution["legal"] and execution["artifact"] == public["final_artifact"],
                           "legal": execution["legal"],
                           "future_log_score": log_score(prediction["future_probabilities"],
                                                        truth["future_artifact"]),
                           "history_class_size": len(truth["equivalence_classes"])},
              "costs": {"training_primitives": acquisition.processing_cost,
                        "library_definition": acquisition.definition_cost,
                        "reconstruction_search_primitives": prediction["reconstruction"]["primitive_evaluations"],
                        "reconstruction_primitives": execution["primitive_cost"],
                        "prior_work_queries": 4},
              "failures": [], "prediction_submitted_at": submitted_at, "scored_at": now()}
    write(root / "units" / f"{unit_id}_points.json", record)
    return record
