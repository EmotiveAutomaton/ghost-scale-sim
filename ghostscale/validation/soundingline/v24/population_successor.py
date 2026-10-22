"""Source-frozen N1 full-kernel search successor.

The controller evaluates whole constructed populations with the retained N1
reader and exact independent checker. It allocates disjoint worlds according
to the V24 eta-three schedule, retains every evaluated cell, and treats the
selected cell as a search output rather than confirmation.
"""
from pathlib import Path
import math

import numpy as np

from ..v23 import checker, ledger, population_pipeline, registry
from ..v23.common import digest, read, write
from .search import successive_halving


def validate_design(plan):
    expected = [variant["id"] for variant in registry.variants("N1")]
    if (plan.get("kind") != "pilot" or plan.get("task") != "v24-n1-search"
            or plan.get("family") != "N1" or plan.get("confirmation") is not False
            or plan.get("production") is not False):
        raise ValueError("excluded V24 N1 search pilot required")
    if plan.get("variant_ids") != expected or len(expected) != 240:
        raise ValueError("complete fixed N1 roster required")
    if plan.get("objective") != "mean law-marginal minus individual log-loss gain":
        raise ValueError("frozen N1 informativeness objective required")
    if plan.get("initial_worlds") != 1 or plan.get("eta") != 3:
        raise ValueError("frozen eta-three allocation required")
    if not 0 < plan.get("cpu_cap_seconds", 0) <= 10800:
        raise ValueError("bounded N1 search CPU cap required")
    if len(plan.get("canary_key", "")) != 64:
        raise ValueError("evaluator-only canary key required")
    return expected


def preflight(plan):
    expected = validate_design(plan)
    controls = population_pipeline.known_answers()
    return dict(passed=bool(controls["passed"]), full_roster=len(expected),
                known_answers=controls, scientific_worlds=0,
                selection_only=True, confirmation=False)


def execute(root, plan, pulse, *, guard=lambda: None):
    """Run the full retained kernel under the predeclared search allocation."""
    roster = validate_design(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    schema, _ = ledger.canonical_schema(plan["schema_directory"], plan["schema_pin"])
    write(root / "CONTROLS.json", population_pipeline.known_answers())
    variants = {variant["id"]: variant for variant in registry.variants("N1")}
    evidence = {identity: [] for identity in roster}

    def evaluate(active, worlds, round_number):
        scores = {}
        for identity in active:
            variant = variants[identity]
            for world in worlds:
                guard()
                result = population_pipeline.retained_world(
                    root / identity / f"world-{world:04d}", variant,
                    "v24-n1-search", world, schema, plan["canary_key"], pulse=pulse)
                evidence[identity].append(result["primary_mean"])
                pulse(phase="v24-n1-search", round=round_number,
                      variant=identity, world=world,
                      evaluated_worlds=len(evidence[identity]))
            scores[identity] = math.fsum(evidence[identity]) / len(evidence[identity])
        return scores

    decision = successive_halving(
        roster, evaluate, initial_worlds=plan["initial_worlds"], eta=plan["eta"])
    summary = dict(
        state="N1_SEARCH_EXECUTED_PENDING_INDEPENDENT_REVIEW",
        family="N1", objective=plan["objective"], search=decision,
        evaluated={identity: dict(worlds=len(values), world_gains=values)
                   for identity, values in evidence.items() if values},
        independent_unit="whole constructed population", selection_only=True,
        scientific_primary_decided=False, confirmation=False, production=False)
    write(root / "SUMMARY.json", summary)
    return summary


def reconstruct(root, plan, pulse=lambda **_: None):
    """Independently rebuild every retained proper score and search decision."""
    roster = validate_design(plan)
    root = Path(root)
    summary = read(root / "SUMMARY.json")
    rebuilt_scores = {}
    for identity, item in summary["evaluated"].items():
        if identity not in roster:
            raise ValueError("unknown retained N1 cell")
        gains = []
        for folder in sorted((root / identity).glob("world-*")):
            for name, expected in read(folder / "RAW_INDEX.json").items():
                if digest(folder / name) != expected:
                    raise ValueError("population raw record changed")
            with (np.load(folder / "prediction_points.npz") as prediction,
                  np.load(folder / "truth_points.npz") as truth,
                  np.load(folder / "score_points.npz") as saved):
                rebuilt = {name: checker.scores(prediction[name], truth["y"])
                           for name in prediction.files}
                for name, metrics in rebuilt.items():
                    for metric, values in metrics.items():
                        if not np.allclose(values, saved[name + "__outside__" + metric],
                                           atol=1e-11, rtol=0):
                            raise ValueError("retained N1 proper score differs")
                gain = rebuilt["individual-only"]["log_loss"] \
                    - rebuilt["law-marginal"]["log_loss"]
                gains.append(math.fsum(map(float, gain)) / len(gain))
        if len(gains) != item["worlds"] or not np.allclose(
                gains, item["world_gains"], atol=1e-11, rtol=0):
            raise ValueError("retained N1 world aggregation differs")
        rebuilt_scores[identity] = gains
        pulse(phase="v24-n1-independent-review", variant=identity, worlds=len(gains))

    positions = {identity: 0 for identity in roster}
    def evaluate(active, worlds, _round):
        result = {}
        for identity in active:
            positions[identity] += len(worlds)
            values = rebuilt_scores[identity][:positions[identity]]
            if len(values) != positions[identity]:
                raise ValueError("N1 search world coverage differs")
            result[identity] = math.fsum(values) / len(values)
        return result
    decision = successive_halving(
        roster, evaluate, initial_worlds=plan["initial_worlds"], eta=plan["eta"])
    if decision != summary["search"]:
        raise ValueError("retained N1 search decision differs")
    return dict(state="N1_SEARCH_INDEPENDENTLY_RECONSTRUCTED",
                cells=len(rebuilt_scores), total_worlds=decision["total_worlds"],
                selected=decision["selected"], selection_only=True,
                scientific_primary_decided=False, confirmation=False, production=False)
