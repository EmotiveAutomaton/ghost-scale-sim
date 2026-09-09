"""Executed acquisition and bounded construction for the K01 discovery packet."""
from __future__ import annotations
from dataclasses import asdict
from itertools import product
import random
from .records import seed_for, digest
from .world import execute, histories
from .learning import learn, expand

DESIGN = {
    "card_id": "K01",
    "question": "Can acquired procedures help construct new combinations at matched primitive-search cost?",
    "mechanism": "A bounded zero-arity learner admits repeated successful two-primitive fragments.",
    "rival": "pooled generic motifs and primitive-only breadth-first search",
    "conditions": [{"id": f"budget-{budget}", "training_attempts": 16, "search_primitive_budget": budget}
                   for budget in [8, 32, 128]],
    "access_arms": {"personal": "own permitted training attempts and feedback",
                    "pooled": "same number of independently sampled generic attempts and feedback",
                    "primitive": "same own training record; no abstraction learning"},
    "target_realization": "Recorded attempts execute; learned macros are absent initially; test targets add a cell never trained in that composition.",
    "primary_estimands": ["personal-minus-pooled-success", "personal-minus-primitive-success"],
    "secondary": ["legality", "training primitives", "definition cost", "search primitives",
                  "execution cost", "compression diagnostic", "errors"],
    "generator_families": ["W1-bounded-search"],
    "paired_unit": "one independent maker acquisition history and the same two held-out target compositions",
    "sample_rule": {"scout_makers_per_condition": 64, "scout_constructors": 8,
                    "expansion_makers_per_condition": 256, "expansion_constructors": 20,
                    "further_makers_per_condition": 1024},
    "dependencies": ["artifact_marginalization", "executable_acquisition", "serialized_access",
                     "craft-cost", "estimand-identity", "restart-reaggregation"],
    "adversaries": ["X01", "X02", "X04", "X07", "X08"],
    "repair_budget": 1,
    "continuation": "Expand for a clean capability or unresolved matched-cost boundary. Retain generic benefit if personal adds nothing; a null does not invalidate machinery.",
    "seed_namespace": "v16-k01-discovery-1",
    "interpretation": "conditional miniature at scout size; no mechanism promotion below 20 redraws",
}


def construct(target, library, *, primitive_budget, start=0, feasible=tuple(range(8))):
    """Cost-bounded breadth-first enumeration; no free legality mask or hidden plan."""
    tokens = tuple(f"m{i}" for i in range(len(library))) + tuple(range(8))
    attempts = []
    cost = 0
    # Enumerate every token sequence at increasing description length. Bad programs
    # spend their attempted primitive cost. Do not prefilter with private feasibility.
    for length in range(4):
        for sequence in product(tokens, repeat=length):
            primitives = expand(sequence, library)
            charged = min(len(primitives), 3)  # third primitive is the checkpoint boundary
            if cost + charged > primitive_budget:
                return {"program": [], "tokens": [], "search_primitives": cost,
                        "considered": len(attempts), "search_timeout": True,
                        "attempted_programs": attempts}
            result = execute(primitives, start=start, feasible=feasible)
            cost += result.primitive_cost
            attempts.append({"tokens": list(sequence), "legal": result.legal,
                             "artifact": result.artifact, "cost": result.primitive_cost})
            if result.legal and result.artifact == target:
                return {"program": list(primitives), "tokens": list(sequence),
                        "search_primitives": cost, "considered": len(attempts),
                        "search_timeout": False, "attempted_programs": attempts}
    return {"program": [], "tokens": [], "search_primitives": cost,
            "considered": len(attempts), "search_timeout": True, "attempted_programs": attempts}


def draw_constructor(namespace, constructor_id):
    rng = random.Random(seed_for(namespace, "constructor", constructor_id))
    permutation = list(range(4))
    rng.shuffle(permutation)
    return {"permutation": permutation, "instruction_reliability": rng.uniform(0.8, 1.0),
            "personal_topic_probability": rng.uniform(0.75, 0.95)}


def acquire(rng, constructor, direction, count, *, generic=False):
    permutation = constructor["permutation"]
    motifs = [tuple(permutation[:2]), tuple(permutation[2:])]
    attempts, targets, dates, instructions = [], [], [], []
    for date in range(count):
        choice = rng.randrange(2) if generic else (
            direction if rng.random() < constructor["personal_topic_probability"] else 1-direction)
        instruction = motifs[choice]
        performed = list(instruction)
        if rng.random() > constructor["instruction_reliability"]:
            performed[rng.randrange(2)] = rng.randrange(8)
        attempts.append(tuple(performed))
        targets.append(execute(instruction).artifact)
        dates.append(date)
        instructions.append(instruction)
    learned = learn(attempts, targets)
    return learned, {"attempts": [list(trace) for trace in attempts], "targets": targets,
                     "feedback": list(learned.feedback), "dates": dates,
                     "instructions": [list(instruction) for instruction in instructions],
                     "initial_library": [], "learned_library": [list(motif) for motif in learned.library],
                     "definition_cost": learned.definition_cost, "processing_cost": learned.processing_cost}


def prepare_unit(condition, index, *, namespace, constructors, evidence_scope):
    constructor_id = f"constructor-{index % constructors:03d}"
    constructor = draw_constructor(namespace, constructor_id)
    generation = random.Random(seed_for(namespace, condition["id"], index, "generation"))
    direction = generation.randrange(2)
    personal, personal_record = acquire(random.Random(seed_for(namespace, condition["id"], index, "training")),
                                        constructor, direction, condition["training_attempts"])
    generic, generic_record = acquire(random.Random(seed_for(namespace, condition["id"], index, "pooled-training")),
                                      constructor, direction, condition["training_attempts"], generic=True)
    permutation = constructor["permutation"]
    learned_pair = permutation[:2] if direction == 0 else permutation[2:]
    new_cells = permutation[2:] if direction == 0 else permutation[:2]
    targets = [sum(1 << cell for cell in [*learned_pair, new]) for new in new_cells]
    unit_id = digest([namespace, condition["id"], index])[:24]
    public = {"task_id": unit_id, "schema_version": "v16.craft.public.1",
              "access_tier": "training-with-feedback", "targets": targets,
              "search_primitive_budget": condition["search_primitive_budget"],
              "arm_training": {"personal": personal_record, "pooled": generic_record,
                               "primitive": personal_record}}
    # The maker's own acquisition record is explicitly permitted for K01 construction,
    # unlike K02's observer, which must infer a library from earlier artifacts.
    private = {"constructor": constructor, "direction": direction,
               "acquisition_record": personal_record, "generic_acquisition_record": generic_record,
               "heldout_target_check": all(target not in personal_record["targets"] for target in targets)}
    return {"unit_id": unit_id, "card_id": "K01", "condition": condition["id"],
            "constructor_id": constructor_id, "maker_history_id": unit_id,
            "evidence_scope": evidence_scope, "lineage": namespace,
            "seed_components": {"index": index, "constructors": constructors},
            "public": public, "private": private}


def construct_public(payload: bytes):
    import json
    public = json.loads(payload)
    if set(public) != {"task_id", "schema_version", "access_tier", "targets",
                       "search_primitive_budget", "arm_training"}:
        raise ValueError("unexpected craft reader fields")
    if public["schema_version"] != "v16.craft.public.1":
        raise ValueError("craft schema mismatch")
    arms = {}
    for name, record in public["arm_training"].items():
        # Independently learn from explicitly permitted attempts. Do not use the
        # serialized learned-library diagnostic as supplied privileged capability.
        learned = learn(record["attempts"], record["targets"])
        library = () if name == "primitive" else learned.library
        submissions = [construct(target, library, primitive_budget=public["search_primitive_budget"])
                       for target in public["targets"]]
        arms[name] = {"submissions": submissions,
                      "costs": {"training_attempts": len(record["attempts"]),
                                "training_primitives": learned.processing_cost,
                                "library_definition": 0 if name == "primitive" else learned.definition_cost},
                      "learned_library": [list(motif) for motif in library]}
    return arms
