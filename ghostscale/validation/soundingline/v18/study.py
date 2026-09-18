"""Thin V18 adapter. Closed V16 pure helpers retain their original science."""
from collections import Counter
from dataclasses import asdict
import json

from ..v16 import attention_craft
from ..v16.craft import construct
from ..v16.purpose_craft import inhibition
from ..v16.records import canonical, digest
from ..v16.world import execute

SCHEMA = "v18.selective-acquisition.1"
QUOTAS = {"broad": (8, 8), "focused": (12, 4)}
CONDITION = {"offers": 32, "instruction": True, "feedback": True}
METHODS = ("primitive", "broad-unchecked", "focused-unchecked", "broad-checked",
           "focused-checked", "broad-checked-extra", "focused-checked-extra")


def offered_view(prepared):
    # No instruction, outcome, success, reliability, or eventual own target.
    return {"schema": SCHEMA, "focal_topic": 0,
            "offers": [{"offer": o["offer"], "topic": o["topic"]} for o in prepared["offers"]]}


def select_offers(payload):
    public = json.loads(payload)
    if set(public) != {"schema", "focal_topic", "offers"} or public["schema"] != SCHEMA:
        raise ValueError("offered-only schema violation")
    offers = public["offers"]
    if public["focal_topic"] != 0 or len(offers) != 32:
        raise ValueError("wrong offered population")
    if any(set(o) != {"offer", "topic"} for o in offers):
        raise ValueError("private field in offered view")
    if [o["offer"] for o in offers] != list(range(32)) or Counter(o["topic"] for o in offers) != {0: 16, 1: 16}:
        raise ValueError("unbalanced or duplicate offers")
    return {name: sorted(o["offer"] for topic, quota in enumerate(quotas)
                         for o in [x for x in offers if x["topic"] == topic][:quota])
            for name, quotas in QUOTAS.items()}


def learn_public(payload):
    data = json.loads(payload)
    if set(data) != {"schema", "processed"} or data["schema"] != SCHEMA:
        raise ValueError("acquisition schema violation")
    processed = data["processed"]
    if len(processed) != 16 or len({x["offer"] for x in processed}) != 16:
        raise ValueError("invalid processed dose")
    fields = {"offer", "topic", "instruction", "proposed", "program", "artifact", "feedback"}
    if any(set(x) != fields or type(x["feedback"]) is not bool or len(x["program"]) != 2 for x in processed):
        raise ValueError("unavailable feedback or invalid admitted record")
    library = attention_craft.learn_processed(processed)
    counts = Counter(tuple(x["program"]) for x in processed if x["feedback"] is not False)
    candidates = sorted((p for p, n in counts.items() if n >= 3), key=lambda p: (-counts[p], p))
    best = counts[candidates[0]] if candidates else None
    return {"library": library, "library_sha256": digest(library),
            "eligible_counts": [{"program": list(p), "count": n} for p, n in sorted(counts.items())],
            "top_ties": [list(p) for p in candidates if counts[p] == best],
            "costs": {"processed_trials": 16, "training_primitives": sum(execute(x["program"]).primitive_cost for x in processed),
                      "instruction_queries": 16, "feedback_queries": 16,
                      "learning_records_scanned": 16, "eligible_programs_sorted": len(candidates),
                      "storage_primitives": sum(map(len, library)), "offered_cues_scanned": 32}}


def make_case(namespace, constructor_index, history_index, constructor_count=64):
    # Local consecutive indices; constructor identity is invariant across histories.
    index = history_index * constructor_count + constructor_index
    prepared = attention_craft.prepare(namespace, CONDITION, index, constructor_count)
    offered = offered_view(prepared)
    allocations = select_offers(canonical(offered))
    prepared["allocations"] = allocations
    # perform seeds realized actions by offer, not arm. Its extraneous public fields
    # are explicitly excluded before the V18 learner sees any bytes.
    performed, private_trials = attention_craft.perform(prepared, CONDITION, index, namespace)
    acquisitions = {}
    for name, selected in allocations.items():
        request = {"schema": SCHEMA, "processed": performed["processed"][name]}
        acquisitions[name] = {"selected": selected, "request": request, "result": learn_public(canonical(request))}
    permutation = prepared["constructor"]["permutation"]
    a, b = permutation[:2], permutation[2:]
    targets = {"compatible": [sum(1 << c for c in [*a, other]) for other in b],
               "changed": [(1 << x) | (1 << y) for x in a for y in b]}
    private = {"constructor": prepared["constructor"], "offers": prepared["offers"], "trials": private_trials}
    return {"schema": SCHEMA, "case_id": digest([namespace, constructor_index, history_index]),
            "namespace": namespace, "constructor_index": constructor_index, "history_index": history_index,
            "constructor_count": constructor_count, "source_index": index,
            "constructor_sha256": digest(prepared["constructor"]),
            "structural_world_sha256": digest({"motifs": [a, b], "primitive_order": list(range(8))}),
            "offered": offered, "allocations": allocations, "acquisitions": acquisitions,
            "transfer_targets": targets, "private": private,
            "ordering": "offered cues -> selection -> realized acquisition -> learn -> disclose own target"}


def use_library(payload):
    public = json.loads(payload)
    if set(public) != {"schema", "library", "target", "budget", "checking", "extra_search"} or public["schema"] != SCHEMA:
        raise ValueError("transfer schema violation")
    library = public["library"]
    active, checks = inhibition(library, public["target"]) if public["checking"] else (library, [])
    checking_cost = sum(x["primitive_cost"] for x in checks)
    budget = public["budget"] - (0 if public["extra_search"] else checking_cost)
    if budget < 0:
        raise ValueError("checking exceeds envelope")
    submission = construct(public["target"], active, primitive_budget=budget)
    execution = asdict(execute(submission["program"]))
    nonempty = next((x for x in submission["attempted_programs"] if x["tokens"]), None)
    first_program = []
    if nonempty:
        for token in nonempty["tokens"]:
            first_program.extend(active[int(token[1:])] if isinstance(token, str) else [token])
    return {"request": public, "active_library": active, "checks": checks,
            "rejected_fragments": len(library) - len(active),
            "submission": submission, "execution": execution,
            "first_nonempty_proposal": {"program": first_program, "attempt": nonempty},
            "success": execution["legal"] and execution["artifact"] == public["target"],
            "invalid": not execution["legal"], "empty_library": not library,
            "costs": {"checking_primitives": checking_cost, "search_primitives": submission["search_primitives"],
                      "submission_primitives": execution["primitive_cost"],
                      "online_primitives": checking_cost + submission["search_primitives"] + execution["primitive_cost"],
                      "search_envelope": budget, "attempts_considered": submission["considered"]}}


def evaluate(case, budget):
    rows = []
    for stratum, targets in sorted(case["transfer_targets"].items()):
        for target_index, target in enumerate(targets):
            for method in METHODS:
                allocation = method.split("-")[0]
                learned = None if method == "primitive" else case["acquisitions"][allocation]["result"]
                request = {"schema": SCHEMA, "library": [] if learned is None else learned["library"],
                           "target": target, "budget": budget, "checking": "-checked" in method,
                           "extra_search": method.endswith("-extra")}
                row = use_library(canonical(request))
                rows.append({"method": method, "stratum": stratum, "target_index": target_index,
                             "budget": budget, **row})
    return rows
