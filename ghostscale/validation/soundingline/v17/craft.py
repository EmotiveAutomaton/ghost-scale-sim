"""First A comparison: actual V16 fragments, bounded episodic adaptation and primitives.

This is the graphic acquisition comparison, explicitly assisted by maker training
and a supplied construction target. It is not the artifact-only observer study.
"""
from collections import Counter
from itertools import combinations
import heapq
import random

from ..v16 import graphic_world
from ..v16.records import digest, seed_for
from .contracts import BudgetExhausted, Costs
from .programs import execute

METHODS = ("primitive_search", "repeated_fragments", "episodic_adaptation")
REGIMES = ("familiar_combinations", "new_combinations", "changed_constraint")
PILOT_BUDGETS = (8, 32, 128, 512, 2048)


def make_case(namespace, constructor_index, history_index, regime):
    if regime not in REGIMES:
        raise ValueError("unknown regime")
    rng = random.Random(seed_for(namespace, "constructor", constructor_index))
    cells = list(range(16))
    rng.shuffle(cells)
    # Physical relabelings are counted separately from structural novelty.
    constructor = {"cells": cells, "kind": "three disjoint two-cell training motifs"}
    rng = random.Random(seed_for(namespace, constructor_index, history_index, "history"))
    pairs = [cells[:2], cells[2:4], cells[4:6]]
    training = []
    counts = rng.choice(((8, 4, 4), (6, 6, 4), (4, 8, 4), (4, 4, 8)))
    for pair_index in ([0]*counts[0] + [1]*counts[1] + [2]*counts[2]):
        program = list(pairs[pair_index])
        if history_index % 2:
            program.reverse()
        result = execute(program)
        training.append({"initial": 0, "program": program, "target": result["artifact"],
                         "feedback": True, "forbidden": []})
    rng.shuffle(training)
    target_cells = cells[:4] if regime != "new_combinations" else [cells[0], cells[2], cells[6]]
    target = sum(1 << c for c in target_cells)
    initial = (1 << cells[0]) if regime == "changed_constraint" else 0
    forbidden = [cells[0]] if regime == "changed_constraint" else []
    action_order = list(range(32))
    rng.shuffle(action_order)
    public = {"schema": "v17.craft.1", "training": training, "initial": initial,
              "target": target, "forbidden": forbidden, "max_steps": 6,
              "action_order": action_order}
    if target in {t["target"] for t in training}:
        raise ValueError("held-out target was acquired")
    witness = [c for c in target_cells if not initial & (1 << c)]
    realized = execute(witness, initial=initial, forbidden=forbidden)
    if not realized["legal"] or realized["artifact"] != target:
        raise ValueError("impossible target")
    private = {"constructor": constructor, "training_history": training,
               "current_goal": target, "earlier_goals": [t["target"] for t in training],
               "actual_constraints": forbidden, "believed_constraints": forbidden,
               "considered_options": action_order, "realized_history": witness,
               "contributor_role": "single maker", "recipient_state": None,
               "recipient_state_reason": "not a recipient task in this A slice"}
    semantic = {"constructor": constructor, "training_multiset": sorted(Counter(digest(t) for t in training).items()),
                "initial": initial, "target": target, "forbidden": forbidden}
    return {"case_id": digest([namespace, constructor_index, history_index, regime]),
            "constructor_id": digest([namespace, constructor_index]),
            "history_id": digest([namespace, constructor_index, history_index]),
            "regime": regime, "namespace": namespace, "public": public, "private": private,
            "public_problem_sha256": digest({k: v for k, v in public.items() if k != "action_order"}),
            "private_construction_sha256": digest(semantic),
            "structural_family": "disjoint pairs; permutation is not a new dependency topology"}


def acquire(training, method, memory_cap):
    if method not in METHODS or type(memory_cap) is not int or memory_cap < 0:
        raise ValueError("unknown method or invalid memory cap")
    costs = Costs()
    # Each deployed maker bears the same physical acquisition cost.
    for example in training:
        costs.charge("training_acquisition", len(example["program"]))
    library, episodes = [], []
    if method == "repeated_fragments":
        # Literal existing baseline, including its >=4 repetitions and one-pair cap.
        learned = graphic_world.learn(training)
        costs.charge("learning", learned["processing_primitives"] + len(training))
        for fragment in learned["library"]:
            size = 2 * len(fragment) + 1  # operation + cell per primitive, sequence token
            if costs.values["definition_storage"] + size <= memory_cap:
                library.append(fragment)
                costs.charge("definition_storage", size)
    elif method == "episodic_adaptation":
        seen = set()
        # First distinct successful episode in acquisition order; never evaluation failures.
        for example in training:
            result = execute(example["program"], initial=example["initial"], forbidden=example["forbidden"])
            costs.charge("learning", result["primitive_cost"] + 1)
            identity = digest(example)
            if result["legal"] and result["artifact"] == example["target"] and identity not in seen:
                seen.add(identity)
                size = 2 * len(example["program"]) + 3 + len(example["forbidden"])
                if costs.values["definition_storage"] + size <= memory_cap:
                    episodes.append({k: example[k] for k in ("initial", "target", "forbidden", "program")})
                    costs.charge("definition_storage", size)
    return {"library": library, "episodes": episodes, "costs": costs.values,
            "retained_training_examples": len(episodes), "memory_cap_tokens": memory_cap}


def proposals(state, representation, public, costs):
    """Charge retrieval, adaptation, generation and deterministic ordering literally."""
    candidates = []
    for action in public["action_order"]:
        costs.charge("proposal_generation")
        candidates.append([action])
    for fragment in representation["library"]:
        costs.charge("retrieval", len(fragment))
        costs.charge("proposal_generation")
        candidates.append(fragment)
    for episode in representation["episodes"]:
        costs.charge("retrieval", 3 + len(episode["forbidden"]) + 2*len(episode["program"]))
        adapted = []
        for action in episode["program"]:
            costs.charge("argument_binding")
            c = action % 16
            present, desired = bool(state & (1 << c)), bool(public["target"] & (1 << c))
            if action not in public["forbidden"] and present != desired and ((action < 16) == desired):
                adapted.append(action)
        costs.charge("proposal_generation")
        if adapted:
            candidates.append(adapted)
    # All methods use one physical first-action order; representations do not get
    # an automatic first slot. Reverse ordering is a required additional pilot check.
    order = {a: i for i, a in enumerate(public["action_order"])}
    for candidate in candidates:
        costs.charge("selection")
    return sorted(candidates, key=lambda p: (order[p[0]], len(p), p))


def solve(public, method, budget, memory_cap=32):
    if set(public) != {"schema", "training", "initial", "target", "forbidden", "max_steps", "action_order"}:
        raise ValueError("unexpected public construction fields")
    if public["schema"] != "v17.craft.1" or sorted(public["action_order"]) != list(range(32)):
        raise ValueError("invalid public construction schema/order")
    if type(budget) is not int or budget < 0:
        raise ValueError("invalid budget")
    representation = acquire(public["training"], method, memory_cap)
    costs = Costs(dict(representation["costs"]), search_budget=budget)
    queue = [((public["initial"] ^ public["target"]).bit_count(), 0, public["initial"], [])]
    seen, serial = {public["initial"]}, 0
    selected, attempted, failed_candidates = None, 0, 0
    evidence = []
    try:
        while queue:
            costs.charge("selection")
            _, _, state, program = heapq.heappop(queue)
            if state == public["target"]:
                selected = program
                break
            for fragment in proposals(state, representation, public, costs):
                if len(program) + len(fragment) > public["max_steps"]:
                    failed_candidates += 1
                    continue
                new_state, legal = state, True
                for action in fragment:
                    costs.charge("hypothetical_execution")
                    transition = execute([action], initial=new_state, forbidden=public["forbidden"], max_steps=1)
                    attempted += 1
                    new_state = transition["artifact"]
                    if not transition["legal"]:
                        failed_candidates += 1
                        legal = False
                        break
                costs.charge("selection")  # evaluation of a considered candidate
                evidence.append([state, fragment, new_state, legal])
                if not legal:
                    continue
                proposed = program + fragment
                if new_state == public["target"]:
                    selected = proposed
                    break
                if new_state not in seen:
                    seen.add(new_state)
                    serial += 1
                    heapq.heappush(queue, ((new_state ^ public["target"]).bit_count(), serial, new_state, proposed))
            if selected is not None:
                break
    except BudgetExhausted:
        pass
    actual = None
    if selected is not None:
        actual = execute(selected, initial=public["initial"], forbidden=public["forbidden"], max_steps=public["max_steps"])
        costs.charge("actual_execution", actual["primitive_cost"])
    return {"method": method, "budget": budget, "memory_cap": memory_cap,
            "program": selected, "execution": actual,
            "task_success": actual is not None and actual["legal"] and actual["artifact"] == public["target"],
            "missing_output": selected is None,
            "invalid_program": actual is not None and not actual["legal"],
            "costs": costs.receipt(deployments=16),
            "representation": {k: v for k, v in representation.items() if k != "costs"},
            "attempted_primitives": attempted, "failed_candidates": failed_candidates,
            "considered_digest": digest(evidence), "considered_candidates": len(evidence)}


def evaluate_case(case, budgets, memory_cap=32):
    return [{**solve(case["public"], method, budget, memory_cap),
             "case_id": case["case_id"], "constructor_id": case["constructor_id"],
             "history_id": case["history_id"], "regime": case["regime"],
             "evidence_tier": "maker_acquisition_and_supplied_target",
             "claim_status": "descriptive"}
            for budget in budgets for method in METHODS]


def select_budgets(rows):
    """Discarded-pilot rule: three distinct budgets nearest pooled 20/50/80% solve."""
    means = {b: sum(r["task_success"] for r in rows if r["budget"] == b) /
             sum(r["budget"] == b for r in rows) for b in PILOT_BUDGETS}
    selected = min(combinations(PILOT_BUDGETS, 3),
                   key=lambda bs: (sum((means[b]-target)**2 for b, target in zip(bs, (.2, .5, .8))), bs))
    return {"budgets": list(selected), "pooled_pilot_solve_rates": means,
            "rule": "minimum squared distance to 0.2, 0.5, 0.8; distinct ascending budgets; lower tuple breaks ties"}
