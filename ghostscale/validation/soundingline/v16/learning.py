"""A native zero-arity, bounded repeated-fragment learner (not Stitch)."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from .world import execute


@dataclass(frozen=True)
class Acquisition:
    attempts: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]
    feedback: tuple[bool, ...]
    library: tuple[tuple[int, ...], ...]
    definition_cost: int
    processing_cost: int


def encoding_cost(program, library):
    costs = [0] + [len(program) + 1] * len(program)
    for end in range(1, len(program) + 1):
        costs[end] = costs[end - 1] + 1
        for motif in library:
            start = end - len(motif)
            if start >= 0 and tuple(program[start:end]) == tuple(motif):
                costs[end] = min(costs[end], costs[start] + 1)
    return costs[-1]


def learn(attempts, targets, *, capacity=2, min_saving=1):
    traces = tuple(tuple(trace) for trace in attempts)
    target_tuple = tuple(targets)
    if len(traces) != len(target_tuple):
        raise ValueError("training needs one feedback target per attempt")
    feedback = tuple(execute(trace).legal and execute(trace).artifact == target
                     for trace, target in zip(traces, target_tuple))
    counts = Counter(fragment for trace, success in zip(traces, feedback) if success
                     for offset in range(len(trace) - 1)
                     for fragment in [trace[offset:offset + 2]])
    # Candidate order, cap, zero arity and strict net savings frozen before evaluation.
    candidates = sorted(counts, key=lambda motif: (-counts[motif], motif))
    library = []
    for motif in candidates:
        old_cost = sum(encoding_cost(trace, library) for trace in traces)
        new_cost = sum(encoding_cost(trace, library + [motif]) for trace in traces) + len(motif)
        if old_cost - new_cost >= min_saving:
            library.append(motif)
        if len(library) == capacity:
            break
    return Acquisition(traces, target_tuple, feedback, tuple(library),
                       sum(map(len, library)), sum(map(len, traces)))


def expand(tokens, library):
    expanded = []
    for token in tokens:
        if isinstance(token, int):
            expanded.append(token)
        elif isinstance(token, str) and token.startswith("m") and token[1:].isdigit():
            index = int(token[1:])
            if index >= len(library):
                raise ValueError("unknown procedure")
            expanded.extend(library[index])
        else:
            raise ValueError("unknown token")
    return tuple(expanded)


def training_library(direction):
    # Two permitted dated instruction curricula; each is executed and learned,
    # not an assigned library. Richer histories are introduced in later packets.
    motif = (0, 1) if direction == 0 else (2, 3)
    attempts = [motif] * 4
    return learn(attempts, [execute(motif).artifact] * 4)


def plan(target, library, *, search_budget=32, feasible=tuple(range(8)), start=0):
    from collections import deque
    tokens = tuple(f"m{i}" for i in range(len(library))) + tuple(range(8))
    frontier = deque([()])
    considered = []
    primitive_evaluations = 0
    while frontier and len(considered) < search_budget:
        program = frontier.popleft()
        primitives = expand(program, library)
        result = execute(primitives, start=start, feasible=feasible)
        primitive_evaluations += result.primitive_cost
        considered.append(program)
        if result.legal and result.artifact == target:
            return {"program": list(primitives), "tokens": list(program), "success": True,
                    "considered": len(considered), "primitive_evaluations": primitive_evaluations}
        if len(primitives) < 3:
            frontier.extend(program + (token,) for token in tokens
                            if len(expand(program + (token,), library)) <= 3)
    return {"program": [], "tokens": [], "success": target == start,
            "considered": len(considered), "primitive_evaluations": primitive_evaluations}
