"""Independent scalar enumeration and execution. Does not import world or readers."""
from __future__ import annotations
from itertools import product
import math


def interpret(program, start=0, feasible=tuple(range(8)), budget=3):
    cells = [bool(start & (2 ** index)) for index in range(4)]
    count = 0
    for action in program:
        if count == budget:
            return {"artifact": sum(2**i for i, cell in enumerate(cells) if cell),
                    "legal": False, "primitive_cost": count, "stopped": False}
        count += 1
        if type(action) is not int or action not in feasible:
            return {"artifact": sum(2**i for i, cell in enumerate(cells) if cell),
                    "legal": False, "primitive_cost": count, "stopped": True}
        if action <= 3:
            cells[action] = True
        else:
            cells[action - 4] = False
    return {"artifact": sum(2**i for i, cell in enumerate(cells) if cell),
            "legal": True, "primitive_cost": count, "stopped": True}


def token_cost(program, motifs):
    # Exhaustive segmentation is intentionally different from the learner's DP.
    def visit(offset):
        if offset == len(program):
            return 0
        candidates = [1 + visit(offset + 1)]
        for motif in motifs:
            if tuple(program[offset:offset + len(motif)]) == tuple(motif):
                candidates.append(1 + visit(offset + len(motif)))
        return min(candidates)
    return visit(0)


def artifact_distribution(motifs, target, beta=1.0, length_penalty=0.3, max_steps=3):
    probabilities = [0.0] * 16
    denominator = 0.0
    for length in range(max_steps + 1):
        for sequence in product(range(8), repeat=length):
            final = interpret(sequence, budget=max_steps)["artifact"]
            mismatch = sum(bool(final & (2**i)) != bool(target & (2**i)) for i in range(4))
            weight = math.exp(-beta * mismatch - length_penalty * token_cost(sequence, motifs))
            probabilities[final] += weight
            denominator += weight
    return [value / denominator for value in probabilities]
