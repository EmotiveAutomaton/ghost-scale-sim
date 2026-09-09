"""W1: a four-cell graphic board with eight place/remove primitives."""
from __future__ import annotations
from itertools import product
from dataclasses import dataclass

ACTIONS = tuple(range(8))
MAX_STEPS = 3


@dataclass(frozen=True)
class Execution:
    artifact: int
    legal: bool
    primitive_cost: int
    stopped: bool
    error: str | None = None


def step(board: int, action: int, feasible: tuple[int, ...] = ACTIONS) -> int:
    if type(action) is not int or action not in feasible:
        raise ValueError("illegal primitive")
    cell = action % 4
    return board | (1 << cell) if action < 4 else board & ~(1 << cell)


def execute(program, *, start: int = 0, feasible=ACTIONS, budget: int = MAX_STEPS) -> Execution:
    board = start
    for index, action in enumerate(program):
        if index >= budget:
            return Execution(board, False, index, False, "timeout")
        try:
            board = step(board, action, tuple(feasible))
        except ValueError:
            return Execution(board, False, index + 1, True, "illegal primitive")
    return Execution(board, True, len(program), True)


def histories(max_steps: int = MAX_STEPS):
    for length in range(max_steps + 1):
        yield from product(ACTIONS, repeat=length)


def distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
