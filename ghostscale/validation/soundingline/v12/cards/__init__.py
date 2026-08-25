"""Card implementations for V12, one module per trunk. Every card is a function
``run_<ID>(card, cfg, workers, lane)`` returning a verdict dict whose ``state`` is one of the
resolved states and which has been written by ``finish``.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import RESOLVED, new_verdict
from ..world import World, make_world, random_params


def _worlds_for_impl(cfg, lane: str = "both", limit: int | None = None) -> list:
    """(world_id, World) for the lane. Discovery world 0 is the default (V11) construction; every
    other world is an independent randomized construction seeded from its lineage id."""
    ids = []
    if lane in ("discovery", "both"):
        ids += C.DISCOVERY_IDS
    if lane in ("confirmation", "both"):
        ids += C.CONFIRMATION_IDS
    if lane == "transfer":
        # a third lineage for fresh-world transfer cards run during discovery (S08, X12), so the
        # confirmation lineage stays untouched until the confirmation pass
        ids += list(range(200, 212))
    if limit is not None:
        ids = ids[:limit]
    out = []
    for wid in ids:
        rng = np.random.default_rng(C.world_seed(wid))
        if wid == 0:
            out.append((wid, make_world(cfg, rng=rng)))
        else:
            out.append((wid, make_world(cfg, params=random_params(rng), rng=rng)))
    return out


def lane_of(wid: int) -> str:
    return "confirmation" if wid >= 100 else "discovery"


def finish(card, verdict: dict, gr: G.GateReport, module_file: str, state: str,
           closure_reason: str = "") -> dict:
    assert state in RESOLVED, state
    verdict["state"] = state
    verdict["closure_reason"] = closure_reason
    verdict["gates_summary"] = gr.to_dict()
    C.write_verdict(card.id, verdict, gr, module_file)
    return verdict


def decide_state(gr: G.GateReport, landed_if_valid: bool = True) -> str:
    return "LANDED" if gr.to_dict()["all_passed"] and landed_if_valid else "INSTRUMENT_FAILED"


def by_lane(pairs: dict) -> dict:
    """Split a {world_id: values} mapping into discovery and confirmation halves."""
    return {"discovery": {k: v for k, v in pairs.items() if int(k) < 100},
            "confirmation": {k: v for k, v in pairs.items() if int(k) >= 100}}


def worlds_for(cfg, lane: str = "both", limit: int | None = None) -> list:
    """The worlds a card runs on. ``GS_V12_WORLD_LIMIT`` caps the count for smoke tests; a
    verdict produced under the cap is never a landed result (the runner refuses the cap)."""
    import os
    env = os.environ.get("GS_V12_WORLD_LIMIT")
    if env:
        cap = int(env)
        limit = cap if limit is None else min(limit, cap)
    return _worlds_for_impl(cfg, lane, limit)
