"""Worker-side execution of one card unit or one coverage cell, with process-tree accounting.

A worker initialises numeric threads to one, lowers its priority, hides accelerators, runs
``unit_<ID>(ctx)`` (or ``coverage_cell(ctx)``) and returns the unit dict together with its own CPU
time and peak resident set, so the parent can *sum* child CPU rather than infer it from wall time.
Spec §9.4 requires descendant CPU in the runtime receipt, and a parent that guesses would let a
stalled pool look busy.
"""
from __future__ import annotations

import importlib
import os
import time
import traceback

from . import common as C
from .schemas import card_from_dict


def init_worker() -> None:
    C.set_numeric_threads(1)
    C.hide_accelerators()
    C.lower_priority()
    os.environ["GS_V15_WORKER"] = "1"


def run_unit(task: dict) -> dict:
    """``task``: {card, module, lane, wid, rep, tier, tier_name, cfg, smoke, attack, n_units,
    item, expected}. Returns {ok, unit, runtime, error}."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    card = card_from_dict(task["card"])
    ctx = {"lane": task["lane"], "wid": task["wid"], "rep": task["rep"], "tier": task["tier"],
           "tier_name": task.get("tier_name"), "card": card, "cfg": task.get("cfg") or {},
           "smoke": bool(task.get("smoke")), "attack": task.get("attack"),
           "n_units": task.get("n_units"), "item": task.get("item"),
           "expected": task.get("expected") or {},
           "upstream_oracle": bool(task.get("upstream_oracle"))}
    try:
        mod = importlib.import_module(task["module"])
        fn = getattr(mod, f"unit_{card.id}")
        unit = C.to_jsonable(fn(ctx))
        ok, err = True, None
    except Exception as exc:                                           # noqa: BLE001
        unit, ok, err = None, False, f"{exc!r}\n{traceback.format_exc()[-3000:]}"
    acc = C.process_accounting()
    return {"ok": ok, "unit": unit, "error": err,
            "runtime": {"wall_s": round(time.perf_counter() - t0, 3),
                        "cpu_s": round(time.process_time() - c0, 3),
                        "peak_rss": acc.get("peak_rss"), "pid": acc.get("pid")}}


def run_coverage(task: dict) -> dict:
    """One balanced-coverage cell (spec §9.5). Cells carry no narrative and are never a packet."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        from .coverage import execute_cell
        out = C.to_jsonable(execute_cell(task["cell"], task.get("tier") or {},
                                         smoke=bool(task.get("smoke"))))
        ok, err = True, None
    except Exception as exc:                                           # noqa: BLE001
        out, ok, err = None, False, f"{exc!r}\n{traceback.format_exc()[-3000:]}"
    return {"ok": ok, "unit": out, "error": err, "cell_id": task.get("cell", {}).get("cell_id"),
            "runtime": {"wall_s": round(time.perf_counter() - t0, 3),
                        "cpu_s": round(time.process_time() - c0, 3)}}
