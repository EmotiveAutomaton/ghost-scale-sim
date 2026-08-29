"""Worker-side execution of one card unit, with process-tree accounting (spec §7.4, §20.3).

A worker process initialises numeric threads to one, lowers its priority, hides accelerators,
runs ``unit_<ID>(ctx)`` and returns the unit dict together with its own CPU time and peak
resident set, so the parent can sum child CPU rather than guess it from wall time.
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
    os.environ["GS_V14_WORKER"] = "1"


def run_unit(task: dict) -> dict:
    """``task``: {card: dict, module: 'pkg.mod', lane, wid, rep, tier, tier_name, cfg, smoke, upstream_oracle,
    attack, n_units, item}. Returns {ok, unit, runtime, error}."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    card = card_from_dict(task["card"])
    ctx = {"lane": task["lane"], "wid": task["wid"], "rep": task["rep"], "tier": task["tier"], "tier_name": task.get("tier_name"),
           "card": card, "cfg": task.get("cfg") or {}, "smoke": bool(task.get("smoke")), "upstream_oracle": bool(task.get("upstream_oracle")),
           "attack": task.get("attack"), "n_units": task.get("n_units"), "item": task.get("item")}
    try:
        mod = importlib.import_module(task["module"])
        fn = getattr(mod, f"unit_{card.id}")
        unit = fn(ctx)
        unit = C.to_jsonable(unit)
        ok, err = True, None
    except Exception as exc:                                           # noqa: BLE001
        unit, ok, err = None, False, f"{exc!r}\n{traceback.format_exc()[-3000:]}"
    acc = C.process_accounting()
    return {"ok": ok, "unit": unit, "error": err,
            "runtime": {"wall_s": round(time.perf_counter() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
                        "peak_rss": acc.get("peak_rss"), "pid": acc.get("pid")}}
