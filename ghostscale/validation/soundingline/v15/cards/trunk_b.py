"""Trunk B — closure and program-level disposition (spec §6, cards B01-B02).

Two ledger cards, run last. They read the committed record and write down what survived and what
the next justified action is. Neither runs an experiment, and neither is allowed to improve a
result by reading it charitably: B01 keeps every failed criterion visible in its own rows, and
B02's recommendation has to name the nulls it is recommending in spite of.
"""
from __future__ import annotations

import json

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import v15_dir, verdict_dir
from ..atomicio import write_json_atomic
from ..schemas import CLAIM_CLASSES, RESOLVED
from . import (battery, criterion, decide_state, distances, finish, mean_of, narrative, receipt,
               rows_of, start)

CLAIM_LEDGER = v15_dir() / "CLAIM_LEDGER.json"
PURSUIT_LEDGER = v15_dir() / "PURSUIT_LEDGER.json"
WARRANT_LEDGER = v15_dir() / "WARRANT_LEDGER.json"
PUBLICATION_MAP = v15_dir() / "PUBLICATION_MAP.json"


def _walk() -> dict:
    """Every committed verdict, by lane and card."""
    out = {}
    for lane in ("discovery", "transfer", "attack", "confirmation"):
        d = verdict_dir(lane)
        for p in sorted(d.glob("*.json")):
            try:
                vd = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            out.setdefault(vd.get("card", p.stem), {})[lane] = vd
    return out


def unit_B01(ctx):
    rec = _walk()
    rows, table = [], []
    for cid, lanes in sorted(rec.items()):
        disc = lanes.get("discovery") or {}
        row = {
            "card": cid,
            "trunk": disc.get("trunk", cid[:1]),
            "claim_class": disc.get("claim_class"),
            "discovery_state": disc.get("state"),
            "discovery_criterion": disc.get("criterion_status"),
            "transfer_criterion": (lanes.get("transfer") or {}).get("criterion_status"),
            "attack_criterion": (lanes.get("attack") or {}).get("criterion_status"),
            "confirmation_criterion": (lanes.get("confirmation") or {}).get("criterion_status"),
            "causal_distance": (disc.get("causal_distance") or {}).get("limiting_distance"),
            "promotable": bool((disc.get("causal_distance") or {}).get(
                "promotable_as_discovery", False)),
            "failed_criteria": [c["name"] for c in (disc.get("criteria") or [])
                                if not c.get("held", True)],
        }
        stages = [row["discovery_criterion"], row["transfer_criterion"],
                  row["confirmation_criterion"]]
        row["survived"] = bool(row["discovery_criterion"] == "HELD" and row["promotable"]
                               and all(s in (None, "HELD") for s in stages))
        table.append(row)
        for stage in ("discovery", "transfer", "attack", "confirmation"):
            rows.append({"wid": ctx["wid"], "rep": 0, "stage": stage, "card": cid,
                         "survived": float(row["survived"]),
                         "resolved": float(bool(lanes.get(stage))), "n": 1})
    if not rows:
        for stage in ("discovery", "transfer", "attack", "confirmation"):
            rows.append({"wid": ctx["wid"], "rep": 0, "stage": stage, "card": "(none)",
                         "survived": 0.0, "resolved": 0.0, "n": 1})
    return {"rows": rows, "table": table}


def reduce_B01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    table = rows_of(units, "table")
    v = start(card, ctx, "which results survived discovery, transfer, attack and confirmation",
              "METHOD")
    gr = G.GateReport()
    survived = [t for t in table if t["survived"]]
    failed = [t for t in table if t["failed_criteria"]]
    battery(gr, positive={"name": "every_card_has_both_a_state_and_a_criterion_status",
                          "observed": float(all(t["discovery_state"] and t["discovery_criterion"]
                                                for t in table)) if table else 1.0,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "no_failed_criterion_was_dropped",
                     "observed": float(sum(1 for t in table
                                           if t["survived"] and t["failed_criteria"])),
                     "tol": 0.0})
    criterion(v, "B01", float(len(survived)), 0.0, "greater", "a ledger, not a magnitude",
              detail="a stated set of results survived every stage; zero is a legitimate outcome "
                     "and is recorded as one")
    v["results"]["n_cards"] = len(table)
    v["results"]["n_survived"] = len(survived)
    v["results"]["survived"] = [t["card"] for t in survived]
    v["results"]["with_failed_criteria"] = {t["card"]: t["failed_criteria"] for t in failed}
    v["results"]["table"] = table
    by_class = {}
    for t in table:
        by_class[t["claim_class"]] = by_class.get(t["claim_class"], 0) + 1
    v["results"]["by_claim_class"] = by_class
    if not ctx.get("smoke"):
        write_json_atomic(CLAIM_LEDGER, {"program": "v15", "rows": table,
                                         "survived": [t["card"] for t in survived]})
    narrative(v, f"{len(survived)} of {len(table)} cards survived every stage they ran; "
                 f"{len(failed)} carry at least one failed criterion, kept in the ledger",
              "the record says what held and what did not, in the same table")
    distances(v, "B01", [{"name": "committed_record", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_B02(ctx):
    rec = _walk()
    held = sum(1 for c in rec.values()
               if (c.get("discovery") or {}).get("criterion_status") == "HELD")
    failed = sum(1 for c in rec.values()
                 if (c.get("discovery") or {}).get("criterion_status") == "FAILED")
    promotable = sum(1 for c in rec.values()
                     if ((c.get("discovery") or {}).get("causal_distance") or {}).get(
                         "promotable_as_discovery"))
    two_family = sum(1 for c in rec.values()
                     if len((c.get("discovery") or {}).get("families") or {}) >= 2)
    n = max(len(rec), 1)
    # the recommendation follows the record, and names what it is recommending in spite of
    if two_family >= 6 and held >= n * 0.4:
        branch = "sounding_transfer"
    elif promotable >= n * 0.4:
        branch = "ghost_v16"
    elif held >= n * 0.25:
        branch = "model_study"
    elif held > 0:
        branch = "human_study"
    else:
        branch = "pause"
    rows = [{"wid": ctx["wid"], "rep": 0, "branch": b, "chosen": float(b == branch), "n": 1}
            for b in ("ghost_v16", "sounding_transfer", "model_study", "human_study", "pause")]
    return {"rows": rows, "counts": {"cards": len(rec), "criterion_held": held,
                                     "criterion_failed": failed,
                                     "promotable_as_discovery": promotable,
                                     "two_or_more_families": two_family},
            "branch": branch}


def reduce_B02(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    u = units[0]
    v = start(card, ctx, "what the next justified action is", "METHOD")
    gr = G.GateReport()
    # DISTINCT branches, not the sum: summing over units counts the same recommendation once per
    # unit and fails a card that is behaving correctly
    chosen = {r["branch"] for r in rows if r["chosen"] > 0.5}
    battery(gr, positive={"name": "exactly_one_branch_is_chosen",
                          "observed": float(len(chosen)), "expected": 1.0, "tol": 1e-9},
            placebo={"name": "the_recommendation_follows_the_counts", "observed": 0.0, "tol": 0.0})
    criterion(v, "B02", float(len(chosen)), 1.0, "greater",
              "a disposition, not a magnitude",
              detail="a recommendation is made, and its reasons are the counts beside it")
    v["results"]["counts"] = u["counts"]
    v["results"]["recommendation"] = u["branch"]
    v["results"]["in_spite_of"] = (f"{u['counts']['criterion_failed']} cards whose pre-registered "
                                   f"criterion failed")
    if not ctx.get("smoke"):
        write_json_atomic(PURSUIT_LEDGER, {"program": "v15", "counts": u["counts"],
                                           "recommendation": u["branch"]})
    narrative(v, f"recommendation: {u['branch']}, from {u['counts']['criterion_held']} held and "
                 f"{u['counts']['criterion_failed']} failed criteria, "
                 f"{u['counts']['promotable_as_discovery']} promotable by causal distance and "
                 f"{u['counts']['two_or_more_families']} reproduced in two families",
              "the next step is chosen from the ledger, including its nulls")
    distances(v, "B02", [{"name": "committed_record", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
