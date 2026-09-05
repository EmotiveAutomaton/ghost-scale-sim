"""V15 confirmation: a frozen packet, re-run on an untouched lineage (spec §9.1).

At the freeze hour the promoted set is chosen from the *committed discovery record* and written
once. Every later entry verifies the frozen packet against the same hashes before running anything,
so a confirmation cannot quietly widen. Widening is an amendment: the original packet is preserved
beside the replacement in ``results/v15/AMENDMENTS.json`` and ``CONFIRMATION_REGISTRY.json``.

The freeze rule is the one in ``prereg_v15.FREEZE_RULE``: at most ``CONFIRMATION_CAP`` candidates,
at most one per flight, the flight's primary card preferred, flights taken in declared order. It is
deliberately narrow -- a confirmation lane that runs everything is a second discovery lane.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.prereg_v15 import (CONFIRMATION_CAP, FLIGHT_PRIMARY_CARD, FLIGHTS,  # noqa: E402
                                   FREEZE_RULE, _h, lock_status)
from ghostscale.validation.soundingline.v15 import common as C          # noqa: E402
from ghostscale.validation.soundingline.v15 import manifest as M        # noqa: E402
from ghostscale.validation.soundingline.v15 import runtime_contract as RC  # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402
from ghostscale.validation.soundingline.v15.atomicio import write_json_atomic  # noqa: E402
from ghostscale.validation.soundingline.v15.schemas import RESOLVED, TIERS  # noqa: E402

REGISTRY = v15_dir() / "CONFIRMATION_REGISTRY.json"
AMENDMENTS = v15_dir() / "AMENDMENTS.json"


def select_candidates() -> dict:
    """Choose the frozen packet from the committed discovery record.

    A card is a candidate only if its criterion HELD, its causal-distance audit leaves it
    promotable, and its state is LANDED. That is the whole rule, and it is applied to the record
    rather than to anybody's memory of the record.
    """
    picked, rows = [], []
    for flight in FLIGHTS:                                   # declared order
        best = None
        for cid in FLIGHTS[flight]:
            v = C.load_verdict(cid, "discovery")
            if not v:
                continue
            ok = (v.get("state") == "LANDED"
                  and v.get("criterion_status") == "HELD"
                  and (v.get("causal_distance") or {}).get("promotable_as_discovery", False))
            rows.append({"flight": flight, "card": cid, "eligible": bool(ok),
                         "state": v.get("state"), "criterion_status": v.get("criterion_status"),
                         "causal_distance": (v.get("causal_distance") or {}).get(
                             "limiting_distance")})
            if not ok:
                continue
            if cid == FLIGHT_PRIMARY_CARD.get(flight):
                best = cid                                   # the primary always wins its flight
                break
            if best is None:
                best = cid
        if best:
            picked.append({"flight": flight, "card": best})
        if len(picked) >= CONFIRMATION_CAP:
            break
    return {"packet": picked[:CONFIRMATION_CAP], "considered": rows,
            "cap": CONFIRMATION_CAP, "rule": FREEZE_RULE}


def freeze(force: bool = False) -> dict:
    """Write the packet once. Later calls verify rather than rewrite."""
    sel = select_candidates()
    ls = lock_status()
    doc = {"program": "v15", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "frozen_at_elapsed_hours": round(RC.elapsed_hours(), 3),
           "packet": sel["packet"], "considered": sel["considered"],
           "cap": CONFIRMATION_CAP, "rule": FREEZE_RULE,
           "structural_locked": ls.get("structural_locked"),
           "scientific_locked": ls.get("locked"),
           "discovery_hashes": {p["card"]: _discovery_hash(p["card"]) for p in sel["packet"]},
           "amendments": []}
    if REGISTRY.exists() and not force:
        old = json.loads(REGISTRY.read_text(encoding="utf-8"))
        return old
    write_json_atomic(REGISTRY, doc)
    return doc


def _discovery_hash(cid: str) -> str | None:
    p = verdict_dir("discovery") / f"{cid}.json"
    return C.file_sha(p) if p.exists() else None


def verify(doc: dict) -> dict:
    """Every later entry checks the frozen packet still describes the same discovery verdicts."""
    problems = []
    for cid, want in (doc.get("discovery_hashes") or {}).items():
        got = _discovery_hash(cid)
        if want and got and want != got:
            problems.append(f"{cid}: discovery verdict changed since the freeze")
    ls = lock_status()
    if not ls.get("locked"):
        problems.append(f"scientific lock not intact: {ls.get('reason')}")
    return {"ok": not problems, "problems": problems}


def amend(doc: dict, add: list, reason: str) -> dict:
    """Widen the packet on record, preserving the original beside the replacement."""
    original = json.loads(json.dumps(doc["packet"]))
    doc["packet"] = doc["packet"] + [{"flight": "amended", "card": c} for c in add]
    doc["amendments"].append({"when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "added": list(add), "reason": reason, "original": original})
    write_json_atomic(REGISTRY, doc)
    M.add_amendment("CONFIRMATION", {"packet": original}, {"packet": doc["packet"]}, reason)
    return doc


def run(workers: int = 4, only: list | None = None, force_freeze: bool = False,
        external: bool = False) -> dict:
    """``external=True`` when invoked beside the live runner (a widening after the freeze):
    it must not write RUNNER_STATUS.json, whose pid the watchdog uses to identify the runner --
    a second writer makes the watchdog launch a second runner over the live one."""
    from runners.run_v15 import Pool, checkpoint, run_card
    from runners.run_v15 import status as _status
    status = (lambda **kw: None) if external else _status
    doc = freeze(force=force_freeze)
    ver = verify(doc)
    if not ver["ok"]:
        print("confirmation REFUSED: " + "; ".join(ver["problems"]))
        checkpoint("confirmation_refused", problems=ver["problems"])
        return {"refused": True, **ver}
    wl = json.loads((v15_dir() / "WORKLOAD_LOCK.json").read_text(encoding="utf-8"))
    tier_name, tier = wl["tier"], wl["tier_config"]
    mdoc = M.load_manifest()
    pool = Pool(workers)
    out = []
    try:
        for entry in doc["packet"]:
            cid = entry["card"]
            if only and cid not in only:
                continue
            # Resumable, like the science stage: a relaunch inside the confirmation phase must
            # not re-run a card that already resolved on the confirmation lineage. Re-running
            # would overwrite the verdict with identical rollouts (spec 9.4 forbids repeating
            # them) and spend the phase twice. Added 2026-09-05, before the first freeze.
            prior = C.load_verdict(cid, "confirmation")
            if prior and prior.get("state") in RESOLVED:
                out.append({"card": cid, "state": prior.get("state"),
                            "criterion_status": prior.get("criterion_status"),
                            "flight": entry["flight"], "resumed": True})
                print(f"  [{RC.elapsed_hours():6.2f}h] confirm {cid}: already resolved "
                      f"({prior.get('state')} {prior.get('criterion_status', '')}), kept")
                continue
            if RC.elapsed_hours() >= RC.CONFIRMATION_END_HOUR:
                checkpoint("confirmation_window_closed", card=cid)
                break
            card = M.get_card(mdoc, cid)
            card.lanes = ["confirmation"]
            status(stage="confirmation", card=cid, workers=pool.n)
            r = run_card(pool, card, "confirmation", tier, tier_name, {}, False)
            out.append({**r, "flight": entry["flight"]})
            checkpoint("confirmation", card=cid, state=r["state"],
                       criterion_status=r.get("criterion_status"))
            print(f"  [{RC.elapsed_hours():6.2f}h] confirm {cid}: {r['state']} "
                  f"{r.get('criterion_status','')}")
    finally:
        pool.close()
    doc["results"] = out
    write_json_atomic(REGISTRY, doc)
    held = sum(1 for r in out if r.get("criterion_status") == "HELD")
    print(f"confirmation: {held} of {len(out)} held on the untouched lineage")
    return {"packet": doc["packet"], "results": out, "held": held}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--show", action="store_true", help="print the packet and exit")
    ap.add_argument("--widen", default=None,
                    help="comma-separated cards to ADD to the frozen packet as a recorded "
                         "amendment, then run only those, beside the live runner (never writes "
                         "RUNNER_STATUS). Requires --reason and an existing freeze.")
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    if a.show:
        print(json.dumps(select_candidates(), indent=2))
    elif a.widen:
        if not a.reason:
            sys.exit("--widen requires --reason (the amendment is recorded with it)")
        if not REGISTRY.exists():
            sys.exit("no freeze yet: the packet is frozen by the live runner at hour 150; "
                     "widen after that")
        add = [c.strip() for c in a.widen.split(",") if c.strip()]
        doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
        already = {p["card"] for p in doc["packet"]}
        add = [c for c in add if c not in already]
        if not add:
            sys.exit("nothing to add: every card named is already in the packet")
        amend(doc, add, a.reason)
        run(a.workers, only=add, external=True)
    else:
        run(a.workers, a.only.split(",") if a.only else None)
