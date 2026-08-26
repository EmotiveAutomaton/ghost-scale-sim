"""V13 confirmation pass (spec wave 5): promoted cards re-run on the untouched confirmation
lineage. Promotion is mechanical: a card is promoted when its discovery verdict LANDED, every
pre-specified criterion it carries passed, and its claim ceiling is METHOD or
CONSTRUCTED_MECHANISM. Confirmation verdicts go to results/validation/soundingline/v13/confirmation/
and the ledger to results/v13/CONFIRMATION.json. The promoted set and its hashes are frozen in
the ledger before the first confirmation world is touched.

    python runners/run_v13_confirmation.py --auto
    python runners/run_v13_confirmation.py --only C04 O02
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v13 as V                          # noqa: E402
from ghostscale.validation.soundingline.v13 import common as C              # noqa: E402
from ghostscale.validation.soundingline.v13 import manifest as M            # noqa: E402
from ghostscale.validation.soundingline.v13.runtime import init_worker      # noqa: E402
from ghostscale.validation.soundingline.v13.schemas import RESOLVED, card_from_dict  # noqa: E402

LEDGER = V.V13_RESULTS / "CONFIRMATION.json"


def criteria_passed(verdict: dict):
    crit = [x for k, x in (verdict or {}).get("results", {}).items() if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x]
    if not crit:
        return None
    return all(bool(x["passed"]) for x in crit)


def promoted(doc: dict) -> list:
    out = []
    for d in doc["cards"]:
        if d["trunk"] in ("I", "L", "X") or d["status"] != "LANDED":
            continue
        v = C.load_verdict(d["id"], "discovery")
        if v is None:
            continue
        if criteria_passed(v) and v.get("claim_ceiling") in ("METHOD", "CONSTRUCTED_MECHANISM"):
            out.append(d["id"])
    return out


def run(doc: dict, workers: int, pool, only=None) -> None:
    from runners.run_v13 import record_runtime, run_card, tier_for
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"cards": {}, "frozen": None}
    ids = list(only or [])
    if not ids:
        ids = promoted(doc)
    if ledger.get("frozen") is None:
        ledger["frozen"] = {"promoted": ids, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "discovery_hashes": {i: C.file_sha(V.verdict_dir("discovery") / f"{i}.json") for i in ids if (V.verdict_dir("discovery") / f"{i}.json").exists()}}
        LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tname, tier = tier_for(doc)
    print(f"confirmation pass on {len(ids)} cards at {tname}: {ids}", flush=True)
    for cid in ids:
        if cid in ledger["cards"] and ledger["cards"][cid].get("state") in RESOLVED:
            print(f"  [{cid}] already confirmed: {ledger['cards'][cid]['state']}")
            continue
        card = card_from_dict(next(d for d in doc["cards"] if d["id"] == cid))
        card.lanes = ["confirmation"]
        disc = C.load_verdict(cid, "discovery") or {}
        t0 = time.perf_counter()
        try:
            out = run_card(doc, card, "confirmation", tname, tier, workers, pool)
            v, acct = out["verdict"], out["accounting"]
            ledger["cards"][cid] = {"state": v["state"], "wall_s": acct["wall_s"], "children_cpu_s": acct["children_cpu_s"],
                                    "criteria_passed_discovery": criteria_passed(disc), "criteria_passed_confirmation": criteria_passed(v),
                                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "closure_reason": v.get("closure_reason", "")}
            record_runtime(cid, "confirmation", acct, v["state"])
            print(f"  [{cid}] -> {v['state']}; criteria {ledger['cards'][cid]['criteria_passed_confirmation']}", flush=True)
        except Exception as exc:                                             # noqa: BLE001
            ledger["cards"][cid] = {"state": "ERROR", "error": repr(exc), "wall_s": round(time.perf_counter() - t0, 1)}
            print(f"  [{cid}] !! ERROR {exc!r}", flush=True)
        LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    held = [c for c, x in ledger["cards"].items() if x.get("criteria_passed_confirmation") is True]
    print(f"\nconfirmed (criteria held on the untouched lineage): {held}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if os.environ.get("GS_V13_SMOKE"):
        sys.exit("refusing to run the confirmation pass under GS_V13_SMOKE")
    workers = args.workers or max(1, min(12, (os.cpu_count() or 2) // 2))
    C.lower_priority()
    C.hide_accelerators()
    doc = M.load_manifest()
    if not args.auto and not args.only:
        sys.exit("nothing to run: pass --auto or --only")
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        run(doc, workers, pool, only=args.only)


if __name__ == "__main__":
    main()
