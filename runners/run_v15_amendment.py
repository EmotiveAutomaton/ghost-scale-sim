"""V15 amendment runner: re-run a card whose *instrument* was repaired, on its original lineage.

Usage::

    python -m runners.run_v15_amendment --cards X23,X24,H03 --reason "..." [--workers 6]

**Module form only** (see ``run_v15.py``). This is not a second scheduler: it uses the same
``run_card``, the same tier from ``WORKLOAD_LOCK.json``, the same lane and therefore the same
seeds (``rng_for`` seeds on lane, card, world and repeat), and writes through the same verdict,
ledger and manifest code the science stage uses. What it adds is the record of the amendment:

* the original verdict and its ``.produced`` receipt are preserved, never deleted, under
  ``<lane verdict dir>/amended/<CARD>.<stamp>.json``;
* ``results/v15/AMENDMENTS.json`` gets one entry per card with the original's state, criterion
  status and hash beside the replacement's, and the reason;
* a ``kind=amendment`` checkpoint line is appended.

It never writes ``RUNNER_STATUS.json``: the watchdog identifies the live runner by that file's
pid, and a second writer would make it relaunch a second runner. It refuses to run if the
scientific lock is not intact, and it refuses cards that have no resolved verdict to amend --
an unrun card is queue work, not an amendment.

An amendment is an instrument repair (a gate that was not a known answer, a receipt that
miscounted units). It is not a way to change a criterion, an estimator, or a factor: those are
lock amendments and a curator decision, and this runner cannot express them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import common as C          # noqa: E402
from ghostscale.validation.soundingline.v15 import manifest as M        # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402
from ghostscale.validation.soundingline.v15.schemas import RESOLVED      # noqa: E402
from runners.run_v15 import Pool, checkpoint, run_card, safe_workers      # noqa: E402


def preserve(lane: str, cid: str, stamp: str) -> dict:
    d = verdict_dir(lane)
    keep = d / "amended"
    keep.mkdir(parents=True, exist_ok=True)
    out = {}
    for suffix in (".json", ".produced"):
        src = d / f"{cid}{suffix}"
        if src.exists():
            dst = keep / f"{cid}.{stamp}{suffix}"
            shutil.copy2(src, dst)
            out[suffix.lstrip(".")] = dst.relative_to(REPO).as_posix()
    return out


def summary(v: dict, path: str | None = None) -> dict:
    return {"state": v.get("state"), "criterion_status": v.get("criterion_status"),
            "claim_class": v.get("claim_class"),
            "failed_gates": (v.get("gates_summary") or {}).get("failed_names"),
            "verdict_sha256": (v.get("produced_by") or {}).get("sha256"),
            "preserved_at": path}


def main() -> int:
    ap = argparse.ArgumentParser(description="V15 amendment runner")
    ap.add_argument("--cards", required=True, help="comma-separated card ids")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--workers", type=int, default=6,
                    help="keep it below the live runner's count; the machine is shared")
    args = ap.parse_args()
    from ghostscale.prereg_v15 import lock_status
    ls = lock_status()
    if not ls.get("locked"):
        print(f"REFUSED: scientific lock is not intact: {ls}")
        return 2
    C.set_numeric_threads(1)
    C.hide_accelerators()
    C.lower_priority()
    wl = json.loads((v15_dir() / "WORKLOAD_LOCK.json").read_text(encoding="utf-8"))
    tier_name, tier = wl["tier"], wl["tier_config"]
    doc = M.load_manifest()
    todo = []
    for cid in [c.strip() for c in args.cards.split(",") if c.strip()]:
        card = M.get_card(doc, cid)
        lanes = [ln for ln in card.lanes if (C.load_verdict(cid, ln) or {}).get("state") in RESOLVED]
        if not lanes:
            print(f"REFUSED {cid}: no resolved verdict to amend (an unrun card is queue work)")
            return 2
        todo.append((card, lanes))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    pool = Pool(safe_workers(args.workers))
    rc = 0
    try:
        for card, lanes in todo:
            for lane in lanes:
                original = C.load_verdict(card.id, lane)
                kept = preserve(lane, card.id, stamp)
                t0 = time.perf_counter()
                r = run_card(pool, card, lane, tier, tier_name, {}, False)
                replacement = C.load_verdict(card.id, lane) or {}
                M.add_amendment(card.id, summary(original, kept.get("json")),
                                summary(replacement), f"{args.reason} [lane={lane}]")
                M.update_card(doc, card.id, status=r["state"],
                              criterion_status=r.get("criterion_status", "UNEVALUATED"),
                              actual={"wall_s": r["wall_s"], "lane": lane, "amended": stamp})
                checkpoint("amendment", card=card.id, lane=lane, state=r["state"],
                           criterion_status=r.get("criterion_status"),
                           original_state=original.get("state"),
                           wall_s=round(time.perf_counter() - t0, 2))
                print(f"  {lane[:4]} {card.id}: {original.get('state')} -> {r['state']} "
                      f"{r.get('criterion_status', '')} {r['wall_s']:.0f}s"
                      + (f"   errors: {r['errors'][:1]}" if r.get("errors") else ""))
                if r["state"] == "VOID":
                    rc = 1
        M.write_coverage(doc)
    finally:
        pool.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
