"""V12 confirmation pass (spec wave 5): re-run promoted cards on the untouched confirmation lineage.

Discovery verdicts stay where they are. Confirmation verdicts are written to
results/validation/soundingline/v12/confirmation/<CARD>.json and a ledger to
results/v12/CONFIRMATION.json, so the two lanes never overwrite each other and the gate walker
in tests/ sees both.

Promotion is mechanical: a card is promoted when its discovery verdict LANDED, every
pre-specified criterion it carries passed, and its claim ceiling is METHOD or
CONSTRUCTED_MECHANISM. Pass --only to override.

    ./.venv/Scripts/python.exe runners/run_v12_confirmation.py --auto
    ./.venv/Scripts/python.exe runners/run_v12_confirmation.py --only S04 S05 B02
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v12 as V                                  # noqa: E402
from ghostscale.config import load_config                                            # noqa: E402
from ghostscale.validation.soundingline.v12 import manifest as M                     # noqa: E402
from ghostscale.validation.soundingline.v12.schemas import RESOLVED                  # noqa: E402

LEDGER = M.MANIFEST.parent / "CONFIRMATION.json"


def lower_priority() -> str:
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return "below_normal"
    except Exception as exc:                                                          # noqa: BLE001
        return f"unchanged ({exc!r})"


def criteria_passed(verdict: dict):
    crit = [x for k, x in verdict.get("results", {}).items()
            if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x]
    if not crit:
        return None
    return all(bool(x["passed"]) for x in crit)


def promoted(doc: dict, discovery_dir: Path) -> list:
    out = []
    for c in doc["cards"]:
        p = discovery_dir / f"{c['id']}.json"
        if c["status"] != "LANDED" or not p.exists():
            continue
        v = json.loads(p.read_text(encoding="utf-8"))
        if criteria_passed(v) and v.get("claim_ceiling") in ("METHOD", "CONSTRUCTED_MECHANISM"):
            out.append(c["id"])
    return out


def main() -> None:
    if os.environ.get("GS_V12_WORLD_LIMIT"):
        sys.exit("refusing to run the confirmation pass under GS_V12_WORLD_LIMIT")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true", help="run every promoted card")
    args = ap.parse_args()
    prio = lower_priority()
    cfg = load_config()
    doc = M.load_manifest()
    discovery_dir = Path(V.V12_VERDICTS)
    conf_dir = discovery_dir / "confirmation"
    conf_dir.mkdir(parents=True, exist_ok=True)
    ids = list(args.only or [])
    if args.auto:
        ids += [i for i in promoted(doc, discovery_dir) if i not in ids]
    if not ids:
        sys.exit("nothing to run: pass --auto or --only")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"priority": prio, "cards": {}}
    V.V12_VERDICTS = conf_dir                       # every finish() now writes into the confirmation directory
    print(f"confirmation pass on {len(ids)} cards: {ids}")
    for cid in ids:
        if cid in ledger["cards"] and ledger["cards"][cid].get("state") in RESOLVED:
            print(f"  [{cid}] already confirmed: {ledger['cards'][cid]['state']}")
            continue
        card = M.get_card(doc, cid)
        mod, fn_name = card.module.split(":")
        fn = getattr(importlib.import_module(mod), fn_name)
        disc = json.loads((discovery_dir / f"{cid}.json").read_text(encoding="utf-8")) if (discovery_dir / f"{cid}.json").exists() else {}
        print(f"\n=== {cid} (confirmation): {card.question}")
        t0 = time.perf_counter()
        try:
            verdict = fn(card, cfg, workers=1, lane="confirmation")
            state = verdict.get("state", "LANDED")
            assert (conf_dir / f"{cid}.produced").exists(), "card finished without its produce marker"
            ledger["cards"][cid] = {"state": state, "wall_seconds": round(time.perf_counter() - t0, 2),
                                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "discovery_state": disc.get("state"),
                                    "criteria_passed_discovery": criteria_passed(disc),
                                    "criteria_passed_confirmation": criteria_passed(verdict),
                                    "closure_reason": verdict.get("closure_reason", "")}
            print(f"    -> {state}; criteria {ledger['cards'][cid]['criteria_passed_confirmation']}")
        except Exception as exc:                                                      # noqa: BLE001
            ledger["cards"][cid] = {"state": "ERROR", "error": repr(exc), "traceback": traceback.format_exc()[-2000:],
                                    "wall_seconds": round(time.perf_counter() - t0, 2)}
            print(f"    !! ERROR {exc!r}")
        LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    confirmed = [c for c, x in ledger["cards"].items() if x.get("criteria_passed_confirmation") is True]
    print(f"\nconfirmed (criteria held on the untouched lineage): {confirmed}")


if __name__ == "__main__":
    main()
