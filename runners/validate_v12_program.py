"""Validate the V12 program's manifest, coverage, lineages, and verdicts (spec section 17.4).

    python runners/validate_v12_program.py             # fails if any mandatory card is unresolved
    python runners/validate_v12_program.py --interim   # reports state without failing on PLANNED/BUILT

Fails when a mandatory card is missing or unresolved, a floor was lowered without an amendment,
two cards share an output, a LANDED verdict lacks gates/provenance/environment/runtime, discovery
and confirmation lineages overlap, a closure has no rule, or a forbidden state appears.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostscale.validation.soundingline.v12 import manifest as M      # noqa: E402
from ghostscale.validation.soundingline.v12 import verdict_dir        # noqa: E402
from ghostscale.validation.soundingline.v12.schemas import FLOORS, RESOLVED, STATES  # noqa: E402

MANDATORY = {c.id for c in M.build_cards() if c.trunk != "X"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim", action="store_true")
    args = ap.parse_args()
    doc = M.load_manifest()
    problems, notes = [], []

    ids = [d["id"] for d in doc["cards"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate card ids")
    missing = MANDATORY - set(ids)
    if missing:
        problems.append(f"mandatory cards missing from manifest: {sorted(missing)}")
    outputs = [d["output"] for d in doc["cards"]]
    if len(outputs) != len(set(outputs)):
        problems.append("two cards share an output path")
    disc, conf = set(doc["lineages"]["discovery"]), set(doc["lineages"]["confirmation"])
    if disc & conf:
        problems.append("discovery and confirmation lineages overlap")

    amended = {a.get("card") for a in doc.get("amendments", [])}
    for d in doc["cards"]:
        if d["status"] not in STATES or d["status"] == "DONE":
            problems.append(f"{d['id']}: forbidden state {d['status']}")
        fl = d.get("floors", {})
        for k, v in FLOORS.items():
            if k in fl and d["id"] not in amended:
                if isinstance(v, list):
                    if not set(v).issubset(set(fl[k])):
                        problems.append(f"{d['id']}: floor {k} reduced without amendment")
                elif fl[k] < v:
                    problems.append(f"{d['id']}: floor {k} reduced without amendment")
        if d["status"] in ("SCIENTIFIC_CLOSED", "INSTRUMENT_FAILED", "RESOURCE_BLOCKED") \
                and not d.get("closure_reason"):
            problems.append(f"{d['id']}: closure without a stated rule")
        if d["status"] == "LANDED" or d["status"] in ("SCIENTIFIC_CLOSED", "INSTRUMENT_FAILED"):
            p = verdict_dir() / f"{d['id']}.json"
            if not p.exists():
                problems.append(f"{d['id']}: resolved but no verdict on disk")
                continue
            v = json.loads(p.read_text(encoding="utf-8"))
            for key in ("gates", "produced_by", "environment", "runtime_seconds", "claim_ceiling"):
                if key not in v or v[key] in (None, {}):
                    problems.append(f"{d['id']}: verdict lacks {key}")
            if not v.get("produced_by", {}).get("sha256"):
                problems.append(f"{d['id']}: verdict lacks a source hash")
            marker = verdict_dir() / f"{d['id']}.produced"
            if not marker.exists():
                problems.append(f"{d['id']}: no produce marker")
        if d["status"] not in RESOLVED and d["trunk"] != "X":
            notes.append(f"{d['id']}: {d['status']}")

    from ghostscale.prereg_v12 import lock_status
    ls = lock_status()
    if not ls.get("locked"):
        problems.append(f"prereg lock missing or stale: {ls}")

    cov = M.write_coverage(doc)
    print(json.dumps({"coverage": cov, "problems": problems,
                      "unresolved": notes if args.interim else len(notes)}, indent=2, default=str))
    if problems:
        return 1
    if notes and not args.interim:
        print(f"{len(notes)} mandatory cards unresolved; the program is not complete.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
