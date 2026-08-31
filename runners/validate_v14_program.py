"""Validate the V14 program: manifest, locks, lineages, cells, ledgers, verdicts (spec §7.5).

    python runners/validate_v14_program.py             # fails while mandatory cards are unresolved
    python runners/validate_v14_program.py --interim   # reports state without failing on unresolved

Fails when: a mandatory card id is missing; a declared factor, level, route, policy or ecology is
absent from a resolved verdict's cells; realized cells fall below the instantiated expected-cell
matrix; two cards share an output path; lanes share a lineage id; a resolved card lacks a ledger
entry, a verdict, gates, provenance or a source hash; the ledger's hashes do not match the files;
a closure has no stated rule; a forbidden state appears; the workload was selected without a
pilot forecast; the scientific lock is stale; an attack verdict names a flight the attack matrix
does not declare it relevant to (or omits one it does); a tier was selected without a window
record; the closure cards resolved without a SHORT_RUN record; or a plain-language record uses
vocabulary that would describe a person rather than a constructed world.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v14 as V                          # noqa: E402
from ghostscale.validation.soundingline.v14 import common as C              # noqa: E402
from ghostscale.validation.soundingline.v14 import manifest as M            # noqa: E402
from ghostscale.validation.soundingline.v14.schemas import RESOLVED, STATES, VERDICT_REQUIRED  # noqa: E402

MANDATORY = set(M.MANDATORY_IDS)
# words that would turn a constructed-world result into a claim about people; the record must not use them
FORBIDDEN_VOCABULARY = ("patient", "clinical", "clinician", "diagnosis", "diagnosed", "therap", "human subject", "participant", "embodied", "brain",
                        "neural", "the person", "people ", "children", "adults", "real people", "in humans")   # "diagnostic" (of evidence) is house vocabulary and stays legal


def attack_flights_named(v: dict) -> set:
    """The flights an attack verdict actually applied to (marked relevant); a flight recorded with
    relevant=False is the required not-silent refusal, not an application."""
    from ghostscale.prereg_v14 import FLIGHTS
    named = set()
    for k, x in (v or {}).get("results", {}).items():
        if isinstance(x, dict):
            for kk, xx in x.items():
                if kk in FLIGHTS and isinstance(xx, dict) and xx.get("relevant") is True:
                    named.add(kk)
                if isinstance(xx, dict):
                    for k3, x3 in xx.items():
                        if k3 in FLIGHTS and isinstance(x3, dict) and x3.get("relevant") is True:
                            named.add(k3)
    return named


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
    lin = doc.get("lineages", {})
    if not C.lineage_disjoint({k: v for k, v in lin.items() if isinstance(v, list)}):
        problems.append("lane lineages overlap")

    cells_doc = M.load_cells()
    ledger = json.loads(C.COMPLETION.read_text(encoding="utf-8")).get("entries", {}) if C.COMPLETION.exists() else {}
    for d in doc["cards"]:
        cid = d["id"]
        if d["status"] not in STATES or d["status"] == "DONE":
            problems.append(f"{cid}: forbidden state {d['status']}")
        if d["status"] in ("SCIENTIFIC_CLOSED", "INSTRUMENT_FAILED", "RESOURCE_BLOCKED", "VOID") and not d.get("closure_reason"):
            v = C.load_verdict(cid, "discovery" if "discovery" in d.get("lanes", ["discovery"]) else "transfer")
            if not (v and v.get("closure_reason")):
                problems.append(f"{cid}: closure without a stated rule")
        if d["status"] in RESOLVED and d["trunk"] != "X":
            lane = "discovery" if "discovery" in d.get("lanes", ["discovery"]) else "transfer"   # transfer-only cards resolve in their own lane
            key = f"{lane}:{cid}"
            p = V.verdict_dir(lane) / f"{cid}.json"
            if not p.exists():
                problems.append(f"{cid}: resolved but no verdict on disk")
                continue
            v = json.loads(p.read_text(encoding="utf-8"))
            for req in VERDICT_REQUIRED:
                if req not in v or v[req] in (None, {}):
                    problems.append(f"{cid}: verdict lacks {req}")
            if not v.get("produced_by", {}).get("sha256"):
                problems.append(f"{cid}: verdict lacks a source hash")
            rec = v.get("record", {})
            if not rec.get("what_happened"):
                problems.append(f"{cid}: plain-language record empty")
            if key not in ledger:
                problems.append(f"{cid}: resolved without a completion-ledger entry")
            elif ledger[key]["verdict_sha256"] != C.file_sha(p):
                problems.append(f"{cid}: ledger hash does not match the verdict on disk")
            if cells_doc and d["status"] == "LANDED":
                exp = cells_doc.get("cards", {}).get(cid, {}).get("discovery")
                rec2 = v.get("expected_cell_receipt", {})
                if exp and not rec2.get("ok", False):
                    problems.append(f"{cid}: expected-cell receipt not met: {rec2}")
            declared = set()
            for lv in d.get("factors", {}).values():
                declared.update(str(x) for x in lv)
            realized = set()
            for cell in v.get("cells", {}):
                realized.update(cell.split("|"))
            absent = declared - realized
            if absent and d["status"] == "LANDED":
                problems.append(f"{cid}: declared levels never realized: {sorted(absent)[:8]}")
        if d["status"] not in RESOLVED and d["trunk"] != "X":
            notes.append(f"{cid}: {d['status']}")
        # attack applicability: what an attack verdict reports on must be what the matrix declares
        if d["trunk"] == "X":
            from ghostscale.prereg_v14 import ATTACK_RELEVANCE
            xv = C.load_verdict(cid, "attack")
            if xv is not None:
                named = attack_flights_named(xv)
                declared = {fl for fl, xs in ATTACK_RELEVANCE.items() if cid in xs}    # the matrix is keyed by flight
                if named and named != declared:
                    problems.append(f"{cid}: attack applied to {sorted(named)} but the matrix declares {sorted(declared)}")
        # forbidden vocabulary in the plain-language record
        for lane_ in (["attack"] if d["trunk"] == "X" else d.get("lanes", ["discovery"])):
            vv = C.load_verdict(cid, lane_)
            if vv is None:
                continue
            text = " ".join(str(vv.get("record", {}).get(k, "")) for k in ("what_happened", "question", "what_it_means")).lower()
            hits = [w for w in FORBIDDEN_VOCABULARY if w in text]
            if hits:
                problems.append(f"{cid} [{lane_}]: record uses forbidden vocabulary {hits}")

    from ghostscale.prereg_v14 import lock_status
    ls = lock_status()
    if not ls.get("structural_locked"):
        problems.append(f"structural lock missing or stale: {ls}")
    if doc.get("selected_tier") and not (V.V14_RESULTS / "PILOT.json").exists():
        problems.append("a tier was selected without a pilot record")
    if doc.get("selected_tier") and not ls.get("locked"):
        problems.append(f"tier selected but the scientific lock is stale or absent: {ls}")
    if doc.get("selected_tier") and not M.DEADLINE.exists():
        problems.append("a tier was selected but no window record (DEADLINE.json) exists")
    if M.DEADLINE.exists():
        w = json.loads(M.DEADLINE.read_text(encoding="utf-8"))
        if not w.get("immutable") or abs((w["deadline_epoch"] - w["opened_epoch"]) / 3600.0 - w["window_hours"]) > 1e-6:
            problems.append("the window record is inconsistent")
    by_id = {d["id"]: d for d in doc["cards"]}
    if all(by_id.get(b, {}).get("status") in RESOLVED for b in ("B01", "B02")) and not (V.V14_RESULTS / "SHORT_RUN.json").exists():
        problems.append("closure cards resolved without a SHORT_RUN record")

    cov = M.write_coverage(doc)
    print(json.dumps({"coverage": cov, "problems": problems, "lock": ls,
                      "unresolved": notes if args.interim else len(notes)}, indent=2, default=str))
    if problems:
        return 1
    if notes and not args.interim:
        print(f"{len(notes)} mandatory cards unresolved; the program is not complete.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
