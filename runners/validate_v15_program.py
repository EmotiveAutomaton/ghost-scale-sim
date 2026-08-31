"""Read-only validator for the V15 program (spec §10, §14).

Checks the *record*, never the prose. It writes nothing and runs no experiment, so it is safe to
call at any point including while the runner is live.

The check that exists because of V14
------------------------------------
V14's bridge packet recorded a card as UNRUN while the committed verdict for that card was LANDED
with its criterion held, because the export was generated before the cards it described had closed.
``stale_export`` below refuses any export whose source verdict hashes predate the closure of the
cards it names. That is the mechanical version of "remember to regenerate".
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import common as C          # noqa: E402
from ghostscale.validation.soundingline.v15 import manifest as M        # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402
from ghostscale.validation.soundingline.v15.schemas import (CLAIM_CLASSES,  # noqa: E402
                                                            CRITERION_STATUS, ENDPOINTS,
                                                            RESOLVED, STATES)

#: Vocabulary that would make a constructed-world result read as a human, clinical or historical
#: one. "diagnostic" is house vocabulary and is deliberately NOT in this list -- V14's validator
#: matched the stem "diagnos" and flagged its own legitimate usage.
FORBIDDEN_VOCABULARY = [
    "patient", "clinical", "diagnosis", "diagnosed", "therapy", "therapeutic",
    "participant", "subject reported", "neural", "brain", "limbic", "cortex", "cortical",
    "felt affect", "lived experience", "embodied", "phenomenal",
    "human evidence", "historical record", "real artist", "real author",
]
SPEC_TRUNK_COUNTS = {"I": 8, "C": 14, "M": 12, "E": 12, "G": 10, "V": 10, "S": 10, "R": 8,
                     "F": 10, "H": 8, "P": 8, "B": 2}


def _load(lane: str) -> dict:
    out = {}
    d = verdict_dir(lane)
    for p in sorted(d.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            out[p.stem] = {"_unreadable": repr(exc)}
    return out


def check_manifest(problems: list) -> dict:
    cards = M.build_cards()
    mand, att = M.mandatory(cards), M.attacks(cards)
    counts = {}
    for c in mand:
        counts[c.trunk] = counts.get(c.trunk, 0) + 1
    if len(mand) != 112:
        problems.append(f"manifest has {len(mand)} mandatory cards, spec says 112")
    if len(att) != 24:
        problems.append(f"manifest has {len(att)} attacks, spec says 24")
    if counts != SPEC_TRUNK_COUNTS:
        problems.append(f"trunk counts {counts} do not match the spec's {SPEC_TRUNK_COUNTS}")
    ids = [c.id for c in cards]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate card ids: {dupes}")
    for c in mand:
        if c.causal and not c.endpoints:
            problems.append(f"{c.id}: a causal card with no declared hidden-event endpoint")
        for e in c.endpoints:
            if e not in ENDPOINTS:
                problems.append(f"{c.id}: undeclared endpoint {e!r}")
        if not c.sesoi_basis:
            problems.append(f"{c.id}: a smallest effect of interest with no stated basis")
        if c.claim_class not in CLAIM_CLASSES:
            problems.append(f"{c.id}: unknown claim class {c.claim_class!r}")
    return {"n_mandatory": len(mand), "n_attacks": len(att), "trunk_counts": counts}


def check_verdicts(problems: list) -> dict:
    out = {}
    for lane in ("discovery", "transfer", "attack", "confirmation"):
        vs = _load(lane)
        out[lane] = {"n": len(vs), "states": {}, "criterion_status": {}}
        for cid, v in vs.items():
            if "_unreadable" in v:
                problems.append(f"{lane}:{cid}: unreadable verdict")
                continue
            st = v.get("state")
            cs = v.get("criterion_status")
            out[lane]["states"][st] = out[lane]["states"].get(st, 0) + 1
            out[lane]["criterion_status"][cs] = out[lane]["criterion_status"].get(cs, 0) + 1
            if st == "DONE":
                problems.append(f"{lane}:{cid}: DONE is not a state")
            if st not in STATES:
                problems.append(f"{lane}:{cid}: unknown state {st!r}")
            if cs not in CRITERION_STATUS:
                problems.append(f"{lane}:{cid}: unknown criterion status {cs!r}")
            if st == "LANDED" and cs == "UNEVALUATED":
                problems.append(f"{lane}:{cid}: LANDED with no criterion evaluated")
            # a gate bar is never a criterion bar
            for g in (v.get("gates") or {}).get("gates", []):
                if g.get("kind") in ("live", "no_oracle") and abs(float(
                        g.get("expected") or 0.0)) > 1e-12:
                    problems.append(f"{lane}:{cid}: gate {g.get('name')} carries a nonzero bar "
                                    f"({g.get('expected')}); a magnitude belongs in the criterion")
            cd = v.get("causal_distance") or {}
            if v.get("claim_class") == "SIMULATOR_DISCOVERY" and cd and not cd.get(
                    "promotable_as_discovery", True):
                problems.append(f"{lane}:{cid}: claims SIMULATOR_DISCOVERY but its causal distance "
                                f"is {cd.get('limiting_distance')}")
    return out


def check_vocabulary(problems: list) -> dict:
    """No human, clinical, neural or historical vocabulary anywhere in the record's prose."""
    hits = []
    roots = [verdict_dir(l) for l in ("discovery", "transfer", "attack", "confirmation")]
    roots.append(v15_dir())
    vdir = REPO / "docs" / "versions" / "v15-boundary-map"
    if vdir.exists():
        roots.append(vdir)
    for root in roots:
        for p in list(root.glob("*.json")) + list(root.glob("*.md")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for word in FORBIDDEN_VOCABULARY:
                if word in text:
                    hits.append({"file": p.name, "word": word})
    for h in hits:
        problems.append(f"forbidden vocabulary {h['word']!r} in {h['file']}")
    return {"hits": hits}


def check_stale_export(problems: list) -> dict:
    """An export whose source verdict hashes predate the closure of the cards it names."""
    out = {"checked": [], "stale": []}
    ledger = {}
    comp = v15_dir() / "COMPLETION.json"
    if comp.exists():
        ledger = json.loads(comp.read_text(encoding="utf-8")).get("entries", {})
    vdir = REPO / "docs" / "versions" / "v15-boundary-map"
    for p in (list(vdir.glob("*.md")) if vdir.exists() else []):
        text = p.read_text(encoding="utf-8", errors="ignore")
        out["checked"].append(p.name)
        for cid in re.findall(r"\b([ICMEGVSRFHPB]\d{2}|X\d{2})\b", text):
            if re.search(rf"{cid}\s*\(\s*UNRUN\s*\)", text, re.I):
                for key, e in ledger.items():
                    if key.endswith(f":{cid}") and e.get("state") in RESOLVED:
                        out["stale"].append({"file": p.name, "card": cid,
                                             "record_state": e.get("state")})
                        problems.append(f"{p.name}: names {cid} as UNRUN while the record has it "
                                        f"{e.get('state')}")
    return out


def check_runtime(problems: list) -> dict:
    out = {}
    for name in ("DEADLINE.json", "WORKER_OCCUPANCY.json", "DEADLINE_OPENING_RECEIPT.json"):
        p = v15_dir() / name
        out[name] = p.exists()
    occ = v15_dir() / "WORKER_OCCUPANCY.json"
    if occ.exists():
        d = json.loads(occ.read_text(encoding="utf-8"))
        out["occupancy_ratio"] = d.get("occupancy_ratio")
        out["RUNTIME_FAILED"] = d.get("RUNTIME_FAILED")
        out["runtime_failed_reasons"] = d.get("runtime_failed_reasons")
        if d.get("RUNTIME_FAILED"):
            # not a validator problem: spec §9.4 allows the results to be reported. It is a
            # problem only if something claims the seven-day contract anyway.
            out["note"] = ("RUNTIME_FAILED is set; results may be reported but the seven-day "
                           "contract may not be claimed")
    return out


def check_lineages(problems: list) -> dict:
    lin = M.lineages()
    if not lin["disjoint"]:
        problems.append("lane id ranges are not disjoint")
    return lin


def check_locks(problems: list) -> dict:
    try:
        from ghostscale.prereg_v15 import lock_status
        ls = lock_status()
    except Exception as exc:                                       # noqa: BLE001
        problems.append(f"lock status unreadable: {exc!r}")
        return {"error": repr(exc)}
    if ls.get("structural_locked") is False:
        problems.append(f"structural lock broken: changed {ls.get('changed')}")
    return ls


def run(quiet: bool = False) -> dict:
    problems: list = []
    out = {"program": "v15", "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "manifest": check_manifest(problems),
           "verdicts": check_verdicts(problems),
           "vocabulary": check_vocabulary(problems),
           "stale_export": check_stale_export(problems),
           "runtime": check_runtime(problems),
           "lineages": check_lineages(problems),
           "locks": check_locks(problems)}
    out["problems"] = problems
    out["ok"] = not problems
    if not quiet:
        print(json.dumps(out, indent=2, default=str))
        print(f"\n{len(problems)} problem(s)")
        for p in problems[:40]:
            print(f"  - {p}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    r = run(a.quiet)
    sys.exit(0 if r["ok"] else 1)
