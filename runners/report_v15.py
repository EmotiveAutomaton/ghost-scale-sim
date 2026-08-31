"""The single V15 curator packet, generated only after the deadline (spec §9.1, §12).

Spec §9.1 is absolute: no result prose, HTML, Markdown summary, bridge packet or curator-facing
chart may be emitted before hour 168. This module refuses to write anything into ``docs/`` until
``runtime_contract.window_closed()`` is true, and ``--draft`` writes to a scratch directory only.
That guard exists because a checkpoint, a dashboard or a bridge file is exactly how an early packet
gets created without anyone deciding to create one.

Two passes, in the spec's order: Pass A is read first and ends with a literal STOP READING HERE;
Pass B is the analyst appendix and carries every failed criterion, every conditional map, the
budgets, the equivalence classes, the attacks, the runtime receipt and the hashes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import common as C          # noqa: E402
from ghostscale.validation.soundingline.v15 import manifest as M        # noqa: E402
from ghostscale.validation.soundingline.v15 import runtime_contract as RC  # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402

VERSION_DIR = REPO / "docs" / "versions" / "v15-boundary-map"
SCRATCH = v15_dir("draft")


def _verdicts(lane: str) -> dict:
    out = {}
    for p in sorted(verdict_dir(lane).glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if abs(x) < 1000 else f"{x:.3g}"
    return str(x)


def build(pass_b: bool = True) -> str:
    disc = _verdicts("discovery")
    tra = _verdicts("transfer")
    atk = _verdicts("attack")
    conf = _verdicts("confirmation")
    occ = {}
    p = v15_dir() / "WORKER_OCCUPANCY.json"
    if p.exists():
        occ = json.loads(p.read_text(encoding="utf-8"))
    win = RC.window() or {}
    cov = {}
    p = v15_dir() / "COVERAGE.json"
    if p.exists():
        cov = json.loads(p.read_text(encoding="utf-8"))

    held = [c for c, v in disc.items() if v.get("criterion_status") == "HELD"]
    failed = [c for c, v in disc.items() if v.get("criterion_status") == "FAILED"]
    L = []
    A = L.append
    A("# V15 — The Boundary Map. Curator packet")
    A("")
    A(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} from the committed record, after the "
      f"168-hour window closed. Every number traces to a verdict under "
      f"`results/validation/soundingline/v15/`. Everything here is a property of a constructed "
      f"world and its stated reader. Nothing describes a person.*")
    A("")
    A("## Pass A")
    A("")
    A("### 1. Where the constructed project world moved")
    A("")
    A(f"{len(disc)} discovery cards resolved, of which **{len(held)} had their pre-registered "
      f"criterion hold and {len(failed)} had it fail**. `LANDED` and a held criterion are separate "
      f"columns throughout: a card that landed with a failed criterion produced a valid verdict "
      f"and a negative result, and both are reported.")
    A("")
    A("### 2. The coupling and access phase map")
    A("")
    A("| card | axis | onset | max advantage (nats) | criterion |")
    A("|---|---|---|---|---|")
    for cid in sorted(c for c in disc if c.startswith("C")):
        v = disc[cid]
        ph = v.get("phase") or {}
        A(f"| {cid} | {ph.get('axis','—')} | {_fmt(ph.get('onset'), 2)} | "
          f"{_fmt(ph.get('max'))} | {v.get('criterion_status','—')} |")
    A("")
    A("### 3. Which V14 positives remained identities")
    A("")
    for cid in ("C01", "S01", "S02", "F01", "H01", "E01", "G01"):
        v = disc.get(cid)
        if v:
            A(f"- **{cid}** ({v.get('claim_class')}): "
              f"{v.get('record', {}).get('what_happened', '—')}")
    A("")
    A("### 4. The strongest results and boundaries")
    A("")
    strong = sorted(
        [(cid, v) for cid, v in disc.items()
         if v.get("criterion_status") == "HELD"
         and (v.get("causal_distance") or {}).get("promotable_as_discovery")],
        key=lambda kv: kv[0])[:6]
    for cid, v in strong:
        A(f"- **{cid}** — {v.get('record', {}).get('what_happened', '—')}")
    if not strong:
        A("- none: no card both held its criterion and cleared the causal-distance audit.")
    A("")
    A("### 5. Prior assertions that gained, lost or stayed unmeasured")
    A("")
    A(f"- gained: {', '.join(held[:12]) if held else 'none'}")
    A(f"- lost: {', '.join(failed[:12]) if failed else 'none'}")
    A("")
    A("### 6. Rediscoveries, deltas and publication seeds")
    A("")
    A("| card | established component | project delta | grade | maturity |")
    A("|---|---|---|---|---|")
    for cid, v in sorted(disc.items()):
        pub = v.get("publication") or {}
        if pub.get("established_component"):
            A(f"| {cid} | {pub['established_component']} | {pub.get('project_specific_delta','—')} "
              f"| {pub.get('evidence_grade','—')} | {pub.get('maturity','—')} |")
    A("")
    A("### 7. What Sounding Line is licensed to import")
    A("")
    p07 = disc.get("P07") or {}
    counts = (p07.get("results") or {}).get("counts") or {}
    A(f"licensed {counts.get('license', 0)}, partial {counts.get('partial', 0)}, "
      f"deferred {counts.get('defer', 0)}, killed {counts.get('kill', 0)}.")
    A("")
    A("### 8. Recommendation")
    A("")
    b02 = disc.get("B02") or {}
    A(f"**{(b02.get('results') or {}).get('recommendation', '—')}** — "
      f"{(b02.get('results') or {}).get('in_spite_of', '')}")
    A("")
    A("### 9. Curator questions")
    A("")
    A("- none that change the next branch; the ledger decides it.")
    A("")
    A("> **STOP READING HERE**")
    A("")
    if not pass_b:
        return "\n".join(L)

    A("## Pass B — analyst appendix")
    A("")
    A("### Runtime receipt")
    A("")
    A(f"- window: {win.get('opened','—')} → {win.get('deadline','—')}")
    A(f"- occupancy: {_fmt(occ.get('occupancy_ratio'), 3)} against a "
      f"{occ.get('occupancy_target', 0.8)} target")
    A(f"- science worker-hours {_fmt(occ.get('science_worker_hours'), 1)}, capacity "
      f"{_fmt(occ.get('capacity_worker_hours'), 1)}")
    A(f"- coverage blocks {occ.get('coverage_blocks_executed', 0)}, cells "
      f"{occ.get('coverage_cells_executed', 0)}")
    A(f"- **RUNTIME_FAILED = {occ.get('RUNTIME_FAILED')}** "
      f"{occ.get('runtime_failed_reasons') or ''}")
    if occ.get("RUNTIME_FAILED"):
        A("")
        A("> The seven-day contract is **not** claimed. Spec §9.4 permits the results below to be "
          "reported and does not permit the contract to be called complete.")
    A("")
    A("### Every card")
    A("")
    A("| card | class | state | criterion | causal distance | failed criteria |")
    A("|---|---|---|---|---|---|")
    for cid, v in sorted(disc.items()):
        cd = (v.get("causal_distance") or {}).get("limiting_distance", "—")
        bad = ", ".join(c["name"] for c in (v.get("criteria") or []) if not c.get("held", True))
        A(f"| {cid} | {v.get('claim_class','—')} | {v.get('state','—')} | "
          f"{v.get('criterion_status','—')} | {cd} | {bad or '—'} |")
    A("")
    A("### Failed criteria, in full")
    A("")
    for cid, v in sorted(disc.items()):
        for c in (v.get("criteria") or []):
            if not c.get("held", True):
                A(f"- **{cid}/{c['name']}**: observed {_fmt(c.get('observed'))} against a bar of "
                  f"{_fmt(c.get('bar'))} ({c.get('direction')}); basis: {c.get('basis','—')}")
    A("")
    A("### Conditional maps (no pooled headlines)")
    A("")
    for cid, v in sorted(disc.items()):
        cm = v.get("conditional_matrix")
        if cm:
            A(f"- **{cid}**: {cm.get('axis_rows','—')} × {cm.get('axis_cols','—')} — "
              f"{cm.get('pooled_headline', '')}")
    A("")
    A("### Architecture budgets")
    A("")
    A("| card | compute matched | median likelihood evaluations |")
    A("|---|---|---|")
    for cid, v in sorted(disc.items()):
        b = v.get("budgets") or {}
        if b:
            A(f"| {cid} | {b.get('compute_matched')} | "
              f"{_fmt(b.get('median_likelihood_evaluations'), 0)} |")
    A("")
    A("### Transfer, attacks and confirmation")
    A("")
    A(f"- transfer: {len(tra)} cards, "
      f"{sum(1 for v in tra.values() if v.get('criterion_status') == 'HELD')} held")
    A(f"- attacks: {len(atk)} run, "
      f"{sum(1 for v in atk.values() if v.get('criterion_status') == 'HELD')} held")
    A(f"- confirmation: {len(conf)} run, "
      f"{sum(1 for v in conf.values() if v.get('criterion_status') == 'HELD')} held")
    A("")
    A("### Coverage")
    A("")
    A(f"- {json.dumps(cov.get('by_state', {}))}")
    A("")
    A("### Hashes")
    A("")
    for name in ("prereg_v15_structural_lock.json", "WORKLOAD_LOCK.json", "prereg_v15_lock.json",
                 "BALANCED_COVERAGE_SEQUENCE.json"):
        f = v15_dir() / name
        if f.exists():
            A(f"- `{name}`: `{C.file_sha(f)[:16]}`")
    A("")
    return "\n".join(L)


def run(force: bool = False, draft: bool = False) -> dict:
    closed = RC.window_closed()
    if not closed and not force and not draft:
        msg = (f"REFUSED: the window is still open ({RC.elapsed_hours():.2f} h of "
               f"{RC.WINDOW_HOURS}). Spec 9.1 permits no packet before the deadline.")
        print(msg)
        return {"written": None, "refused": True, "reason": msg}
    text = build(pass_b=True)
    if draft or (not closed and force):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        out = SCRATCH / "RESULTS_PACKET_DRAFT.md"
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"draft (not a curator packet) -> {out}")
        return {"written": str(out), "draft": True}
    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    out = VERSION_DIR / "RESULTS_PACKET.md"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return {"written": str(out), "draft": False}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--draft", action="store_true",
                    help="write to a scratch directory; never a curator packet")
    a = ap.parse_args()
    run(a.force, a.draft)
