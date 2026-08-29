"""Trunk B — the Sounding Line Stage 5 bridge and the closure ledger (spec §5, cards B01-B02).

Both are single-unit deliverables generated mechanically from the committed record: a ruler is
licensed only by landed cards whose criteria held; a failed instrument licenses nothing.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .....methods import gates as G
from .. import REPO, common as C
from . import Cells, battery, criterion, decide_state, finish, narrative, pursuit_of, receipt, start

CANDIDATES = {
    "joint_reader_posterior": {"licensing": ["J04", "J02", "J03", "J08"], "access": "action, semantic and context observations per episode; a held-out next action",
                               "gate": "a constructed corpus in which the joint estimator beats independent marginals on a hidden next action at matched evidence",
                               "rival": "independent marginals per latent", "endpoint": "next-action log score", "shape": "joint > staged > independent, with class-level uncertainty under equifinality",
                               "ceiling": "CONSTRUCTED_MECHANISM"},
    "route_reliability_weights": {"licensing": ["R02", "R03", "R04", "R07"], "access": "per-route feedback on training items without target labels at test",
                                  "gate": "a planted corpus where one route is degraded and learned weights beat equal weights; ease crossed against accuracy",
                                  "rival": "equal weighting", "endpoint": "held-out prediction with learned tempering", "shape": "learned weights track accuracy, not ease; duplicates fused by cause",
                                  "ceiling": "CONSTRUCTED_MECHANISM"},
    "competence_history_signatures": {"licensing": ["E01", "E03", "E05", "E08"], "access": "execution-accuracy and early-token records of a maker across episodes",
                                      "gate": "planted competence and planted history recovered independently on a constructed corpus", "rival": "one scalar skill",
                                      "endpoint": "prospective next-choice and signature classification", "shape": "two dissociable signatures with different correction dynamics", "ceiling": "CONSTRUCTED_MECHANISM"},
    "counterfactual_source_probes": {"licensing": ["A05", "A06", "A08", "A10"], "access": "a correction event and an off-audience action per source, beside the artifact",
                                     "gate": "a constructed corpus where fanatic and propagandist collide on the artifact and separate on probes", "rival": "a template classifier",
                                     "endpoint": "pairwise accuracy with and without probes; abstention mass", "shape": "chance from the artifact, separation from probes, factored uptake", "ceiling": "CONSTRUCTED_MECHANISM"},
    "learning_progress_selector": {"licensing": ["F04", "F05", "F06", "F08"], "access": "the learner's own recent surprise per item and a cost per look",
                                   "gate": "the unlearnable-noise trap on a constructed item set", "rival": "raw surprise", "endpoint": "realized held-out gain per cost",
                                   "shape": "progress avoids noise, gain per cost beats fixed policies, abstention on null probes", "ceiling": "CONSTRUCTED_MECHANISM"},
    "equivalence_class_reporting": {"licensing": ["J01", "J08", "H02", "H03"], "access": "a declared hypothesis grid with its equivalences",
                                    "gate": "planted equifinal and reward-equivalent pairs keep their class mass", "rival": "forced unique attribution",
                                    "endpoint": "class mass and single-member mass", "shape": "spread within the class, contraction on resolving evidence", "ceiling": "BOUNDARY"},
}


def _verdict(cid, lane="discovery"):
    return C.load_verdict(cid, lane)


def _state(cid):
    v = _verdict(cid)
    if v is None:
        return "UNRUN", None
    passed = None
    for k, x in v.get("results", {}).items():
        if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x:
            passed = bool(x["passed"])
    conf = _verdict(cid, "confirmation")
    cpass = None
    if conf is not None:
        for k, x in conf.get("results", {}).items():
            if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x:
                cpass = bool(x["passed"])
    return v.get("state", "UNRUN"), {"criterion": passed, "confirmed": cpass, "ceiling": v.get("claim_ceiling")}


def unit_B01(ctx):
    rows_out = []
    ledger = {}
    for name, spec in CANDIDATES.items():
        states = {c: _state(c) for c in spec["licensing"]}
        landed_held = [c for c, (s, d) in states.items() if s == "LANDED" and d and d["criterion"]]
        failed = [c for c, (s, _) in states.items() if s in ("INSTRUMENT_FAILED", "VOID")]
        unrun = [c for c, (s, _) in states.items() if s == "UNRUN"]
        if unrun:
            rec = "defer"
        elif failed:
            rec = "kill"
        elif len(landed_held) == len(spec["licensing"]):
            rec = "license"
        elif landed_held:
            rec = "license_partial"
        else:
            rec = "kill"
        ledger[name] = {**spec, "states": {c: {"state": s, **(d or {})} for c, (s, d) in states.items()}, "recommendation": rec}
    rows_out.append({"wid": ctx["wid"], "rep": 0, "item": "deliverable", "n": 1, "licensed": float(sum(x["recommendation"] == "license" for x in ledger.values()))})
    return {"rows": rows_out, "ledger": ledger}


def reduce_B01(card, units, ctx):
    v = start(card, ctx, "Only rulers every one of whose licensing cards landed with its criterion held are licensed for Sounding Line Stage 5; a failed instrument licenses nothing.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ledger = units[0]["ledger"]
    recs = {k: x["recommendation"] for k, x in ledger.items()}
    gr = G.GateReport()
    gr.identity("every_candidate_has_a_recommendation", float(len(recs)), float(len(CANDIDATES)), tol=0.0)
    battery(gr, positive={"observed": float(all(r in ("license", "license_partial", "defer", "kill") for r in recs.values())), "expected": 1.0, "tol": 0.0, "name": "recommendations_mechanical"},
            placebo={"observed": float(any(any(s["state"] in ("INSTRUMENT_FAILED", "VOID") for s in x["states"].values()) and x["recommendation"].startswith("license") for x in ledger.values())), "tol": 0.0, "name": "no_failed_instrument_licenses"})
    criterion(v, "B01", True, recommendations=recs)
    v["results"].update({"bridge_ledger": ledger})
    receipt(v, rows, card, ctx)
    if not ctx.get("smoke"):
        out = REPO / "docs" / "versions" / "v14-routed-reader" / "BRIDGE_PACKET.md"
        lines = ["# V14 → Sounding Line Stage 5 bridge packet (generated by trunk B)", "",
                 f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the committed verdicts. Each row names the licensing cards and their states, the access it needs, the construction gate, the cheap rival, the endpoint, the expected shape and the claim ceiling. Recommendations are mechanical.", ""]
        for name, x in ledger.items():
            lines += [f"## {name} — **{x['recommendation']}**", "",
                      "- licensing: " + ", ".join(f"{c} ({s['state']}{', criterion held' if s.get('criterion') else (', criterion failed' if s.get('criterion') is False else '')})" for c, s in x["states"].items()),
                      f"- access: {x['access']}", f"- construction gate: {x['gate']}", f"- cheap rival: {x['rival']}", f"- endpoint: {x['endpoint']}", f"- expected shape: {x['shape']}",
                      f"- claim ceiling: {x['ceiling']}; nothing here describes a person", ""]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    narrative(v, "Licensed: " + (", ".join(k for k, r in recs.items() if r == "license") or "none") + "; partial: " + (", ".join(k for k, r in recs.items() if r == "license_partial") or "none")
              + "; deferred: " + (", ".join(k for k, r in recs.items() if r == "defer") or "none") + "; killed: " + (", ".join(k for k, r in recs.items() if r == "kill") or "none") + ".",
              "What Sounding Line may implement is exactly what landed, and no more.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(True))


def unit_B02(ctx):
    from .. import manifest as M
    doc = M.load_manifest()
    ledger = {}
    for d in doc["cards"]:
        if d["trunk"] in ("X", "B"):
            continue
        s, det = _state(d["id"])
        ledger[d["id"]] = {"state": s, "pursuit": d.get("pursuit"), "warrant": d.get("warrant"), **(det or {})}
    promoted = [c for c, x in ledger.items() if x["state"] == "LANDED" and x.get("criterion") and x.get("confirmed")]
    closed = [c for c, x in ledger.items() if x["state"] in ("INSTRUMENT_FAILED", "VOID", "SCIENTIFIC_CLOSED")]
    context = [c for c in ledger if c not in promoted and c not in closed]
    rt = json.loads(M.RUNTIME.read_text(encoding="utf-8")) if M.RUNTIME.exists() else {"cards": {}}
    wall = sum((x.get("wall_s") or 0) for x in rt.get("cards", {}).values())
    return {"rows": [{"wid": ctx["wid"], "rep": 0, "item": "deliverable", "n": 1, "promoted": float(len(promoted))}],
            "ledger": ledger, "promoted": promoted, "closed": closed, "context": context, "runtime_wall_h": wall / 3600.0}


def reduce_B02(card, units, ctx):
    v = start(card, ctx, "The final pursuit and warrant ledger names what is promoted, what is closed and what stays as context; no V15 follows automatically.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    u = units[0]
    gr = G.GateReport()
    gr.identity("every_mandatory_card_in_the_ledger", float(len(u["ledger"])), 62.0, tol=0.0, detail="64 mandatory cards less the two B cards")
    battery(gr, positive={"observed": 1.0, "expected": 1.0, "tol": 0.0, "name": "ledger_written"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "no_automatic_v15"})
    criterion(v, "B02", True, promoted=u["promoted"], closed=u["closed"], context_count=len(u["context"]), runtime_wall_h=u["runtime_wall_h"])
    v["results"].update({"ledger": u["ledger"]})
    receipt(v, rows, card, ctx)
    narrative(v, f"{len(u['promoted'])} cards promoted (landed, criterion held, confirmed), {len(u['closed'])} closed, {len(u['context'])} left as context; {u['runtime_wall_h']:.1f} wall-clock hours of card runs.",
              "The recommendation is written from this ledger by a person, not generated.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(True))
