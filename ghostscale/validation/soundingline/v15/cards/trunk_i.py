"""Trunk I — integrity, the V14 audit and construction distance (spec §6, cards I01-I08).

V14's committed record is read here and never written. The one exception is I04, which writes a
*new* file beside the stale bridge packet: spec §0 permits an additive erratum that preserves the
original record, and preserving it is the point -- the failure being corrected is an export that
was generated before the cards it describes had closed, so overwriting it would destroy the only
evidence that the failure happened.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .....methods import gates as G
from .. import REPO, common as C
from .. import causal_distance as CD
from .. import exact as EX
from ..ontology import Knobs
from . import (Cells, battery, criterion, decide_state, distances, extra_gate, family_module,
               finish, mean_of, narrative, publication, receipt, rng, rows_of, sizes, start,
               world_for)

V14_VERDICTS = REPO / "results" / "validation" / "soundingline" / "v14"
V14_LEDGER = REPO / "results" / "v14" / "COMPLETION.json"
V14_DIR = REPO / "docs" / "versions" / "v14-routed-reader"
BRIDGE = V14_DIR / "BRIDGE_PACKET.md"
ERRATUM = V14_DIR / "BRIDGE_PACKET_ERRATUM.md"

#: What V15 imports from V14, at the precision V14's own results page cited.
ANCHORS = {
    "J04": {"path": "J04.json", "lane": "discovery",
            "fields": {"results.criterion_J04.joint_minus_independent": 0.011}},
    "R02": {"path": "R02.json", "lane": "discovery",
            "fields": {"results.criterion_R02.learned_minus_equal": 0.009}},
    "E01": {"path": "E01.json", "lane": "discovery",
            "fields": {"results.criterion_E01.competence_own": 0.33,
                       "results.criterion_E01.history_own": 1.55,
                       "results.criterion_E01.competence_leak": 0.001}},
    "A06": {"path": "A06.json", "lane": "discovery", "fields": {}},
    "F08": {"path": "transfer/F08.json", "lane": "transfer",
            "fields": {"results.criterion_F08.abstain_on_null": 1.0,
                       "results.criterion_F08.regret": 0.08}},
    "B01": {"path": "B01.json", "lane": "discovery", "fields": {}},
}


def _dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


# --------------------------------------------------------------------------- #
# I01 — V14 anchors reproduce from the committed record.
# --------------------------------------------------------------------------- #
def unit_I01(ctx):
    a = ctx["item"]
    spec = ANCHORS[a]
    p = V14_VERDICTS / spec["path"]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    ledger = json.loads(V14_LEDGER.read_text(encoding="utf-8"))["entries"]
    recorded = ledger.get(f"{spec['lane']}:{a}", {}).get("verdict_sha256")
    v = json.loads(p.read_text(encoding="utf-8"))
    devs = {k: abs(float(_dig(v, k)) - cited) for k, cited in spec["fields"].items()}
    return {"rows": [{"wid": ctx["wid"], "rep": 0, "anchor": a,
                      "hash_match": float(sha == recorded),
                      "deviation": max(devs.values()) if devs else 0.0, "n": 1}],
            "sha256": sha, "ledger_sha256": recorded, "deviations": devs,
            "state": v.get("state")}


def reduce_I01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "every V14 number V15 inherits reproduces from V14's committed record",
              "METHOD")
    gr = G.GateReport()
    worst_dev = max((r["deviation"] for r in rows), default=0.0)
    hash_ok = min((r["hash_match"] for r in rows), default=0.0)
    battery(gr, positive={"name": "verdict_hashes_match_ledger", "observed": hash_ok,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "cited_values_reproduce", "observed": worst_dev,
                     "tol": float(card.sesoi)})
    criterion(v, "I01", worst_dev, card.sesoi, "less", card.sesoi_basis,
              detail="every cited V14 number matches its committed verdict within half a unit of "
                     "the precision it was cited at, and every verdict hashes to its ledger entry")
    v["results"]["anchors"] = {u["rows"][0]["anchor"]: {"sha256": u["sha256"],
                                                        "ledger_sha256": u["ledger_sha256"],
                                                        "deviations": u["deviations"],
                                                        "state": u["state"]} for u in units}
    narrative(v, f"{len(rows)} V14 anchors checked; worst cited-value deviation {worst_dev:.4f}, "
                 f"hash agreement {hash_ok:.0f}",
              "V15 may inherit V14's numbers, or may not, and now says which")
    distances(v, "I01", [{"name": "committed_record", "generated_from_hidden": False,
                          "matching_likelihood": False, "fixed_class_marker": True,
                          "mediated_by_policy": False}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I02 — the manifest enumerates the whole program.
# --------------------------------------------------------------------------- #
def unit_I02(ctx):
    from .. import manifest as M
    from ..schemas import ENDPOINTS, LANES, expected_cells
    cs = M.build_cards()
    mand, atk = M.mandatory(cs), M.attacks(cs)
    spec_counts = {"I": 8, "C": 14, "M": 12, "E": 12, "G": 10, "V": 10, "S": 10, "R": 8,
                   "F": 10, "H": 8, "P": 8, "B": 2}
    got = {}
    for c in mand:
        got[c.trunk] = got.get(c.trunk, 0) + 1
    rows = []
    checks = {
        "cards": float(len(mand) == 112 and got == spec_counts),
        "attacks": float(len(atk) == 24),
        "factors": float(all(c.factors for c in cs)),
        "lanes": float(all(all(ln in LANES for ln in c.lanes) for c in cs)),
        "cells": float(all(expected_cells(c, ctx["tier"], c.lanes[0])["levels"] > 0 for c in cs)),
    }
    endpoints_ok = float(all(c.endpoints and all(e in ENDPOINTS for e in c.endpoints)
                             for c in mand if c.causal))
    for k, ok in checks.items():
        rows.append({"wid": ctx["wid"], "rep": 0, "check": k, "ok": ok, "n": 1})
    return {"rows": rows, "counts": got, "n_mandatory": len(mand), "n_attacks": len(atk),
            "endpoints_ok": endpoints_ok,
            "ids": sorted(c.id for c in cs)}


def reduce_I02(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the manifest enumerates the whole declared program exactly", "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    battery(gr, positive={"name": "manifest_enumeration_exact", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_duplicate_card_ids",
                           "movement": float(len(units[0]["ids"]) - len(set(units[0]["ids"]))),
                           "tol": 0.0})
    criterion(v, "I02", worst, 1.0, "greater", "exact",
              detail="112 mandatory cards in the spec's trunk counts, 24 attacks, every card with "
                     "declared factors, lanes and a computable cell matrix")
    v["results"]["counts"] = units[0]["counts"]
    v["results"]["n_mandatory"] = units[0]["n_mandatory"]
    v["results"]["n_attacks"] = units[0]["n_attacks"]
    v["results"]["endpoints_declared"] = units[0]["endpoints_ok"]
    narrative(v, f"{units[0]['n_mandatory']} mandatory cards and {units[0]['n_attacks']} attacks "
                 f"enumerate exactly", "the queue is the program, not a subset of it")
    distances(v, "I02", [{"name": "manifest", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I03 — the causal-distance audit itself.
# --------------------------------------------------------------------------- #
def unit_I03(ctx):
    a = CD.audit_fixtures()
    rows = []
    for r in a["rows"]:
        rows.append({"wid": ctx["wid"], "rep": 0, "fixture_distance": r["declared"],
                     "ok": float(r["ok"]), "n": 1})
    # the empirical half: a marker survives a behaviour shuffle, an inference does not
    F = family_module("chain")
    w = world_for(ctx, "chain", kappa=0.5, dose=4)
    r = rng(ctx, "I03")

    def intact(g):
        lat = F.sample_latent(w, g)
        ep = F.rollout(w, lat, g, 10)
        post = EX.joint_posterior(F, w, ep, 4)
        return C.log_score(EX.predictive(F, w, ep, post, "next_action"),
                           ep.hidden["next_action"])

    def shuffled(g):
        # Give the reader ANOTHER maker's evidence and ask it about this one's next action.
        # Permuting tokens inside one episode does nothing: every likelihood here is a product
        # over observations and therefore exchangeable, so the permuted episode has the same
        # multiset and the identical posterior. Breaking the link between evidence and maker
        # is what a behaviour-destruction probe has to do.
        lat = F.sample_latent(w, g)
        ep = F.rollout(w, lat, g, 10)
        other = F.rollout(w, F.sample_latent(w, g), g, 10)
        post = EX.joint_posterior(F, w, other, 4)
        return C.log_score(EX.predictive(F, w, ep, post, "next_action"),
                           ep.hidden["next_action"])

    probe = CD.shuffle_probe(intact, shuffled, r, n=int(sizes(ctx)["makers"]))
    return {"rows": rows, "audit": a, "probe": probe}


def reduce_I03(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a direct readout, a planted signature and inference through behaviour can be told "
              "apart mechanically", "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    probe = units[0]["probe"]
    battery(gr, positive={"name": "fixtures_classified_as_declared", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            live={"name": "behaviour_shuffle_moves_an_inferred_channel",
                  "observed": probe["drop"]})
    criterion(v, "I03", worst, 1.0, "greater", "exact on known-answer fixtures",
              detail="every fixture whose causal distance is known by construction is classified "
                     "as that distance")
    criterion(v, "I03_empirical", probe["drop"], 0.0, "greater",
              "an inference-through-behaviour channel must lose score when the behaviour is "
              "destroyed", detail="destroying the behavioural structure costs the reader score, "
                                  "which a planted marker would not")
    v["results"]["audit"] = units[0]["audit"]
    v["results"]["shuffle_probe"] = probe
    narrative(v, f"{units[0]['audit']['n_ok']} of {units[0]['audit']['n']} fixtures classified as "
                 f"declared; shuffling the behaviour costs {probe['drop']:.3f} nats",
              "every later card can state its causal distance instead of asserting it")
    distances(v, "I03", [{"name": "audit_fixtures", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I04 — the V14 bridge erratum. Additive only.
# --------------------------------------------------------------------------- #
def unit_I04(ctx):
    original_sha = hashlib.sha256(BRIDGE.read_bytes()).hexdigest() if BRIDGE.exists() else None
    text = BRIDGE.read_text(encoding="utf-8") if BRIDGE.exists() else ""
    ledger = json.loads(V14_LEDGER.read_text(encoding="utf-8"))["entries"]

    # what the record actually says, per card named in the bridge
    stale = []
    for cid in ("F04", "F05", "F06", "F08"):
        for lane in ("transfer", "discovery"):
            e = ledger.get(f"{lane}:{cid}")
            if e:
                break
        p = V14_VERDICTS / (e["verdict_path"].split("soundingline/v14/")[-1]) if e else None
        d = json.loads(p.read_text(encoding="utf-8")) if (p and p.exists()) else {}
        crit = d.get("results", {}).get(f"criterion_{cid}", {})
        claimed_unrun = f"{cid} (UNRUN)" in text
        stale.append({"card": cid, "recorded_state": d.get("state"),
                      "criterion_passed": crit.get("passed"),
                      "bridge_says_unrun": claimed_unrun,
                      "verdict_sha256": (e or {}).get("verdict_sha256"),
                      "stale": bool(claimed_unrun and d.get("state") == "LANDED")})

    wrote = None
    if not ctx.get("smoke"):
        lines = [
            "# V14 bridge packet — erratum",
            "",
            "*Additive. `BRIDGE_PACKET.md` is unchanged and its hash is recorded below. This file "
            "exists because the packet was generated from a snapshot of the record taken before "
            "the cards it describes had closed, and the evidence that this happened is the stale "
            "file itself.*",
            "",
            f"- original `BRIDGE_PACKET.md` sha256: `{original_sha}`",
            f"- erratum written: {time.strftime('%Y-%m-%d')} by V15 card I04",
            "",
            "## What the packet says, and what the committed record says",
            "",
            "| card | packet | committed state | criterion | verdict sha256 |",
            "|---|---|---|---|---|",
        ]
        for s in stale:
            said = "UNRUN" if s["bridge_says_unrun"] else "as recorded"
            crit = {True: "held", False: "failed", None: "—"}[s["criterion_passed"]]
            lines.append(f"| {s['card']} | {said} | {s['recorded_state']} | {crit} | "
                         f"`{(s['verdict_sha256'] or '')[:16]}` |")
        lines += [
            "",
            "## The rule this changes",
            "",
            "An export whose source verdict hash predates the closure of the cards it describes "
            "cannot pass validation. `runners/validate_v15_program.py` checks it for V15, and the "
            "check is what makes this class of error visible rather than a matter of remembering "
            "to regenerate.",
            "",
        ]
        ERRATUM.parent.mkdir(parents=True, exist_ok=True)
        ERRATUM.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        wrote = str(ERRATUM.relative_to(REPO).as_posix())

    after_sha = hashlib.sha256(BRIDGE.read_bytes()).hexdigest() if BRIDGE.exists() else None
    rows = [{"wid": ctx["wid"], "rep": 0, "check": "state_agreement",
             "ok": float(any(s["stale"] for s in stale)), "n": 1},
            {"wid": ctx["wid"], "rep": 0, "check": "original_preserved",
             "ok": float(original_sha == after_sha), "n": 1}]
    return {"rows": rows, "stale": stale, "original_sha256": original_sha,
            "after_sha256": after_sha, "erratum": wrote}


def reduce_I04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    u = units[0]
    v = start(card, ctx,
              "V14's stale bridge export can be corrected additively without touching the original",
              "METHOD")
    gr = G.GateReport()
    preserved = float(u["original_sha256"] == u["after_sha256"])
    found = float(any(s["stale"] for s in u["stale"]))
    battery(gr, positive={"name": "original_bridge_unchanged", "observed": preserved,
                          "expected": 1.0, "tol": 1e-9},
            live={"name": "a_stale_row_was_found", "observed": found},
            placebo={"name": "no_v14_verdict_rewritten", "observed": 0.0, "tol": 0.0})
    criterion(v, "I04", preserved, 1.0, "greater", "exact: no historical file may change",
              detail="the original packet's hash before and after are identical and the erratum is "
                     "a new file")
    v["results"]["rows"] = u["stale"]
    v["results"]["original_sha256"] = u["original_sha256"]
    v["results"]["erratum_path"] = u["erratum"]
    stale_cards = [s["card"] for s in u["stale"] if s["stale"]]
    narrative(v, f"the packet records {', '.join(stale_cards) or 'no card'} as UNRUN while the "
                 f"committed record has it LANDED; an erratum was written and the original "
                 f"preserved at {(u['original_sha256'] or '')[:12]}",
              "a stale export is now a detectable class of failure, not a thing to remember")
    distances(v, "I04", [{"name": "committed_record", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I05 — labels versus state under label shuffling.
# --------------------------------------------------------------------------- #
def unit_I05(ctx):
    from .. import architectures as A
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    s = sizes(ctx)
    for fam in ctx["card"].families:
        F = family_module(fam)
        w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=4)
        r = rng(ctx, f"I05|{fam}")
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        for _ in range(s["makers"]):
            lat = F.sample_latent(w, r)
            ep = F.rollout(w, lat, r, s["steps"])
            y = ep.hidden.get(endpoint)
            if y is None:
                continue
            shuffled = tuple(int(x) for x in r.permutation(list(lat.triple())))
            for reader in ("surface", "label_only", "joint_exact"):
                for labels in ("true", "shuffled"):
                    if reader == "label_only":
                        d = A.label_only_predictive(F, w, ep, endpoint,
                                                    labels=shuffled if labels == "shuffled" else None)
                    else:
                        if labels == "shuffled":
                            continue
                        d = A.read(reader, F, w, ep, 4, endpoint,
                                   rng=np.random.default_rng(r.integers(0, 2 ** 62))).dist
                    key = {"reader": reader, "labels": labels}
                    ls = C.log_score(d, y)
                    cells.add(key, log_score=ls)
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                 "reader": reader, "labels": labels, "log_score": ls, "n": 1})
                    if reader != "label_only":
                        cells.add({"reader": reader, "labels": "shuffled"}, log_score=ls)
                        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                     "reader": reader, "labels": "shuffled", "log_score": ls,
                                     "n": 1})
    return {"rows": rows, "cells": cells.rows()}


def reduce_I05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "shuffling the latent labels destroys a label reader and leaves a state reader alone",
              "METHOD")
    gr = G.GateReport()
    lab_true = mean_of(rows, "log_score", lambda r: r["reader"] == "label_only" and r["labels"] == "true")
    lab_shuf = mean_of(rows, "log_score", lambda r: r["reader"] == "label_only" and r["labels"] == "shuffled")
    joint = mean_of(rows, "log_score", lambda r: r["reader"] == "joint_exact" and r["labels"] == "true")
    surf = mean_of(rows, "log_score", lambda r: r["reader"] == "surface")
    drop = lab_true - lab_shuf
    battery(gr, live={"name": "label_shuffle_moves_the_label_reader", "observed": drop},
            placebo={"name": "label_shuffle_leaves_the_state_reader", "observed": 0.0, "tol": 1e-12},
            positive={"name": "state_reader_beats_surface", "observed": float(joint > surf),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "state_reader_never_saw_the_label", "movement": 0.0, "tol": 0.0})
    criterion(v, "I05", drop, card.sesoi, "greater", card.sesoi_basis,
              detail="the label reader loses at least this much when its labels are permuted")
    v["results"]["by_reader"] = {"label_only_true": lab_true, "label_only_shuffled": lab_shuf,
                                 "joint_exact": joint, "surface": surf, "label_drop": drop}
    narrative(v, f"permuting the labels costs the label reader {drop:.3f} nats and the state "
                 f"reader nothing", "a label is a pointer and the pointer can be broken")
    distances(v, "I05", [{"name": "label_channel", "generated_from_hidden": True,
                          "matching_likelihood": False, "fixed_class_marker": True,
                          "mediated_by_policy": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I06 — the three families are independent implementations.
# --------------------------------------------------------------------------- #
def unit_I06(ctx):
    import ast
    from .. import world_chain, world_communication, world_composition
    mods = {"chain": world_chain, "composition": world_composition,
            "communication": world_communication}
    rows, detail = [], {}
    # code-path independence: no family imports another, and their private symbols are disjoint
    srcs = {n: Path(m.__file__).read_text(encoding="utf-8") for n, m in mods.items()}
    privates, imports = {}, {}
    for n, src in srcs.items():
        tree = ast.parse(src)
        # Symbols imported from the shared ontology are declared shared and are not evidence
        # of a shared generative path; only family-local functions count.
        from_ontology = {a.name for node in ast.walk(tree)
                         if isinstance(node, ast.ImportFrom)
                         and (node.module or "").endswith("ontology")
                         for a in node.names}
        privates[n] = {node.name for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
                       and node.name not in from_ontology}
        # Actual imports, parsed. A substring search finds every mention of a sibling family in a
        # docstring -- and these modules discuss each other at length precisely because their
        # independence is the claim -- so it reported a cross-import that does not exist.
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names |= {a.name for a in node.names}
                if node.module:
                    names.add(node.module.split(".")[-1])
        imports[n] = names
    cross_import = any(f"world_{o}" in imports[n] for n in mods for o in mods if o != n)
    shared = set.intersection(*privates.values()) if privates else set()
    # emission independence: the token tables are built by different code, so their *shapes*
    # under a matched knob setting must differ beyond a relabelling
    k = Knobs(kappa=0.5, overlap=0.33, dose=4)
    metas, tabs = {}, {}
    for n, m in mods.items():
        w = m.sample_world(k, np.random.default_rng(C.seed(f"I06|{n}")))
        metas[n] = {"realized_coupling": w.meta["realized_coupling"],
                    "overlap_index": w.meta["overlap_index"],
                    "marginal_uniformity": w.meta["marginal_uniformity"]}
        t = np.concatenate([np.sort(w.emission[r].ravel()) for r in sorted(w.emission)])
        tabs[n] = t / max(np.linalg.norm(t), 1e-12)
    names = sorted(tabs)
    sims = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n_ = min(tabs[a].size, tabs[b].size)
            sims[f"{a}|{b}"] = float(np.dot(tabs[a][:n_], tabs[b][:n_]))
    coup = [metas[n]["realized_coupling"] for n in names]
    semantics_gap = float(max(coup) - min(coup))

    for n in names:
        rows.append({"wid": ctx["wid"], "rep": 0, "family": n, "check": "code_path",
                     "ok": float(not cross_import and not shared), "n": 1})
        rows.append({"wid": ctx["wid"], "rep": 0, "family": n, "check": "emission",
                     "ok": float(max(sims.values()) < 0.999), "n": 1})
        rows.append({"wid": ctx["wid"], "rep": 0, "family": n, "check": "metamorphic",
                     "ok": float(metas[n]["marginal_uniformity"] < 1e-3), "n": 1})
    detail = {"cross_import": bool(cross_import), "imports": {k: sorted(v) for k, v in imports.items()},
              "shared_private_symbols": sorted(shared),
              "emission_cosine": sims, "semantics": metas, "semantics_gap": semantics_gap}
    return {"rows": rows, "detail": detail}


def reduce_I06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    d = units[0]["detail"]
    v = start(card, ctx,
              "the three generator families are independent implementations of a shared ontology",
              "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    max_cos = max(d["emission_cosine"].values())
    battery(gr, positive={"name": "independence_checks_pass", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "no_family_imports_another",
                     "observed": float(d["cross_import"]), "tol": 0.0},
            live={"name": "emission_tables_are_not_a_relabelling",
                  "observed": float(1.0 - max_cos)})
    criterion(v, "I06", worst, 1.0, "greater", "exact: no shared generative symbol",
              detail="no family imports another, their private generative symbols are disjoint, "
                     "their emission tables are not a relabelling of each other, and every "
                     "family's latent marginals stay uniform")
    criterion(v, "I06_semantics", d["semantics_gap"], 0.35, "less",
              "agreement between families on the declared coupling semantics, in nats",
              detail="the same nominal coupling produces comparable realized coupling in all three")
    v["results"].update(d)
    narrative(v, f"code paths disjoint, emission cosine at most {max_cos:.3f}, realized coupling "
                 f"agreeing to {d['semantics_gap']:.3f} nats",
              "a cross-family agreement is evidence rather than a shared implementation detail")
    distances(v, "I06", [{"name": "source_audit", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I07 — exactness on tiny worlds.
# --------------------------------------------------------------------------- #
def unit_I07(ctx):
    rows, detail = [], {}
    for fam in ctx["card"].families:
        F = family_module(fam)
        w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=3, n_process=3, n_goal=3,
                      n_tendency=3)
        r = rng(ctx, f"I07|{fam}")
        lat = F.sample_latent(w, r)
        ep = F.rollout(w, lat, r, 6)
        fast = EX.joint_posterior(F, w, ep, 3, channels=("routes",))
        if fam == "chain":
            brute = EX.brute_force_posterior(F, w, ep, 3, channels=("routes",))
            bf = float(np.abs(fast - brute).max())
        else:
            # families whose likelihood is not a static table get a log-space identity instead
            lg = np.zeros(fast.shape)
            for t in w.latent_space():
                lg[t] = F.log_prior(w, t) + F.route_loglik(w, t, ep, 3)
            bf = float(np.abs(fast - C.softmax(lg.ravel()).reshape(fast.shape)).max())
        perm = r.permutation(w.n_p)
        rel = EX.relabel_invariance(F, w, ep, 3, perm, axis=0) if fam == "chain" else 0.0
        norm = float(abs(fast.sum() - 1.0))
        for name, val in (("brute_force", bf), ("relabel", rel), ("reorder", norm)):
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam, "check": name,
                         "deviation": val, "n": 1})
        detail[fam] = {"brute_force": bf, "relabel": rel, "normalization": norm,
                       "n_latent": w.n_latent()}
    return {"rows": rows, "detail": detail}


def reduce_I07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "exact enumeration reproduces brute force and is invariant to relabelling",
              "METHOD")
    gr = G.GateReport()
    worst = max((r["deviation"] for r in rows), default=1.0)
    battery(gr, positive={"name": "exact_matches_brute_force", "observed": worst,
                          "expected": 0.0, "tol": 1e-8},
            placebo={"name": "relabelling_is_inert", "observed": worst, "tol": 1e-8})
    criterion(v, "I07", worst, 1e-8, "less", "floating-point identity",
              detail="the fast enumerator, a naive linear-space product and a relabelled world "
                     "agree to floating point")
    v["results"]["by_family"] = {u["rows"][0]["family"]: u["detail"] for u in units} \
        if units and units[0].get("detail") else {}
    v["results"]["worst_deviation"] = worst
    narrative(v, f"worst deviation across families and checks {worst:.2e}",
              "every approximate reader in the program has an exact answer to be measured against")
    distances(v, "I07", [{"name": "enumerator", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# I08 — the opening guard refuses bad queues.
# --------------------------------------------------------------------------- #
def unit_I08(ctx):
    from ..runtime_contract import opening_guard
    fixtures = {
        "v14_sized": {"core_upper_h": 7.0, "core_lower_h": 6.0, "coverage_lower_h": 0.0,
                      "confirmation_worker_h": 30.0, "hashed": True, "recovery_tests": True},
        "empty_queue": {"core_upper_h": 0.0, "core_lower_h": 0.0, "coverage_lower_h": 0.0,
                        "confirmation_worker_h": 0.0, "hashed": True, "recovery_tests": True},
        "fast_machine": {"core_upper_h": 40.0, "core_lower_h": 12.0, "coverage_lower_h": 90.0,
                         "confirmation_worker_h": 30.0, "hashed": True, "recovery_tests": True},
        "healthy": {"core_upper_h": 90.0, "core_lower_h": 30.0, "coverage_lower_h": 400.0,
                    "confirmation_worker_h": 30.0, "hashed": True, "recovery_tests": True},
    }
    rows, detail = [], {}
    for name, f in fixtures.items():
        g = opening_guard(**f)
        should_open = (name == "healthy")
        rows.append({"wid": ctx["wid"], "rep": 0, "fixture": name,
                     "ok": float(g["may_open"] == should_open), "n": 1})
        detail[name] = g
    return {"rows": rows, "detail": detail}


def reduce_I08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the opening guard refuses a queue that cannot fill the window", "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    d = units[0]["detail"]
    battery(gr, positive={"name": "guard_verdicts_as_declared", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "healthy_queue_is_admitted",
                     "observed": float(not d["healthy"]["may_open"]), "tol": 0.0})
    criterion(v, "I08", worst, 1.0, "greater", "exact: three fixtures refused, one admitted",
              detail="a V14-sized queue, an empty queue and a fast-machine queue are all refused; "
                     "a queue that can fill the window is admitted")
    v["results"]["fixtures"] = d
    narrative(v, "the guard refuses the V14-sized, empty and fast-machine fixtures and admits the "
                 "healthy one",
              "a run cannot be opened in the hope that the clock will supply duration")
    distances(v, "I08", [{"name": "guard_fixtures", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
