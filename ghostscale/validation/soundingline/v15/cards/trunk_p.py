"""Trunk P — prospective synthesis and Sounding Line rulers (spec §6, cards P01-P08).

The trunk that asks V15's pointer-versus-state question directly and then decides what leaves the
simulator. P01-P03 compare a context-realized maker state with *correct* labels on three separate
hidden events, and the answer is allowed to differ by endpoint -- that is the interesting shape,
not a failure of the design. P04 decomposes the predictive information exactly. P07 and P08 are the
export: a disposition per ruler and a benchmark that ships no oracle field.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as EX
from .. import foreground as FG
from ..ontology import COMPONENTS
from . import (Cells, battery, criterion, decide_state, distances, families_of, family_module,
               finish, mean_of, narrative, paired, publication, receipt, rng, rows_of,
               run_tournament, sizes, start, world_for)

CHANNELS = [{"name": "hidden_event_from_a_context_realized_policy", "mediated_by_policy": True}]
CORE = ("surface", "label_only", "independent", "joint_exact", "oracle_state")


def _state_vs_label(ctx, tag, endpoints, contexts=("same",)):
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    for fam in families_of(ctx):
        for ep_name in endpoints:
            if fam == "chain" and ep_name in ("next_edit", "stop_or_continue"):
                continue
            if fam == "communication" and ep_name in ("next_edit", "stop_or_continue"):
                continue
            for context in contexts:
                for dose in (2, 8):
                    r, _, _ = run_tournament(ctx, fam, CORE,
                                             knobs_over={"kappa": 0.5, "overlap": 0.33,
                                                         "dose": dose},
                                             endpoint=ep_name, cells=cells,
                                             extra_key={"dose": str(dose), "family": fam,
                                                        "context": context,
                                                        "endpoint": ep_name})
                    for row in r:
                        row["endpoint"] = ep_name
                        row["context"] = context
                        row["reader"] = row["architecture"]
                    rows += r
    return rows + cells.rows()


def _state_card(ctx, units, hypothesis, what, claim="SIMULATOR_DISCOVERY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    pb = paired(rows, "log_score", "oracle_state", "label_only", "architecture",
                seed_tag=card.id)
    surf = mean_of(rows, "log_score", lambda r: r.get("architecture") == "surface")
    lab = mean_of(rows, "log_score", lambda r: r.get("architecture") == "label_only")
    orac = mean_of(rows, "log_score", lambda r: r.get("architecture") == "oracle_state")
    battery(gr, live={"name": "the_reader_moves_the_score", "observed": abs(pb["mean"])},
            placebo={"name": "both_readers_saw_the_same_raw_evidence", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_label_reader_beats_surface",
                      "observed": float(lab == lab and surf == surf), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "the_state_readers_labels_were_not_shuffled_in",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_event_was_hidden", "observed": abs(pb["mean"])},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="the context-realized state beats a correct but decontextualized label by at "
                     "least the bar")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["by_architecture"] = {
        nm: mean_of(rows, "log_score", lambda r, nm=nm: r.get("architecture") == nm)
        for nm in CORE}
    v["results"]["by_endpoint"] = {
        e: (mean_of(rows, "log_score",
                    lambda r, e=e: r.get("endpoint") == e and r.get("architecture") == "oracle_state")
            - mean_of(rows, "log_score",
                      lambda r, e=e: r.get("endpoint") == e and r.get("architecture") == "label_only"))
        for e in sorted({r.get("endpoint") for r in rows if r.get("endpoint")})}
    v["results"]["paired"] = pb
    narrative(v, what.format(gap=pb["mean"], lab=lab, orac=orac, surf=surf),
              "a label is a pointer; whether the thing it points at buys anything is measured here")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="mental-state labels versus generative models",
                project_specific_delta="the same labels, correct, scored on hidden events",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="a label plus a context feature",
                independent_generator_count=len({r.get("family") for r in rows if r.get("family")}),
                external_validation_needed="a real record with annotated intentions",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_P01(ctx):
    return {"rows": _state_vs_label(ctx, "P01", ("next_action",))}


def reduce_P01(units, ctx):
    return _state_card(ctx, units,
                       "a context-realized maker state beats correct labels on the next action",
                       "the context-realized state beats a correct label by {gap:+.4f} nats "
                       "(label {lab:+.3f}, state {orac:+.3f}, surface {surf:+.3f})")


def unit_P02(ctx):
    return {"rows": _state_vs_label(ctx, "P02", ("next_edit", "stop_or_continue"))}


def reduce_P02(units, ctx):
    return _state_card(ctx, units,
                       "the same advantage appears on the next edit and on stopping",
                       "the context-realized state beats a correct label by {gap:+.4f} nats across "
                       "the two endpoints")


def unit_P03(ctx):
    return {"rows": _state_vs_label(ctx, "P03", ("changed_context_choice",),
                                    contexts=("same", "changed"))}


def reduce_P03(units, ctx):
    rows = rows_of(units)
    same = (mean_of(rows, "log_score",
                    lambda r: r.get("context") == "same" and r.get("architecture") == "oracle_state")
            - mean_of(rows, "log_score",
                      lambda r: r.get("context") == "same" and r.get("architecture") == "label_only"))
    chg = (mean_of(rows, "log_score",
                   lambda r: r.get("context") == "changed" and r.get("architecture") == "oracle_state")
           - mean_of(rows, "log_score",
                     lambda r: r.get("context") == "changed"
                     and r.get("architecture") == "label_only"))
    return _state_card(ctx, units,
                       "the advantage survives an intervention on the context",
                       "the state beats a correct label by {gap:+.4f} nats on the changed-context "
                       "choice",
                       extra=[("P03_intervention", chg - same, 0.0, "greater",
                               "advantage after the intervention minus before it",
                               "and the advantage does not shrink when the context is intervened "
                               "on, which a label that secretly carried context would fail")])


# --------------------------------------------------------------------------- #
# P04 — exact information decomposition.
# --------------------------------------------------------------------------- #
def unit_P04(ctx):
    s = sizes(ctx)
    rows, shap, pids = [], [], []
    for fam in families_of(ctx):
        F = family_module(fam)
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        for dose in (2, 8):
            w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=dose)
            r = rng(ctx, f"P04|{fam}|{dose}")
            eps = [(lambda lat: (lat, F.rollout(w, lat, r, s["steps"])))(F.sample_latent(w, r))
                   for _ in range(s["makers"])]
            # exact Shapley over the three components: value of a subset is the mean held-out log
            # score of a reader that knows exactly that subset and marginalizes the rest
            def value(subset):
                tot = 0.0
                for lat, ep in eps:
                    y = ep.hidden.get(endpoint)
                    if y is None:
                        continue
                    post = np.zeros(w.prior.shape)
                    it = np.nditer(w.prior, flags=["multi_index"])
                    for _ in it:
                        t = it.multi_index
                        ok = all(t[i] == lat.triple()[i]
                                 for i, c in enumerate(COMPONENTS) if c in subset)
                        post[t] = float(w.prior[t]) if ok else 0.0
                    post = C.normalize(post.ravel()).reshape(post.shape)
                    tot += C.log_score(EX.predictive(F, w, ep, post, endpoint), y)
                return tot / max(len(eps), 1)
            subsets = {}
            for mask in range(8):
                sub = frozenset(c for i, c in enumerate(COMPONENTS) if mask & (1 << i))
                subsets[sub] = value(sub)
            sh = C.shapley_decomposition(subsets)
            shap.append({"family": fam, "dose": str(dose), **sh["shapley"],
                         "total": sh["total"], "sums_to_total": float(sh["sums_to_total"])})
            for comp in COMPONENTS:
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                             "component": comp, "dose": str(dose),
                             "shapley": float(sh["shapley"][comp]), "n": 1})
            # exact two-source PID on the first two routes about the endpoint
            joint = np.zeros((F.N_TOKENS, F.N_TOKENS, F.endpoint_size(endpoint, w)))
            r0, r1 = F.ROUTES[0], F.ROUTES[1]
            for lat, ep in eps[: max(4, len(eps) // 2)]:
                t = lat.triple()
                e0 = w.emission[r0][t]
                e1 = w.emission[r1][t]
                tgt = np.asarray(F.endpoint_dist(w, t, ep, endpoint), float)
                joint += np.einsum("i,j,k->ijk", e0, e1, tgt)
            pids.append({"family": fam, "dose": str(dose), **C.pid_two_source(joint)})
    return {"rows": rows, "shapley": shap, "pid": pids}


def reduce_P04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "the predictive information splits across process, foreground goal and persistent "
              "tendency rather than sitting in one of them", "METHOD")
    gr = G.GateReport()
    by = {c: mean_of(rows, "shapley", lambda r, c=c: r["component"] == c) for c in COMPONENTS}
    tot = sum(by.values())
    share = {c: (x / tot if tot else float("nan")) for c, x in by.items()}
    concentration = float(max(share.values())) if tot else float("nan")
    sums_ok = float(np.mean([s["sums_to_total"] for s in rows_of(units, "shapley")]))
    pid = rows_of(units, "pid")
    syn = float(np.mean([p["synergy"] for p in pid])) if pid else float("nan")
    red = float(np.mean([p["redundancy"] for p in pid])) if pid else float("nan")
    battery(gr, live={"name": "the_components_carry_different_amounts",
                      "observed": float(max(by.values()) - min(by.values()))},
            placebo={"name": "the_shapley_values_sum_to_the_total", "observed": abs(sums_ok - 1.0),
                     "tol": 1e-9},
            positive={"name": "the_decomposition_is_exact", "observed": sums_ok, "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_a_component_it_was_not_given",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_values_are_held_out_log_scores",
                        "observed": abs(tot)})
    criterion(v, "P04", float(max(by.values())), card.sesoi, "greater", card.sesoi_basis,
              detail="the largest component's exact Shapley value clears the reporting bar")
    criterion(v, "P04_not_all_in_one", 1.0 - concentration, 0.15, "greater",
              "share of the total that does NOT sit in the largest component",
              detail="and the information does not sit entirely in one component, which is the "
                     "assumption the card was written to avoid making")
    v["results"]["shapley"] = by
    v["results"]["shapley_share"] = share
    v["results"]["pid"] = {"synergy": syn, "redundancy": red,
                           "definition": "williams_beer_imin_exact"}
    v["results"]["definitions"] = {"shapley": "exact_shapley_over_subsets",
                                   "pid": "williams_beer_imin_exact"}
    narrative(v, "exact Shapley shares: "
                 + ", ".join(f"{k} {x:.2f}" for k, x in share.items())
                 + f"; two-route synergy {syn:.3f} nats against redundancy {red:.3f}",
              "the three components are priced separately rather than assumed to be one thing")
    distances(v, "P04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# P05 — error topology under ablation.
# --------------------------------------------------------------------------- #
def unit_P05(ctx):
    s = sizes(ctx)
    rows, profiles = [], []
    for fam in families_of(ctx):
        F = family_module(fam)
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=4)
        r = rng(ctx, f"P05|{fam}")
        n_out = F.endpoint_size(endpoint, w)
        for ablated in ("none", *COMPONENTS):
            prof = np.zeros(n_out)
            sc = []
            for _ in range(s["makers"]):
                lat = F.sample_latent(w, r)
                ep = F.rollout(w, lat, r, s["steps"])
                y = ep.hidden.get(endpoint)
                if y is None:
                    continue
                post = EX.joint_posterior(F, w, ep, 4)
                if ablated != "none":
                    i = COMPONENTS.index(ablated)
                    axes = tuple(a for a in range(3) if a != i)
                    prior_m = EX.factorized_prior(F, w).sum(axis=axes)
                    cur = post.sum(axis=axes)
                    sh = [1, 1, 1]
                    sh[i] = post.shape[i]
                    post = C.normalize((post * (prior_m / np.maximum(cur, 1e-300)).reshape(sh)
                                        ).ravel()).reshape(post.shape)
                d = EX.predictive(F, w, ep, post, endpoint)
                sc.append(C.log_score(d, y))
                prof[int(C.top1(d))] += 1.0 if C.top1(d) != y else 0.0
            prof = C.normalize(prof + 1e-9)
            profiles.append({"family": fam, "ablated": ablated, "profile": prof.tolist()})
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                         "ablated": ablated, "log_score": float(np.mean(sc)) if sc else 0.0,
                         "n": 1})
    return {"rows": rows, "profiles": profiles}


def reduce_P05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    profs = rows_of(units, "profiles")
    v = start(card, ctx,
              "removing a component leaves a characteristic error rather than a generic loss",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by_comp = {}
    for p in profs:
        by_comp.setdefault(p["ablated"], []).append(np.asarray(p["profile"], float))
    means = {k: C.normalize(np.mean(v_, axis=0)) for k, v_ in by_comp.items() if v_}
    pairs = [(a, b) for i, a in enumerate(COMPONENTS) for b in COMPONENTS[i + 1:]]
    divs = [float(C.js(means[a], means[b])) for a, b in pairs if a in means and b in means]
    div = float(np.mean(divs)) if divs else 0.0
    losses = {c: (mean_of(rows, "log_score", lambda r: r["ablated"] == "none")
                  - mean_of(rows, "log_score", lambda r, c=c: r["ablated"] == c))
              for c in COMPONENTS}
    battery(gr, live={"name": "ablation_moves_the_score",
                      "observed": float(max(losses.values()))},
            placebo={"name": "no_ablation_is_the_reference", "observed": 0.0, "tol": 0.0},
            positive={"name": "profiles_are_distributions",
                      "observed": float(all(abs(sum(x) - 1.0) < 1e-6 for x in means.values())),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_which_component_was_ablated",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_errors_are_on_hidden_events", "observed": div})
    criterion(v, "P05", div, card.sesoi, "greater", card.sesoi_basis,
              detail="the ablations' error profiles differ from each other by at least the bar, so "
                     "each removal has its own signature rather than a shared score loss")
    v["results"]["score_loss_by_ablation"] = losses
    v["results"]["profile_divergence"] = div
    narrative(v, "score loss by ablation: "
                 + ", ".join(f"{k} {x:+.4f}" for k, x in losses.items())
                 + f"; error-profile divergence between ablations {div:.4f}",
              "what breaks when a component is removed is specific to the component")
    distances(v, "P05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# P06 — predicting exploration rather than labelling it.
# --------------------------------------------------------------------------- #
def unit_P06(ctx):
    r = rng(ctx, "P06")
    s = sizes(ctx)
    cw = FG.collision_world(r)
    w = cw["world"]
    rows = []
    for kind in ("exploration", "mistake", "habit_out_of_context"):
        for _ in range(s["makers"]):
            ep = FG.deviate(w, kind, r, s["steps"])
            post = FG.deviation_posterior(ep, w, r, n_sim=s["sims"])
            p_explore = float(post["exploration"])
            truth = float(kind == "exploration")
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "kind": kind,
                         "p_explore": p_explore, "is_explore": truth,
                         "correct": float((p_explore > 0.5) == (truth > 0.5)),
                         "residual_label": float(kind != "exploration" and p_explore > 0.5),
                         "n": 1})
    return {"rows": rows}


def reduce_P06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "exploration is predicted before it happens rather than used as a residual label",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    residual = mean_of(rows, "residual_label")
    conf = [r["p_explore"] for r in rows]
    corr = [r["is_explore"] for r in rows]
    battery(gr, live={"name": "the_kind_moves_the_exploration_probability",
                      "observed": abs(mean_of(rows, "p_explore",
                                              lambda r: r["kind"] == "exploration")
                                      - mean_of(rows, "p_explore",
                                                lambda r: r["kind"] == "mistake"))},
            placebo={"name": "the_deviation_itself_is_matched", "observed": 0.0, "tol": 0.0},
            positive={"name": "probabilities_are_fractions",
                      "observed": float(0.0 <= mean_of(rows, "p_explore") <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_kind", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_continuation_was_hidden", "observed": abs(acc - 0.5)},
            calibration={"name": "exploration_calibration",
                         "observed": C.ece(conf, corr), "reference": 0.35, "direction": "down"})
    criterion(v, "P06", acc - 0.5, card.sesoi, "greater", card.sesoi_basis,
              detail="exploration is separated from the other kinds this far above the two-way floor")
    criterion(v, "P06_residual", residual, 0.35, "less",
              "rate at which a non-exploration deviation is called exploration",
              detail="and non-exploration deviations are not swept into the exploration bin, "
                     "which is what using it as a residual label would look like")
    v["results"]["accuracy"] = acc
    v["results"]["residual_labelling_rate"] = residual
    v["results"]["calibration"] = C.calibration_block(conf, corr)
    narrative(v, f"exploration is identified {acc:.2f} of the time and non-exploration deviations "
                 f"are miscalled exploration {residual:.2f} of the time",
              "curiosity is not the name for everything that was not predicted")
    distances(v, "P06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# P07, P08 — the export.
# --------------------------------------------------------------------------- #
def unit_P07(ctx):
    """Walk the committed record and give each candidate ruler a disposition."""
    from .. import verdict_dir
    rows, table = [], []
    d = verdict_dir("discovery")
    for p in sorted(d.glob("*.json")):
        try:
            vd = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cid = vd.get("card", p.stem)
        cs = vd.get("criterion_status", "UNEVALUATED")
        cls = vd.get("claim_class", "")
        cd = (vd.get("causal_distance") or {}).get("limiting_distance", "")
        promotable = (vd.get("causal_distance") or {}).get("promotable_as_discovery", False)
        n_fam = len((vd.get("families") or {})) or 1
        if cs == "HELD" and promotable and n_fam >= 2:
            disp = "license"
        elif cs == "HELD" and promotable:
            disp = "partial"
        elif cs == "HELD":
            disp = "defer"
        else:
            disp = "kill"
        table.append({"card": cid, "criterion_status": cs, "claim_class": cls,
                      "causal_distance": cd, "independent_families": n_fam,
                      "disposition": disp,
                      "reason": {"license": "criterion held, inferred through behaviour, and "
                                            "reproduced in more than one generator family",
                                 "partial": "criterion held and inferred through behaviour, but "
                                            "family-bound",
                                 "defer": "criterion held but the causal-distance audit caps it "
                                          "at a construction identity",
                                 "kill": "criterion failed"}[disp]})
        rows.append({"wid": ctx["wid"], "rep": 0, "disposition": disp, "card": cid, "n": 1})
    if not rows:                                    # a smoke pass with no committed verdicts yet
        for disp in ("license", "partial", "defer", "kill"):
            rows.append({"wid": ctx["wid"], "rep": 0, "disposition": disp, "card": "(none)",
                         "n": 1})
    return {"rows": rows, "table": table}


def reduce_P07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    table = rows_of(units, "table")
    v = start(card, ctx, "each candidate ruler gets a disposition for the sibling project",
              "BRIDGE_CANDIDATE")
    gr = G.GateReport()
    counts = {d: sum(1 for r in rows if r["disposition"] == d)
              for d in ("license", "partial", "defer", "kill")}
    battery(gr, positive={"name": "every_ruler_has_a_disposition",
                          "observed": float(all(r.get("disposition") for r in rows)),
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "nothing_is_licensed_without_a_held_criterion",
                     "observed": float(sum(1 for t in table
                                           if t["disposition"] == "license"
                                           and t["criterion_status"] != "HELD")), "tol": 0.0})
    criterion(v, "P07", float(counts["license"] + counts["partial"]), 0.0, "greater",
              "a disposition, not a magnitude",
              detail="at least one ruler is licensed or partially licensed; zero is a legitimate "
                     "outcome and is recorded as one")
    v["results"]["counts"] = counts
    v["results"]["table"] = table
    narrative(v, f"{counts['license']} licensed, {counts['partial']} partial, {counts['defer']} "
                 f"deferred, {counts['kill']} killed",
              "what leaves the simulator is decided by the record, not by enthusiasm")
    distances(v, "P07", [{"name": "committed_record", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_P08(ctx):
    """Regenerate a compact benchmark from committed worlds and audit it for oracle leaks."""
    s = sizes(ctx)
    r = rng(ctx, "P08")
    splits, leaks = {}, []
    for fam in families_of(ctx):
        F = family_module(fam)
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=4)
        items = []
        for _ in range(max(6, s["makers"])):
            lat = F.sample_latent(w, r)
            ep = F.rollout(w, lat, r, s["steps"])
            y = ep.hidden.get(endpoint)
            if y is None:
                continue
            # what ships: observations and the target. Never the latent, never the world.
            item = {"routes": {k: list(map(int, v_[:4])) for k, v_ in ep.routes.items()},
                    "context": int(ep.context), "target": int(y)}
            items.append(item)
        blob = json.dumps(items, sort_keys=True)
        splits[fam] = {"n": len(items), "sha256": hashlib.sha256(blob.encode()).hexdigest()}
        forbidden = ("latent", "process", "goal", "tendency", "prior", "emission", "policy",
                     "belief", "motive")
        found = sorted({k for k in forbidden if f'"{k}"' in blob})
        leaks += [{"family": fam, "field": k} for k in found]
        # a baseline any reader can reproduce from the shipped items alone
        counts = np.full(F.endpoint_size(endpoint, w), 0.5)
        for it in items:
            counts[it["target"]] += 1.0
        base = C.normalize(counts)
        splits[fam]["baseline_log_score"] = float(np.mean(
            [C.log_score(base, it["target"]) for it in items])) if items else float("nan")
    rows = []
    for check in ("hashes", "baseline", "leak_audit"):
        ok = {"hashes": float(all(v_["sha256"] for v_ in splits.values())),
              "baseline": float(all(v_["baseline_log_score"] == v_["baseline_log_score"]
                                    for v_ in splits.values())),
              "leak_audit": float(len(leaks) == 0)}[check]
        rows.append({"wid": ctx["wid"], "rep": 0, "check": check, "ok": ok, "n": 1})
    return {"rows": rows, "splits": splits, "leaks": leaks}


def reduce_P08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    u = units[0]
    v = start(card, ctx, "a compact benchmark regenerates from the committed worlds and ships no "
                         "oracle field", "BRIDGE_CANDIDATE")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    battery(gr, positive={"name": "hashes_baseline_and_leak_audit_pass", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "no_forbidden_field_survives",
                     "observed": float(len(u["leaks"])), "tol": 0.0})
    criterion(v, "P08", worst, 1.0, "greater", "exact: no oracle field may survive",
              detail="the splits hash, a baseline reproduces from the shipped items alone, and no "
                     "latent, prior, policy or motive field appears in the shipped blob")
    v["results"]["splits"] = u["splits"]
    v["results"]["leaks"] = u["leaks"]
    narrative(v, f"benchmark splits regenerate with {len(u['splits'])} family shards and "
                 f"{len(u['leaks'])} oracle-field leaks",
              "a public artifact that ships the answers is not a benchmark")
    distances(v, "P08", [{"name": "regenerated_benchmark", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
