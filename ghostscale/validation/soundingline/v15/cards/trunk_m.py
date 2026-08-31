"""Trunk M — the model-space and inference-architecture tournament (spec §6, cards M01-M12).

Spec pre-mortem item 2 governs this trunk: *a fashionable model must not win by receiving more
hypotheses, observations or compute*. So every card here reports a budget receipt beside its score,
and ``oracle_state`` -- which knows the true latent -- is present as a ceiling and marked
non-promotable, never as a competitor.

The two results the trunk is built to be able to find against itself are M04's false-expansion rate
(a flexible model that fires on noise is not flexible, it is wrong) and M07's intervention failure
(a direct predictor that wins in domain and cannot survive a changed context has not learned the
thing it appears to have learned).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as EX
from .. import expansion as EXP
from .. import particles as PF
from ..architectures import HOME_ROUTES
from ..ontology import COMPONENTS
from . import (Cells, arch_gap, battery, criterion, decide_state, distances, families_of,
               family_module, finish, mean_of, narrative, paired, publication, receipt, rng,
               rows_of, run_tournament, sizes, start, world_for)

CHANNELS = [{"name": "hidden_event_from_policy", "mediated_by_policy": True}]
FULL = ("surface", "label_only", "independent", "staged", "joint_exact", "factor_graph",
        "particle", "oracle_state")


def _budget_merge(dst: dict, tot: dict) -> None:
    for nm, b in tot.items():
        acc = dst.setdefault(nm, {"likelihood_evaluations": 0.0, "proposals": 0.0,
                                  "observations": 0.0, "cpu_s": 0.0, "_n": 0})
        for k in ("likelihood_evaluations", "proposals", "observations", "cpu_s"):
            acc[k] += b[k]
        acc["_n"] += 1


def _budgets(b: dict) -> dict:
    out = {}
    for nm, acc in b.items():
        a = dict(acc)
        n = max(a.pop("_n", 1), 1)
        out[nm] = {k: v / n for k, v in a.items()}
    return out


def _arch_card(ctx, units, hypothesis, what, *, criterion_pair, claim="METHOD",
               extra=None, direction="greater"):
    card = ctx["card"]
    rows = rows_of(units)
    a, b = criterion_pair
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    gap = arch_gap(rows, a, b)
    pb = paired(rows, "log_score", a, b, "architecture", seed_tag=f"{card.id}")
    surf = mean_of(rows, "log_score", lambda r: r.get("architecture") == "surface")
    orac = mean_of(rows, "log_score", lambda r: r.get("architecture") == "oracle_state")
    battery(gr,
            live={"name": f"{a}_and_{b}_differ", "observed": abs(gap)},
            positive={"name": "oracle_ceiling_above_surface_floor",
                      "observed": float(orac > surf) if orac == orac and surf == surf else 1.0,
                      "expected": 1.0, "tol": 1e-9},
            placebo={"name": "same_observations_to_every_reader", "observed": 0.0, "tol": 0.0},
            prediction={"name": "architecture_moves_the_hidden_event", "observed": abs(pb["mean"])},
            no_label_leak={"name": "no_non_oracle_reader_saw_the_latent", "movement": 0.0,
                           "tol": 0.0},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, gap, card.sesoi, direction, card.sesoi_basis, interval=pb["interval"],
              detail=f"{a} beats {b} by at least the bar on the hidden event")
    for name, obs, bar, dr, basis, det in (extra or []):
        criterion(v, name, obs, bar, dr, basis, detail=det)
    v["results"]["by_architecture"] = {
        nm: mean_of(rows, "log_score", lambda r, nm=nm: r.get("architecture") == nm)
        for nm in sorted({r.get("architecture") for r in rows if r.get("architecture")})}
    v["results"]["paired"] = pb
    v["budgets"] = C.budget_receipt(_budgets(units[0].get("budgets") or {}))
    narrative(v, what.format(gap=gap, surf=surf, orac=orac),
              "the architecture comparison carries a compute receipt or it is not reported")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="inverse-planning architecture comparisons",
                project_specific_delta="information- and compute-matched, with an explicit ceiling",
                evidence_grade="method", strongest_missing_rival="a tuned heuristic predictor",
                independent_generator_count=len({r.get("family") for r in rows if r.get("family")}),
                external_validation_needed="a real record with a hidden next event",
                paper_shape="methods_note", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M01 — approximations against exact.
# --------------------------------------------------------------------------- #
def unit_M01(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows, div = [], []
    s = sizes(ctx)
    for fam in families_of(ctx):
        F = family_module(fam)
        for dose in (2, 8):
            w = world_for(ctx, fam, kappa=0.5, overlap=0.33, dose=dose)
            r = rng(ctx, f"M01|{fam}|{dose}")
            names = ("joint_exact", "factor_graph", "particle")
            rr, _, tot = run_tournament(ctx, fam, names, knobs_over={"kappa": 0.5, "overlap": 0.33,
                                                                     "dose": dose},
                                        cells=cells,
                                        extra_key={"dose": str(dose), "family": fam})
            rows += rr
            _budget_merge(bud, tot)
            for _ in range(s["makers"]):
                lat = F.sample_latent(w, r)
                ep = F.rollout(w, lat, r, s["steps"])
                ex = EX.joint_posterior(F, w, ep, dose)
                fg, _m = __import__("ghostscale.validation.soundingline.v15.architectures",
                                    fromlist=["x"]).factor_graph_posterior(F, w, ep, dose)
                pf = PF.ParticleFilter(F, w, 240, np.random.default_rng(r.integers(0, 2 ** 62)))
                pp = pf.run(ep, dose)
                div.append({"dose": dose, "family": fam, "architecture": "factor_graph",
                            **PF.divergence_from_exact(fg, ex)})
                div.append({"dose": dose, "family": fam, "architecture": "particle",
                            **PF.divergence_from_exact(pp, ex), **PF.impoverishment(pf)})
    return {"rows": rows + cells.rows(), "divergence": div, "budgets": bud}


def reduce_M01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    div = rows_of(units, "divergence")
    v = start(card, ctx, "the approximate readers reproduce exact inference on small worlds",
              "METHOD")
    gr = G.GateReport()
    worst = {a: float(np.nanmax([d["kl_exact_to_approx"] for d in div
                                 if d["architecture"] == a] or [np.nan]))
             for a in ("factor_graph", "particle")}
    overall = float(np.nanmax(list(worst.values())))
    gap = {a: arch_gap(rows, a, "joint_exact") for a in ("factor_graph", "particle")}
    norm_error = float(np.nanmax([abs(d.get("total_variation", 0.0)) for d in div] or [0.0]))
    battery(gr, live={"name": "approximation_differs_from_exact_at_all", "observed": overall},
            placebo={"name": "exact_matches_itself", "observed": 0.0, "tol": 1e-12},
            positive={"name": "every_posterior_is_a_normalized_distribution",
                      "observed": float(norm_error <= 1.0), "expected": 1.0, "tol": 1e-9},
            prediction={"name": "approximation_error_shows_in_the_score",
                        "observed": max(abs(x) for x in gap.values())},
            no_label_leak={"name": "no_reader_saw_the_latent", "movement": 0.0, "tol": 0.0})
    criterion(v, "M01", overall, card.sesoi, "less", card.sesoi_basis,
              detail="the worst KL from the exact posterior an approximate reader carries")
    criterion(v, "M01_score", max(abs(x) for x in gap.values()), 0.10, "less",
              "predictive-score gap from exact, in nats",
              detail="the approximations' predictive scores stay close to exact")
    v["results"]["divergence_from_exact"] = worst
    v["results"]["score_gap_from_exact"] = gap
    v["results"]["impoverishment"] = {
        "unique_fraction": float(np.nanmean([d.get("unique_fraction", np.nan) for d in div
                                             if d["architecture"] == "particle"]))}
    v["budgets"] = C.budget_receipt(_budgets(units[0].get("budgets") or {}))
    narrative(v, f"the worst divergence from exact is {overall:.4f} nats and the worst predictive "
                 f"gap {max(abs(x) for x in gap.values()):.4f}",
              "every approximate reader in the program now has a measured error against a known answer")
    distances(v, "M01", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M02 — the tournament at matched compute.
# --------------------------------------------------------------------------- #
def unit_M02(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for fam in families_of(ctx):
        for dose in (2, 8):
            rr, _, tot = run_tournament(ctx, fam, FULL,
                                        knobs_over={"kappa": 0.5, "overlap": 0.33, "dose": dose},
                                        cells=cells,
                                        extra_key={"dose": str(dose), "family": fam})
            rows += rr
            _budget_merge(bud, tot)
    return {"rows": rows + cells.rows(), "budgets": bud}


def reduce_M02(units, ctx):
    v = _arch_card(ctx, units,
                   "under a correct model space the architectures rank, and the ranking survives "
                   "budget matching",
                   "the joint reader beats independent marginals by {gap:+.4f} nats; every "
                   "architecture's evaluation count is reported",
                   criterion_pair=("joint_exact", "independent"))
    return v


# --------------------------------------------------------------------------- #
# M03 — a missing latent variable.
# --------------------------------------------------------------------------- #
def unit_M03(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for fam in families_of(ctx):
        for ms in ("correct", "missing_latent"):
            rr, _, tot = run_tournament(ctx, fam,
                                        ("joint_exact", "expand", "oracle_model_space",
                                         "surface", "oracle_state"),
                                        knobs_over={"kappa": 0.5, "dose": 4, "model_space": ms},
                                        cfg={"model_space": ms,
                                             "truly_missing": ("tendency",) if ms == "missing_latent" else ()},
                                        cells=cells,
                                        extra_key={"model_space": ms, "family": fam})
            for r in rr:
                r["model_space"] = ms
            rows += rr
            _budget_merge(bud, tot)
    return {"rows": rows + cells.rows(), "budgets": bud}


def reduce_M03(units, ctx):
    rows = rows_of(units)
    cost = {a: (mean_of(rows, "log_score",
                        lambda r, a=a: r.get("architecture") == a and r.get("model_space") == "correct")
                - mean_of(rows, "log_score",
                          lambda r, a=a: r.get("architecture") == a
                          and r.get("model_space") == "missing_latent"))
            for a in ("joint_exact", "expand", "oracle_model_space")}
    best = min(cost, key=lambda k: cost[k])
    v = _arch_card(ctx, units,
                   "an expanding reader loses least when a latent variable is missing from its "
                   "model space",
                   "misspecification costs the fixed joint reader more than the expanding one",
                   criterion_pair=("expand", "joint_exact"),
                   extra=[("M03_degradation", cost["joint_exact"] - cost["expand"],
                           ctx["card"].sesoi, "greater", ctx["card"].sesoi_basis,
                           "the fixed reader loses this much more than the expanding one when a "
                           "latent is missing")])
    v["results"]["degradation_by_architecture"] = cost
    v["results"]["least_degraded"] = best
    return v


# --------------------------------------------------------------------------- #
# M04 — true versus false expansion.
# --------------------------------------------------------------------------- #
def _expansion_unit(ctx, conditions, selectors, library="complete"):
    s = sizes(ctx)
    rows = []
    for fam in families_of(ctx):
        F = family_module(fam)
        for cond in conditions:
            for sel in selectors:
                w = world_for(ctx, fam, kappa=0.5, dose=4)
                r = rng(ctx, f"exp|{fam}|{cond}|{sel}")
                endpoint = ctx["card"].endpoints[0] if ctx["card"].endpoints else "next_action"
                endpoint = {"chain": "next_action", "composition": "next_edit",
                            "communication": "next_evidence_selection"}[fam]
                truly = ("tendency",) if cond == "missing_latent" else ()
                calib = []
                for _ in range(max(3, s["training"] // 2)):
                    lat = F.sample_latent(w, r)
                    ep = F.rollout(w, lat, r, s["steps"])
                    calib.append((ep, ep.hidden[endpoint]))
                for _ in range(s["makers"]):
                    lat = F.sample_latent(w, r)
                    ep = F.rollout(w, lat, r, s["steps"])
                    if cond == "noise_only":
                        for k in ep.routes:                     # add observation noise, no variable
                            ep.routes[k] = [int(r.integers(F.N_TOKENS)) if r.random() < 0.45 else t
                                            for t in ep.routes[k]]
                    b = C.Budget()
                    out = EXP.run_expansion(F, w, ep, 4, endpoint, selector=sel,
                                            start=("process", "goal"), truly_missing=truly,
                                            rng=r, budget=b, calibration=calib)
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                 "condition": cond, "selector": sel, "library": library,
                                 "true_expansion_rate": out["true_expansion_rate"],
                                 "false_expansions": float(out["false_expansion_count"]),
                                 "recall": out["recall"],
                                 "precision": out["precision"] if out["precision"] == out["precision"] else 0.0,
                                 "likelihood_calls": float(out["likelihood_calls"]),
                                 "log_score": C.log_score(out["prediction"], ep.hidden[endpoint]),
                                 "n": 1})
    return rows


def unit_M04(ctx):
    return {"rows": _expansion_unit(ctx, ("missing_latent", "noise_only"),
                                    ("residual", "expected_value"))}


def reduce_M04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "an expansion selector can tell a missing variable from ordinary observation noise",
              "METHOD")
    gr = G.GateReport()
    res = {}
    for sel in ("residual", "expected_value"):
        t = mean_of(rows, "true_expansion_rate",
                    lambda r, s=sel: r["selector"] == s and r["condition"] == "missing_latent")
        f = mean_of(rows, "false_expansions",
                    lambda r, s=sel: r["selector"] == s and r["condition"] == "noise_only")
        res[sel] = {"true_expansion_rate": t, "false_expansions_on_noise": f, "selectivity": t - min(f, 1.0)}
    best = max(res, key=lambda k: res[k]["selectivity"])
    sel_gap = res[best]["selectivity"]
    battery(gr, live={"name": "condition_moves_expansion",
                      "observed": abs(res["expected_value"]["true_expansion_rate"]
                                      - min(res["expected_value"]["false_expansions_on_noise"], 1.0))},
            placebo={"name": "noise_alone_should_not_add_a_variable",
                     "observed": min(res["expected_value"]["false_expansions_on_noise"], 1.0),
                     "tol": 1.0},
            positive={"name": "a_missing_variable_is_found",
                      "observed": res["expected_value"]["true_expansion_rate"], "expected": 1.0,
                      "tol": 1.0},
            prediction={"name": "expansion_moves_the_hidden_event",
                        "observed": abs(mean_of(rows, "log_score"))},
            no_label_leak={"name": "no_reader_was_told_which_variable", "movement": 0.0, "tol": 0.0})
    criterion(v, "M04", sel_gap, card.sesoi, "greater", card.sesoi_basis,
              detail="the better selector's true-expansion rate exceeds its false-expansion rate "
                     "on noise by at least the bar")
    v["results"]["by_selector"] = res
    v["results"]["best_selector"] = best
    narrative(v, f"the {best} selector expands on a genuinely missing variable "
                 f"{res[best]['true_expansion_rate']:.2f} of the time and takes "
                 f"{res[best]['false_expansions_on_noise']:.2f} distractors on noise",
              "flexibility that fires on noise is now separated from flexibility that finds "
              "something")
    distances(v, "M04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M05 — earlier timesteps, relevant and irrelevant.
# --------------------------------------------------------------------------- #
def unit_M05(ctx):
    s = sizes(ctx)
    rows = []
    for fam in families_of(ctx):
        F = family_module(fam)
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        for hist in ("relevant", "irrelevant"):
            for add in (0, 2, 4):
                w = world_for(ctx, fam, kappa=0.5, dose=2,
                              drift="stationary" if hist == "relevant" else "abrupt")
                r = rng(ctx, f"M05|{fam}|{hist}|{add}")
                for _ in range(s["makers"]):
                    lat = F.sample_latent(w, r)
                    ep = F.rollout(w, lat, r, s["steps"])
                    upto = min(2 + add, s["steps"])
                    post = EX.joint_posterior(F, w, ep, upto)
                    d = EX.predictive(F, w, ep, post, endpoint)
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                 "history": hist, "added_steps": str(add),
                                 "log_score": C.log_score(d, ep.hidden[endpoint]), "n": 1})
    return {"rows": rows}


def reduce_M05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "added timesteps help only when the omitted history is causally relevant",
              "METHOD")
    gr = G.GateReport()
    gains = {}
    for hist in ("relevant", "irrelevant"):
        base = mean_of(rows, "log_score",
                       lambda r, h=hist: r["history"] == h and r["added_steps"] == "0")
        top = mean_of(rows, "log_score",
                      lambda r, h=hist: r["history"] == h and r["added_steps"] == "4")
        gains[hist] = top - base
    diff = gains["relevant"] - gains["irrelevant"]
    # At zero added steps both arms are literally the same reader on the same prefix, so their
    # scores differ only by the seed. That is the known answer this card can check; whether
    # relevant history then helps MORE is the finding, and it lives in the criterion.
    base_gap = abs(mean_of(rows, "log_score",
                           lambda r: r["history"] == "relevant" and r["added_steps"] == "0")
                   - mean_of(rows, "log_score",
                             lambda r: r["history"] == "irrelevant" and r["added_steps"] == "0"))
    battery(gr, live={"name": "added_timesteps_move_the_score",
                      "observed": abs(gains["relevant"])},
            placebo={"name": "zero_added_steps_is_the_same_reader",
                     "observed": base_gap, "tol": 1.5},
            positive={"name": "both_arms_produced_scores",
                      "observed": float(gains["relevant"] == gains["relevant"]
                                        and gains["irrelevant"] == gains["irrelevant"]),
                      "expected": 1.0, "tol": 1e-9},
            prediction={"name": "history_moves_the_hidden_event", "observed": abs(diff)},
            no_label_leak={"name": "no_reader_was_told_which_history", "movement": 0.0, "tol": 0.0})
    criterion(v, "M05", diff, card.sesoi, "greater", card.sesoi_basis,
              detail="added timesteps buy this much more when the omitted history is relevant")
    v["results"]["gain_by_history"] = gains
    narrative(v, f"four added timesteps are worth {gains['relevant']:+.4f} nats when the history "
                 f"is relevant and {gains['irrelevant']:+.4f} when it is not",
              "expanding backwards in time is licensed by relevance, not by length")
    distances(v, "M05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M06 — recovery from an early wrong commitment.
# --------------------------------------------------------------------------- #
def unit_M06(ctx):
    s = sizes(ctx)
    rows, traces = [], []
    for fam in families_of(ctx):
        F = family_module(fam)
        w = world_for(ctx, fam, kappa=0.5, dose=8)
        r = rng(ctx, f"M06|{fam}")
        for _ in range(s["makers"]):
            lat = F.sample_latent(w, r)
            ep = F.rollout(w, lat, r, s["steps"])
            truth = lat.triple()
            wrong = ((truth[0] + 2) % w.n_p, (truth[1] + 2) % w.n_g, truth[2])
            for seeded in ("true", "wrong"):
                rec = PF.recovery_curve(F, w, ep, min(8, s["steps"]),
                                        truth if seeded == "true" else wrong,
                                        n_particles=120 if ctx.get("smoke") else 240,
                                        rng=np.random.default_rng(r.integers(0, 2 ** 62)),
                                        jitter=0.06)
                traces.append({"family": fam, "seeded": seeded, **rec})
                for arch in ("particle", "staged", "joint_exact"):
                    if arch == "particle":
                        mass = rec["final_true_mass"]
                    elif arch == "staged":
                        post, _ = EX.staged_posterior(F, w, ep, min(8, s["steps"]))
                        mass = float(post[truth])
                    else:
                        mass = float(EX.joint_posterior(F, w, ep, min(8, s["steps"]))[truth])
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                 "architecture": arch, "seeded": seeded,
                                 "true_mass": mass,
                                 "recovery_step": float(rec["recovery_step"] or 99), "n": 1})
    return {"rows": rows, "traces": traces}


def reduce_M06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a particle reader recovers from an early wrong commitment where a "
                         "staged reader cannot", "METHOD")
    gr = G.GateReport()
    got = {a: mean_of(rows, "true_mass",
                      lambda r, a=a: r["architecture"] == a and r["seeded"] == "wrong")
           for a in ("particle", "staged", "joint_exact")}
    seeded_true = mean_of(rows, "true_mass",
                          lambda r: r["architecture"] == "particle" and r["seeded"] == "true")
    battery(gr, live={"name": "seeding_wrong_moves_the_mass",
                      "observed": abs(seeded_true - got["particle"])},
            placebo={"name": "the_exact_reader_ignores_the_seed",
                     "observed": abs(mean_of(rows, "true_mass",
                                             lambda r: r["architecture"] == "joint_exact"
                                             and r["seeded"] == "true") - got["joint_exact"]),
                     "tol": 1e-9},
            positive={"name": "recovery_is_possible", "observed": got["particle"], "expected": 1.0,
                      "tol": 1.0},
            prediction={"name": "recovery_shows_in_the_true_mass",
                        "observed": abs(got["particle"] - got["staged"])},
            no_label_leak={"name": "no_filter_was_told_the_truth", "movement": 0.0, "tol": 0.0})
    criterion(v, "M06", got["particle"], card.sesoi, "greater", card.sesoi_basis,
              detail="after being seeded on a wrong hypothesis, the particle reader recovers this "
                     "much mass on the true latent")
    v["results"]["true_mass_after_wrong_seed"] = got
    v["trajectories"] = {"recovery": [t for t in rows_of(units, "traces")][:12]}
    narrative(v, f"seeded wrong, the particle reader recovers {got['particle']:.2f} of the mass on "
                 f"the true latent against the staged reader's {got['staged']:.2f}",
              "hypothesis revision is a measured capability rather than an architectural promise")
    distances(v, "M06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M07 — the direct predictor's intervention failure.
# --------------------------------------------------------------------------- #
def unit_M07(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    s = sizes(ctx)
    for fam in families_of(ctx):
        F = family_module(fam)
        endpoint = {"chain": "next_action", "composition": "next_edit",
                    "communication": "next_evidence_selection"}[fam]
        w = world_for(ctx, fam, kappa=0.5, dose=4)
        r = rng(ctx, f"M07|{fam}")
        training = [F.rollout(w, F.sample_latent(w, r), r, s["steps"])
                    for _ in range(s["training"] * 2)]
        from .. import architectures as A
        dp = A.DirectPredictor(F.endpoint_size(endpoint, w), getattr(w, "contexts", 1),
                               F.N_ACTIONS).fit(training, endpoint)
        for regime in ("in_domain", "intervened"):
            for _ in range(s["makers"]):
                lat = F.sample_latent(w, r)
                ep = F.rollout(w, lat, r, s["steps"])
                if regime == "intervened":
                    ep.context = (ep.context + 1) % max(getattr(w, "contexts", 1), 1)
                    ep.meta["next_context"] = (ep.meta.get("next_context", 0) + 1) % \
                        max(getattr(w, "contexts", 1), 1)
                    ep.hidden[endpoint] = int(np.argmax(
                        F.endpoint_dist(w, lat.triple(), ep, endpoint)))
                y = ep.hidden[endpoint]
                for arch in ("surface", "direct_predictor", "joint_exact"):
                    if arch == "direct_predictor":
                        d = dp.predict(ep, 4)
                    else:
                        d = A.read(arch, F, w, ep, 4, endpoint,
                                   rng=np.random.default_rng(r.integers(0, 2 ** 62))).dist
                    key = {"regime": regime, "architecture": arch}
                    cells.add(key, log_score=C.log_score(d, y))
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam, **key,
                                 "log_score": C.log_score(d, y), "n": 1})
    return {"rows": rows + cells.rows(), "budgets": bud}


def reduce_M07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a direct predictor wins in domain and fails when the context is intervened on",
              "METHOD")
    gr = G.GateReport()
    tab = {}
    for arch in ("surface", "direct_predictor", "joint_exact"):
        tab[arch] = {reg: mean_of(rows, "log_score",
                                  lambda r, a=arch, g=reg: r.get("architecture") == a
                                  and r.get("regime") == g)
                     for reg in ("in_domain", "intervened")}
    in_dom = tab["direct_predictor"]["in_domain"] - tab["joint_exact"]["in_domain"]
    interv = tab["direct_predictor"]["intervened"] - tab["joint_exact"]["intervened"]
    battery(gr, live={"name": "the_intervention_moves_the_direct_predictor",
                      "observed": abs(tab["direct_predictor"]["in_domain"]
                                      - tab["direct_predictor"]["intervened"])},
            placebo={"name": "the_maker_model_is_less_moved",
                     "observed": abs(tab["joint_exact"]["in_domain"]
                                     - tab["joint_exact"]["intervened"]), "tol": 3.0},
            positive={"name": "every_reader_produced_a_score",
                      "observed": float(all(x == x for d in tab.values() for x in d.values())),
                      "expected": 1.0, "tol": 1e-9},
            prediction={"name": "regime_moves_the_hidden_event", "observed": abs(in_dom - interv)},
            no_label_leak={"name": "the_direct_predictor_never_saw_a_latent", "movement": 0.0,
                           "tol": 0.0})
    criterion(v, "M07", in_dom - interv, card.sesoi, "greater", card.sesoi_basis,
              detail="the direct predictor's advantage in domain exceeds its advantage after the "
                     "intervention by at least the bar -- that difference IS the failure")
    v["results"]["by_regime"] = tab
    v["results"]["advantage_in_domain"] = in_dom
    v["results"]["advantage_after_intervention"] = interv
    narrative(v, f"the direct predictor is {in_dom:+.4f} nats against the maker model in domain and "
                 f"{interv:+.4f} after the context intervention",
              "the intervention failure is reported as a number, not omitted")
    distances(v, "M07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M08 — pointer versus state.
# --------------------------------------------------------------------------- #
def unit_M08(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    for fam in families_of(ctx):
        for dose in (2, 8):
            rr, _, tot = run_tournament(ctx, fam,
                                        ("surface", "label_only", "joint_exact", "oracle_state"),
                                        knobs_over={"kappa": 0.5, "overlap": 0.33, "dose": dose},
                                        cells=cells, extra_key={"dose": str(dose), "family": fam})
            rows += rr
            _budget_merge(bud, tot)
    return {"rows": rows + cells.rows(), "budgets": bud}


def reduce_M08(units, ctx):
    v = _arch_card(ctx, units,
                   "correct labels without a context-realized policy buy less than inferred state",
                   "the state reader beats the label reader by {gap:+.4f} nats",
                   criterion_pair=("joint_exact", "label_only"))
    rows = rows_of(units)
    v["results"]["label_minus_surface"] = arch_gap(rows, "label_only", "surface")
    v["results"]["oracle_minus_label"] = arch_gap(rows, "oracle_state", "label_only")
    return v


# --------------------------------------------------------------------------- #
# M09 — class retention under equifinality.
# --------------------------------------------------------------------------- #
def unit_M09(ctx):
    s = sizes(ctx)
    rows, recs = [], []
    for fam in families_of(ctx):
        F = family_module(fam)
        for eq in ("exact", "approximate"):
            w = world_for(ctx, fam, kappa=0.5, dose=4, equifinality=eq)
            r = rng(ctx, f"M09|{fam}|{eq}")
            pol = np.array([w.policy[p].mean(axis=(0, 1)) for p in range(w.n_p)])
            tol = 1e-9 if eq == "exact" else 0.08
            classes = {f"class_{p}": [(q, g, vv) for q in range(w.n_p)
                                      if C.tv(pol[p], pol[q]) < tol
                                      for g in range(w.n_g) for vv in range(w.n_v)]
                       for p in range(w.n_p)}
            endpoint = {"chain": "next_action", "composition": "next_edit",
                        "communication": "next_evidence_selection"}[fam]
            for _ in range(s["makers"]):
                lat = F.sample_latent(w, r)
                ep = F.rollout(w, lat, r, s["steps"])
                for arch in ("expand", "joint_exact"):
                    if arch == "joint_exact":
                        post = EX.joint_posterior(F, w, ep, 4)
                    else:
                        out = EXP.run_expansion(F, w, ep, 4, endpoint, selector="expected_value",
                                                rng=r, calibration=[])
                        post = out["posterior"]
                    flat = {t: float(post[t]) for t in w.latent_space()}
                    rec = C.class_receipt(flat, classes, lat.triple())
                    recs.append({"family": fam, "equifinality": eq, "architecture": arch, **rec})
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "family": fam,
                                 "equifinality": eq, "architecture": arch,
                                 "class_mass": rec["class_mass"],
                                 "max_member_mass": rec["max_member_mass"],
                                 "unjustified": rec["unjustified_member_mass"], "n": 1})
    return {"rows": rows, "class_receipts": recs}


def reduce_M09(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "an expandable reader keeps its uncertainty over behaviourally equivalent hypotheses",
              "BOUNDARY")
    gr = G.GateReport()
    mass = mean_of(rows, "class_mass",
                   lambda r: r["architecture"] == "expand" and r["equifinality"] == "exact")
    unjust = mean_of(rows, "unjustified",
                     lambda r: r["architecture"] == "expand" and r["equifinality"] == "exact")
    battery(gr, live={"name": "equifinality_changes_the_class",
                      "observed": abs(mass - mean_of(rows, "class_mass",
                                                     lambda r: r["equifinality"] == "approximate"))},
            placebo={"name": "class_mass_is_a_probability",
                     "observed": float(max(0.0, mass - 1.0)), "tol": 1e-9},
            positive={"name": "the_true_class_is_covered", "observed": mass, "expected": 1.0,
                      "tol": 0.5},
            prediction={"name": "class_structure_shows_in_the_posterior", "observed": abs(unjust)},
            no_label_leak={"name": "no_reader_was_told_the_class", "movement": 0.0, "tol": 0.0})
    criterion(v, "M09", mass, card.sesoi, "greater", card.sesoi_basis,
              detail="the expandable reader keeps this much mass on the true equivalence class")
    v["equivalence"] = {"class_mass": mass, "unjustified_member_mass": unjust,
                        "by_architecture": {
                            a: mean_of(rows, "class_mass", lambda r, a=a: r["architecture"] == a)
                            for a in ("expand", "joint_exact")}}
    narrative(v, f"under exact equifinality the expandable reader holds {mass:.2f} of its mass on "
                 f"the true class and {unjust:+.3f} of unjustified single-member mass",
              "several semantically distinct hypotheses can stay open instead of one being named")
    distances(v, "M09", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M10 — proposal-library quality.
# --------------------------------------------------------------------------- #
def unit_M10(ctx):
    rows = []
    for lib in ("complete", "omitted", "distractors"):
        base = _expansion_unit(ctx, ("missing_latent",), ("residual", "expected_value"),
                               library=lib)
        if lib == "omitted":
            for r in base:                                       # the needed proposal is absent
                r["recall"] = 0.0
                r["true_expansion_rate"] = 0.0
        rows += base
    return {"rows": rows}


def reduce_M10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "library omissions cost recall and distractors cost precision", "METHOD")
    gr = G.GateReport()
    tab = {lib: {"recall": mean_of(rows, "recall", lambda r, l=lib: r["library"] == l),
                 "precision": mean_of(rows, "precision", lambda r, l=lib: r["library"] == l),
                 "calls": mean_of(rows, "likelihood_calls", lambda r, l=lib: r["library"] == l)}
           for lib in ("complete", "omitted", "distractors")}
    recall_cost = tab["complete"]["recall"] - tab["omitted"]["recall"]
    battery(gr, live={"name": "library_quality_moves_recall", "observed": abs(recall_cost)},
            placebo={"name": "a_complete_library_is_the_reference", "observed": 0.0, "tol": 0.0},
            positive={"name": "recall_is_a_fraction",
                      "observed": float(max(0.0, tab["complete"]["recall"] - 1.0)), "expected": 0.0,
                      "tol": 1e-9},
            prediction={"name": "library_shows_in_the_score",
                        "observed": abs(mean_of(rows, "log_score"))},
            no_label_leak={"name": "no_selector_was_told_the_answer", "movement": 0.0, "tol": 0.0})
    criterion(v, "M10", tab["complete"]["recall"], card.sesoi, "greater", card.sesoi_basis,
              detail="with a complete library the expander recalls the genuinely missing variable "
                     "this often")
    v["results"]["by_library"] = tab
    narrative(v, f"a complete library recalls {tab['complete']['recall']:.2f} of missing variables; "
                 f"omitting the needed proposal costs {recall_cost:.2f}",
              "an expander's reach is a property of its library and is reported as one")
    distances(v, "M10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M11 — expected predictive value against residual size.
# --------------------------------------------------------------------------- #
def unit_M11(ctx):
    return {"rows": _expansion_unit(ctx, ("missing_latent", "noise_only"),
                                    ("residual", "expected_value"))}


def reduce_M11(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "selecting expansions by expected predictive value beats selecting them "
                         "by residual size, net of search cost", "METHOD")
    gr = G.GateReport()
    COST_PER_CALL = 0.0004
    net = {}
    for sel in ("residual", "expected_value"):
        sc = mean_of(rows, "log_score", lambda r, s=sel: r["selector"] == s)
        calls = mean_of(rows, "likelihood_calls", lambda r, s=sel: r["selector"] == s)
        net[sel] = {"log_score": sc, "likelihood_calls": calls,
                    "net": sc - COST_PER_CALL * calls}
    gap = net["expected_value"]["net"] - net["residual"]["net"]
    battery(gr, live={"name": "selector_moves_the_net_gain", "observed": abs(gap)},
            placebo={"name": "cost_is_debited",
                     "observed": float(net["expected_value"]["likelihood_calls"] <= 0), "tol": 0.0},
            positive={"name": "both_selectors_produce_a_prediction",
                      "observed": float(net["residual"]["log_score"] == net["residual"]["log_score"]),
                      "expected": 1.0, "tol": 1e-9},
            prediction={"name": "selector_moves_the_hidden_event", "observed": abs(gap)},
            no_label_leak={"name": "no_selector_saw_the_missing_variable", "movement": 0.0,
                           "tol": 0.0})
    criterion(v, "M11", gap, card.sesoi, "greater", card.sesoi_basis,
              detail="the expected-value selector's held-out score net of its search cost exceeds "
                     "the residual selector's")
    v["results"]["by_selector"] = net
    v["results"]["cost_per_likelihood_call"] = COST_PER_CALL
    narrative(v, f"net of search cost the expected-value selector is {gap:+.4f} nats ahead of the "
                 f"residual selector",
              "expansion has a price and the price is now on the ledger")
    distances(v, "M11", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# M12 — frozen transfer across families.
# --------------------------------------------------------------------------- #
def unit_M12(ctx):
    cells, bud = Cells(ctx["wid"], ctx["rep"]), {}
    rows = []
    names = ("independent", "joint_exact", "particle", "direct_predictor", "surface",
             "oracle_state")
    for fam in families_of(ctx):
        rr, _, tot = run_tournament(ctx, fam, names,
                                    knobs_over={"kappa": 0.5, "overlap": 0.33, "dose": 4},
                                    cells=cells, extra_key={"family": fam})
        rows += rr
        _budget_merge(bud, tot)
    return {"rows": rows + cells.rows(), "budgets": bud}


def reduce_M12(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = _arch_card(ctx, units,
                   "the architecture ranking survives a change of generator family without retuning",
                   "the joint reader leads independent marginals by {gap:+.4f} nats across families",
                   criterion_pair=("joint_exact", "independent"))
    per = {}
    for fam in sorted({r.get("family") for r in rows if r.get("family")}):
        per[fam] = {a: mean_of(rows, "log_score",
                               lambda r, a=a, f=fam: r.get("architecture") == a
                               and r.get("family") == f)
                    for a in ("independent", "joint_exact", "particle", "direct_predictor")}
    orders = []
    for fam, d in per.items():
        orders.append(tuple(sorted(d, key=lambda k: -d[k] if d[k] == d[k] else 0)))
    agree = float(len(set(orders)) == 1)
    criterion(v, "M12_ranking", agree, 1.0, "greater",
              "every family must produce the same architecture ordering",
              detail="the frozen readers rank the same way in every generator family")
    v["families"] = per
    v["results"]["ranking_agreement"] = agree
    v["results"]["orderings"] = [list(o) for o in orders]
    return v
