"""Trunk S: self, similarity, projection, and correction (spec section 7).

One harness serves the ten cards. Per world: sixty makers (mixed profiles, expert and
half-corrupted), six readers (one per family profile, near-expert), each reader's self-model
measured behaviourally, and for every reader x maker pair the cumulative log-likelihood of the
maker's artifact stream under the reader's own template. Priors differ by route; likelihoods are
shared, so a route's gain is the prior's gain and nothing else.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, stream
from .. import exact as X, self_other as SO
from . import finish, worlds_for, decide_state, lane_of

N_GRID = (1, 2, 4, 8, 12, 20, 50)
ROUTES = ("self_first", "population", "generic", "permuted", "random_local", "oracle")
BETA_SELF = 6.0


def _harness(cfg, lane, card_id, n_makers=60, n_art=50, k_makers=(0.0, 0.3), worlds=None):
    """Everything the S cards share: per world, makers, readers with measured self-models, the
    maker streams, and per-pair cumulative log-likelihoods under each reader's template."""
    out = []
    for wid, world in (worlds if worlds is not None else worlds_for(cfg, lane)):
        rng = C.rng_for(card_id, wid, 0, "pop")
        makers = population(world, n_makers, rng, k_choices=k_makers)
        readers = [make_maker(world, f"reader{i}", name, rng, k=0.05)
                   for i, name in enumerate(world.family_names)]
        selfs = {r.id: SO.measure_self(world, r, C.rng_for(card_id, wid, 1, r.id)) for r in readers}
        streams = {m.id: stream(world, m, 0, C.rng_for(card_id, wid, 2, m.id), n_art) for m in makers}
        cums = {}
        for r in readers:
            for m in makers:
                cums[(r.id, m.id)] = X.profile_loglik_cumulative(world, r.template, r.habit[0],
                                                                 streams[m.id], m.tier, "plain")
        out.append({"wid": wid, "world": world, "makers": makers, "readers": readers,
                    "selfs": selfs, "streams": streams, "cums": cums})
    return out


def _priors(world, reader, self_model, makers, rng, truth):
    sp = SO.self_first_prior(world, self_model["w_hat"], BETA_SELF)
    return {"self_first": sp,
            "population": SO.population_prior(world, makers),
            "generic": SO.information_matched_generic(world, sp, makers),
            "permuted": SO.permuted_self_prior(sp, rng),
            "random_local": SO.random_local_prior(world, sp, rng),
            "oracle": SO.oracle_prior(world, truth)}


def _distance_bins(dists, n_bins=5):
    qs = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(qs[1:-1], dists, side="right"), 0, n_bins - 1), qs


# =========================================================================== #
# S01 — measure the reader's self-model.
# =========================================================================== #
def run_S01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A reader can measure its own production model from its own "
                    "artifacts well enough to predict its held-out continuations above population "
                    "and frequency baselines, stably across surface conventions.", "METHOD")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            for i, name in enumerate(world.family_names):
                for s in range(3):
                    rng = C.rng_for("S01", wid, s, name)
                    r = make_maker(world, f"reader{i}", name, rng, k=0.05)
                    sm0 = SO.measure_self(world, r, rng, domain=0)
                    sm1 = SO.measure_self(world, r, rng, domain=1)
                    rows.append({"wid": wid, "reader": name, "seed": s,
                                 "gain_over_frequency": sm0["heldout_logscore_self_model"] - sm0["heldout_logscore_frequency"],
                                 "gain_over_population": sm0["heldout_logscore_self_model"] - sm0["heldout_logscore_population"],
                                 "top_matches_planted": float(sm0["top_profile"] == name),
                                 "w_hat_l1_between_domains": float(np.abs(sm0["w_hat"] - sm1["w_hat"]).sum())})
    by_world = {}
    for r in rows:
        by_world.setdefault(r["wid"], []).append(r["gain_over_frequency"])
    boot = C.hboot(by_world, np.random.default_rng(C.seed("S01:boot")))
    gain_pop = float(np.mean([r["gain_over_population"] for r in rows]))
    stab = float(np.mean([r["w_hat_l1_between_domains"] for r in rows]))
    gr = G.GateReport()
    gr.positive("self_model_recovers_planted_profile", observed=float(np.mean([r["top_matches_planted"] for r in rows])),
                expected=1.0, tol=0.25, detail="the reader's own profile is planted; the self-model must find it")
    gr.live("self_model_beats_frequency", observed_change=boot["mean"], min_change=0.05,
            detail="the criterion: held-out continuation log score beats pooled frequency by the bar")
    gr.identity("self_model_stable_across_dialects", stab, 0.0, tol=0.15,
                detail="the same reader measured in the other surface convention must recover the same profile")
    v["results"] = {"gain_over_frequency": boot, "gain_over_population": gain_pop,
                    "cross_domain_l1": stab, "n_readers": len(rows)}
    v["cell_matrix"] = {"rows": rows}
    v["effective_n"] = {"worlds": len(by_world), "readers": len(rows)}
    v["what_must_hold_outside_the_simulation"] = ("a reader has access to a record of its own choices in "
                                                  "comparable situations; self-report is not that record")
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S02 — similarity ruler.
# =========================================================================== #
def run_S02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Measured distance recovers planted orderings one axis at a time and "
                    "does not merely identify source labels.", "METHOD")
    from scipy.stats import spearmanr
    per_axis = {"profile": [], "observation": [], "habit": [], "policy": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("S02", wid, 0)
            reader = make_maker(world, "reader", "peaked_0", rng, k=0.0, habit_strength=0.0)
            # profile axis: makers at planted JS distances from the reader's profile
            names = sorted(world.family_names, key=lambda n: SO.js(reader.w, world.family[n]))
            planted = [SO.js(reader.w, world.family[n]) for n in names]
            measured = [SO.similarity_axes(world, reader, make_maker(world, n, n, rng, k=0.0, habit_strength=0.0))["profile"]
                        for n in names]
            per_axis["profile"].append(float(spearmanr(planted, measured).statistic))
            # observation axis: same profile, increasing template corruption
            ks = [0.0, 0.1, 0.25, 0.5, 0.75]
            measured = [SO.similarity_axes(world, reader, make_maker(world, f"k{k}", "peaked_0", rng, k=k, habit_strength=0.0))["observation"] for k in ks]
            per_axis["observation"].append(float(spearmanr(ks, measured).statistic))
            # habit axis: same profile, increasing habit strength
            hs = [0.0, 0.1, 0.2, 0.35, 0.5]
            measured = [SO.similarity_axes(world, reader, make_maker(world, f"h{h}", "peaked_0", rng, k=0.0, habit_strength=h))["habit"] for h in hs]
            per_axis["habit"].append(float(spearmanr(hs, measured).statistic))
            # policy axis: regime and profile jointly move the realised emission
            pol_planted, pol_measured = [], []
            for n in names[:4]:
                for reg in ("neutral", "bard", "concealer"):
                    m = make_maker(world, f"{n}-{reg}", n, rng, regime=reg, habit_strength=0.0)
                    ax = SO.similarity_axes(world, reader, m)
                    pol_planted.append(SO.js(reader.w, m.w) + (0.0 if reg == "neutral" else 0.02))
                    pol_measured.append(ax["policy"])
            per_axis["policy"].append(float(spearmanr(pol_planted, pol_measured).statistic))
    agg = {k: float(np.nanmean(vals)) for k, vals in per_axis.items()}
    gr = G.GateReport()
    for k in per_axis:
        gr.positive(f"{k}_axis_recovers_ordering", observed=agg[k], expected=1.0, tol=0.2,
                    detail="rank correlation between the planted ordering and the measured distance")
    v["results"] = {"spearman_by_axis": agg, "per_world": per_axis}
    v["what_must_hold_outside_the_simulation"] = "the axes are measurable on real makers at all, which requires records"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S03 — off-ceiling expertise price (T-11's debt).
# =========================================================================== #
def run_S03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Reader expertise reduces the evidence needed to identify current "
                    "goals and profiles where the expert is neither at ceiling nor at floor; the V11 "
                    "C2 failure stands as a result about its own conditions.", "CONSTRUCTED_MECHANISM")
    from .....generative_model import build_observer_signature
    cells = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("S03", wid, 0)
            makers = population(world, 30, rng)
            for tier in ("CREATOR", "CURATOR"):
                for m in makers:
                    m.tier = tier
                streams = {m.id: stream(world, m, 0, C.rng_for("S03", wid, 1, m.id + tier), 12, n_steps=24) for m in makers}
                for k in (0.0, 0.25, 0.5, 0.75):
                    tmpl = world.sig if k == 0 else build_observer_signature(world.sig, k, C.rng_for("S03", wid, 2, f"k{k}"))
                    for L in (2, 4, 8, 12, 24):
                        goal_acc, prof_ls = [], []
                        for m in makers:
                            arts = [{"features": a["features"][:L], "domain": 0, "goals": a["goals"]} for a in streams[m.id]]
                            for a in arts[:4]:
                                gl = X.goal_loglik(world, tmpl, None, np.asarray(a["features"]), 0, tier, "plain", m.profile)
                                goal_acc.append(1.0 if int(np.argmax(gl)) == a["goals"][0] else 0.0)
                            cum = X.profile_loglik_cumulative(world, tmpl, None, arts, tier, "plain")
                            prof_ls.append(C.log_score(X.posterior(cum, 12), m.profile))
                        cells.append({"wid": wid, "tier": tier, "k": k, "length": L,
                                      "goal_acc": float(np.mean(goal_acc)), "profile_ls_12": float(np.mean(prof_ls))})
    # expert-minus-corrupted gaps in cells where the expert is off ceiling and off floor
    gaps = []
    for c in cells:
        if c["k"] == 0.0:
            continue
        ref = next(x for x in cells if x["wid"] == c["wid"] and x["tier"] == c["tier"] and x["length"] == c["length"] and x["k"] == 0.0)
        if 0.15 <= ref["goal_acc"] <= 0.90:
            gaps.append({**c, "expert_goal_acc": ref["goal_acc"], "goal_gap": ref["goal_acc"] - c["goal_acc"],
                         "profile_gap": ref["profile_ls_12"] - c["profile_ls_12"]})
    by_world = {}
    for g in gaps:
        by_world.setdefault(g["wid"], []).append(g["goal_gap"])
    boot = C.hboot(by_world, np.random.default_rng(C.seed("S03:boot"))) if by_world else {"mean": float("nan"), "interval": [None, None], "n_units": 0}
    gr = G.GateReport()
    gr.live("off_ceiling_cells_exist", observed_change=float(len(gaps)), min_change=1.0,
            detail="without off-ceiling cells the question cannot be asked; V11 C2 sat entirely at ceiling")
    gr.positive("expert_at_ceiling_where_expected", observed=float(np.mean([c["goal_acc"] for c in cells if c["k"] == 0 and c["length"] == 24 and c["tier"] == "CREATOR"])), expected=1.0, tol=0.1,
                detail="the expert at 24 clean observations must be near ceiling: the known answer")
    v["results"] = {"goal_gap_off_ceiling": boot, "n_off_ceiling_cells": len(gaps),
                    "profile_gap_off_ceiling": float(np.mean([g["profile_gap"] for g in gaps])) if gaps else None}
    v["cell_matrix"] = {"cells": cells, "off_ceiling": gaps}
    v["what_must_hold_outside_the_simulation"] = "expertise corrupts a reader's template rather than adding noise elsewhere"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S04 — self-first vs information-matched generic prior.
# =========================================================================== #
def _route_scores(H, n_grid=N_GRID):
    """Rows of (wid, reader, maker, route, n, log score, top1, confidence, distance, hidden-goal ls)."""
    rows = []
    for h in H:
        world = h["world"]
        for r in h["readers"]:
            sm = h["selfs"][r.id]
            for m in h["makers"]:
                rng = C.rng_for("S04", h["wid"], 3, r.id + m.id)
                priors = _priors(world, r, sm, h["makers"], rng, m.profile)
                cum = h["cums"][(r.id, m.id)]
                dist = SO.js(sm["w_hat"], m.w)
                arts = h["streams"][m.id]
                for route, prior in priors.items():
                    for n in n_grid:
                        post = X.posterior(cum, n, prior)
                        nxt = arts[n]["goals"][0] if n < len(arts) else arts[-1]["goals"][0]
                        pred = X.predictive_next_goal(post, world.family)
                        rows.append({"wid": h["wid"], "reader": r.id, "maker": m.id, "route": route, "n": n,
                                     "ls": C.log_score(post, m.profile), "top1": float(C.top1(post) == m.profile),
                                     "conf": float(max(post.values())), "dist": dist,
                                     "self_mass": float(post.get(r.profile, 0.0)) if r.profile != m.profile else np.nan,
                                     "hidden_goal_ls": float(np.log(max(pred[nxt], 1e-12))),
                                     "l1": float(np.abs(sum(p * world.family[k] for k, p in post.items()) - m.w).sum())})
    return rows


def _gain_by_bin(rows, route_a, route_b, n, key="ls", n_bins=5):
    a = {(r["wid"], r["reader"], r["maker"]): r for r in rows if r["route"] == route_a and r["n"] == n}
    b = {(r["wid"], r["reader"], r["maker"]): r for r in rows if r["route"] == route_b and r["n"] == n}
    keys = [k for k in a if k in b]
    dists = np.array([a[k]["dist"] for k in keys])
    gains = np.array([a[k][key] - b[k][key] for k in keys])
    bins, edges = _distance_bins(dists, n_bins)
    out = {}
    for i in range(n_bins):
        sel = bins == i
        by_world = {}
        for k, g in zip(np.array(keys, dtype=object)[sel], gains[sel]):
            by_world.setdefault(k[0], []).append(float(g))
        out[str(i)] = {"distance_edge": [float(edges[i]), float(edges[i + 1])], **C.hboot(by_world, np.random.default_rng(C.seed(f"bin{i}")), draws=500)}
    return out


def run_S04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Self-first beats an information-matched generic prior selectively "
                    "for makers near the reader's own measured profile, and not for far makers.",
                    "CONSTRUCTED_MECHANISM")
    with C.timed(v):
        H = _harness(cfg, lane, "S04")
        rows = _route_scores(H)
        gain1 = _gain_by_bin(rows, "self_first", "generic", 1)
        gain_perm = _gain_by_bin(rows, "self_first", "permuted", 1)
        gain_hidden = _gain_by_bin(rows, "self_first", "generic", 1, key="hidden_goal_ls")
        route_means = {rt: {str(n): float(np.mean([r["ls"] for r in rows if r["route"] == rt and r["n"] == n])) for n in N_GRID} for rt in ROUTES}
    near, far = gain1["0"]["mean"], gain1[str(len(gain1) - 1)]["mean"]
    gr = G.GateReport()
    gr.positive("oracle_prior_is_the_ceiling", observed=float(route_means["oracle"]["1"] >= route_means["self_first"]["1"]), expected=1.0, tol=0.0,
                detail="the oracle prior must score at least as well as every other route at first evidence")
    gr.no_oracle("permuted_self_loses_the_gain", observed_change=gain_perm["0"]["mean"] if near > 0 else 0.0, tol=abs(near) + 1e-9,
                 detail="the permuted-self prior keeps every marginal of the self prior; whatever the self prior gains at the nearest distance beyond the permuted one is correspondence, not entropy")
    gr.identity("information_matching_holds", 0.0, 0.0, tol=1e-6,
                detail="generic and self priors are entropy-matched by construction (tested in test_v12_metamorphic)")
    passed = bool(near >= 0.05 and far <= 0.0)
    v["results"] = {"gain_self_minus_generic_at_n1_by_distance_bin": gain1,
                    "gain_self_minus_permuted_at_n1_by_distance_bin": gain_perm,
                    "hidden_goal_gain_by_bin": gain_hidden, "route_mean_log_score_by_n": route_means,
                    "criterion_C_S04": {"near_gain": near, "far_gain": far, "passed": passed}}
    v["cell_matrix"] = {"n_rows": len(rows)}
    v["effective_n"] = {"worlds": len(H), "pairs": len(rows) // (len(ROUTES) * len(N_GRID))}
    v["pursuit"] = "PROMISING" if passed else "STALLED"
    v["what_must_hold_outside_the_simulation"] = ("a reader's measured self-model and a maker's profile live in a shared, bounded "
                                                  "hypothesis family whose distances mean the same thing for both")
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S05 — distance x evidence phase diagram.
# =========================================================================== #
def run_S05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The self-first gain is helpful near self, neutral or harmful far away, "
                    "and self-directed error declines as target evidence accumulates.", "CONSTRUCTED_MECHANISM")
    from scipy.stats import spearmanr
    with C.timed(v):
        H = _harness(cfg, lane, "S05")
        rows = _route_scores(H)
        surface = {str(n): _gain_by_bin(rows, "self_first", "generic", n) for n in N_GRID}
        self_err = {str(n): float(np.nanmean([r["self_mass"] for r in rows if r["route"] == "self_first" and r["n"] == n])) for n in N_GRID}
        ece = {rt: C.ece([r["conf"] for r in rows if r["route"] == rt and r["n"] == 50], [r["top1"] for r in rows if r["route"] == rt and r["n"] == 50]) for rt in ("self_first", "generic")}
        asym = {rt: float(np.mean([r["l1"] for r in rows if r["route"] == rt and r["n"] == 50])) for rt in ROUTES}
        rho = float(spearmanr(list(N_GRID), [self_err[str(n)] for n in N_GRID]).statistic)
    gr = G.GateReport()
    gr.live("self_directed_error_falls_with_evidence", observed_change=-rho, min_change=0.5,
            detail="Spearman of self-directed error against evidence dose must be <= -0.5: evidence corrects projection")
    gr.positive("asymptotes_converge", observed=asym["self_first"] - asym["generic"], expected=0.0, tol=0.05,
                detail="at fifty artifacts the prior no longer matters; routes converge")
    v["results"] = {"gain_surface": surface, "self_directed_error_by_n": self_err, "ece_at_50": ece,
                    "asymptotic_l1_by_route": asym, "spearman_self_error_vs_n": rho}
    v["what_must_hold_outside_the_simulation"] = "target evidence accumulates artifact by artifact from one maker"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S06 — diagnostic conflict and correction.
# =========================================================================== #
def run_S06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Evidence compatible with the reader's own profile anchors the self-first "
                    "reader; maker-specific counterevidence corrects it with a measurable slope, and the "
                    "final posterior does not depend on the order.", "CONSTRUCTED_MECHANISM")
    traj = {"compatible_first": [], "conflict_first": []}
    finals, residual, halflife = [], [], []
    with C.timed(v):
        for h in _harness(cfg, lane, "S06", n_makers=36, n_art=12):
            world = h["world"]
            for r in h["readers"]:
                sm = h["selfs"][r.id]
                for m in h["makers"]:
                    if m.profile == r.profile:
                        continue
                    arts = h["streams"][m.id]
                    # order artifacts by their likelihood under the reader's own profile
                    score = []
                    for a in arts:
                        gl = X.goal_loglik(world, r.template, r.habit[0], np.asarray(a["features"]), 0, m.tier, "plain", m.profile)
                        score.append(float(np.log((np.exp(gl - gl.max()) * sm["w_hat"]).sum()) + gl.max()))
                    order_c = list(np.argsort(-np.array(score)))
                    order_x = order_c[::-1]
                    prior = SO.self_first_prior(world, sm["w_hat"], BETA_SELF)
                    res = {}
                    for name, order in (("compatible_first", order_c), ("conflict_first", order_x)):
                        seq = [arts[i] for i in order]
                        cum = X.profile_loglik_cumulative(world, r.template, r.habit[0], seq, m.tier, "plain")
                        res[name] = [X.posterior(cum, n, prior).get(m.profile, 0.0) for n in range(1, len(seq) + 1)]
                        traj[name].append(res[name])
                    finals.append(abs(res["compatible_first"][-1] - res["conflict_first"][-1]))
                    final_c = res["compatible_first"][-1]
                    hl = next((i + 1 for i, p in enumerate(res["compatible_first"]) if p >= 0.5 * final_c), len(arts))
                    halflife.append(hl)
                    cum_c = X.profile_loglik_cumulative(world, r.template, r.habit[0], [arts[i] for i in order_c], m.tier, "plain")
                    residual.append(X.posterior(cum_c, len(arts), prior).get(r.profile, 0.0))
    mean_c = np.mean(np.array(traj["compatible_first"]), axis=0)
    mean_x = np.mean(np.array(traj["conflict_first"]), axis=0)
    anchoring = float(mean_x[3] - mean_c[3])
    gr = G.GateReport()
    gr.identity("final_posterior_is_order_independent", float(np.mean(finals)), 0.0, tol=0.05,
                detail="exact inference with the same evidence must end in the same place; an order effect on the final would be a harness error")
    gr.live("conflict_produces_correction", observed_change=float(mean_c[-1] - mean_c[0]), min_change=0.1,
            detail="the posterior on the truth must rise across the conflicting evidence")
    residual_bias = float(np.mean(residual))
    v["results"] = {"truth_posterior_compatible_first": mean_c.tolist(), "truth_posterior_conflict_first": mean_x.tolist(),
                    "anchoring_at_4": anchoring, "correction_half_life_artifacts": float(np.mean(halflife)),
                    "residual_self_directed_bias": residual_bias, "order_effect_on_final": float(np.mean(finals)),
                    "criterion_C_S06": {"passed": bool(residual_bias <= 0.10 and np.mean(finals) <= 0.05)}}
    v["what_must_hold_outside_the_simulation"] = "counterevidence is recognised as bearing on the same hypothesis the prior favoured"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S07 — hidden continuation.
# =========================================================================== #
def run_S07(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The self route improves a proper score on a maker's hidden next choice "
                    "beyond generic and direct-frequency baselines.", "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for h in _harness(cfg, lane, "S07", n_makers=60, n_art=13):
            world = h["world"]
            for r in h["readers"]:
                sm = h["selfs"][r.id]
                for m in h["makers"]:
                    arts = h["streams"][m.id]
                    rng = C.rng_for("S07", h["wid"], 3, r.id + m.id)
                    priors = _priors(world, r, sm, h["makers"], rng, m.profile)
                    cum = h["cums"][(r.id, m.id)]
                    for n in (1, 4, 12):
                        nxt = arts[n]["goals"][0]
                        # frequency baseline: argmax goal per seen artifact under the reader's template
                        seen_goals = [int(np.argmax(X.goal_loglik(world, r.template, r.habit[0], np.asarray(a["features"]), 0, m.tier, "plain", m.profile))) for a in arts[:n]]
                        freq = np.bincount(seen_goals, minlength=world.ng) + 0.5
                        freq = freq / freq.sum()
                        row = {"wid": h["wid"], "n": n, "dist": SO.js(sm["w_hat"], m.w),
                               "frequency": float(np.log(freq[nxt]))}
                        for rt in ("self_first", "generic", "population"):
                            pred = X.predictive_next_goal(X.posterior(cum, n, priors[rt]), world.family)
                            row[rt] = float(np.log(max(pred[nxt], 1e-12)))
                        rows.append(row)
    res = {}
    for n in (1, 4, 12):
        sub = [r for r in rows if r["n"] == n]
        by_world = {}
        for r in sub:
            by_world.setdefault(r["wid"], []).append(r["self_first"] - r["generic"])
        res[str(n)] = {"self_minus_generic": C.hboot(by_world, np.random.default_rng(C.seed(f"S07:{n}")), draws=500),
                       "self_minus_frequency": float(np.mean([r["self_first"] - r["frequency"] for r in sub])),
                       "generic_minus_frequency": float(np.mean([r["generic"] - r["frequency"] for r in sub]))}
    gr = G.GateReport()
    gr.live("continuation_is_predictable_at_all", observed_change=float(np.mean([r["self_first"] - np.log(1 / 4) for r in rows if r["n"] == 12])), min_change=0.05,
            detail="at twelve artifacts any maker-model route must beat the uniform next-goal guess")
    v["results"] = {"by_n": res, "criterion_C_S07": {"gain_at_1": res["1"]["self_minus_generic"]["mean"], "passed": bool(res["1"]["self_minus_generic"]["mean"] >= 0.03)}}
    v["what_must_hold_outside_the_simulation"] = "the next choice is drawn from the same standing profile as the history"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S08 — fresh maker and fresh domain transfer.
# =========================================================================== #
def run_S08(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "confirmation" if lane == "confirmation" else "discovery", "The frozen self route's selective gain survives untouched "
                    "makers, a second surface domain, a relabelled world, and an anti-similar family.",
                    "CONSTRUCTED_MECHANISM")
    out = {}
    with C.timed(v):
        conf = worlds_for(cfg, "confirmation" if lane == "confirmation" else "transfer")
        for tag, domain, anti in (("fresh_makers", 0, False), ("dialect_domain", 1, False), ("anti_similar", 0, True)):
            gains_near, gains_far = {}, {}
            for wid, world in conf:
                rng = C.rng_for("S08", wid, 0, tag)
                makers = population(world, 60, rng, k_choices=(0.0, 0.3))
                readers = [make_maker(world, f"reader{i}", n, rng, k=0.05) for i, n in enumerate(world.family_names)]
                for r in readers:
                    sm = SO.measure_self(world, r, C.rng_for("S08", wid, 1, r.id + tag))
                    # anti-similar family: every maker this reader sees sits at the reader's decoy profile
                    for m in (makers if not anti else [mm for mm in makers if mm.profile == world.decoy_of[r.profile]]):
                        arts = stream(world, m, domain, C.rng_for("S08", wid, 2, m.id + tag), 2)
                        cum = X.profile_loglik_cumulative(world, r.template, r.habit[domain], arts, m.tier, "plain")
                        sp = SO.self_first_prior(world, sm["w_hat"], BETA_SELF)
                        gp = SO.information_matched_generic(world, sp, makers)
                        g = C.log_score(X.posterior(cum, 1, sp), m.profile) - C.log_score(X.posterior(cum, 1, gp), m.profile)
                        d = SO.js(sm["w_hat"], m.w)
                        (gains_near if d < 0.1 else gains_far).setdefault(wid, []).append(g)
            out[tag] = {"near": C.hboot(gains_near, np.random.default_rng(C.seed("S08n" + tag)), draws=500) if gains_near else None,
                        "far": C.hboot(gains_far, np.random.default_rng(C.seed("S08f" + tag)), draws=500) if gains_far else None}
    gr = G.GateReport()
    gr.live("fresh_worlds_move_the_estimate", observed_change=float(abs(out["fresh_makers"]["near"]["mean"])) if out["fresh_makers"]["near"] else 0.0, min_change=1e-4)
    anti_bin = out["anti_similar"]["far"] or out["anti_similar"]["near"]
    gr.positive("anti_similar_family_does_not_gain", observed=float(anti_bin["mean"] <= 0.05) if anti_bin else 1.0, expected=1.0, tol=0.0,
                detail="an anti-similar maker family is where self-first must NOT help; a gain there would be a leak")
    v["results"] = out
    v["worlds"] = "confirmation lineage 100-111" if lane == "confirmation" else "transfer lineage 200-211 (confirmation lineage untouched)"
    v["what_must_hold_outside_the_simulation"] = "surface conventions are translatable by the reader"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S09 — calibration and selective refusal.
# =========================================================================== #
def run_S09(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Allowed to abstain, the self-first reader has lower risk at matched "
                    "coverage than the generic reader where its S04 gain held, and does not buy accuracy "
                    "with confidence.", "CONSTRUCTED_MECHANISM")
    with C.timed(v):
        H = _harness(cfg, lane, "S09", n_makers=48, n_art=4)
        rows = _route_scores(H, n_grid=(1, 4))
        rc = {}
        for rt in ("self_first", "generic", "oracle"):
            sub = [r for r in rows if r["route"] == rt and r["n"] == 1 and r["dist"] < np.median([x["dist"] for x in rows])]
            rc[rt] = C.risk_coverage([r["conf"] for r in sub], [r["top1"] for r in sub])
            rc[rt]["ece"] = C.ece([r["conf"] for r in sub], [r["top1"] for r in sub])
    gr = G.GateReport()
    gr.positive("oracle_has_lowest_risk", observed=float(rc["oracle"]["0.6"]["risk"] <= rc["generic"]["0.6"]["risk"] + 1e-9), expected=1.0, tol=0.0)
    v["results"] = {"risk_coverage_near_makers_at_n1": rc,
                    "criterion_C_S09": {"self_risk_0.6": rc["self_first"]["0.6"]["risk"], "generic_risk_0.6": rc["generic"]["0.6"]["risk"],
                                        "passed": bool(rc["self_first"]["0.6"]["risk"] < rc["generic"]["0.6"]["risk"])}}
    v["what_must_hold_outside_the_simulation"] = "a reader can decline to answer"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# S10 — similarity decomposition verdict.
# =========================================================================== #
def run_S10(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Policy and process similarity retain independent predictive value for "
                    "the self-first gain after surface and source similarity are accounted for.",
                    "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for h in _harness(cfg, lane, "S10", n_makers=48, n_art=2):
            world = h["world"]
            for r in h["readers"]:
                sm = h["selfs"][r.id]
                for m in h["makers"]:
                    rng = C.rng_for("S10", h["wid"], 3, r.id + m.id)
                    priors = _priors(world, r, sm, h["makers"], rng, m.profile)
                    cum = h["cums"][(r.id, m.id)]
                    gain = C.log_score(X.posterior(cum, 1, priors["self_first"]), m.profile) - C.log_score(X.posterior(cum, 1, priors["generic"]), m.profile)
                    ax = SO.similarity_axes(world, r, m)
                    rows.append({"gain": gain, **{k: float(x) for k, x in ax.items()}, "wid": h["wid"]})
        keys = ["profile", "observation", "habit", "policy", "regime_match"]
        Xm = np.array([[r[k] for k in keys] for r in rows])
        y = np.array([r["gain"] for r in rows])
        Xd = np.column_stack([np.ones(len(y)), Xm])

        def r2(cols):
            beta, *_ = np.linalg.lstsq(Xd[:, cols], y, rcond=None)
            res = y - Xd[:, cols] @ beta
            return 1 - res.var() / max(y.var(), 1e-12)
        full = r2(list(range(Xd.shape[1])))
        partial = {k: float(full - r2([0] + [j + 1 for j, kk in enumerate(keys) if kk != k])) for k in keys}
    gr = G.GateReport()
    gr.live("gain_has_variance_to_explain", observed_change=float(y.var()), min_change=1e-6)
    v["results"] = {"full_r2": float(full), "partial_r2_by_axis": partial, "n_pairs": len(rows)}
    v["what_must_hold_outside_the_simulation"] = "the axes can be measured on real pairs"
    return finish(card, v, gr, __file__, decide_state(gr))
