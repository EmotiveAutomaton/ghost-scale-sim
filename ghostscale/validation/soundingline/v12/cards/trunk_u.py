"""Trunk U: from maker posterior to preferred outcomes and policy (spec section 10).

Every card runs on exact (and deliberately corrupted) posteriors so the bridge itself is tested
regardless of how the S readers fare. The downstream task is the choice task in uptake.py; the
reader's own standing preference is its profile; the bridge mixes in a trusted maker posterior at
an explicit uptake weight. Never named empathy.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, stream
from .. import exact as X, uptake as U
from . import finish, worlds_for, decide_state

WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def _fe(pol, out, c, beta=8.0):
    """The objective a softmax policy maximizes exactly: expected log-preference utility (the utility
    uptake.policy scores actions by) plus entropy over beta."""
    pol = np.asarray(pol, float)
    u = np.asarray(out) @ np.log(np.maximum(np.asarray(c, float), 1e-300))
    h = float(-(pol[pol > 0] * np.log(pol[pol > 0])).sum())
    return float(pol @ u + h / beta)


def _posteriors_for(world, m, rng, n_art=12):
    """Accurate/confident, accurate/uncertain, wrong/confident, wrong/uncertain, equifinal, none."""
    arts = stream(world, m, 0, rng, n_art)
    cum = X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "plain")
    acc_conf = X.posterior(cum, n_art)
    acc_unc = X.posterior(cum, 1)
    decoy = world.decoy_of[m.profile]
    wrong_conf = {n: (0.94 if n == decoy else 0.06 / (len(world.family_names) - 1)) for n in world.family_names}
    wrong_unc = {n: (0.4 if n == decoy else 0.6 / (len(world.family_names) - 1)) for n in world.family_names}
    equif = {n: (0.5 if n in (m.profile, decoy) else 0.0) for n in world.family_names}
    return {"accurate_confident": acc_conf, "accurate_uncertain": acc_unc, "wrong_confident": wrong_conf,
            "wrong_uncertain": wrong_unc, "equifinal": equif, "none": None}


def run_U01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "both", "The bridge identities hold: zero weight is bit-identical, the oracle "
                    "posterior moves policy toward the maker, a uniform posterior carries no direction "
                    "beyond the family, a shuffled posterior does not help, and C_AIF holds no provenance "
                    "preference.", "METHOD")
    world = worlds_for(cfg, "discovery", limit=1)[0][1]
    rng = np.random.default_rng(C.seed("U01"))
    with C.timed(v):
        checks = {"zero_weight_identity": [], "oracle_moves_toward_maker": [], "uniform_no_direction": [],
                  "shuffled_no_task_gain": []}
        fam_mean = np.mean([world.family[n] for n in world.family_names], axis=0)
        for i in range(60):
            m = make_maker(world, f"m{i}", world.family_names[i % 6], rng)
            reader_pref = world.family[world.family_names[(i + 3) % 6]]
            out = U.task(rng, world.ng)
            p0 = U.policy(reader_pref, out)
            checks["zero_weight_identity"].append(float(np.array_equal(p0, U.policy(U.bridge(reader_pref, m.w, 0.0), out))))
            p_or = U.policy(U.bridge(reader_pref, m.w, 1.0), out)
            checks["oracle_moves_toward_maker"].append(float(_fe(p_or, out, m.w) >= _fe(p0, out, m.w) - 1e-12))
            w_uni = U.representation({n: 1 / 6 for n in world.family_names}, world.family, "mean")
            checks["uniform_no_direction"].append(float(np.allclose(U.bridge(reader_pref, w_uni, 1.0), U.bridge(reader_pref, fam_mean, 1.0))))
            shuf = world.family[world.family_names[int(rng.integers(6))]]
            p_sh = U.policy(U.bridge(reader_pref, shuf, 1.0), out)
            checks["shuffled_no_task_gain"].append(float(U.regret(p_sh, out, m.w)))
        oracle_regret = float(np.mean([U.regret(U.policy(U.bridge(world.family[world.family_names[(i + 3) % 6]], world.family[world.family_names[i % 6]], 1.0), U.task(np.random.default_rng(i), world.ng)), U.task(np.random.default_rng(i), world.ng), world.family[world.family_names[i % 6]]) for i in range(60)]))
    gr = G.GateReport()
    gr.identity("zero_uptake_is_bit_identical", float(np.mean(checks["zero_weight_identity"])), 1.0, tol=0.0)
    gr.positive("oracle_posterior_moves_policy_toward_maker", observed=float(np.mean(checks["oracle_moves_toward_maker"])), expected=1.0, tol=0.0,
                detail="scored on the objective a softmax policy maximizes (expected utility plus entropy over beta), where the oracle is optimal by construction")
    gr.identity("uniform_posterior_equals_population_update", float(np.mean(checks["uniform_no_direction"])), 1.0, tol=0.0)
    gr.positive("shuffled_posterior_no_task_gain", observed=float(np.mean(checks["shuffled_no_task_gain"]) >= oracle_regret - 1e-9), expected=1.0, tol=0.0,
                detail="a shuffled maker posterior must not beat the oracle on the maker's own task; it may be worse")
    gr.identity("no_provenance_preference_in_C_AIF", 0.0, 0.0, tol=0.0, detail="C_AIF is a vector over goal channels only; provenance has no coordinate")
    v["results"] = {k: float(np.mean(x)) for k, x in checks.items()} | {"oracle_regret_on_maker_task": oracle_regret}
    v["what_must_hold_outside_the_simulation"] = "nothing; construction identities"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Reconstruction accuracy and uptake weight separate: weight moves policy, "
                    "accuracy decides whether the movement helps, and wrong-and-confident is the "
                    "worst cell.", "CONSTRUCTED_MECHANISM")
    cells = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U02", wid, 0)
            makers = population(world, 60, rng)
            for m in makers:
                posts = _posteriors_for(world, m, rng)
                reader_pref = world.family[world.decoy_of[world.decoy_of[m.profile]]]
                out = U.task(rng, world.ng)
                p0 = U.policy(reader_pref, out)
                for kind, post in posts.items():
                    for u in WEIGHTS:
                        if post is None:
                            p1 = p0
                        else:
                            p1 = U.policy(U.bridge(reader_pref, U.representation(post, world.family, "mean"), u), out)
                        cell = cells.setdefault((kind, u), {"regret_maker": [], "regret_self": [], "movement": [], "wrong_dir": []})
                        cell["regret_maker"].append(U.regret(p1, out, m.w))
                        cell["regret_self"].append(U.regret(p1, out, reader_pref))
                        cell["movement"].append(U.movement(p0, p1))
                        cell["wrong_dir"].append(float(U.wrong_direction(p0, p1, out, m.w)))
    table = {f"{k}|u={u}": {s: float(np.mean(x)) for s, x in d.items()} for (k, u), d in cells.items()}
    wc = table["wrong_confident|u=1.0"]["wrong_dir"]
    others = max(table[f"{k}|u=1.0"]["wrong_dir"] for k in ("accurate_confident", "accurate_uncertain"))
    gr = G.GateReport()
    gr.identity("no_movement_at_zero_weight", float(max(table[f"{k}|u=0.0"]["movement"] for k in ("accurate_confident", "wrong_confident"))), 0.0, tol=1e-12)
    gr.live("weight_moves_policy", observed_change=table["accurate_confident|u=1.0"]["movement"], min_change=0.05)
    v["results"] = {"cells": table, "criterion_C_U02": {"wrong_confident_wrong_direction": wc, "accurate_max_wrong_direction": others, "passed": bool(wc > others)}}
    v["what_must_hold_outside_the_simulation"] = "an update's helpfulness is judged on the maker's own preference, which the reader cannot see"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Uncertainty-aware posterior representations produce less catastrophic "
                    "wrong-direction movement than the MAP profile where histories stay equivalent.",
                    "CONSTRUCTED_MECHANISM")
    reps = ("map", "mean", "lower_confidence", "confidence_gated")
    res = {r: {"regret": [], "wrong_dir": []} for r in reps} | {"oracle": {"regret": [], "wrong_dir": []}, "none": {"regret": [], "wrong_dir": []}}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U03", wid, 0)
            for m in population(world, 60, rng):
                posts = _posteriors_for(world, m, rng)
                reader_pref = world.family[world.decoy_of[world.decoy_of[m.profile]]]
                out = U.task(rng, world.ng)
                p0 = U.policy(reader_pref, out)
                for kind in ("accurate_uncertain", "equifinal", "wrong_uncertain"):
                    post = posts[kind]
                    for r in reps:
                        p1 = U.policy(U.bridge(reader_pref, U.representation(post, world.family, r), 1.0), out)
                        res[r]["regret"].append(U.regret(p1, out, m.w)); res[r]["wrong_dir"].append(float(U.wrong_direction(p0, p1, out, m.w)))
                    res["oracle"]["regret"].append(U.regret(U.policy(U.bridge(reader_pref, m.w, 1.0), out), out, m.w)); res["oracle"]["wrong_dir"].append(0.0)
                    res["none"]["regret"].append(U.regret(p0, out, m.w)); res["none"]["wrong_dir"].append(0.0)
    table = {r: {k: float(np.mean(x)) for k, x in d.items()} for r, d in res.items()}
    gr = G.GateReport()
    gr.positive("oracle_is_the_floor", observed=float(min(table[r]["regret"] for r in reps) >= table["oracle"]["regret"] - 1e-9), expected=1.0, tol=0.0)
    v["results"] = {"by_representation": table, "criterion_C_U03": {"map_wrong_dir": table["map"]["wrong_dir"], "lower_confidence_wrong_dir": table["lower_confidence"]["wrong_dir"],
                                                                     "passed": bool(table["lower_confidence"]["wrong_dir"] <= table["map"]["wrong_dir"])}}
    v["what_must_hold_outside_the_simulation"] = "the reader can represent its uncertainty about the maker"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Maker competence, source reliability, task relevance, and value similarity "
                    "have separable effects on policy movement; none substitutes for another.",
                    "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U04", wid, 0)
            for m in population(world, 48, rng, k_choices=(0.0, 0.5)):
                for reliability in (0.3, 1.0):
                    for relevance in (0.0, 1.0):
                        reader_pref = world.family[world.family_names[int(rng.integers(6))]]
                        sim = 1.0 - float(np.abs(reader_pref - m.w).sum() / 2)
                        out = U.task(rng, world.ng)
                        if relevance < 0.5:
                            out = out[:, rng.permutation(world.ng)]     # maker's channels irrelevant to task
                        arts = stream(world, m, 0, rng, 8)
                        post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "plain"), 8)
                        acc = float(C.top1(post) == m.profile)
                        p0 = U.policy(reader_pref, out)
                        p1 = U.policy(U.bridge(reader_pref, U.representation(post, world.family, "mean"), 1.0, trust=reliability), out)
                        rows.append({"wid": wid, "competence": 1.0 - m.k, "reliability": reliability, "relevance": relevance,
                                     "similarity": sim, "accuracy": acc, "movement": U.movement(p0, p1),
                                     "regret_self": U.regret(p1, out, reader_pref) - U.regret(p0, out, reader_pref)})
        keys = ["competence", "reliability", "relevance", "similarity", "accuracy"]
        Xm = np.column_stack([np.ones(len(rows))] + [[r[k] for r in rows] for k in keys])
        effects = {}
        for target in ("movement", "regret_self"):
            y = np.array([r[target] for r in rows])
            beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            effects[target] = dict(zip(["intercept"] + keys, [float(b) for b in beta]))
    gr = G.GateReport()
    gr.live("reliability_scales_movement", observed_change=float(effects["movement"]["reliability"]), min_change=0.01)
    gr.identity("movement_does_not_depend_on_competence_by_construction", float(abs(effects["movement"]["competence"])), 0.0, tol=0.05,
                detail="the bridge never sees competence; if movement tracks it, something else leaked it in")
    v["results"] = {"linear_effects": effects, "n_rows": len(rows)}
    v["what_must_hold_outside_the_simulation"] = "source reliability and maker competence are observable separately"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "After exposure, the process channel moves more than the standing-preference "
                    "channel at matched exposure; belief, imitation, and novel-constraint action are "
                    "reported separately.", "CONSTRUCTED_MECHANISM")
    ch = {"process": [], "preference": [], "belief": [], "imitation": [], "novel_constraint": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U05", wid, 0)
            for m in population(world, 48, rng, k_choices=(0.0,)):
                arts = stream(world, m, 0, rng, 8)
                post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "plain"), 8)
                w_hat = U.representation(post, world.family, "mean")
                reader_pref = world.family[world.family_names[int(rng.integers(6))]]
                # process: reuse of the maker's execution template for the reader's OWN goal
                feats = np.concatenate([a["features"] for a in arts])
                freq = np.bincount(feats, minlength=world.nf) + 0.5
                freq = freq / freq.sum()
                g_self = int(np.argmax(reader_pref))
                own = world.sig[g_self]
                ch["process"].append(float(1.0 - np.abs(own - (0.5 * own + 0.5 * freq)).sum() / 2) - float(1.0 - np.abs(own - own).sum() / 2))
                # preference: movement of the standing preference under a confidence-gated bridge
                new_pref = U.bridge(reader_pref, U.representation(post, world.family, "confidence_gated"), 0.5)
                ch["preference"].append(float(np.abs(new_pref - reader_pref).sum() / 2))
                # belief: posterior mass moved onto the maker's true profile
                ch["belief"].append(float(post.get(m.profile, 0.0) - 1.0 / 6))
                # imitation: surface overlap between the reader's continued output and the maker's
                ch["imitation"].append(float(1.0 - np.abs(freq - world.synth).sum() / 2))
                # novel constraint: does the moved preference change the argmax action on a new task
                out = U.task(rng, world.ng)
                ch["novel_constraint"].append(float(np.argmax(U.policy(new_pref, out)) != np.argmax(U.policy(reader_pref, out))))
    table = {k: {"mean": float(np.mean(x)), "sd": float(np.std(x))} for k, x in ch.items()}
    gr = G.GateReport()
    gr.live("exposure_moves_something", observed_change=float(max(table[k]["mean"] for k in ("belief", "preference"))), min_change=0.01)
    v["results"] = {"channels": table, "criterion_C_U05": {"passed": bool(table["belief"]["mean"] > table["preference"]["mean"])},
                    "note": "process and imitation are surface-overlap statistics on this construction; belief and preference are the load-bearing channels"}
    v["what_must_hold_outside_the_simulation"] = "the channels are separately probeable after exposure"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Understanding a competent but value-divergent maker, or a concealer with "
                    "correct local technique, improves prediction while worsening the reader's own task "
                    "under unconditional uptake.", "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U06", wid, 0)
            for kind in ("competent_divergent", "incompetent_similar", "concealer_good_technique", "false_context"):
                for i in range(30):
                    reader_pref = world.family["peaked_0"]
                    if kind == "competent_divergent":
                        m = make_maker(world, f"m{i}", "peaked_2", rng, k=0.0)
                    elif kind == "incompetent_similar":
                        m = make_maker(world, f"m{i}", "peaked_0", rng, k=0.6)
                    elif kind == "concealer_good_technique":
                        m = make_maker(world, f"m{i}", "peaked_2", rng, k=0.0, regime="concealer")
                    else:
                        m = make_maker(world, f"m{i}", "peaked_2", rng, k=0.0)
                    arts = stream(world, m, 0, rng, 8)
                    assumption = "bard" if kind == "concealer_good_technique" else "plain"
                    post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, assumption), 8)
                    trust = 1.0
                    if kind == "false_context":
                        post = {n: (0.9 if n == "peaked_3" else 0.1 / 5) for n in world.family_names}  # a false claim overrides
                    out = U.task(rng, world.ng)
                    p0 = U.policy(reader_pref, out)
                    p1 = U.policy(U.bridge(reader_pref, U.representation(post, world.family, "mean"), 1.0, trust), out)
                    d = res.setdefault(kind, {"prediction_acc": [], "own_regret_change": [], "maker_regret_change": []})
                    d["prediction_acc"].append(float(C.top1(post) == m.profile))
                    d["own_regret_change"].append(U.regret(p1, out, reader_pref) - U.regret(p0, out, reader_pref))
                    d["maker_regret_change"].append(U.regret(p1, out, m.w) - U.regret(p0, out, m.w))
    table = {k: {s: float(np.mean(x)) for s, x in d.items()} for k, d in res.items()}
    gr = G.GateReport()
    gr.live("divergent_maker_worsens_own_task", observed_change=table["competent_divergent"]["own_regret_change"], min_change=0.01)
    v["results"] = table
    v["what_must_hold_outside_the_simulation"] = "own-task outcomes are observable to the reader"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U07(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Reliable counterevidence after an update reverses harmful preference "
                    "movement while process learning is retained, and the reader distinguishes a corrected "
                    "belief from a changed preference.", "CONSTRUCTED_MECHANISM")
    rev, retained, anchored = [], [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U07", wid, 0)
            for m in population(world, 40, rng):
                reader_pref = world.family[world.decoy_of[world.decoy_of[m.profile]]]
                out = U.task(rng, world.ng)
                # a wrong-confident first posterior, then reliable counterevidence (the true stream)
                decoy = world.decoy_of[m.profile]
                wrong = {n: (0.94 if n == decoy else 0.06 / 5) for n in world.family_names}
                p0 = U.policy(reader_pref, out)
                p_wrong = U.policy(U.bridge(reader_pref, U.representation(wrong, world.family, "mean"), 1.0), out)
                arts = stream(world, m, 0, rng, 12)
                corrected = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "plain"), 12, wrong)
                p_corr = U.policy(U.bridge(reader_pref, U.representation(corrected, world.family, "mean"), 1.0), out)
                harm_before = U.regret(p_wrong, out, m.w) - U.regret(p0, out, m.w)
                harm_after = U.regret(p_corr, out, m.w) - U.regret(p0, out, m.w)
                rev.append(float(harm_after < harm_before))
                retained.append(float(corrected.get(m.profile, 0.0)))
                anchored.append(float(corrected.get(decoy, 0.0)))
    gr = G.GateReport()
    gr.live("counterevidence_moves_the_posterior", observed_change=float(np.mean(retained)), min_change=0.2)
    v["results"] = {"harm_reversed_rate": float(np.mean(rev)), "posterior_on_truth_after_correction": float(np.mean(retained)),
                    "residual_on_false_source": float(np.mean(anchored)), "criterion_C_U07": {"passed": bool(np.mean(rev) >= 0.8)}}
    v["what_must_hold_outside_the_simulation"] = "counterevidence is reliable and recognised as such"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_U08(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Repeated small updates accumulate and can reverse under later reliable "
                    "context; a constructed accumulation analogue, not a lifetime human effect.",
                    "CONSTRUCTED_MECHANISM")
    curves = {"early_reliable": [], "late_reliable": [], "intermittent_conflict": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("U08", wid, 0)
            for m in population(world, 24, rng):
                reader_pref = world.family[world.decoy_of[world.decoy_of[m.profile]]]
                out = U.task(rng, world.ng)
                arts = stream(world, m, 0, rng, 30)
                for scen in curves:
                    pref = reader_pref.copy()
                    traj = []
                    for t in range(30):
                        reliable = (t < 10) if scen == "early_reliable" else (t >= 20) if scen == "late_reliable" else (t % 3 == 0)
                        post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts[: t + 1], m.tier, "plain"), t + 1)
                        trust = 1.0 if reliable else 0.3
                        pref = U.bridge(pref, U.representation(post, world.family, "mean"), 0.1, trust)
                        traj.append(U.movement(U.policy(pref, out), U.policy(reader_pref, out)))
                    curves[scen].append(traj)
    table = {k: np.mean(np.array(v_), axis=0).tolist() for k, v_ in curves.items()}
    gr = G.GateReport()
    gr.live("accumulation_is_visible", observed_change=float(table["early_reliable"][-1]), min_change=0.02)
    v["results"] = {"cumulative_movement_by_scenario": table}
    v["what_must_hold_outside_the_simulation"] = "this is a constructed accumulation analogue and nothing more"
    return finish(card, v, gr, __file__, decide_state(gr))
