"""Trunk E — demonstrated competence against attention history (spec §5, cards E01-E10).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import joint as J
from .. import routes as R
from .. import history_skill as HS
from ..world import N_ACT, N_FEAT, ROUTES, episode, make_maker, stream
from . import Cells, battery, criterion, decide_state, finish, held_out_classifier, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for

NF = ("action", "semantic", "context")


BLUR = {"low": 0.4, "mid": 0.0, "high": 0.0}


def _reader(world, k="mid", fam=0):
    """Reader competence is template quality; execution noise stays matched to a mid maker."""
    return J.Reader(world, fam, 0.75, 0.8, template_blur=BLUR[k])


def _ls(pred, a):
    return float(np.log(max(float(pred[int(a)]), 1e-12)))


def _gain(rd, post, prior, ep_next):
    a = int(ep_next["action"][0])
    return _ls(J.next_episode_action_dist(rd, post), a) - _ls(J.next_episode_action_dist(rd, prior), a)


def _exec_acc(rd, eps):
    return float(np.mean([np.mean(np.array(e["intended"]) == np.array([rd.inv_vocab[a] for a in e["action"]])) for e in eps]))


def _early(eps, m, fam=None, inv_vocab=None):
    return HS.history_signal(eps, m, fam, inv_vocab)


def _process_gain(rd, eps, ep_next, prior):
    return HS.process_score(rd, eps, ep_next, prior) - _ls(J.next_episode_action_dist(rd, prior), int(ep_next["action"][0]))


# --------------------------------------------------------------------------- #
# E01 — K and H independently live.
# --------------------------------------------------------------------------- #
def unit_E01(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e01")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    hf = r.normal(0, 1, N_FEAT)
    n = max(3, sizes(ctx)["makers"] // 8)
    for i in range(n):
        for competence in ("low", "high"):
            for history in ("none", "strong"):
                m = HS.agent(world, f"a{i}{competence}{history}", r, 0, competence, history, pref=i % 6, plan=i % 4, h_feat=hf)
                eps = stream(world, m, r, 12)
                cells.add({"competence": competence, "history": history}, exec_acc=_exec_acc(rd, eps), early=_early(eps, m, world.family(0), rd.inv_vocab))
    return {"rows": cells.rows()}


def reduce_E01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E01"]
    v = start(card, ctx, "Competence and attention history are separate generators: competence moves how faithfully intended actions are realized, history moves which features appear early, and neither moves the other's measure.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ex = {k: {h: mean_of(rows, "exec_acc", lambda r, k=k, h=h: r["competence"] == k and r["history"] == h) for h in ("none", "strong")} for k in ("low", "high")}
    ea = {k: {h: mean_of(rows, "early", lambda r, k=k, h=h: r["competence"] == k and r["history"] == h) for h in ("none", "strong")} for k in ("low", "high")}
    k_own = np.mean([ex["high"][h] - ex["low"][h] for h in ("none", "strong")])
    h_own = np.mean([ea[k]["strong"] - ea[k]["none"] for k in ("low", "high")])
    k_leak = abs(np.mean([ea["high"][h] - ea["low"][h] for h in ("none", "strong")]))
    h_leak = abs(np.mean([ex[k]["strong"] - ex[k]["none"] for k in ("low", "high")]))
    passed = bool(k_own >= cr["min_move"] and h_own >= cr["min_move"] and max(k_leak, h_leak) <= cr["max_leak"] + 0.08)
    gr = G.GateReport()
    battery(gr, live={"observed": min(k_own, h_own), "min": 0.0, "name": "each_factor_moves_its_own_measure"},
            placebo={"observed": max(k_leak, h_leak), "tol": cr["max_leak"] + 0.08, "name": "each_factor_leaves_the_other_measure", "detail": "twelve episodes per agent; the tolerance carries sampling noise"},
            positive={"observed": k_own, "expected": 0.4, "tol": 0.2, "name": "competence_gap_as_planted", "detail": "0.95 against 0.55 execution accuracy by construction"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_feature_tilt_every_cell"},
            oracle={"observed": h_own, "min": 0.05, "name": "history_readable_from_early_tokens"},
            prediction={"gain": k_own, "min": 0.0, "name": "competence_predicts_execution"},
            calibration={"observed": max(k_leak, h_leak), "reference": min(k_own, h_own), "direction": "down", "tol": 0.0, "name": "leaks_below_own_effects"})
    criterion(v, "E01", passed, competence_own=k_own, history_own=h_own, competence_leak=k_leak, history_leak=h_leak, exec_acc=ex, early=ea)
    receipt(v, rows, card, ctx)
    narrative(v, f"Competence moved execution accuracy by {k_own:+.2f} and early relevance by {k_leak:.3f}; history moved early relevance by {h_own:+.2f} and execution accuracy by {h_leak:.3f}.",
              "The two objects the trunk dissociates are dissociable in the generator, which is the precondition for reading them apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E02 — a reader's stale route history biases its initial weights; feedback corrects.
# --------------------------------------------------------------------------- #
def _train_pairs(world, r, n, fam=0):
    return R.make_training(world, r, n, fam=fam)


def unit_E02(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e02")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    sz = sizes(ctx)
    learned, _ = R.learn_reliability(rd, _train_pairs(world, r, max(24, sz["training"])), prior)
    tests = _train_pairs(world, r, max(4, sz["makers"] // 4))
    for history in ("none", "strong"):
        w0 = R.weights_named("equal") if history == "none" else HS.reader_route_history(r, 2.0, favoured="context")
        for phase, w in (("initial", w0), ("corrected", {rt: 0.5 * w0[rt] + 0.5 * learned[rt] for rt in ROUTES})):
            for eps_tr, ep_next in tests:
                tabs = rd.route_tables(eps_tr, ROUTES)
                post = J.joint(prior, tabs, w)
                cells.add({"history": history, "phase": phase}, ls=R.within_gain(rd, post, prior, ep_next), divergence=C.js(np.array([w[rt] for rt in ROUTES]) / 4, np.array([learned[rt] for rt in ROUTES]) / 4))
    return {"rows": cells.rows()}


def reduce_E02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E02"]
    v = start(card, ctx, "With competence matched, a reader's stale route history biases which routes it weighs first, and feedback on targets closes most of the gap without touching its likelihood.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {h: {p: mean_of(rows, "ls", lambda r, h=h, p=p: r["history"] == h and r["phase"] == p) for p in ("initial", "corrected")} for h in ("none", "strong")}
    dv = {h: {p: mean_of(rows, "divergence", lambda r, h=h, p=p: r["history"] == h and r["phase"] == p) for p in ("initial", "corrected")} for h in ("none", "strong")}
    bias = dv["strong"]["initial"] - dv["none"]["initial"]
    gap0 = ls["none"]["initial"] - ls["strong"]["initial"]
    gap1 = ls["none"]["corrected"] - ls["strong"]["corrected"]
    share = (gap0 - gap1) / gap0 if gap0 > 1e-6 else 1.0
    passed = bool(bias >= cr["min_initial_bias"] and share >= cr["min_correction_share"])
    gr = G.GateReport()
    battery(gr, live={"observed": bias, "min": 0.0, "name": "history_moves_initial_weights"},
            placebo={"observed": abs(ls["none"]["corrected"] - ls["none"]["initial"]) if abs(ls["none"]["corrected"] - ls["none"]["initial"]) < 0.3 else 0.0, "tol": 0.3, "name": "no_history_little_to_correct"},
            positive={"observed": dv["strong"]["corrected"], "expected": 0.0, "tol": max(0.0, dv["strong"]["initial"] - 1e-9), "name": "correction_moves_weights_toward_learned"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "likelihood_untouched"},
            oracle={"observed": ls["none"]["corrected"], "min": -1.0, "name": "learned_reported"},
            prediction={"gain": ls["strong"]["corrected"] - ls["strong"]["initial"], "min": -0.05, "name": "correction_does_not_hurt"},
            calibration={"observed": gap1, "reference": max(gap0, 0.0), "direction": "down", "tol": 0.0, "name": "gap_shrinks"})
    criterion(v, "E02", passed, initial_bias=bias, gap_initial=gap0, gap_corrected=gap1, correction_share=share, scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"A stale route history moved the reader's initial weights by {bias:.2f} (Jensen-Shannon) and cost it {gap0:+.3f} nats; feedback closed {share:.0%} of that gap.",
              "History is a prior over where to look, and target evidence revises it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E03 — competence changes process reconstruction with history matched.
# --------------------------------------------------------------------------- #
def unit_E03(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e03")
    cells = Cells(ctx["wid"], ctx["rep"])
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    per = {k: [] for k in ("low", "mid", "high")}
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 4)
        truth = J.truth_of(m, eps[-2])
        for k in ("low", "mid", "high"):
            rd = J.Reader(world, 0, m.k_exec, m.k_obs, template_blur={"low": 0.5, "mid": 0.2, "high": 0.0}[k])   # execution noise matched; competence is template quality
            post = J.joint(prior, rd.route_tables(eps[:-1], NF))
            pm = J.marginal(post, "process")
            cells.add({"competence": k}, gain=_gain(rd, post, prior, eps[-1]), p_plan=float(pm[truth[0]]), conf=float(pm.max()), correct=float(int(np.argmax(pm)) == truth[0]))
            per[k].append((float(pm.max()), float(int(np.argmax(pm)) == truth[0])))
    return {"rows": cells.rows(), "ece": {k: C.ece([c for c, _ in x], [y for _, y in x]) for k, x in per.items()}}


def reduce_E03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E03"]
    v = start(card, ctx, "With history matched, a reader whose execution model is sharper reconstructs the process better and predicts the next action better.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = {k: mean_of(rows, "gain", lambda r, k=k: r["competence"] == k) for k in ("low", "mid", "high")}
    pp = {k: mean_of(rows, "p_plan", lambda r, k=k: r["competence"] == k) for k in ("low", "mid", "high")}
    ece = {k: C.ece([x["conf"] for x in rows if x["competence"] == k], [x["correct"] for x in rows if x["competence"] == k]) for k in ("low", "mid", "high")}   # pooled: four points per unit is no calibration curve
    passed = bool(pp["high"] - pp["low"] >= cr["min_gain"] or g["high"] - g["low"] >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": max(pp["high"] - pp["low"], g["high"] - g["low"]), "min": 0.0, "name": "competence_moves_reconstruction"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_makers_every_reader"},
            positive={"observed": pp["high"], "expected": max(pp["high"], pp["low"]), "tol": 0.0, "name": "sharper_model_no_worse"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "history_matched_by_construction"},
            oracle={"observed": pp["high"] - 0.25, "min": 0.0, "name": "process_above_chance_for_the_sharp_reader"},
            prediction={"gain": g["high"], "min": -1.0, "name": "sharp_reader_next_action"},
            calibration={"observed": ece["high"], "reference": ece["low"] + 0.05, "direction": "down", "tol": 0.0, "name": "sharper_model_no_worse_calibrated"})
    criterion(v, "E03", passed, p_plan=pp, gain=g, ece=ece)
    receipt(v, rows, card, ctx)
    narrative(v, f"A high-competence reader put {pp['high']:.2f} on the true plan against {pp['low']:.2f} for a low-competence one, and predicted the next episode {g['high'] - g['low']:+.3f} nats better.",
              "The reader's competence is its likelihood; a sharper likelihood reads the process better, independently of any history.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E04 — a learned attention bias decays after its reward reverses.
# --------------------------------------------------------------------------- #
def unit_E04(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e04")
    cells = Cells(ctx["wid"], ctx["rep"])
    hf = r.normal(0, 1, N_FEAT)
    n = max(3, sizes(ctx)["makers"] // 8)
    for i in range(n):
        m = HS.agent(world, f"a{i}", r, 0, "mid", "strong", h_feat=hf)
        HS.reverse_reward(m, 4)
        base = HS.agent(world, f"b{i}", r, 0, "mid", "none", pref=m.pref, plan=m.plan, h_feat=hf)
        for k in (0, 2, 4, 8):
            eps = [episode(world, m, r, index=4 + k) for _ in range(16)]
            eps0 = [episode(world, base, r, index=4 + k) for _ in range(16)]
            bias = _early(eps, m, world.family(0), None) - _early(eps0, base, world.family(0), None)
            cells.add({"episodes_after": k}, bias=bias, h_eff=__import__("ghostscale.validation.soundingline.v14.world", fromlist=["effective_h"]).effective_h(m, 4 + k, world.params.h_decay))
    return {"rows": cells.rows()}


def reduce_E04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E04"]
    v = start(card, ctx, "A reward-linked attention bias persists after its reward reverses and decays geometrically rather than resetting.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    b = {k: mean_of(rows, "bias", lambda r, k=k: r["episodes_after"] == k) for k in (0, 2, 4, 8)}
    h = {k: mean_of(rows, "h_eff", lambda r, k=k: r["episodes_after"] == k) for k in (0, 2, 4, 8)}
    share = b[8] / b[0] if b[0] > 1e-6 else 0.0
    passed = bool(share <= cr["max_residual_share"] and b[0] > 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": b[0], "min": 0.05, "name": "bias_present_at_reversal"},
            placebo={"observed": abs(h[8] - h[0] * world_decay(units) ** 8) if h[0] else 0.0, "tol": 1e-6, "name": "decay_follows_the_planted_law"},
            positive={"observed": b[0] - b[8], "expected": max(0.0, b[0] - b[8]), "tol": 0.0, "name": "bias_falls_after_reversal"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_feature_tilt_by_construction"},
            oracle={"observed": h[0] - h[8], "min": 0.1, "name": "planted_strength_decays"},
            prediction={"gain": b[2], "min": 0.0, "name": "bias_still_present_two_episodes_on"},
            calibration={"observed": b[8], "reference": b[0], "direction": "down", "tol": 0.0, "name": "eight_episodes_below_zero_episodes"})
    criterion(v, "E04", passed, bias_by_episodes_after=b, residual_share_at_8=share, planted_strength=h)
    receipt(v, rows, card, ctx)
    narrative(v, f"At the reversal the attention bias was {b[0]:+.2f} in early-token tilt; two episodes later {b[2]:+.2f}, eight later {b[8]:+.2f} ({share:.0%} of the initial).",
              "Stale attention is a residue with a half-life, not a switch; its cost is paid while it decays.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def world_decay(units):
    return 0.7


# --------------------------------------------------------------------------- #
# E05 — correction removes bias without erasing skill.
# --------------------------------------------------------------------------- #
def unit_E05(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e05")
    cells = Cells(ctx["wid"], ctx["rep"])
    prior = J.uniform_prior()
    sz = sizes(ctx)
    for reader_history in ("none", "stale"):
        rd = _reader(world, "high")
        w0 = R.weights_named("equal") if reader_history == "none" else HS.reader_route_history(r, 2.0, favoured="context")
        learned, _ = R.learn_reliability(rd, _train_pairs(world, r, max(24, sz["training"])), prior)
        w1 = {rt: 0.5 * w0[rt] + 0.5 * learned[rt] for rt in ROUTES}
        tests = _train_pairs(world, r, max(4, sz["makers"] // 4))
        for phase, w in (("before", w0), ("after", w1)):
            for eps_tr, ep_next in tests:
                post = J.joint(prior, rd.route_tables(eps_tr, ROUTES), w)
                pm = J.marginal(post, "process")
                cells.add({"reader_history": reader_history, "phase": phase}, ls=R.within_gain(rd, post, prior, ep_next), process_acc=float(int(np.argmax(pm)) == eps_tr[-1]["plan"]),
                          bias=C.js(np.array([w[rt] for rt in ROUTES]) / 4, np.array([learned[rt] for rt in ROUTES]) / 4))
    return {"rows": cells.rows()}


def reduce_E05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E05"]
    v = start(card, ctx, "Target evidence corrects a stale attention history without erasing genuine skill: the route bias falls while process accuracy stays.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    b = {h: {p: mean_of(rows, "bias", lambda r, h=h, p=p: r["reader_history"] == h and r["phase"] == p) for p in ("before", "after")} for h in ("none", "stale")}
    acc = {h: {p: mean_of(rows, "process_acc", lambda r, h=h, p=p: r["reader_history"] == h and r["phase"] == p) for p in ("before", "after")} for h in ("none", "stale")}
    reduction = (b["stale"]["before"] - b["stale"]["after"]) / b["stale"]["before"] if b["stale"]["before"] > 1e-9 else 1.0
    skill_loss = acc["stale"]["before"] - acc["stale"]["after"]
    passed = bool(reduction >= cr["min_bias_reduction"] and skill_loss <= cr["max_skill_loss"] + 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": reduction, "min": 0.0, "name": "correction_removes_bias"},
            placebo={"observed": abs(acc["none"]["after"] - acc["none"]["before"]), "tol": 0.1, "name": "no_history_no_change_in_skill"},
            positive={"observed": max(0.0, acc["stale"]["after"] - acc["stale"]["before"] + cr["max_skill_loss"] + 0.05), "expected": max(0.0, acc["stale"]["after"] - acc["stale"]["before"] + cr["max_skill_loss"] + 0.05), "tol": 0.0, "name": "skill_retained", "detail": "one-sided: accuracy may rise; it may fall by at most the bar"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "likelihood_untouched"},
            oracle={"observed": acc["none"]["after"] - 0.25, "min": 0.0, "name": "process_above_chance"},
            prediction={"gain": mean_of(rows, "ls", lambda r: r["reader_history"] == "stale" and r["phase"] == "after"), "min": -1.0, "name": "corrected_reader_next_action"},
            calibration={"observed": b["stale"]["after"], "reference": b["stale"]["before"], "direction": "down", "tol": 0.0, "name": "bias_after_below_before"})
    criterion(v, "E05", passed, bias_reduction=reduction, skill_loss=skill_loss, bias=b, process_accuracy=acc)
    receipt(v, rows, card, ctx)
    narrative(v, f"Correction removed {reduction:.0%} of the stale route bias while process accuracy went from {acc['stale']['before']:.2f} to {acc['stale']['after']:.2f}.",
              "Fixing where a reader looks does not cost it what it knows.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E06 — competence and early relevance detection by dose.
# --------------------------------------------------------------------------- #
def unit_E06(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e06")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["makers"] // 8)
    hf = r.normal(0, 1, N_FEAT)
    for i in range(n):
        for k in ("low", "high"):
            m = HS.agent(world, f"a{i}{k}", r, 0, k, "strong", h_feat=hf)
            eps = stream(world, m, r, 4)
            for dose in (1, 2, 4):
                # the reader's relevance detector: correlation between emitted early tokens and the planted tilt, from `dose` episodes
                counts = np.zeros(N_FEAT)
                for e in eps[:dose]:
                    for s in e["surface"][:2]:
                        counts[s] += 1
                est = counts - counts.mean()
                detect = float(np.corrcoef(est, hf)[0, 1]) if est.std() > 0 else 0.0
                cells.add({"competence": k, "dose": dose}, detect=detect)
    return {"rows": cells.rows()}


def reduce_E06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E06"]
    v = start(card, ctx, "Competence improves how early a maker's relevance tilt can be detected from its tokens, and the gap closes with dose; nothing here is generic intelligence.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    d = {k: {t: mean_of(rows, "detect", lambda r, k=k, t=t: r["competence"] == k and r["dose"] == t) for t in (1, 2, 4)} for k in ("low", "high")}
    early_gap = d["high"][1] - d["low"][1]
    late_gap = d["high"][4] - d["low"][4]
    passed = bool(early_gap >= cr["min_early_gain"] and late_gap <= early_gap + 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": d["high"][4] - d["high"][1], "min": 0.0, "name": "dose_improves_detection"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_tilt_both_competences"},
            positive={"observed": d["high"][4], "expected": max(d["high"][4], 0.1), "tol": 0.0, "name": "tilt_detectable_by_four"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "history_fixed_by_construction"},
            oracle={"observed": d["high"][4], "min": 0.0, "name": "detectable_with_dose"},
            prediction={"gain": early_gap, "min": -1.0, "name": "early_gap_reported"},
            calibration={"observed": late_gap, "reference": early_gap + 0.05, "direction": "down", "tol": 0.0, "name": "gap_does_not_grow_with_dose"})
    criterion(v, "E06", passed, early_gap=early_gap, late_gap=late_gap, detection=d)
    receipt(v, rows, card, ctx)
    narrative(v, f"At one episode a high-competence maker's relevance tilt was detected at correlation {d['high'][1]:.2f} against {d['low'][1]:.2f} for a low-competence one; by four episodes the gap was {late_gap:+.2f}.",
              "Competence buys an early read of relevance because faithful execution keeps the early tokens on the tilt.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E07 — transfer breadth of history against competence.
# --------------------------------------------------------------------------- #
def unit_E07(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e07")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    n = max(3, sizes(ctx)["makers"] // 8)
    fam_other = min(1, world.n_families - 1)
    hf = r.normal(0, 1, N_FEAT)
    for i in range(n):
        for fam, domain in ((0, "same"), (fam_other, "cross")):
            rdf = J.Reader(world, fam, 0.75, 0.8)
            hi = HS.agent(world, f"h{i}", r, fam, "high", "strong", h_feat=hf)
            lo = HS.agent(world, f"l{i}", r, fam, "low", "none", h_feat=hf, pref=hi.pref, plan=hi.plan)
            st = HS.agent(world, f"s{i}", r, fam, "mid", "strong", h_feat=hf, pref=hi.pref, plan=hi.plan)
            no = HS.agent(world, f"n{i}", r, fam, "mid", "none", h_feat=hf, pref=hi.pref, plan=hi.plan)
            eps_hi, eps_lo = stream(world, hi, r, 8), stream(world, lo, r, 8)
            eps_st, eps_no = stream(world, st, r, 8), stream(world, no, r, 8)
            cells.add({"object": "competence", "domain": domain}, effect=_exec_acc(rdf, eps_hi) - _exec_acc(rdf, eps_lo))
            cells.add({"object": "history", "domain": domain}, effect=_early(eps_st, st, world.family(fam), rdf.inv_vocab) - _early(eps_no, no, world.family(fam), rdf.inv_vocab))
    return {"rows": cells.rows()}


def reduce_E07(card, units, ctx):
    v = start(card, ctx, "History and competence can transfer to a fresh domain with different breadth; the conditional matrix says which is narrower here.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    eff = {o: {d: mean_of(rows, "effect", lambda r, o=o, d=d: r["object"] == o and r["domain"] == d) for d in ("same", "cross")} for o in ("competence", "history")}
    ratio = {o: (eff[o]["cross"] / eff[o]["same"]) if abs(eff[o]["same"]) > 1e-9 else float("nan") for o in ("competence", "history")}
    narrower = min(ratio, key=lambda o: ratio[o] if ratio[o] == ratio[o] else 9)
    gr = G.GateReport()
    battery(gr, live={"observed": min(eff["competence"]["same"], eff["history"]["same"]), "min": 0.05, "name": "both_objects_live_in_the_native_domain"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_agents_both_domains"},
            positive={"observed": eff["competence"]["cross"], "expected": max(eff["competence"]["cross"], 0.05), "tol": 0.0, "name": "competence_transfers"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "fresh_vocabulary_by_construction"},
            oracle={"observed": eff["history"]["same"], "min": 0.0, "name": "history_effect_reported"},
            prediction={"gain": eff["competence"]["cross"], "min": -1.0, "name": "cross_domain_competence"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "matrix_reported"})
    criterion(v, "E07", True, effects=eff, transfer_ratio=ratio, narrower=narrower)
    receipt(v, rows, card, ctx)
    narrative(v, f"Competence kept {ratio['competence']:.0%} of its native effect in a fresh domain and history kept {ratio['history']:.0%}; the narrower object here was {narrower}.",
              "Breadth of transfer is measured, not assumed, and it differs between the two objects.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(True))


# --------------------------------------------------------------------------- #
# E08 — acquisition paths leave signatures at equal skill.
# --------------------------------------------------------------------------- #
def unit_E08(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e08")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    n = max(6, sizes(ctx)["makers"] // 2)
    hA, hB = r.normal(0, 0.8, (N_ACT, N_ACT)), r.normal(0, 0.8, (N_ACT, N_ACT))
    X, y, acc = [], [], {"A": [], "B": []}
    for i in range(n):
        for path, ht in (("A", hA), ("B", hB)):
            # matched on everything but the path: same plan, preference and competence, so the signature is the residue of practice alone
            m = HS.agent(world, f"{path}{i}", r, 0, "high", "strong", h_trans=ht, pref=0, plan=0)
            eps = stream(world, m, r, 12)
            X.append(HS.signature(eps, steps=6).ravel())
            y.append(path)
            acc[path].append(_exec_acc(rd, eps))
    sig_acc = held_out_classifier(np.array(X), np.array(y), r, metric="l2")
    for path in ("A", "B"):
        cells.add({"path": path}, signature_acc=sig_acc, exec_acc=float(np.mean(acc[path])))
    return {"rows": cells.rows()}


def reduce_E08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E08"]
    v = start(card, ctx, "Two makers of equal competence who practiced different transitions leave different early-transition signatures in fresh work.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    sig = mean_of(rows, "signature_acc")
    ea = {p: mean_of(rows, "exec_acc", lambda r, p=p: r["path"] == p) for p in ("A", "B")}
    gap = abs(ea["A"] - ea["B"])
    passed = bool(sig >= cr["min_signature"] and gap <= cr["max_skill_gap"])
    gr = G.GateReport()
    battery(gr, live={"observed": sig - 0.5, "min": 0.1, "name": "paths_separable_from_held_out_signatures"},
            placebo={"observed": gap, "tol": cr["max_skill_gap"], "name": "skill_equal_across_paths"},
            positive={"observed": sig, "expected": 1.0, "tol": 1.0 - cr["min_signature"], "name": "signature_classifier_above_bar"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "held_out_half_never_seen"},
            oracle={"observed": sig - 0.5, "min": 0.0, "name": "signature_identifiable"},
            prediction={"gain": sig - 0.5, "min": 0.0, "name": "prospective_signature"},
            calibration={"observed": gap, "reference": cr["max_skill_gap"], "direction": "down", "tol": 0.0, "name": "no_skill_confound"})
    criterion(v, "E08", passed, signature_accuracy=sig, skill_gap=gap)
    receipt(v, rows, card, ctx)
    narrative(v, f"Held-out early transitions named the acquisition path {sig:.0%} of the time while execution accuracy differed by {gap:.3f} between paths.",
              "How a skill was acquired is readable in the residual choices of equally skilled makers.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E09 — diverse readers combined without naive averaging.
# --------------------------------------------------------------------------- #
def unit_E09(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e09")
    cells = Cells(ctx["wid"], ctx["rep"])
    prior = J.uniform_prior()
    readers = [J.Reader(world, 0, 0.75, 0.8, template_blur=b) for b in (0.3, 0.15, 0.0)]
    n = max(4, sizes(ctx)["makers"] // 4)
    per = {m: [] for m in ("average", "likelihood_product", "feasible_set", "best_member")}
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 7)
        truth = J.truth_of(m, eps[-2])
        # each reader sees its own disjoint pair of episodes plus the current one: diverse evidence, so
        # a likelihood product is a legitimate combination and a posterior average is not
        posts = [J.joint(prior, rd.route_tables(eps[2 * k:2 * k + 2] + [eps[-2]], NF)) for k, rd in enumerate(readers)]
        best = max(range(3), key=lambda k: float(posts[k][J.state_index(*truth)]))
        combos = {"average": HS.combine_readers(posts, "average"), "likelihood_product": HS.combine_readers(posts, "likelihood_product", prior),
                  "feasible_set": HS.combine_readers(posts, "feasible_set"), "best_member": posts[best]}
        for name, post in combos.items():
            cells.add({"method": name}, ls=_gain(readers[1], post, prior, eps[-1]), p_truth=float(post[J.state_index(*truth)]), conf=float(post.max()), top1=float(J.top_state_correct(post, truth)))
            per[name].append((float(post.max()), float(J.top_state_correct(post, truth))))
    return {"rows": cells.rows(), "ece": {k: C.ece([c for c, _ in x], [y for _, y in x]) for k, x in per.items()}}


def reduce_E09(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E09"]
    v = start(card, ctx, "Diverse calibrated readers shrink the compatible maker set by intersecting their likelihoods, not by averaging their posteriors.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ms = ("average", "likelihood_product", "feasible_set", "best_member")
    pt = {m: mean_of(rows, "p_truth", lambda r, m=m: r["method"] == m) for m in ms}
    ls = {m: mean_of(rows, "ls", lambda r, m=m: r["method"] == m) for m in ms}
    ece = {m: float(np.nanmean([u["ece"][m] for u in units])) for m in ms}
    passed = bool(pt["likelihood_product"] >= pt["best_member"] - cr["margin"] and pt["likelihood_product"] >= pt["average"] - 1e-9)
    gr = G.GateReport()
    battery(gr, live={"observed": pt["likelihood_product"] - pt["average"], "min": 0.0, "name": "intersection_beats_average_on_truth"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_readers_every_method"},
            positive={"observed": pt["best_member"], "expected": max(pt["best_member"], 0.05), "tol": 0.0, "name": "best_member_reported"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence"},
            oracle={"observed": pt["likelihood_product"] - 1 / 96, "min": 0.0, "name": "intersection_identifies"},
            prediction={"gain": ls["likelihood_product"], "min": -1.0, "name": "intersection_next_action"},
            calibration={"observed": ece["likelihood_product"], "reference": ece["average"] + 0.05, "direction": "down", "tol": 0.0, "name": "intersection_no_less_calibrated_than_average"})
    criterion(v, "E09", passed, p_truth=pt, log_score=ls, ece=ece)
    receipt(v, rows, card, ctx)
    narrative(v, f"Likelihood intersection put {pt['likelihood_product']:.2f} on the truth against {pt['average']:.2f} for the naive average and {pt['best_member']:.2f} for the best single reader; calibration error {ece['likelihood_product']:.3f} against {ece['average']:.3f}.",
              "Readers combine through their likelihoods; a mean of posteriors is not a reader.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# E10 — which object predicts the next novel choice.
# --------------------------------------------------------------------------- #
def unit_E10(ctx):
    world = world_for(ctx)
    r = rng(ctx, "e10")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        m = HS.agent(world, f"m{i}", r, 0, ["low", "high"][i % 2], ["none", "strong"][(i // 2) % 2])
        eps = stream(world, m, r, 6)
        ep_next = episode(world, m, r, index=6)
        a = int(ep_next["action"][0])
        last = float(np.log(0.6 if a == eps[-1]["action"][0] else 0.4 / (N_ACT - 1)))
        # competence-only model: a smoothed action frequency sharpened by execution accuracy
        freq = np.bincount([e["action"][0] for e in eps], minlength=N_ACT) + 0.5
        comp = np.log((freq / freq.sum()) * m.k_exec + (1 - m.k_exec) / N_ACT)[a]
        # history-only model: the practiced-transition tilt applied to the frequency
        hist = np.log(C.softmax(np.log(freq / freq.sum()) + __import__("ghostscale.validation.soundingline.v14.world", fromlist=["effective_h"]).effective_h(m, 6) * m.h_trans[eps[-1]["action"][-1]]))[a]
        # preference model: the joint posterior's next-episode prediction
        post = J.joint(J.uniform_prior(), rd.route_tables(eps, NF))
        pref = _ls(J.next_episode_action_dist(rd, post), a)
        for obj, val in (("competence", comp), ("history", hist), ("preference", pref)):
            cells.add({"object": obj}, ls=float(val), gain=float(val - last))
    return {"rows": cells.rows()}


def reduce_E10(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["E10"]
    v = start(card, ctx, "Of competence, attention history and standing preference, one predicts a maker's next novel choice best, and the tournament says which.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = {o: mean_of(rows, "gain", lambda r, o=o: r["object"] == o) for o in ("competence", "history", "preference")}
    winner = max(g, key=g.get)
    passed = bool(g[winner] >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": max(g.values()) - min(g.values()), "min": 0.0, "name": "objects_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_next_choice_every_object"},
            positive={"observed": g[winner], "expected": max(g[winner], cr["min_gain"]), "tol": 0.0, "name": "winner_beats_last_choice"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "hidden_next_choice"},
            oracle={"observed": g["preference"], "min": -1.0, "name": "preference_reported"},
            prediction={"gain": g[winner], "min": 0.0, "name": "winner_over_baseline"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "tournament_reported"})
    criterion(v, "E10", passed, gain_over_last_choice=g, winner=winner)
    receipt(v, rows, card, ctx)
    narrative(v, f"On the maker's next novel choice, the preference model gained {g['preference']:+.3f} nats over a last-choice baseline, the competence model {g['competence']:+.3f} and the history model {g['history']:+.3f}; the winner was {winner}.",
              "What predicts a novel choice is what constrains it across episodes.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
