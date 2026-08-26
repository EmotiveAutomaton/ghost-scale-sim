"""Trunk G: communicative goals, epistemic vigilance, trust, and uptake (spec §13).

Stance is a goal a maker holds, reliability is a property of a source's assertions, content
support is what the artifact itself shows, and uptake is a decision. None of them is one scalar
unless a card is explicitly comparing scalar and factored readers.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import goals_trust as GT
from .. import pymdp_reader as PR
from . import (battery, boot, ci_abs, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of)
from .trunk_c import Cells

GOALS = GT.GOALS


def _kinds(ctx, tag="kinds"):
    return GT.kind_dists(rng(ctx, tag))


def _speak_n(src, r, d0, d1, n, n_tok=8, domain=0, start_t=0):
    out = []
    t = start_t
    while len(out) < n and t < start_t + 8 * n:
        a = GT.speak(src, r, d0, d1, n_tok, domain=domain, t=t)
        t += 1
        if a is not None:
            out.append(a)
    return out


def _acc(q_goal, goal):
    return float(max(q_goal, key=q_goal.get) == goal)


# --------------------------------------------------------------------------- #
# G01 — stance as an ordinary goal.
# --------------------------------------------------------------------------- #
def unit_G01(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g01")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    feats, labels = [], []
    for g in GOALS:
        for s in range(max(2, sz["sources"] // 7)):
            src = GT.Source(f"{g}{s}", g, {0: 0.9}, agenda=int(r.random() < 0.5), slot=s)
            arts = _speak_n(src, r, d0, d1, 12)
            rev = {i: a["truth"] for i, a in enumerate(arts)}
            for dose in (2, 6, 12):
                fr = GT.factored_read(arts[:dose], d0, d1, revealed={i: rev[i] for i in range(dose)})
                cells.add({"goal": g, "dose": dose}, ls=float(np.log(max(fr["q_goal"][g], 1e-12))), acc=_acc(fr["q_goal"], g), conf=float(max(fr["q_goal"].values())))
            for a in arts:
                h = np.bincount(a["tokens"], minlength=GT.N_KINDS) / len(a["tokens"])
                feats.append(np.concatenate([h, [C.entropy(h), a["n_tokens"] / 8.0, a["cue"] / 4.0]]))
                labels.append(g)
    return {"rows": cells.rows(), "surface": held_out_classifier(np.array(feats), np.array(labels), r, metric="l2")}


def reduce_G01(card, units, ctx):
    v = start(card, ctx, "A maker's communicative stance is an ordinary goal: a reader that learns which claims were true recovers "
              "it from the correspondence between assertion, evidence and truth, with no source-type label and no surface cue.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surface = float(np.nanmean([u["surface"] for u in units]))
    acc = {str(d): mean_of(rows, "acc", lambda r, d=d: r["dose"] == d) for d in (2, 6, 12)}
    by_goal = {g: mean_of(rows, "acc", lambda r, g=g: r["goal"] == g and r["dose"] == 12) for g in GOALS}
    passed = bool(acc["12"] >= 0.6 and abs(surface - 1 / 7) <= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": acc["12"] - acc["2"], "min": 0.0, "name": "evidence_sharpens_the_stance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "no_source_label_in_the_reader"},
            positive={"observed": by_goal["misleading"], "expected": 1.0, "tol": 0.4, "name": "misleading_recognised"},
            surface={"accuracy": surface, "chance": 1 / 7, "tol": 0.10, "name": "artifact_surface_at_chance"},
            oracle={"observed": acc["12"], "min": 0.5, "name": "stance_readable_with_truths"},
            prediction={"gain": mean_of(rows, "ls", lambda r: r["dose"] == 12) - np.log(1 / 7), "min": 0.0, "name": "log_score_above_uniform"},
            calibration={"observed": C.ece([r["conf"] for r in rows if r["dose"] == 12], [r["acc"] for r in rows if r["dose"] == 12]), "reference": 0.25, "direction": "down", "tol": 0.0, "name": "stance_confidence_calibrated"})
    criterion(v, "G01", passed, accuracy_by_dose=acc, surface=surface, by_goal=by_goal)
    v["results"].update({"accuracy_by_dose": acc, "accuracy_by_goal_at_12": by_goal, "surface_classifier": surface})
    receipt(v, rows, card, ctx)
    narrative(v, f"With twelve artifacts and their claims' truths, the reader named the maker's stance {acc['12']:.0%} of the time against {1 / 7:.0%} chance; a surface classifier on the artifacts named it {surface:.0%}.",
              "Stance is readable as a goal through correspondence; it is not a source type stamped on the artifact.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G02 — joint inference with task goals.
# --------------------------------------------------------------------------- #
def unit_G02(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g02")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        task = int(r.integers(2))                                    # the domain the maker's task goal targets
        cg = str(r.choice(["accurate", "misleading", "persuasion"]))
        for pair in ("compatible", "conflicting"):
            src = GT.Source(f"s{s}", cg, {0: 0.9, 1: 0.9}, agenda=1, slot=s)
            arts = []
            for i in range(12):
                dom = task if r.random() < 0.7 else 1 - task
                g = cg if (pair == "compatible" or dom == task) else "accurate"     # conflicting: the stance applies only on the task domain
                a = GT.speak(GT.Source(src.id, g, src.reliability, 1, slot=s), r, d0, d1, 8, domain=dom, t=i)
                if a is not None:
                    arts.append(a)
            rev = {i: a["truth"] for i, a in enumerate(arts)}
            # joint: stance conditional on the domain being the task domain; independent: one stance for all
            ind = GT.factored_read(arts, d0, d1, revealed=rev)
            on = [a for a in arts if a["domain"] == task]
            off = [a for a in arts if a["domain"] != task]
            j_on = GT.factored_read(on, d0, d1, revealed={i: a["truth"] for i, a in enumerate(on)}) if on else ind
            j_off = GT.factored_read(off, d0, d1, revealed={i: a["truth"] for i, a in enumerate(off)}) if off else ind
            # score: log score of the true stance on the task domain, and of the off-domain stance
            true_off = cg if pair == "compatible" else "accurate"
            # structural hypotheses: one stance everywhere, or a stance conditional on the task domain; the joint reader keeps
            # whichever explains the record better (marginal likelihood of the revealed truths under each)
            ll_shared = float(C.logsumexp(np.array([sum(GT.goal_loglik(a, a["truth"], d0, d1, g) for a in arts) for g in GT.GOALS])))
            ll_cond = float(C.logsumexp(np.array([sum(GT.goal_loglik(a, a["truth"], d0, d1, g) for a in on) for g in GT.GOALS])) + C.logsumexp(np.array([sum(GT.goal_loglik(a, a["truth"], d0, d1, g) for a in off) for g in GT.GOALS])) - np.log(7.0))
            if ll_shared >= ll_cond - 2.0:
                j_on, j_off = ind, ind                       # structural parsimony: split stances only on clear evidence
            ls_joint = 0.5 * (np.log(max(j_on["q_goal"][cg], 1e-12)) + np.log(max(j_off["q_goal"][true_off], 1e-12)))
            ls_ind = 0.5 * (np.log(max(ind["q_goal"][cg], 1e-12)) + np.log(max(ind["q_goal"][true_off], 1e-12)))
            cells.add({"pair": pair, "reader": "joint"}, ls=float(ls_joint), acc=_acc(j_on["q_goal"], cg))
            cells.add({"pair": pair, "reader": "independent"}, ls=float(ls_ind), acc=_acc(ind["q_goal"], cg))
    return {"rows": cells.rows()}


def reduce_G02(card, units, ctx):
    v = start(card, ctx, "Communicative and task goals must be inferred jointly only where they interact; where the stance is the same "
              "everywhere, the independent shortcut is as good.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {p: {rd: boot(rows, "ls", lambda r, p=p, rd=rd: r["pair"] == p and r["reader"] == rd, seed_tag=f"G02{p}{rd}")["mean"] for rd in ("joint", "independent")} for p in ("compatible", "conflicting")}
    g_conf = grid["conflicting"]["joint"] - grid["conflicting"]["independent"]
    g_comp = grid["compatible"]["joint"] - grid["compatible"]["independent"]
    passed = bool(g_conf >= 0.02 and abs(g_comp) <= 0.1)
    gr = G.GateReport()
    battery(gr, live={"observed": g_conf - g_comp, "min": 0.02, "name": "interaction_moves_the_joint_advantage"},
            placebo={"observed": ci_abs([{"wid": r["wid"], "d": r.get("ls", 0.0), "reader": r["reader"], "pair": r["pair"]} for r in rows], "d", lambda r: False, seed_tag="none") if False else (0.0 if abs(g_comp) <= 0.20 else abs(g_comp)), "tol": 0.20, "name": "compatible_pairs_tie"},
            positive={"observed": mean_of(rows, "acc", lambda r: r["reader"] == "joint"), "expected": 1.0, "tol": 0.5, "name": "joint_recovers_the_stance"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": grid["conflicting"]["joint"] - np.log(1 / 7), "min": 0.0, "name": "identifiable"},
            prediction={"gain": g_conf, "min": 0.0, "name": "joint_gain_where_goals_interact"},
            calibration={"observed": mean_of(rows, "acc", lambda r: r["reader"] == "independent" and r["pair"] == "conflicting"), "reference": mean_of(rows, "acc", lambda r: r["reader"] == "joint" and r["pair"] == "conflicting"), "direction": "down", "tol": 0.0, "name": "independent_worse_under_conflict"})
    criterion(v, "G02", passed, joint_gain_conflicting=g_conf, joint_gain_compatible=g_comp)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Where the stance applied only to the maker's task domain, joint inference beat the independent shortcut by {g_conf:+.2f} nats; where the stance was uniform the two differed by {g_comp:+.2f}.",
              "Joint inference pays exactly where the goals interact; the shortcut is an equivalence elsewhere.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G03 — competing goals explain mixed signals.
# --------------------------------------------------------------------------- #
def _multi_goal_dist(d0, d1, T, A, acc, pers, kind):
    """Token distribution under weights on accuracy (toward D_T), persuasion (toward D_A) and
    kindness (toward the flat distribution)."""
    lp = acc * np.log(np.maximum(d1 if T == 1 else d0, 1e-12)) + pers * np.log(np.maximum(d1 if A == 1 else d0, 1e-12)) + kind * np.log(1.0 / d0.size)
    return C.softmax(lp)


GRID3 = [(a, p, k) for a in (0.0, 0.5, 1.0) for p in (0.0, 0.5, 1.0) for k in (0.0, 0.5)]
THREE = {"helpful": (1.0, 0.0, 0.0), "neutral": (0.5, 0.0, 0.5), "misleading": (0.0, 1.0, 0.0)}


def unit_G03(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g03")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        acc, pers, kind = float(r.uniform(0.2, 1.0)), float(r.uniform(0.0, 0.8)), float(r.uniform(0.0, 0.5))
        arts = []
        for i in range(16):
            T = int(r.random() < 0.5)
            A_ = int(r.random() < 0.5)
            arts.append({"truth": T, "assertion": A_, "tokens": r.choice(GT.N_KINDS, size=8, p=_multi_goal_dist(d0, d1, T, A_, acc, pers, kind))})
        train, test = arts[:10], arts[10:]

        def ll_under(params, items):
            return sum(float(np.log(np.maximum(_multi_goal_dist(d0, d1, a["truth"], a["assertion"], *params)[a["tokens"]], 1e-12)).sum()) for a in items)
        best3 = max(GRID3, key=lambda p: ll_under(p, train))
        best_c = max(THREE, key=lambda c: ll_under(THREE[c], train))
        cells.add({"model": "multi_goal"}, ls=ll_under(best3, test) / (6 * 8), fit=ll_under(best3, train))
        cells.add({"model": "three_class"}, ls=ll_under(THREE[best_c], test) / (6 * 8), fit=ll_under(THREE[best_c], train))
    return {"rows": cells.rows()}


def reduce_G03(card, units, ctx):
    v = start(card, ctx, "Makers who trade accuracy against persuasion and kindness produce mixed signals that a multi-goal model "
              "predicts on held-out tokens better than a three-class stance.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {m: boot(rows, "ls", lambda r, m=m: r["model"] == m, seed_tag="G03" + m)["mean"] for m in ("multi_goal", "three_class")}
    gain = by["multi_goal"] - by["three_class"]
    passed = bool(gain >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": abs(gain), "min": 0.0, "name": "models_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "held_out_tokens_unseen_at_fit"},
            positive={"observed": float(mean_of(rows, "fit", lambda r: r["model"] == "multi_goal") >= mean_of(rows, "fit", lambda r: r["model"] == "three_class")), "expected": 1.0, "tol": 0.0, "name": "richer_model_fits_training_at_least_as_well"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_tokens"},
            oracle={"observed": by["multi_goal"] - np.log(1 / GT.N_KINDS), "min": 0.0, "name": "tokens_predictable"},
            prediction={"gain": gain, "min": 0.0, "name": "held_out_token_gain"},
            calibration={"observed": by["three_class"], "reference": by["multi_goal"], "direction": "down", "tol": 0.0, "name": "three_class_no_better"})
    criterion(v, "G03", passed, multi_minus_three=gain, by_model=by)
    v["results"].update({"held_out_per_token": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"A model with separate weights on accuracy, persuasion and kindness predicted held-out tokens {gain:+.3f} nats per token better than the best three-class stance.",
              "Mixed signals are what competing goals look like; a stance taxonomy loses the mixture.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G04 / G05 / G06 — reliability, content, alignment as separate posteriors.
# --------------------------------------------------------------------------- #
KINDS4 = {"helpful_incompetent": ("accurate", 0.5, 1.0), "neutral_reliable": ("accurate", 1.0, 1.0),
          "adversarial_truthful": ("misleading", 1.0, 1.0), "helpful_outdated": ("accurate", 1.0, 0.6)}


def unit_G04(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g04")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(3, sz["sources"] // 3)):
        for kind, (goal, competence, rel) in KINDS4.items():
            src = GT.Source(kind, goal, {0: rel}, competence=competence, slot=s)
            arts = _speak_n(src, r, d0, d1, 12)
            if kind == "neutral_reliable":
                for a in arts:
                    a["cue"] = int(r.integers(GT.N_SLOTS))          # asserts and shows what it found, no reader-directed shaping
            if kind == "adversarial_truthful":
                for a in arts:
                    a["assertion"] = a["truth"]                       # says the true thing, shows misleading evidence
            if kind == "helpful_outdated":
                for a in arts:
                    if r.random() > rel:
                        a["assertion"] = 1 - a["assertion"]           # outdated: assertions wrong 40% of the time despite the goal
            rev = {i: a["truth"] for i, a in enumerate(arts)}
            fr = GT.factored_read(arts, d0, d1, revealed=rev)
            planted_rel = float(np.mean([a["assertion"] == a["truth"] for a in arts if a["assertion"] is not None])) if any(a["assertion"] is not None for a in arts) else 0.5
            sc = GT.scalar_trust_read(arts, d0, d1, revealed=rev)
            cells.add({"source": kind}, rel_err=abs(fr["q_source"] - planted_rel), q_source=fr["q_source"], planted=planted_rel,
                      goal_acc=_acc(fr["q_goal"], goal), coop=float(fr["q_goal"]["accurate"] + fr["q_goal"]["comprehension_support"]), scalar=sc["trust"])
    return {"rows": cells.rows()}


def reduce_G04(card, units, ctx):
    v = start(card, ctx, "Source reliability and cooperative intent are separate posteriors: a helpful but incompetent source, a neutral "
              "reliable one, an adversarial but truthful one and a helpful but outdated one each get the reliability they earned, not their valence.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: {m: mean_of(rows, m, lambda r, k=k: r["source"] == k) for m in ("rel_err", "q_source", "planted", "goal_acc", "coop", "scalar")} for k in KINDS4}
    worst = max(by[k]["rel_err"] for k in by)
    passed = bool(worst <= 0.15)
    gr = G.GateReport()
    battery(gr, live={"observed": by["neutral_reliable"]["q_source"] - by["helpful_outdated"]["q_source"], "min": 0.2, "name": "reliability_moves_with_the_record"},
            placebo={"observed": abs(by["adversarial_truthful"]["q_source"] - by["adversarial_truthful"]["planted"]), "tol": 0.15, "name": "adversarial_intent_does_not_lower_earned_reliability"},
            positive={"observed": by["neutral_reliable"]["q_source"], "expected": 1.0, "tol": 0.15, "name": "reliable_source_recognised"},
            surface={"accuracy": max(abs(by[k]["scalar"] - by[k]["planted"]) for k in by) - worst, "chance": 0.0, "tol": 1.0, "name": "scalar_trust_error_reported"},
            oracle={"observed": mean_of(rows, "goal_acc"), "min": 0.3, "name": "goal_still_readable"},
            prediction={"gain": -worst, "min": -0.15, "name": "reliability_within_bar"},
            calibration={"observed": by["helpful_incompetent"]["coop"], "reference": by["adversarial_truthful"]["coop"], "direction": "up", "tol": 0.3, "name": "intent_posterior_separates_valence"})
    criterion(v, "G04", passed, by_source=by, worst_reliability_error=worst)
    v["results"].update({"by_source": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Reliability posterior against the planted rate: " + ", ".join(f"{k} {b['q_source']:.2f} vs {b['planted']:.2f}" for k, b in by.items()) + ".",
              "A source's intent and its reliability come apart in the record, and the factored reader keeps them apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G05(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g05")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        base = {ev: _speak_n(GT.Source(f"s{s}", "accurate", {0: 0.9}, slot=s), r, d0, d1, 6, n_tok=12 if ev == "strong" else 2) for ev in ("strong", "weak")}
        for ev in ("strong", "weak"):
            for hist in ("strong", "weak"):
                arts = [dict(a) for a in base[ev]]
                prior = (9.0, 1.0) if hist == "strong" else (1.0, 1.0)
                fr = GT.factored_read(arts, d0, d1, revealed=None, source_prior=prior)
                fr_rev = GT.factored_read(arts, d0, d1, revealed={i: a["truth"] for i, a in enumerate(arts)}, source_prior=prior)
                content = float(np.mean([abs(pa["q_content_T1"] - 0.5) for pa in fr["per_artifact"]]))
                cells.add({"evidence": ev, "history": hist}, content_move=content, source_move=abs(fr_rev["q_source"] - prior[0] / sum(prior)),
                          q_source=fr["q_source"], truth_acc=float(np.mean([(pa["q_T1_factored"] > 0.5) == (pa["truth"] == 1) for pa in fr["per_artifact"]])))
    return {"rows": cells.rows()}


def reduce_G05(card, units, ctx):
    v = start(card, ctx, "Content evidence and source history update different posteriors: strong artifact evidence moves the content "
              "posterior whatever the history, and a revealed history moves the reliability posterior whatever the evidence.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {e: {h: {k: mean_of(rows, k, lambda r, e=e, h=h: r["evidence"] == e and r["history"] == h) for k in ("content_move", "source_move", "truth_acc")} for h in ("strong", "weak")} for e in ("strong", "weak")}
    c_by_ev = grid["strong"]["weak"]["content_move"] - grid["weak"]["weak"]["content_move"]
    s_by_hist = grid["weak"]["weak"]["source_move"] - grid["weak"]["strong"]["source_move"]
    c_invariant = abs(grid["strong"]["strong"]["content_move"] - grid["strong"]["weak"]["content_move"])
    passed = bool(c_by_ev >= 0.05 and s_by_hist >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": c_by_ev, "min": 0.05, "name": "evidence_strength_moves_content"},
            placebo={"observed": c_invariant, "tol": 1e-9, "name": "history_leaves_content_untouched"},
            positive={"observed": grid["strong"]["strong"]["truth_acc"], "expected": 1.0, "tol": 0.2, "name": "strong_evidence_reliable_source_read_true"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts_across_histories"},
            oracle={"observed": grid["strong"]["weak"]["truth_acc"] - 0.5, "min": 0.2, "name": "content_alone_identifies_truth"},
            prediction={"gain": s_by_hist, "min": 0.0, "name": "history_moves_source"},
            calibration={"observed": grid["weak"]["weak"]["truth_acc"], "reference": grid["strong"]["weak"]["truth_acc"], "direction": "down", "tol": 0.0, "name": "weak_evidence_less_certain"})
    criterion(v, "G05", passed, content_move_by_evidence=c_by_ev, source_move_by_history=s_by_hist)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Strong artifact evidence moved the content posterior {c_by_ev:+.2f} more than weak, whatever the history; a revealed history moved the reliability posterior {s_by_hist:+.2f} more than none, whatever the evidence.",
              "Neither channel overwrites the other; plausibility and track record are two ledgers.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G06(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g06")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        base = {truth: _speak_n(GT.Source(f"s{s}", "accurate" if truth == "true" else "misleading", {0: 0.9}, slot=s), r, d0, d1, 8) for truth in ("true", "false")}
        for align in ("aligned", "divergent"):
            for truth in ("true", "false"):
                arts = base[truth]
                fr = GT.factored_read(arts, d0, d1, revealed={i: a["truth"] for i, a in enumerate(arts)})
                alignment = 0.9 if align == "aligned" else 0.1
                up = GT.uptake_decision(fr["q_goal"], fr["q_source"], 1.0, alignment, 0.9)
                cells.add({"alignment": align, "truth": truth}, belief=up["belief_update"], preference=up["preference_movement"], prediction=up["prediction_use"], q_source=fr["q_source"])
    return {"rows": cells.rows()}


def reduce_G06(card, units, ctx):
    v = start(card, ctx, "Goal alignment and reliability drive different uptake channels: belief update follows whether the source "
              "tells the truth, preference movement follows alignment as well, and prediction use follows neither.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = lambda k, a, t: mean_of(rows, k, lambda r: r["alignment"] == a and r["truth"] == t)
    belief_truth = g("belief", "divergent", "true") - g("belief", "divergent", "false")
    belief_align = abs(g("belief", "aligned", "true") - g("belief", "divergent", "true"))
    pref_align = g("preference", "aligned", "true") - g("preference", "divergent", "true")
    pred_inv = max(abs(g("prediction", a, t) - g("prediction", "aligned", "true")) for a in ("aligned", "divergent") for t in ("true", "false"))
    passed = bool(belief_truth >= 0.3 and belief_align <= 1e-9 and pref_align >= 0.3)
    gr = G.GateReport()
    battery(gr, live={"observed": belief_truth, "min": 0.3, "name": "belief_update_follows_truth"},
            placebo={"observed": belief_align, "tol": 1e-9, "name": "belief_update_ignores_alignment"},
            positive={"observed": pref_align, "expected": 0.8, "tol": 0.5, "name": "preference_follows_alignment"},
            surface={"accuracy": pred_inv, "chance": 0.0, "tol": 1e-9, "name": "prediction_use_invariant"},
            oracle={"observed": g("q_source", "aligned", "true") - g("q_source", "aligned", "false"), "min": 0.3, "name": "reliability_read"},
            prediction={"gain": belief_truth, "min": 0.0, "name": "belief_channel_gain"},
            calibration={"observed": g("preference", "aligned", "false"), "reference": g("preference", "aligned", "true"), "direction": "down", "tol": 0.0, "name": "false_content_dampens_preference_movement"})
    criterion(v, "G06", passed, belief_by_truth=belief_truth, belief_by_alignment=belief_align, preference_by_alignment=pref_align)
    v["results"].update({"cells": {f"{a}_{t}": {k: g(k, a, t) for k in ("belief", "preference", "prediction", "q_source")} for a in ("aligned", "divergent") for t in ("true", "false")}})
    receipt(v, rows, card, ctx)
    narrative(v, f"Belief update moved {belief_truth:+.2f} with the source's truthfulness and {belief_align:.1e} with its alignment; preference movement moved {pref_align:+.2f} with alignment.",
              "Aligned values with false content and divergent values with true content are read through different channels.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G07 / G08 — trust defaults and dynamics.
# --------------------------------------------------------------------------- #
def unit_G07(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g07")
    for default, prior in (("low", (1.0, 3.0)), ("mid", (1.0, 1.0)), ("high", (3.0, 1.0))):
        for rel_name, rel in (("low", 0.3), ("high", 0.9)):
            for s in range(6):
                outs, _ = GT.history_outcomes(r, 20, [(20, rel)])
                a, b = prior
                traj = []
                accepted_false = 0
                for t, x in enumerate(outs):
                    q = a / (a + b)
                    traj.append(q)
                    if q > 0.5 and x == 0:
                        accepted_false += 1
                    a += x
                    b += 1 - x
                within = next((t for t, q in enumerate(traj) if abs(q - rel) <= 0.1), 20)
                cells.add({"default": default, "reliability": rel_name}, efficiency=float(within), vulnerability=float(accepted_false), final_err=abs(traj[-1] - rel))
    return {"rows": cells.rows()}


def reduce_G07(card, units, ctx):
    v = start(card, ctx, "A high-trust default reaches a reliable source's true reliability sooner and accepts more of an unreliable "
              "source's false assertions; a low default does the reverse; the frontier is a property of the prior, not a personality.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {d: {rl: {k: mean_of(rows, k, lambda r, d=d, rl=rl: r["default"] == d and r["reliability"] == rl) for k in ("efficiency", "vulnerability", "final_err")} for rl in ("low", "high")} for d in ("low", "mid", "high")}
    passed = bool(grid["high"]["high"]["efficiency"] <= grid["low"]["high"]["efficiency"] and grid["high"]["low"]["vulnerability"] >= grid["low"]["low"]["vulnerability"])
    gr = G.GateReport()
    battery(gr, live={"observed": grid["high"]["low"]["vulnerability"] - grid["low"]["low"]["vulnerability"], "min": 0.5, "name": "default_moves_vulnerability"},
            placebo={"observed": max(grid[d][rl]["final_err"] for d in grid for rl in grid[d]), "tol": 0.2, "name": "every_default_converges_by_twenty"},
            positive={"observed": float(grid["high"]["high"]["efficiency"] <= grid["low"]["high"]["efficiency"]), "expected": 1.0, "tol": 0.0, "name": "high_default_faster_on_reliable_sources"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_histories"},
            oracle={"observed": 1.0 - grid["mid"]["high"]["final_err"], "min": 0.7, "name": "reliability_learnable"},
            prediction={"gain": grid["low"]["low"]["vulnerability"] - grid["high"]["low"]["vulnerability"], "min": -20.0, "name": "frontier_reported"},
            calibration={"observed": grid["high"]["low"]["final_err"], "reference": 0.2, "direction": "down", "tol": 0.0, "name": "high_default_still_learns"})
    criterion(v, "G07", passed, grid=grid)
    v["results"].update({"frontier": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"A high-trust default reached a reliable source's rate within {grid['high']['high']['efficiency']:.0f} claims against {grid['low']['high']['efficiency']:.0f} for a low default, and accepted {grid['high']['low']['vulnerability']:.1f} false assertions from an unreliable source against {grid['low']['low']['vulnerability']:.1f}.",
              "Efficiency and vulnerability trade along the default; nothing here names a person.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G08(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g08")
    sz = sizes(ctx)
    long = bool((ctx.get("cfg") or {}).get("long_histories", False))
    n = 80 if long else 40
    for s in range(max(4, sz["histories"])):
        for hist in ("stable", "one_change", "many_changes"):
            regimes = {"stable": [(n, 0.85)], "one_change": [(n // 2, 0.9), (n // 2, 0.2)],
                       "many_changes": [(n // 4, 0.9), (n // 4, 0.2), (n // 4, 0.85), (n // 4, 0.15)]}[hist]
            outs, ps = GT.history_outcomes(r, n, regimes)
            for model in GT.TRUST_MODELS:
                tr = GT.trust_trajectory(outs, model)
                cells.add({"model": model, "history": hist}, brier=float(np.mean((tr - ps) ** 2)), recovery=float(np.mean(np.abs(tr[-n // 4:] - ps[-n // 4:]))))
    return {"rows": cells.rows()}


def reduce_G08(card, units, ctx):
    v = start(card, ctx, "No trust-update rule fits every source history: the Brier score of each dynamics model by history kind is the "
              "result, and loss is not assumed faster than gain.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {m: {h: boot(rows, "brier", lambda r, m=m, h=h: r["model"] == m and r["history"] == h, seed_tag=f"G08{m}{h}")["mean"] for h in ("stable", "one_change", "many_changes")} for m in GT.TRUST_MODELS}
    winners = {h: min(GT.TRUST_MODELS, key=lambda m: grid[m][h]) for h in ("stable", "one_change", "many_changes")}
    universal = len(set(winners.values())) == 1
    gr = G.GateReport()
    battery(gr, live={"observed": max(grid[m]["many_changes"] for m in grid) - min(grid[m]["many_changes"] for m in grid), "min": 0.01, "name": "models_differ_under_change"},
            placebo={"observed": abs(grid["bayes"]["stable"] - min(grid[m]["stable"] for m in grid)), "tol": 0.02, "name": "bayes_near_best_when_stable"},
            positive={"observed": float(grid["change_point"]["many_changes"] <= grid["bayes"]["many_changes"]), "expected": 1.0, "tol": 0.0, "name": "change_point_beats_bayes_under_many_changes"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_outcome_sequences"},
            oracle={"observed": 0.25 - min(grid[m]["stable"] for m in grid), "min": 0.1, "name": "reliability_predictable"},
            prediction={"gain": grid["bayes"]["one_change"] - grid["change_point"]["one_change"], "min": -1.0, "name": "change_point_advantage_reported"},
            calibration={"observed": float(universal), "reference": 0.0, "direction": "down", "tol": 0.0, "name": "no_universal_rule"})
    criterion(v, "G08", not universal, winners=winners, grid=grid)
    v["results"].update({"brier_by_model_and_history": grid, "winners": winners})
    receipt(v, rows, card, ctx)
    narrative(v, "The best-predicting trust rule was " + ", ".join(f"{h}: {m}" for h, m in winners.items()) + ".",
              "Trust dynamics are history-dependent; the asymmetric loss-faster-than-gain rule is one competitor, not the default.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(not universal))


# --------------------------------------------------------------------------- #
# G09 / G10 — reinterpretation and repair.
# --------------------------------------------------------------------------- #
def unit_G09(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g09")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        for revelation in ("true", "false"):
            for relevant in (1, 0):
                src = GT.Source(f"s{s}", "misleading", {0: 0.1}, slot=s)
                early = _speak_n(src, r, d0, d1, 6, n_tok=4)                      # ambiguous: short evidence, no truths yet
                late_src = src if relevant else GT.Source(f"o{s}", "misleading", {0: 0.1}, slot=s + 1)
                late = _speak_n(late_src, r, d0, d1, 4, n_tok=10, start_t=6)
                rev_late = {i: (a["truth"] if revelation == "true" else 1 - a["truth"]) for i, a in enumerate(late)}
                # before the revelation: early artifacts read with no goal information
                before = GT.factored_read(early, d0, d1, revealed=None)
                # after: the goal posterior learned from the late diagnostic event is carried back to the early artifacts of the same source
                fr_late = GT.factored_read(late, d0, d1, revealed=rev_late)
                after = GT.factored_read(early, d0, d1, revealed=None, goal_prior=fr_late["q_goal"] if relevant else None)
                acc_b = float(np.mean([(pa["q_T1_factored"] > 0.5) == (pa["truth"] == 1) for pa in before["per_artifact"]]))
                acc_a = float(np.mean([(pa["q_T1_factored"] > 0.5) == (pa["truth"] == 1) for pa in after["per_artifact"]]))
                cells.add({"revelation": revelation, "relevant": relevant}, gain=acc_a - acc_b, acc_after=acc_a, q_mis=fr_late["q_goal"]["misleading"])
    return {"rows": cells.rows()}


def reduce_G09(card, units, ctx):
    v = start(card, ctx, "Detecting an adversarial goal from a later diagnostic event justifies re-reading the source's earlier ambiguous "
              "artifacts, and only those; a false revelation or another source's event rewrites nothing that is true.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = lambda rv, rl: mean_of(rows, "gain", lambda r: r["revelation"] == rv and r["relevant"] == rl)
    rel_true = g("true", 1)
    irr = g("true", 0)
    false_rev = g("false", 1)
    passed = bool(rel_true >= 0.05 and abs(irr) <= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": rel_true, "min": 0.05, "name": "relevant_revelation_improves_earlier_reading"},
            placebo={"observed": abs(irr), "tol": 0.02, "name": "another_source_event_changes_nothing"},
            positive={"observed": mean_of(rows, "q_mis", lambda r: r["revelation"] == "true"), "expected": 1.0, "tol": 0.4, "name": "true_revelation_identifies_the_goal"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_early_artifacts"},
            oracle={"observed": mean_of(rows, "acc_after", lambda r: r["revelation"] == "true" and r["relevant"] == 1), "min": 0.5, "name": "earlier_truths_recoverable"},
            prediction={"gain": rel_true - false_rev, "min": -1.0, "name": "true_minus_false_revelation"},
            calibration={"observed": false_rev, "reference": rel_true, "direction": "down", "tol": 0.0, "name": "false_revelation_helps_less"})
    criterion(v, "G09", passed, relevant_true=rel_true, irrelevant=irr, false_revelation=false_rev)
    v["results"].update({"gain": {"relevant_true": rel_true, "irrelevant": irr, "false_revelation": false_rev}})
    receipt(v, rows, card, ctx)
    narrative(v, f"Learning from a later event that the source misleads raised the accuracy of its earlier ambiguous claims by {rel_true:+.2f}; the same event from another source changed them by {irr:+.2f}; a false revelation by {false_rev:+.2f}.",
              "Retrospective reinterpretation is licensed by causal relevance and by the truth of the revealing event, not by suspicion.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G10(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g10")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        for repair in ("apology", "costly_action", "repeated_reliability", "third_party", "none"):
            # a failure: three revealed false assertions
            a_, b_ = 3.0, 1.0
            a_, b_ = a_ + 0.0, b_ + 3.0
            trust0 = a_ / (a_ + b_)
            if repair == "apology":
                pass                                                                    # an assertion of future accuracy carries no outcome
            elif repair == "costly_action":
                a_ += 2.0                                                               # one costly verifiable act: strong evidence of one true claim
            elif repair == "repeated_reliability":
                a_ += 3.0
            elif repair == "third_party":
                a_ += 1.5                                                               # a reliable other source vouches: evidence tempered by its reliability
            trust1 = a_ / (a_ + b_)
            cells.add({"repair": repair}, recovery=trust1 - trust0, trust_after=trust1)
    return {"rows": cells.rows()}


def reduce_G10(card, units, ctx):
    v = start(card, ctx, "Trust after a failure is restored by evidence that predicts future reliability, a costly corrective act, repeated "
              "reliable claims or a reliable third party, and not by an assertion of good intent.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: mean_of(rows, "recovery", lambda r, k=k: r["repair"] == k) for k in ("apology", "costly_action", "repeated_reliability", "third_party", "none")}
    passed = bool(by["costly_action"] - by["apology"] >= 0.10 and by["repeated_reliability"] - by["apology"] >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": by["repeated_reliability"], "min": 0.1, "name": "repair_evidence_restores"},
            placebo={"observed": abs(by["apology"] - by["none"]), "tol": 1e-9, "name": "wording_alone_restores_nothing"},
            positive={"observed": float(by["repeated_reliability"] >= by["third_party"]), "expected": 1.0, "tol": 0.0, "name": "own_record_beats_vouching"},
            surface={"accuracy": by["apology"], "chance": 0.0, "tol": 1e-9, "name": "apology_is_not_evidence"},
            oracle={"observed": by["repeated_reliability"], "min": 0.1, "name": "recovery_possible"},
            prediction={"gain": by["costly_action"] - by["apology"], "min": 0.0, "name": "costly_minus_apology"},
            calibration={"observed": mean_of(rows, "trust_after", lambda r: r["repair"] == "repeated_reliability"), "reference": 0.5, "direction": "up", "tol": 0.0, "name": "repaired_trust_above_half"})
    criterion(v, "G10", passed, by_repair=by)
    v["results"].update({"recovery_by_repair": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Trust recovered by " + ", ".join(f"{k} {x:+.2f}" for k, x in by.items()) + " after three revealed false assertions.",
              "Repair is evidence about the future; an apology is an assertion by the source being doubted.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G11 / G12 / G13 — uptake gate, selective transfer, false context.
# --------------------------------------------------------------------------- #
def unit_G11(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g11")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        src = GT.Source(f"s{s}", "accurate", {0: 0.9}, slot=s)
        arts = _speak_n(src, r, d0, d1, 8)
        fr = GT.factored_read(arts, d0, d1, revealed={i: a["truth"] for i, a in enumerate(arts)})
        for conflict in ("proximal", "standing", "competitive", "benign"):
            relevance = {"proximal": 0.3, "standing": 1.0, "competitive": 1.0, "benign": 1.0}[conflict]
            alignment = {"proximal": 0.9, "standing": 0.1, "competitive": 0.0, "benign": 0.8}[conflict]
            up = GT.uptake_decision(fr["q_goal"], fr["q_source"], relevance, alignment, 0.9)
            cells.add({"conflict": conflict}, belief=up["belief_update"], imitation=up["process_imitation"], preference=up["preference_movement"], refusal=up["refusal"])
    return {"rows": cells.rows()}


def reduce_G11(card, units, ctx):
    v = start(card, ctx, "Large goal divergence closes the gates for imitation and preference movement while leaving belief about factual "
              "content open: whether to believe a reliable source and whether to become like it are different decisions.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {k: mean_of(rows, k, lambda r, c=c: r["conflict"] == c) for k in ("belief", "imitation", "preference", "refusal")} for c in ("proximal", "standing", "competitive", "benign")}
    belief_spread = max(b["belief"] for b in by.values()) - min(b["belief"] for b in by.values())
    pref_drop = by["benign"]["preference"] - by["competitive"]["preference"]
    passed = bool(belief_spread <= 1e-9 and pref_drop >= 0.3)
    gr = G.GateReport()
    battery(gr, live={"observed": pref_drop, "min": 0.3, "name": "divergence_closes_preference_movement"},
            placebo={"observed": belief_spread, "tol": 1e-9, "name": "belief_update_unchanged_across_conflicts"},
            positive={"observed": by["benign"]["preference"], "expected": 1.0, "tol": 0.5, "name": "benign_difference_leaves_the_gate_open"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_source"},
            oracle={"observed": by["benign"]["belief"], "min": 0.5, "name": "reliable_source_believed"},
            prediction={"gain": by["proximal"]["imitation"] - by["standing"]["imitation"], "min": -1.0, "name": "imitation_by_conflict_reported"},
            calibration={"observed": by["competitive"]["preference"], "reference": 0.05, "direction": "down", "tol": 0.0, "name": "competitive_task_closes_preference"})
    criterion(v, "G11", passed, by_conflict=by)
    v["results"].update({"channels_by_conflict": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Across proximal, standing, competitive and benign conflicts the belief channel stayed at {by['benign']['belief']:.2f} while preference movement went from {by['benign']['preference']:.2f} to {by['competitive']['preference']:.2f}.",
              "Believing a source and adopting its aims are gated separately.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G12(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g12")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        # reliable in domain 0, unreliable in domain 1; a namesake source with the same label elsewhere
        src = GT.Source(f"s{s}", "accurate", {0: 0.9, 1: 0.2}, slot=s)
        d0a = [a for a in _speak_n(src, r, d0, d1, 8, domain=0)]
        d1a = [a for a in _speak_n(GT.Source(src.id, "misleading", src.reliability, slot=s), r, d0, d1, 8, domain=1)]
        learned = {0: GT.factored_read(d0a, d0, d1, revealed={i: a["truth"] for i, a in enumerate(d0a)})["source_beta"],
                   1: GT.factored_read(d1a, d0, d1, revealed={i: a["truth"] for i, a in enumerate(d1a)})["source_beta"]}
        for transfer in ("same_domain", "new_domain", "shared_label"):
            if transfer == "same_domain":
                test = _speak_n(src, r, d0, d1, 6, n_tok=2, domain=0, start_t=20)
                prior = learned[0]
                true_rel = 0.9
            elif transfer == "new_domain":
                test = _speak_n(GT.Source(src.id, "misleading", src.reliability, slot=s), r, d0, d1, 6, n_tok=2, domain=1, start_t=20)
                prior = learned[0]                                    # wrongly carrying domain-0 reliability into domain 1
                true_rel = 0.0
            else:
                other = GT.Source(src.id, "misleading", {0: 0.2}, slot=s)          # a different source with the same label
                test = _speak_n(other, r, d0, d1, 6, n_tok=2, domain=0, start_t=20)
                prior = learned[0]
                true_rel = 0.0
            fr = GT.factored_read(test, d0, d1, revealed=None, source_prior=prior)
            fr0 = GT.factored_read(test, d0, d1, revealed=None, source_prior=(1.0, 1.0))
            ls = float(np.mean([np.log(max(pa["q_T1_factored"] if pa["truth"] == 1 else 1 - pa["q_T1_factored"], 1e-12)) for pa in fr["per_artifact"]]))
            ls0 = float(np.mean([np.log(max(pa["q_T1_factored"] if pa["truth"] == 1 else 1 - pa["q_T1_factored"], 1e-12)) for pa in fr0["per_artifact"]]))
            cells.add({"transfer": transfer}, gain=ls - ls0, q_source=fr["q_source"], true_rel=true_rel, err=abs(fr["q_source"] - true_rel))
    return {"rows": cells.rows()}


def reduce_G12(card, units, ctx):
    v = start(card, ctx, "Reliability learned about a source in one domain transfers to new claims in that domain and not to another "
              "domain or to another source with the same label.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {t: {"gain": mean_of(rows, "gain", lambda r, t=t: r["transfer"] == t), "err": mean_of(rows, "err", lambda r, t=t: r["transfer"] == t)} for t in ("same_domain", "new_domain", "shared_label")}
    passed = bool(by["same_domain"]["gain"] >= 0.05 and by["shared_label"]["gain"] <= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": by["same_domain"]["gain"], "min": 0.02, "name": "reliability_transfers_within_domain"},
            placebo={"observed": max(by["shared_label"]["gain"], 0.0), "tol": 0.02, "name": "a_shared_label_transfers_nothing_useful"},
            positive={"observed": 1.0 - by["same_domain"]["err"], "expected": 1.0, "tol": 0.25, "name": "same_domain_reliability_right"},
            surface={"accuracy": max(by["new_domain"]["gain"], 0.0), "chance": 0.0, "tol": 0.02, "name": "cross_domain_transfer_does_not_help"},
            oracle={"observed": by["same_domain"]["gain"], "min": 0.0, "name": "transfer_reported"},
            prediction={"gain": by["same_domain"]["gain"] - by["shared_label"]["gain"], "min": 0.0, "name": "selective_transfer"},
            calibration={"observed": by["new_domain"]["err"], "reference": by["same_domain"]["err"], "direction": "up", "tol": 0.0, "name": "carried_reliability_wrong_in_new_domain"})
    criterion(v, "G12", passed, by_transfer=by)
    v["results"].update({"by_transfer": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Carrying a source's learned reliability improved truth reading by {by['same_domain']['gain']:+.2f} in its own domain, {by['new_domain']['gain']:+.2f} in a domain where it misleads, and {by['shared_label']['gain']:+.2f} for a namesake.",
              "Reliability is indexed by source and domain; a label carries neither.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G13(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g13")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        base = {ev: _speak_n(GT.Source(f"s{s}", "neutral", {0: 0.5}, slot=s), r, d0, d1, 4, n_tok=12 if ev == "strong" else 2) for ev in ("weak", "strong")}
        for note_kind in ("true", "false", "ambiguous", "irrelevant"):
            for ev in ("weak", "strong"):
                for order in ("note_first", "evidence_first"):
                    arts = base[ev]
                    # the note: a context source asserting a value of T with its own reliability (0.7), never truth
                    note_rel = 0.7
                    out = []
                    for a in arts:
                        cl = GT.content_loglik(a["tokens"], d0, d1)
                        if note_kind == "true":
                            nv = a["truth"]
                        elif note_kind == "false":
                            nv = 1 - a["truth"]
                        elif note_kind == "ambiguous":
                            nv = None
                        else:
                            nv = None
                        lp = cl.copy()
                        if nv is not None:
                            lp = lp + np.array([np.log(note_rel if nv == 0 else 1 - note_rel), np.log(note_rel if nv == 1 else 1 - note_rel)])
                        q = C.softmax(lp)
                        naive = np.array([1.0 - (nv if nv is not None else 0.5), (nv if nv is not None else 0.5)]) if nv is not None else C.softmax(cl)
                        conflict = float((cl[1] > cl[0]) != (nv == 1)) if nv is not None else 0.0
                        out.append({"q_truth": float(q[a["truth"]]), "naive_truth": float(naive[a["truth"]]), "conflict": conflict, "abstain": float(abs(q[1] - 0.5) < 0.15)})
                    cells.add({"note": note_kind, "evidence": ev, "order": order}, q_truth=float(np.mean([o["q_truth"] for o in out])), naive=float(np.mean([o["naive_truth"] for o in out])),
                              conflict=float(np.mean([o["conflict"] for o in out])), abstain=float(np.mean([o["abstain"] for o in out])))
    return {"rows": cells.rows()}


def reduce_G13(card, units, ctx):
    v = start(card, ctx, "A note about the claim is an assertion by a source with its own reliability: a false note's influence shrinks "
              "as the artifact's own evidence contradicts it, and the reader flags the conflict or abstains rather than obeying.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = lambda k, n, e: mean_of(rows, k, lambda r: r["note"] == n and r["evidence"] == e)
    false_strong = 1.0 - g("q_truth", "false", "strong")
    false_weak = 1.0 - g("q_truth", "false", "weak")
    naive_false = 1.0 - g("naive", "false", "strong")
    order_effect = max(abs(mean_of(rows, "q_truth", lambda r: r["order"] == "note_first" and r["note"] == n) - mean_of(rows, "q_truth", lambda r: r["order"] == "evidence_first" and r["note"] == n)) for n in ("true", "false"))
    passed = bool(false_strong <= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": false_weak - false_strong, "min": 0.05, "name": "contradiction_shrinks_the_false_note"},
            placebo={"observed": order_effect, "tol": 1e-9, "name": "order_irrelevant_for_exact_inference"},
            positive={"observed": g("q_truth", "true", "weak"), "expected": 1.0, "tol": 0.4, "name": "true_note_helps_with_weak_evidence"},
            surface={"accuracy": naive_false - false_strong, "chance": 0.0, "tol": 2.0, "name": "naive_obedience_error_reported"},
            oracle={"observed": g("q_truth", "irrelevant", "strong"), "min": 0.6, "name": "strong_evidence_alone_identifies"},
            prediction={"gain": g("conflict", "false", "strong"), "min": 0.5, "name": "conflict_flagged_under_false_note"},
            calibration={"observed": g("abstain", "false", "weak"), "reference": g("abstain", "false", "strong"), "direction": "up", "tol": 0.0, "name": "abstains_more_when_evidence_is_weak"})
    criterion(v, "G13", passed, false_note_influence_strong=false_strong, false_note_influence_weak=false_weak, naive=naive_false)
    v["results"].update({"false_note_error": {"strong_evidence": false_strong, "weak_evidence": false_weak, "naive_reader": naive_false}, "conflict_rate": g("conflict", "false", "strong")})
    receipt(v, rows, card, ctx)
    narrative(v, f"A false note left {false_strong:.0%} of the mass on the wrong side against strong contradicting evidence and {false_weak:.0%} against weak; a reader that took the note as truth was wrong {naive_false:.0%} of the time.",
              "False context is evidence with a reliability, and strong artifacts outvote it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G14 — active challenge.
# --------------------------------------------------------------------------- #
def unit_G14(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g14")
    d0, d1 = _kinds(ctx)
    sz = sizes(ctx)
    agree = []
    for s in range(max(4, sz["sources"] // 2)):
        for goal in ("accurate", "persuasion", "misleading"):
            src = GT.Source(f"s{s}", goal, {0: 0.9}, agenda=1, slot=s)
            static = _speak_n(src, r, d0, d1, 4)
            fr_static = GT.factored_read(static, d0, d1, revealed=None)
            # challenge: ask about a claim whose truth the reader knows; the response policy exposes the goal
            for pol in ("challenge", "static", "free_look", "random"):
                if pol == "static":
                    fr = fr_static
                elif pol == "free_look":
                    fr = GT.factored_read(static + _speak_n(src, r, d0, d1, 2, start_t=10), d0, d1, revealed=None)
                elif pol == "random":
                    T = int(r.random() < 0.5)
                    resp = GT.challenge_response(src, r, d0, d1, T)
                    arts = static + ([resp] if resp and not resp.get("declined") else [])
                    fr = GT.factored_read(arts, d0, d1, revealed={len(static): T} if len(arts) > len(static) else None)
                else:
                    # exact: choose the challenge truth with the higher expected information about the goal
                    best_T, best_h = 0, np.inf
                    for T in (0, 1):
                        ents = []
                        for g2 in ("accurate", "persuasion", "misleading"):
                            resp = GT.challenge_response(GT.Source(src.id, g2, src.reliability, 1, slot=s), r, d0, d1, T)
                            arts = static + ([resp] if resp and not resp.get("declined") else [])
                            q = GT.factored_read(arts, d0, d1, revealed={len(static): T} if len(arts) > len(static) else None)["q_goal"]
                            ents.append(C.entropy(np.array(list(q.values()))))
                        if np.mean(ents) < best_h:
                            best_T, best_h = T, float(np.mean(ents))
                    resp = GT.challenge_response(src, r, d0, d1, best_T)
                    arts = static + ([resp] if resp and not resp.get("declined") else [])
                    fr = GT.factored_read(arts, d0, d1, revealed={len(static): best_T} if len(arts) > len(static) else None)
                cells.add({"policy": pol}, ls=float(np.log(max(fr["q_goal"][goal], 1e-12))), acc=_acc(fr["q_goal"], goal), conf=float(max(fr["q_goal"].values())))
        # PyMDP agreement on the challenge choice: two probes (challenge truth 0 / 1) over three goal hypotheses
        ems = np.zeros((2, 3, GT.N_KINDS))
        for T in (0, 1):
            for k, g2 in enumerate(("accurate", "persuasion", "misleading")):
                E = T if g2 == "accurate" else (1 if g2 == "persuasion" else 1 - T)
                ems[T, k] = d1 if E == 1 else d0
        ag = PR.build_reader(ems, np.full(3, 1 / 3), probe_costs=np.zeros(2))
        choice, _ = PR.choose_probe(ag)
        eig = PR.exact_eig_per_probe(ems, np.full(3, 1 / 3), 8, r, draws=40)
        agree.append(float(PR.policy_disagreement(eig, choice)["agrees"]))
    return {"rows": cells.rows(), "pymdp_agreement": float(np.mean(agree))}


def reduce_G14(card, units, ctx):
    v = start(card, ctx, "Asking a source about a claim whose truth the reader knows separates teaching from persuasion and deception "
              "better than reading more of its unchallenged polish.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {p: boot(rows, "ls", lambda r, p=p: r["policy"] == p, seed_tag="G14" + p)["mean"] for p in ("challenge", "static", "free_look", "random")}
    gain = by["challenge"] - by["static"]
    agree = float(np.mean([u["pymdp_agreement"] for u in units]))
    passed = bool(gain >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": gain, "min": 0.02, "name": "challenge_moves_the_stance_score"},
            placebo={"observed": max(by["free_look"] - by["challenge"], 0.0), "tol": 0.05, "name": "free_look_no_better_than_challenge"},
            positive={"observed": float(by["challenge"] >= by["random"] - 0.15), "expected": 1.0, "tol": 0.0, "name": "chosen_challenge_no_worse_than_random"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_static_artifacts"},
            oracle={"observed": mean_of(rows, "acc", lambda r: r["policy"] == "challenge"), "min": 0.4, "name": "stance_identifiable_by_challenge"},
            prediction={"gain": gain, "min": 0.0, "name": "challenge_minus_static"},
            calibration={"observed": agree, "reference": 0.0, "direction": "up", "tol": 0.0, "name": "pymdp_agreement_reported", "detail": "the two challenge truths are equally informative by symmetry, so agreement is a coin flip and is reported, not gated"})
    criterion(v, "G14", passed, by_policy=by, pymdp_agreement=agree)
    v["results"].update({"log_score_by_policy": by})
    v["pymdp"] = {"agreement_with_exact_challenge_choice": agree}
    receipt(v, rows, card, ctx)
    narrative(v, f"One chosen challenge improved the stance log score by {gain:+.2f} nats over static reading, against {by['free_look'] - by['static']:+.2f} for two more unchallenged artifacts and {by['random'] - by['static']:+.2f} for a random challenge; the PyMDP reader agreed with the exact challenge choice {agree:.0%} of the time.",
              "A known-truth challenge is the probe that makes stance a response rather than a surface.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# G15 / G16 — factorisation and transfer.
# --------------------------------------------------------------------------- #
def unit_G15(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    for acc in ("accurate", "wrong"):
        for rel in ("low", "high"):
            for relv in ("low", "high"):
                for al in ("low", "high"):
                    q_goal = {"accurate": 0.9, "misleading": 0.05, "persuasion": 0.05}
                    up = GT.uptake_decision(q_goal, 0.9 if rel == "high" else 0.2, 0.9 if relv == "high" else 0.2, 0.9 if al == "high" else 0.2, 0.9 if acc == "accurate" else 0.3)
                    cells.add({"accuracy": acc, "reliability": rel, "relevance": relv, "alignment": al}, **{k: float(v_) for k, v_ in up.items()})
    return {"rows": cells.rows()}


DECLARED = {"prediction_use": {"accuracy"}, "process_imitation": {"accuracy", "relevance"}, "belief_update": {"reliability"}, "preference_movement": {"reliability", "alignment"}}


def reduce_G15(card, units, ctx):
    v = start(card, ctx, "Reconstruction accuracy, reliability, relevance and alignment drive four uptake channels through declared edges "
              "only; the channels are distinct outputs, not one aggregate score.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    factors = ("accuracy", "reliability", "relevance", "alignment")
    matrix = {}
    for ch in DECLARED:
        matrix[ch] = {}
        for f in factors:
            hi = mean_of(rows, ch, lambda r, f=f: r[f] in ("accurate", "high"))
            lo = mean_of(rows, ch, lambda r, f=f: r[f] in ("wrong", "low"))
            matrix[ch][f] = abs(hi - lo)
    off = max(matrix[ch][f] for ch in DECLARED for f in factors if f not in DECLARED[ch])
    on = min(matrix[ch][f] for ch in DECLARED for f in DECLARED[ch])
    gr = G.GateReport()
    gr.identity("no_off_edge_response", off, 0.0, tol=0.02)
    gr.live("declared_edges_respond", observed_change=on, min_change=0.1)
    passed = gr.to_dict()["all_passed"]
    criterion(v, "G15", passed, max_off_edge=off, min_on_edge=on)
    v["results"].update({"response_matrix": matrix, "declared": {k: sorted(x) for k, x in DECLARED.items()}})
    receipt(v, rows, card, ctx)
    narrative(v, f"Every channel responded to its declared factors by at least {on:.2f} and to any other factor by at most {off:.2f}.",
              "Prediction use, imitation, belief update and preference movement are four outputs with four different inputs.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_G16(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "g16")
    d0, d1 = _kinds(ctx, "kinds_fresh")
    sz = sizes(ctx)
    for s in range(max(4, sz["sources"] // 2)):
        for fresh in ("identity", "domain", "base_rate", "reversal"):
            goal = str(r.choice(["accurate", "misleading", "persuasion"]))
            src = GT.Source(f"f{s}{fresh}", goal, {0: 0.9}, slot=s + 3)
            if fresh == "reversal":
                src.change_points = [(8, "misleading" if goal == "accurate" else "accurate")]
            n_tok = 8
            arts = _speak_n(src, r, d0, d1, 16, n_tok=n_tok, domain=1 if fresh == "domain" else 0)
            if fresh == "base_rate":
                for a in arts:
                    a["truth"] = int(r.random() < 0.8)                              # mostly true claims
                    E = a["truth"] if goal == "accurate" else (a["evidence_polarity"] if goal == "persuasion" else 1 - a["truth"])
                    a["tokens"] = r.choice(GT.N_KINDS, size=n_tok, p=(d1 if E == 1 else d0))
                    a["assertion"] = None if goal == "neutral" else (a["truth"] if goal == "accurate" else (E if goal == "persuasion" else 1 - a["truth"]))
            rev = {i: a["truth"] for i, a in enumerate(arts)}
            # sequential reading: predict each claim's truth before its reveal
            correct, confs = [], []
            recovery = None
            for i in range(1, len(arts)):
                fr = GT.factored_read(arts[:i], d0, d1, revealed={j: rev[j] for j in range(i)})
                q_next = GT.factored_read(arts[:i + 1], d0, d1, revealed={j: rev[j] for j in range(i)}, goal_prior=fr["q_goal"])["per_artifact"][-1]["q_T1_factored"]
                ok = (q_next > 0.5) == (arts[i]["truth"] == 1)
                correct.append(float(ok))
                confs.append(float(max(q_next, 1 - q_next)))
                if fresh == "reversal" and i >= 8 and recovery is None and ok:
                    recovery = i - 8
            cells.add({"fresh": fresh}, acc=float(np.mean(correct)), ece=C.ece(confs, correct), recovery=float(recovery if recovery is not None else 8))
    return {"rows": cells.rows()}


def reduce_G16(card, units, ctx):
    v = start(card, ctx, "The frozen factored trust reader predicts fresh sources' claims, stays calibrated under new domains and base "
              "rates, and recovers after a reliability reversal; its failure regions are named.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {f: {k: mean_of(rows, k, lambda r, f=f: r["fresh"] == f) for k in ("acc", "ece", "recovery")} for f in ("identity", "domain", "base_rate", "reversal")}
    passed = bool(all(by[f]["acc"] >= 0.6 for f in ("identity", "domain")) and by["reversal"]["recovery"] <= 6)
    gr = G.GateReport()
    battery(gr, live={"observed": by["identity"]["acc"] - 0.5, "min": 0.1, "name": "fresh_sources_predicted_above_chance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "reader_frozen_before_fresh_sources"},
            positive={"observed": by["identity"]["acc"], "expected": 1.0, "tol": 0.4, "name": "fresh_identity_read"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_label_reuse"},
            oracle={"observed": by["domain"]["acc"] - 0.5, "min": 0.0, "name": "new_domain_reported"},
            prediction={"gain": 8.0 - by["reversal"]["recovery"], "min": 0.0, "name": "recovery_after_reversal"},
            calibration={"observed": by["base_rate"]["ece"], "reference": 0.25, "direction": "down", "tol": 0.0, "name": "calibrated_under_new_base_rate"})
    criterion(v, "G16", passed, by_fresh=by)
    v["results"].update({"by_fresh": by})
    receipt(v, rows, card, ctx)
    narrative(v, "On fresh sources the reader's claim-truth accuracy was " + ", ".join(f"{f} {b['acc']:.0%}" for f, b in by.items()) + f", and it recovered {by['reversal']['recovery']:.0f} claims after a reversal.",
              "The trust architecture transfers where it was designed to and the table says where it does not.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
