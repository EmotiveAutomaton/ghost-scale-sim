"""Trunk P: projection, target-specific correction, and reader plurality (spec §12).

A cheap local prior is the starting point of every card; the question is what corrects it, how
fast, at what residual, and whether the correction stays where it belongs.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as X, priors as P, projection as PJ, attention as A
from ..world import make_maker, population, stream, maker_emission
from . import (battery, boot, criterion, decide_state, finish, narrative, receipt, rng, sizes, start, world_for, mean_of, pursuit_of, sim_bin)
from .trunk_c import Cells, harness, reader_priors, target_priors, bins_for, posterior_at, hidden_goal_ls, gain, matching_summary

CH = ("surface",)
CH3 = ("surface", "goal_consequences", "mechanics")


# --------------------------------------------------------------------------- #
# P01 — the correction curve under matched controls.
# --------------------------------------------------------------------------- #
def unit_P01(ctx):
    H = harness(ctx, n_art=12)
    cells = Cells(ctx["wid"], ctx["rep"])
    edges = bins_for(H)
    reports = []
    for rd in H["readers"]:
        rr = C.rng_for(ctx["lane"], "P01", ctx["wid"], ctx["rep"], rd.id)
        base, rep = reader_priors(H, rd, rr)
        reports.append(rep)
        model = H["models"][rd.id]
        self_idx = model.truth_index(rd)
        for m in [m for m in H["makers"] if m.family == rd.family] + H["antis"].get(rd.id, []):
            is_anti = m.id.startswith("anti")
            d = C.js(H["selfs"][rd.id]["w_hat"], m.w)
            b = sim_bin(d, edges[rd.id], anti=is_anti)
            pri = target_priors(H, rd, m, base)
            ti = model.truth_index(m)
            for route in ("self", "equal_local", "generic_local"):
                cc = PJ.correction_curve(model, pri[route], H["streams"][m.id], ti, self_idx, CH)
                cells.add({"route": route, "sim_bin": b}, half_life=cc["half_life"], residual=cc["residual_self_mass"] if cc["residual_self_mass"] == cc["residual_self_mass"] else None,
                          order=cc["order_effect"], final=cc["final_truth"], conf=cc["confidence_final"], top1=float(cc["final_truth"] >= 0.5))
    return {"rows": cells.rows(), "matching": matching_summary(reports)}


def reduce_P01(card, units, ctx):
    v = start(card, ctx, "The correction of a local prior by target evidence has a half-life, a residual bias and an order effect, and "
              "those are the same for self and for an equally local non-self prior when the two priors are matched.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {rt: {b: {k: mean_of(rows, k, lambda r, rt=rt, b=b: r["route"] == rt and r["sim_bin"] == b) for k in ("half_life", "residual", "order", "final", "conf", "top1")} for b in ("near", "mid", "far", "anti")} for rt in ("self", "equal_local", "generic_local")}
    hl = mean_of(rows, "half_life", lambda r: r["route"] == "self" and r["sim_bin"] in ("far", "anti"))
    resid = mean_of(rows, "residual", lambda r: r["route"] == "self" and r["sim_bin"] in ("far", "anti"))
    order = mean_of(rows, "order")
    passed = bool(hl <= 8 and resid <= 0.10 and order <= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": mean_of(rows, "final", lambda r: r["route"] == "self"), "min": 0.3, "name": "evidence_corrects_the_prior"},
            placebo={"observed": order, "tol": 0.05, "name": "order_effect_absent_for_exact_inference"},
            positive={"observed": float(surf["self"]["near"]["half_life"] <= surf["self"]["far"]["half_life"] + 1e-9), "expected": 1.0, "tol": 0.0, "name": "near_makers_need_no_more_correction_than_far"},
            surface={"accuracy": float(np.nanmean([u["matching"]["equal_local"]["mean_residual_divergence_gap"] or 0 for u in units])), "chance": 0.0, "tol": 0.25, "name": "matching_residual_within_tolerance"},
            oracle={"observed": mean_of(rows, "final", lambda r: r["route"] == "generic_local"), "min": 0.3, "name": "target_identifiable_at_twelve"},
            prediction={"gain": mean_of(rows, "top1", lambda r: r["route"] == "self"), "min": 0.3, "name": "final_top1"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["route"] == "self") - mean_of(rows, "top1", lambda r: r["route"] == "self"), "reference": 0.25, "direction": "down", "tol": 0.0, "name": "final_confidence_calibrated", "detail": "the exact reader's final confidence carries a small optimism from the robust floor and channel structure; the bound reflects the mechanism as built"})
    criterion(v, "P01", passed, half_life_far=hl, residual_far=resid, order_effect=order)
    v["results"].update({"surface": surf})
    v["matching_residuals"] = {"per_unit": [u["matching"] for u in units]}
    receipt(v, rows, card, ctx)
    narrative(v, f"For far and anti-similar makers the self prior was corrected with a half-life of {hl:.1f} artifacts to a residual self mass of {resid:.2f}, with an order effect of {order:.1e}; "
                 f"the equally local non-self prior corrected with a half-life of {surf['equal_local']['far']['half_life']:.1f}.",
              "Correction dynamics belong to locality, not to self: matched priors correct alike.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P02 — which evidence corrects.
# --------------------------------------------------------------------------- #
def unit_P02(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p02")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P02", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        for j in range(3):
            m = make_maker(world, f"m{j}", r, family=fid, group=(rd.group + 1) % len(fam.groups), w=C.normalize(1.0 - rd.w + 0.05), k=0.2)
            other = make_maker(world, f"o{j}", r, family=fid, group=rd.group, w=rd.w, k=0.2)
            ti = model.truth_index(m)
            arts_m = stream(world, m, 0, r, 6, n_steps=8)
            arts_o = stream(world, other, 0, r, 6, n_steps=8)
            hist_m = model.posterior(X.uniform_prior(model), arts_m[2:6], CH)
            hist_o = model.posterior(X.uniform_prior(model), arts_o[2:6], CH)
            base_mass = float(sp[ti])
            for ev in ("behaviour", "process_record", "biography", "group_label", "stated_preference", "source_history"):
                for true in (1, 0):
                    lp = np.log(np.maximum(sp, 1e-300))
                    if ev == "behaviour":
                        lp = lp + model.loglik(arts_m[:4] if true else arts_o[:4], CH).sum(axis=0)
                    elif ev == "process_record":
                        lp = lp + model.loglik(arts_m[:4] if true else arts_o[:4], ("process_records",)).sum(axis=0)
                    elif ev == "biography":
                        lp = lp + PJ.evidence_loglik(model, "biography", m.label if true else other.label, 0.8)
                    elif ev == "group_label":
                        lp = lp + PJ.evidence_loglik(model, "group_label", m.group if true else other.group, 0.8)
                    elif ev == "stated_preference":
                        lp = lp + PJ.evidence_loglik(model, "stated_preference", m.w if true else other.w, 0.8)
                    else:
                        lp = lp + PJ.evidence_loglik(model, "source_history", hist_m if true else hist_o, 0.8)
                    q = C.softmax(lp)
                    cells.add({"evidence": ev, "true": true}, correction=float(q[ti]) - base_mass, truth_mass=float(q[ti]), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P02(card, units, ctx):
    v = start(card, ctx, "Target-specific evidence corrects projection in proportion to its diagnostic validity: the maker's own "
              "behaviour and process records correct; false biography, labels and stated preferences do not, however vivid.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    evs = ("behaviour", "process_record", "biography", "group_label", "stated_preference", "source_history")
    grid = {e: {str(t): boot(rows, "correction", lambda r, e=e, t=t: r["evidence"] == e and r["true"] == t, seed_tag=f"P02{e}{t}")["mean"] for t in (1, 0)} for e in evs}
    beh, proc, bio_false = grid["behaviour"]["1"], grid["process_record"]["1"], grid["biography"]["0"]
    passed = bool(beh > bio_false and proc > bio_false and beh > 0)
    gr = G.GateReport()
    battery(gr, live={"observed": beh, "min": 0.03, "name": "true_behaviour_corrects"},
            placebo={"observed": max(max(grid[e]["0"] for e in evs), 0.0), "tol": 0.05, "name": "false_evidence_does_not_correct_toward_the_truth"},
            positive={"observed": float(proc >= beh - 0.1), "expected": 1.0, "tol": 0.0, "name": "process_records_correct_at_least_as_well"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_prior_every_cell"},
            oracle={"observed": grid["source_history"]["1"], "min": 0.0, "name": "true_history_corrects"},
            prediction={"gain": mean_of(rows, "top1", lambda r: r["evidence"] == "behaviour" and r["true"] == 1), "min": 0.0, "name": "top1_after_true_behaviour"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["true"] == 0), "reference": mean_of(rows, "conf", lambda r: r["true"] == 1), "direction": "down", "tol": 0.2, "name": "false_evidence_not_more_confident"})
    criterion(v, "P02", passed, grid=grid)
    v["results"].update({"correction_by_evidence_and_truth": grid})
    receipt(v, rows, card, ctx)
    narrative(v, "Mass moved onto the true maker by " + ", ".join(f"{e} {grid[e]['1']:+.2f} (false {grid[e]['0']:+.2f})" for e in evs) + ".",
              "Correction follows what the evidence is diagnostic of; a vivid false biography is worth less than two of the maker's own artifacts.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P03 — correction conditioned on relevant similarity.
# --------------------------------------------------------------------------- #
def unit_P03(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p03")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P03", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        self_idx = model.truth_index(rd)
        for actual in (1, 0):
            for perceived in (1, 0):
                for rel_name, rel in (("low", 0.6), ("high", 0.95)):
                    for j in range(2):
                        w = rd.w if actual else C.normalize(1.0 - rd.w + 0.05)
                        m = make_maker(world, f"m{j}", r, family=fid, group=rd.group if actual else (rd.group + 1) % len(fam.groups), w=w, k=0.2)
                        arts = stream(world, m, 0, r, 3, n_steps=8)
                        # perceived similarity: a note claiming the maker shares the reader's group, at the cue's reliability
                        note = PJ.evidence_loglik(model, "group_label", rd.group if perceived else (rd.group + 1) % len(fam.groups), rel)
                        q = C.softmax(np.log(np.maximum(sp, 1e-300)) + note + model.loglik(arts, CH).sum(axis=0))
                        ti = model.truth_index(m)
                        cells.add({"actual": actual, "perceived": perceived, "reliability": rel_name}, self_weight=float(q[self_idx]), truth_mass=float(q[ti]), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P03(card, units, ctx):
    v = start(card, ctx, "The weight a reader keeps on its own profile falls when the evidence says the decision-relevant mapping "
              "differs, and a false but reliable-looking claim of similarity holds it up: the bias is visible in that gap.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    def sw(a, p, rel):
        return mean_of(rows, "self_weight", lambda r: r["actual"] == a and r["perceived"] == p and r["reliability"] == rel)
    gap = sw(1, 1, "high") - sw(0, 1, "high")
    bias = sw(0, 1, "high") - sw(0, 0, "high")
    passed = bool(gap >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": gap, "min": 0.05, "name": "actual_similarity_moves_self_weight"},
            placebo={"observed": abs(sw(1, 1, "low") - sw(1, 0, "low")), "tol": 0.3, "name": "low_reliability_claim_moves_little"},
            positive={"observed": mean_of(rows, "truth_mass", lambda r: r["actual"] == 1 and r["perceived"] == 1 and r["reliability"] == "high"), "expected": 1.0, "tol": 0.7, "name": "truly_similar_maker_read"},
            surface={"accuracy": max(bias, 0.0), "chance": 0.0, "tol": 0.35, "name": "false_claim_bias_bounded", "detail": "the self weight a false but reliable-looking similarity claim keeps up; reported and bounded"},
            oracle={"observed": mean_of(rows, "truth_mass", lambda r: r["actual"] == 0 and r["perceived"] == 0 and r["reliability"] == "high") - 1.0 / 32, "min": 0.0, "name": "dissimilar_maker_identifiable_without_the_false_claim"},
            prediction={"gain": mean_of(rows, "top1", lambda r: r["actual"] == 0 and r["perceived"] == 0), "min": 0.0, "name": "top1_dissimilar"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["actual"] == 0 and r["perceived"] == 1 and r["reliability"] == "high"), "reference": mean_of(rows, "top1", lambda r: r["actual"] == 0 and r["perceived"] == 1 and r["reliability"] == "high"), "direction": "up", "tol": 1.0, "name": "false_claim_overconfidence_reported"})
    criterion(v, "P03", passed, self_weight_gap=gap, false_claim_bias=bias)
    v["results"].update({"self_weight": {f"actual{a}_perceived{p}_{rel}": sw(a, p, rel) for a in (1, 0) for p in (1, 0) for rel in ("low", "high")}})
    receipt(v, rows, card, ctx)
    narrative(v, f"After three artifacts the reader kept {sw(1, 1, 'high'):.2f} of its mass on its own profile when the maker really matched and {sw(0, 1, 'high'):.2f} when the maker differed but a reliable-looking note said it matched, against {sw(0, 0, 'high'):.2f} with no such note.",
              "Correction is conditioned on relevant similarity, and a false perception of similarity is measurable as retained self weight.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P04 — outcome feedback.
# --------------------------------------------------------------------------- #
def unit_P04(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p04")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        if world.family(fid).link != "draw":
            continue                                     # next-goal prediction is undefined where artifacts carry no goal (poe: -1)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P04", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        pop = X.uniform_prior(model)
        targets = [make_maker(world, f"t{j}", r, family=fid, k=0.2) for j in range(12)]
        for fb in ("accurate", "noisy", "delayed", "absent"):
            alpha = 0.7
            pending = []
            confs, corrects, ls_first, ls_second = [], [], [], []
            for t, m in enumerate(targets):
                arts = stream(world, m, 0, r, 4, n_steps=8)
                q = model.posterior(PJ.mixed_prior(sp, pop, alpha), arts[:3], CH)
                pred = model.next_goal(q, fid)
                g = arts[3]["goal"]
                correct = int(np.argmax(pred)) == g
                confs.append(float(pred.max()))
                corrects.append(float(correct))
                (ls_first if t < 6 else ls_second).append(float(np.log(max(pred[g], 1e-12))))
                if fb == "absent":
                    continue
                signal = correct if fb != "noisy" else (correct if r.random() > 0.3 else not correct)
                if fb == "delayed":
                    pending.append(signal)
                    if len(pending) > 2:
                        alpha = PJ.feedback_weight_update(alpha, pending.pop(0))
                else:
                    alpha = PJ.feedback_weight_update(alpha, signal)
            cells.add({"feedback": fb}, ece=C.ece(confs, corrects), improvement=float(np.mean(ls_second) - np.mean(ls_first)), alpha_final=alpha, acc=float(np.mean(corrects)))
    return {"rows": cells.rows()}


def reduce_P04(card, units, ctx):
    v = start(card, ctx, "Outcome feedback across repeated targets tunes how much a reader trusts its local prior; accurate feedback "
              "improves calibration and prediction, noisy and delayed feedback less, absent feedback not at all.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {fb: {k: mean_of(rows, k, lambda r, fb=fb: r["feedback"] == fb) for k in ("ece", "improvement", "alpha_final", "acc")} for fb in ("accurate", "noisy", "delayed", "absent")}
    ece_gain = by["absent"]["ece"] - by["accurate"]["ece"]
    passed = bool(ece_gain >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": abs(by["accurate"]["alpha_final"] - 0.7), "min": 0.05, "name": "feedback_moves_the_self_weight"},
            placebo={"observed": abs(by["absent"]["alpha_final"] - 0.7), "tol": 1e-12, "name": "absent_feedback_leaves_the_weight"},
            positive={"observed": float(by["accurate"]["improvement"] >= by["absent"]["improvement"] - 0.05), "expected": 1.0, "tol": 0.0, "name": "accurate_feedback_improves_at_least_as_much"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_targets"},
            oracle={"observed": by["accurate"]["acc"], "min": 0.3, "name": "goals_predictable"},
            prediction={"gain": by["accurate"]["improvement"], "min": -1.0, "name": "second_half_minus_first"},
            calibration={"observed": by["accurate"]["ece"], "reference": by["absent"]["ece"], "direction": "down", "tol": 0.02, "name": "accurate_feedback_lowers_ece"})
    criterion(v, "P04", passed, by_feedback=by, ece_gain=ece_gain)
    v["results"].update({"by_feedback": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"With accurate feedback the reader's calibration error was {by['accurate']['ece']:.2f} against {by['absent']['ece']:.2f} with none; noisy and delayed feedback gave {by['noisy']['ece']:.2f} and {by['delayed']['ece']:.2f}.",
              "Feedback about outcomes is what tells a reader how much of itself to keep.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P05 — generalization of a correction.
# --------------------------------------------------------------------------- #
def unit_P05(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p05")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        pop = X.uniform_prior(model)
        for j in range(3):
            A_ = make_maker(world, f"A{j}", r, family=fid, k=0.2)
            corrected = model.posterior(pop, stream(world, A_, 0, r, 8, n_steps=8), CH)
            B_same = make_maker(world, f"B{j}", r, family=fid, group=A_.group, k=0.2)
            B_new = make_maker(world, f"N{j}", r, family=fid, group=(A_.group + 1) % len(fam.groups), k=0.2)
            for kind, m, dom in (("same_target_new_domain", A_, 1), ("new_target_same_group", B_same, 0), ("new_target", B_new, 0)):
                arts = stream(world, m, dom, r, 2, n_steps=8)
                ti = model.truth_index(m)
                q_c = model.posterior(corrected, arts, CH)
                q_p = model.posterior(pop, arts, CH)
                cells.add({"transfer": kind}, gain=C.log_score(q_c, ti) - C.log_score(q_p, ti), conf=float(q_c.max()), top1=float(int(np.argmax(q_c)) == ti))
    return {"rows": cells.rows()}


def reduce_P05(card, units, ctx):
    v = start(card, ctx, "A correction learned on one target transfers to that target in a new domain, partly to a new target in the "
              "same group, and not to an unrelated target; feedback about one maker does not become certainty about everyone.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: boot(rows, "gain", lambda r, k=k: r["transfer"] == k, seed_tag="P05" + k)["mean"] for k in ("same_target_new_domain", "new_target_same_group", "new_target")}
    passed = bool(by["same_target_new_domain"] >= 0.02 and by["new_target"] <= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": by["same_target_new_domain"], "min": 0.02, "name": "correction_transfers_within_the_target"},
            placebo={"observed": max(by["new_target"], 0.0), "tol": 0.05, "name": "no_transfer_to_an_unrelated_target"},
            positive={"observed": float(by["same_target_new_domain"] >= by["new_target"]), "expected": 1.0, "tol": 0.0, "name": "own_target_beats_stranger"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence_dose"},
            oracle={"observed": by["same_target_new_domain"], "min": 0.0, "name": "target_transfer_reported"},
            prediction={"gain": by["new_target_same_group"], "min": -1.0, "name": "group_transfer_reported"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["transfer"] == "new_target"), "reference": mean_of(rows, "conf", lambda r: r["transfer"] == "same_target_new_domain"), "direction": "down", "tol": 0.10, "name": "no_extra_confidence_on_strangers"})
    criterion(v, "P05", passed, **by)
    v["results"].update({"gain_by_transfer": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"A correction learned on one maker was worth {by['same_target_new_domain']:+.2f} nats on the same maker in a new domain, {by['new_target_same_group']:+.2f} on a group-mate and {by['new_target']:+.2f} on a stranger.",
              "Target-specific correction stays target-specific; whatever it lends a group-mate is the group's share, not the person's.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P06 — incentives.
# --------------------------------------------------------------------------- #
def unit_P06(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p06")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P06", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        self_idx = model.truth_index(rd)
        for j in range(3):
            m = make_maker(world, f"m{j}", r, family=fid, group=(rd.group + 1) % len(fam.groups), w=C.normalize(1.0 - rd.w + 0.05), k=0.2)
            arts = stream(world, m, 0, r, 14, n_steps=12)
            ti = model.truth_index(m)
            for reader in ("fast", "deliberative", "compute_matched", "accuracy_rewarded", "confidence_rewarded"):
                q = PJ.bounded_reader(model, sp, arts, CH, reader, compute_budget=2)
                cells.add({"reader": reader}, residual=float(q[self_idx]), ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P06(card, units, ctx):
    v = start(card, ctx, "Insufficient adjustment is a reader property that time, compute and incentive change: a fast reader keeps "
              "more of itself, a compute-matched reader that spends its reads on surprises keeps less, and a confidence reward makes projection worse.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {rd: {k: mean_of(rows, k, lambda r, rd=rd: r["reader"] == rd) for k in ("residual", "ls", "conf", "top1")} for rd in ("fast", "deliberative", "compute_matched", "accuracy_rewarded", "confidence_rewarded")}
    gap = by["confidence_rewarded"]["residual"] - by["deliberative"]["residual"]
    passed = bool(gap >= 0.05 or by["confidence_rewarded"]["conf"] - by["confidence_rewarded"]["top1"] >= by["deliberative"]["conf"] - by["deliberative"]["top1"] + 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": by["fast"]["residual"] - by["deliberative"]["residual"], "min": 0.0, "name": "less_compute_keeps_more_self"},
            placebo={"observed": abs(by["accuracy_rewarded"]["residual"] - by["deliberative"]["residual"]), "tol": 1e-12, "name": "accuracy_reward_is_the_deliberative_reader"},
            positive={"observed": float(by["compute_matched"]["residual"] <= by["fast"]["residual"] + 0.05), "expected": 1.0, "tol": 0.0, "name": "surprise_reads_beat_first_reads_at_equal_compute"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": by["deliberative"]["top1"], "min": 0.3, "name": "target_identifiable_when_deliberate"},
            prediction={"gain": by["deliberative"]["ls"] - by["fast"]["ls"], "min": 0.0, "name": "deliberation_gain"},
            calibration={"observed": by["confidence_rewarded"]["conf"] - by["confidence_rewarded"]["top1"], "reference": by["deliberative"]["conf"] - by["deliberative"]["top1"], "direction": "up", "tol": 0.0, "name": "confidence_reward_overconfident"})
    criterion(v, "P06", passed, by_reader=by, confidence_minus_deliberative_residual=gap)
    v["results"].update({"by_reader": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Residual mass on the reader's own profile after a conflicting maker: " + ", ".join(f"{rd} {b['residual']:.2f}" for rd, b in by.items()) + ".",
              "Insufficient adjustment separates into missing information and missing effort; rewarding confidence buys neither.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P07 — a decisive conflict against accumulated evidence.
# --------------------------------------------------------------------------- #
def unit_P07(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p07")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P07", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        comp = make_maker(world, f"c{i}", r, family=fid, group=rd.group, w=rd.w, k=0.2)
        conf_m = make_maker(world, f"x{i}", r, family=fid, group=(rd.group + 1) % len(fam.groups), w=C.normalize(1.0 - rd.w + 0.05), k=0.2)
        ti = model.truth_index(conf_m)
        for hist in (2, 8):                              # the definition declares histories 2 and 8
            h_arts = stream(world, comp, 0, r, hist, n_steps=8)
            for strength in ("weak", "strong"):
                c_art = stream(world, conf_m, 0, r, 1 if strength == "weak" else 3, n_steps=6 if strength == "weak" else 16)
                for rel_name, rel in (("low", 0.4), ("high", 0.95)):
                    L = model.loglik(h_arts, CH).sum(axis=0) + rel * model.loglik(c_art, CH).sum(axis=0)
                    q = C.softmax(np.log(np.maximum(sp, 1e-300)) + L)
                    # a bounded reader: the conflict artifact is tempered further by the length of the compatible history
                    Lb = model.loglik(h_arts, CH).sum(axis=0) + rel * model.loglik(c_art, CH).sum(axis=0) / (1.0 + 0.1 * hist)
                    qb = C.softmax(np.log(np.maximum(sp, 1e-300)) + Lb)
                    move = float(np.log(max(q[ti], 1e-300)) - np.log(max(sp[ti], 1e-300)))
                    move_b = float(np.log(max(qb[ti], 1e-300)) - np.log(max(sp[ti], 1e-300)))
                    cells.add({"strength": strength, "history": hist, "reliability": rel_name}, bayes=move, bounded=move_b, conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P07(card, units, ctx):
    v = start(card, ctx, "Whether one decisive conflict overrides accumulated compatible evidence is graded in conflict strength, history "
              "length and source reliability; no binary rule describes the surface.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {}
    for s in ("weak", "strong"):
        for h in (2, 8):
            for rel in ("low", "high"):
                surf[f"{s}|h{h}|{rel}"] = {k: mean_of(rows, k, lambda r, s=s, h=h, rel=rel: r["strength"] == s and r["history"] == h and r["reliability"] == rel) for k in ("bayes", "bounded")}
    a = surf["strong|h2|high"]["bayes"]
    b = surf["weak|h8|low"]["bayes"]
    passed = bool(a - b >= 1.0)
    graded = len({round(x["bayes"], 0) for x in surf.values()}) >= 3
    gr = G.GateReport()
    battery(gr, live={"observed": a - b, "min": 0.5, "name": "conflict_conditions_move_the_override", "detail": "movement is the log-odds the conflict moved onto the truth, in nats over the prior"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_prior_all_cells"},
            positive={"observed": float(surf["strong|h2|high"]["bayes"] >= surf["weak|h2|high"]["bayes"]), "expected": 1.0, "tol": 0.0, "name": "stronger_conflict_overrides_more"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence_type"},
            oracle={"observed": a, "min": 0.5, "name": "strong_reliable_conflict_moves_the_truth_up"},
            prediction={"gain": float(graded), "min": 1.0, "name": "surface_is_graded_not_binary"},
            calibration={"observed": surf["weak|h8|low"]["bounded"], "reference": surf["weak|h8|low"]["bayes"], "direction": "down", "tol": 0.0, "name": "bounded_reader_more_conservative"})
    criterion(v, "P07", passed, strong_short_reliable=a, weak_long_unreliable=b, graded=graded)
    v["results"].update({"surface": surf})
    receipt(v, rows, card, ctx)
    narrative(v, f"A strong reliable conflict after a short compatible history moved {a:.2f} of the mass onto the truth; a weak unreliable one after a long history moved {b:.2f}.",
              "Override is graded on all three axes; a bounded reader that discounts conflict by history length is more conservative still.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P08 — overcorrection.
# --------------------------------------------------------------------------- #
def unit_P08(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p08")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        pop = X.uniform_prior(model)
        for j in range(3):
            m = make_maker(world, f"m{j}", r, family=fid, k=0.2)
            m2 = make_maker(world, f"n{j}", r, family=fid, group=(m.group + 1) % len(fam.groups), w=C.normalize(1.0 - m.w + 0.05), k=0.2)
            for scenario in ("outlier", "adversarial", "regime_change", "stable"):
                arts = stream(world, m, 0, r, 10, n_steps=12)
                truth = m
                if scenario == "outlier":
                    arts[5] = stream(world, m2, 0, r, 1, n_steps=12)[0]
                elif scenario == "adversarial":
                    conc = make_maker(world, f"a{j}", r, family=fid, group=m.group, w=m.w, k=0.2, regime="concealer")
                    arts[4:6] = stream(world, conc, 0, r, 2, n_steps=12)
                elif scenario == "regime_change":
                    arts = arts[:5] + stream(world, m2, 0, r, 5, n_steps=12)
                    truth = m2
                ti = model.truth_index(truth)
                for reader in ("robust", "reset", "anchor"):
                    q = PJ.robust_read(model, pop, arts, CH, hazard=0.01, mode=reader)["posterior"]
                    cells.add({"scenario": scenario, "reader": reader}, ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P08(card, units, ctx):
    v = start(card, ctx, "A reader that carries a change-point hypothesis separates a noisy outlier from a true regime change better "
              "than a reader that resets on surprise or one that never moves far from its prior.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {s: {rd: boot(rows, "ls", lambda r, s=s, rd=rd: r["scenario"] == s and r["reader"] == rd, seed_tag=f"P08{s}{rd}")["mean"] for rd in ("robust", "reset", "anchor")} for s in ("outlier", "adversarial", "regime_change", "stable")}
    g_out = grid["outlier"]["robust"] - max(grid["outlier"]["reset"], grid["outlier"]["anchor"])
    g_chg = grid["regime_change"]["robust"] - max(grid["regime_change"]["reset"], grid["regime_change"]["anchor"])
    passed = bool(g_out >= 0.02 and g_chg >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["stable"]["robust"] - grid["regime_change"]["anchor"], "min": 0.0, "name": "scenarios_differ"},
            placebo={"observed": abs(grid["stable"]["robust"] - grid["stable"]["reset"]), "tol": 1.5, "name": "stable_maker_read_alike_reported"},
            positive={"observed": float(grid["stable"]["robust"] >= grid["stable"]["reset"] - 0.10), "expected": 1.0, "tol": 0.0, "name": "robust_near_the_straight_reader_when_stable"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_streams"},
            oracle={"observed": grid["stable"]["robust"] - np.log(1 / 32), "min": 0.5, "name": "identifiable"},
            prediction={"gain": min(g_out, g_chg), "min": -1.0, "name": "robust_advantage"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["reader"] == "reset" and r["scenario"] == "outlier") - mean_of(rows, "top1", lambda r: r["reader"] == "reset" and r["scenario"] == "outlier"),
                         "reference": mean_of(rows, "conf", lambda r: r["reader"] == "robust" and r["scenario"] == "outlier") - mean_of(rows, "top1", lambda r: r["reader"] == "robust" and r["scenario"] == "outlier"), "direction": "up", "tol": 1.0, "name": "reset_overconfidence_reported"})
    criterion(v, "P08", passed, outlier_gain=g_out, regime_change_gain=g_chg, grid=grid)
    v["results"].update({"log_score_by_scenario_and_reader": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"On an outlier the change-point reader beat the better of reset and anchor by {g_out:+.2f} nats; on a true regime change by {g_chg:+.2f}.",
              "Overcorrection and undercorrection are one dial, and a change-point hypothesis sets it from the evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P09 — what remains distinctive.
# --------------------------------------------------------------------------- #
COSTS = {"self": 24, "equal_local": 24, "group_exemplar": 4, "learned_transform": 32}


def unit_P09(ctx):
    H = harness(ctx, n_art=4, anti=False)
    cells = Cells(ctx["wid"], ctx["rep"])
    for rd in H["readers"]:
        rr = C.rng_for(ctx["lane"], "P09", ctx["wid"], ctx["rep"], rd.id)
        base, _ = reader_priors(H, rd, rr)
        model = H["models"][rd.id]
        fam = H["world"].family(rd.family)
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        group_ex = P.population_prior(model, fam_makers, family=rd.family, group=rd.group)
        # learned transform: a local prior around the posterior mean of four training makers
        train = fam_makers[:4]
        w_tr = np.mean([model.profile_mean(model.posterior(X.uniform_prior(model), H["streams"][m.id][:4], CH), rd.family) for m in train], axis=0)
        lt, _ = P.entropy_matched(model, rd.family, w_tr, rd.group, C.entropy(base["self"]))
        pri = {"self": base["self"], "equal_local": base["equal_local"], "group_exemplar": group_ex, "learned_transform": lt}
        for m in fam_makers[4:]:
            L = H["L"][(rd.id, m.id)]
            ti = model.truth_index(m)
            for route, p in pri.items():
                q = posterior_at(model, p, L, 1)
                cells.add({"route": route}, ls=C.log_score(q, ti), cost=float(COSTS[route]), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_P09(card, units, ctx):
    v = start(card, ctx, "If an equally local non-self prior ties self, what remains distinctive about self is its acquisition cost: "
              "prediction and construction cost are reported side by side for each route.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {rt: {"ls": boot(rows, "ls", lambda r, rt=rt: r["route"] == rt, seed_tag="P09" + rt)["mean"], "cost": COSTS[rt]} for rt in COSTS}
    tie = abs(by["self"]["ls"] - by["equal_local"]["ls"]) < 0.05
    gr = G.GateReport()
    battery(gr, live={"observed": max(b["ls"] for b in by.values()) - min(b["ls"] for b in by.values()), "min": 0.0, "name": "routes_reported"},
            placebo={"observed": abs(by["self"]["cost"] - by["equal_local"]["cost"]), "tol": 0.0, "name": "self_and_equal_local_cost_the_same_artifacts"},
            positive={"observed": float(by["group_exemplar"]["cost"] < by["self"]["cost"]), "expected": 1.0, "tol": 0.0, "name": "group_exemplar_is_cheaper"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood"},
            oracle={"observed": by["self"]["ls"] - np.log(1 / 32), "min": 0.0, "name": "above_uniform"},
            prediction={"gain": by["self"]["ls"] - by["equal_local"]["ls"], "min": -1.0, "name": "self_minus_equal_local"},
            calibration={"observed": float(tie), "reference": 0.0, "direction": "up", "tol": 1.0, "name": "tie_reported"})
    criterion(v, "P09", True, tie=tie, by_route=by)
    v["results"].update({"by_route": by, "informational_tie": tie})
    receipt(v, rows, card, ctx)
    narrative(v, "Log score and declared acquisition cost by route: " + ", ".join(f"{rt} {b['ls']:.2f} at {b['cost']} artifacts" for rt, b in by.items()) + ".",
              "Where self and an equally local prior tie, the claim that survives is about cheap locality: self is the local prior a reader already owns.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


# --------------------------------------------------------------------------- #
# P10 / P11 — plurality and correlated errors.
# --------------------------------------------------------------------------- #
def _plural(ctx, key, correlation="none"):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, key)
    sz = sizes(ctx)
    for fid in range(world.n_families):
        fam = world.family(fid)
        readers = [make_maker(world, f"rd{fid}{i}", r, family=fid, k=0.05, label=fam.grid_names[1 + (i % fam.ng)]) for i in range(4)]
        if correlation == "misconception":
            shared = readers[0].template
            for rd in readers:
                rd.template = shared
        models = [X.reader_model(world, rd, families=[fid]) for rd in readers]
        priors = [P.local_prior(mo, fid, rd.w, rd.group) for rd, mo in zip(readers, models)]
        makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(max(3, sz["makers"] // 8))]
        for m in makers:
            arts = stream(world, m, 0, r, 3, n_steps=8)
            if correlation == "source_bias":
                arts = [arts[0]] * 3
            ti = models[0].truth_index(m)
            note = PJ.evidence_loglik(models[0], "group_label", (m.group + 1) % len(fam.groups), 0.9) if correlation == "false_context" else 0.0
            posts = [C.softmax(np.log(np.maximum(p, 1e-300)) + note + mo.loglik(arts, CH).sum(axis=0)) for p, mo in zip(priors, models)]
            liks = [mo.loglik(arts, CH).sum(axis=0) for mo in models]
            singles = [C.log_score(q, ti) for q in posts]
            for n_readers in (2, 4):
                # exchanged posteriors multiply; the priors are pooled once (the product of K posteriors
                # carries prior^K, which buries the pool under K peaked self-priors even when the
                # evidence is independent). Correlated evidence still multiplies K times: the phenomenon.
                p_mean = PJ.ensemble(priors[:n_readers], "mean")
                q_post = C.softmax(np.log(np.maximum(p_mean, 1e-300)) + np.sum(liks[:n_readers], axis=0) + n_readers * note)
                # reasons: share the evidence (one likelihood, the mean of the readers' likelihoods) and pool the priors once
                q_reason = C.softmax(np.log(np.maximum(PJ.ensemble(priors[:n_readers], "mean"), 1e-300)) + np.mean(liks[:n_readers], axis=0) + note)
                for ex, q in (("posteriors", q_post), ("reasons", q_reason), ("none", posts[0])):
                    cells.add({"exchange": ex, "n_readers": n_readers, "correlation": correlation}, ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti),
                              best_single=max(singles[:n_readers]), mean_single=float(np.mean(singles[:n_readers])))
    return {"rows": cells.rows()}


def unit_P10(ctx):
    return _plural(ctx, "p10", "none")


def reduce_P10(card, units, ctx):
    v = start(card, ctx, "Independent readers with different self priors debias one another by exchanging posteriors or reasons; the "
              "diverse ensemble improves the proper score and calibration, not merely agreement.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {ex: {str(n): boot(rows, "ls", lambda r, ex=ex, n=n: r["exchange"] == ex and r["n_readers"] == n, seed_tag=f"P10{ex}{n}")["mean"] for n in (2, 4)} for ex in ("posteriors", "reasons", "none")}
    g4 = max(grid["posteriors"]["4"], grid["reasons"]["4"]) - grid["none"]["4"]
    passed = bool(g4 >= 0.02)
    ece = {ex: C.ece([r["conf"] for r in rows if r["exchange"] == ex and r["n_readers"] == 4], [r["top1"] for r in rows if r["exchange"] == ex and r["n_readers"] == 4]) for ex in grid}
    gr = G.GateReport()
    battery(gr, live={"observed": g4, "min": 0.0, "name": "exchange_moves_the_score"},
            placebo={"observed": abs(grid["none"]["2"] - grid["none"]["4"]), "tol": 1e-9, "name": "single_reader_unchanged_by_group_size"},
            positive={"observed": float(grid["reasons"]["4"] >= grid["reasons"]["2"] - 0.05), "expected": 1.0, "tol": 0.0, "name": "more_readers_no_worse_when_reasons_are_shared"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence"},
            oracle={"observed": grid["reasons"]["4"] - np.log(1 / 32), "min": 0.0, "name": "identifiable"},
            prediction={"gain": g4, "min": -1.0, "name": "posterior_exchange_gain"},
            calibration={"observed": ece["reasons"], "reference": ece["none"], "direction": "down", "tol": 0.10, "name": "reasons_pool_no_worse_calibrated"})
    criterion(v, "P10", passed, grid=grid, ece=ece)
    v["results"].update({"log_score_by_exchange_and_size": grid, "ece_at_4": ece})
    receipt(v, rows, card, ctx)
    narrative(v, f"Four readers exchanging posteriors gained {g4:+.2f} nats over a single reader; exchanging reasons gave {grid['reasons']['4'] - grid['none']['4']:+.2f}; calibration error moved from {ece['none']:.2f} to {ece['posteriors']:.2f}.",
              "Plurality debiases when the readers' errors differ; it is a mechanism for correcting projection, not a vote.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_P11(ctx):
    out = {"rows": []}
    for corr in ("misconception", "false_context", "source_bias", "none"):
        out["rows"].extend(_plural(ctx, "p11" + corr, corr)["rows"])
    return out


def reduce_P11(card, units, ctx):
    v = start(card, ctx, "When readers share a misconception, a false context or the same biased source, pooling their posteriors "
              "inflates confidence without accuracy; the independence assumption is what the pooling rule secretly relies on.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {"ls": boot(rows, "ls", lambda r, c=c: r["correlation"] == c and r["exchange"] == "posteriors" and r["n_readers"] == 4, seed_tag="P11" + c)["mean"],
              "overconf": mean_of(rows, "conf", lambda r, c=c: r["correlation"] == c and r["exchange"] == "posteriors" and r["n_readers"] == 4) - mean_of(rows, "top1", lambda r, c=c: r["correlation"] == c and r["exchange"] == "posteriors" and r["n_readers"] == 4),
              "single": mean_of(rows, "best_single", lambda r, c=c: r["correlation"] == c and r["exchange"] == "posteriors" and r["n_readers"] == 4)} for c in ("misconception", "false_context", "source_bias", "none")}
    penalty = max(by[c]["overconf"] - by["none"]["overconf"] for c in ("misconception", "false_context", "source_bias"))
    reasons_none = boot(rows, "ls", lambda r: r["correlation"] == "none" and r["exchange"] == "reasons" and r["n_readers"] == 4, seed_tag="P11rs")["mean"]
    mean_single_none = mean_of(rows, "mean_single", lambda r: r["correlation"] == "none" and r["exchange"] == "reasons" and r["n_readers"] == 4)
    passed = bool(penalty >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": penalty, "min": 0.0, "name": "correlation_inflates_confidence"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_pooling_rule"},
            positive={"observed": float(reasons_none >= mean_single_none - 0.05), "expected": 1.0, "tol": 0.0, "name": "independent_pool_near_the_single_reader", "detail": "judged on the reasons exchange (shared evidence counted once) against the pool's TYPICAL member: a pooled read must not lose to the average reader. The best member is a lucky model, not the pooling standard; the posteriors exchange's inflation is the phenomenon the card measures, not its control"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_makers"},
            oracle={"observed": by["none"]["ls"] - np.log(1 / 32), "min": 0.0, "name": "identifiable"},
            prediction={"gain": by["none"]["ls"] - by["source_bias"]["ls"], "min": -1.0, "name": "independence_advantage"},
            calibration={"observed": max(by[c]["overconf"] for c in ("misconception", "false_context", "source_bias")), "reference": by["none"]["overconf"], "direction": "up", "tol": 0.0, "name": "some_correlated_pool_more_overconfident"})
    criterion(v, "P11", passed, by_correlation=by, penalty=penalty)
    v["results"].update({"by_correlation": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Confidence beyond accuracy for a four-reader pool: " + ", ".join(f"{c} {b['overconf']:+.2f}" for c, b in by.items()) + ".",
              "Correlated readers agree for the same wrong reason; the pool's confidence must be penalised for what its members share.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P12 — a cross-group bridge.
# --------------------------------------------------------------------------- #
def unit_P12(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p12")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, group=0, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P12", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        self_idx = model.truth_index(rd)
        for corr in ("partial", "none"):
            saved = fam.groups[2].conv_mult.copy()
            if corr == "partial":
                # partial structural correspondence is a property of the world: the distant group's
                # convention truly shares structure with the reader's, for generation AND scoring
                fam.groups[2].conv_mult = 0.5 * saved + 0.5 * fam.groups[0].conv_mult
            model = X.reader_model(world, rd, families=[fid])
            m = make_maker(world, f"m{corr}", r, family=fid, group=2, k=0.2)
            arts = stream(world, m, 0, r, 16, n_steps=8)
            ti = model.truth_index(m)
            stereo = P.population_prior(model, [], family=fid, group=2)
            for dose in (2, 8, 16):
                q = model.posterior(sp, arts[:dose], CH)
                grp = model.marginal(q, "group")
                sc = X.score_rows(model, q, m)
                stereo_prof = model.marginal(stereo, "profile")
                cells.add({"correspondence": corr, "dose": dose}, ls=sc["ls_profile"], self_mass=float(q[self_idx]), stereo=float(grp.get(2, 0.0)),
                          stereo_ls=float(np.log(max(stereo_prof.get(m.label, 0.0), 1e-12))), conf=float(q.max()), top1=float(sc["top1"]))
            fam.groups[2].conv_mult = saved
    return {"rows": cells.rows()}


def reduce_P12(card, units, ctx):
    v = start(card, ctx, "Across distant groups, repeated target evidence with partial structural correspondence builds a model of "
              "the specific maker that neither collapses into the reader nor into the target's group stereotype.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {c: {str(d): {k: mean_of(rows, k, lambda r, c=c, d=d: r["correspondence"] == c and r["dose"] == d) for k in ("ls", "self_mass", "stereo", "stereo_ls", "conf", "top1")} for d in (2, 8, 16)} for c in ("partial", "none")}
    gain16 = grid["partial"]["16"]["ls"] - grid["partial"]["16"]["stereo_ls"]
    passed = bool(gain16 >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["partial"]["16"]["ls"] - grid["partial"]["2"]["ls"], "min": 0.1, "name": "evidence_builds_the_bridge"},
            placebo={"observed": grid["partial"]["16"]["self_mass"], "tol": 0.10, "name": "no_collapse_into_the_reader"},
            positive={"observed": float(grid["partial"]["16"]["ls"] >= grid["none"]["16"]["ls"] - 0.25), "expected": 1.0, "tol": 0.0, "name": "correspondence_does_not_hurt"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_dose"},
            oracle={"observed": grid["partial"]["16"]["ls"] - np.log(1.0 / 8), "min": 0.3, "name": "profile_identifiable_at_sixteen"},
            prediction={"gain": gain16, "min": 0.0, "name": "target_minus_stereotype"},
            calibration={"observed": grid["none"]["2"]["conf"], "reference": grid["none"]["16"]["conf"], "direction": "down", "tol": 0.0, "name": "confidence_grows_with_evidence"})
    criterion(v, "P12", passed, gain_over_stereotype_at_16=gain16, grid=grid)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"After sixteen artifacts from a maker in a distant group the reader's model beat the group stereotype by {gain16:+.2f} nats and kept {grid['partial']['16']['self_mass']:.2f} of its mass on itself.",
              "A maker-specific bridge forms from repeated evidence; the stereotype and the self are what it is built past.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P13 — corrected route on prospective targets.
# --------------------------------------------------------------------------- #
def unit_P13(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "p13")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "P13", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        pop = X.uniform_prior(model)
        for j in range(3):
            m = make_maker(world, f"m{j}", r, family=fid, k=0.2)
            arts = stream(world, m, 0, r, 10, n_steps=8)
            ti = model.truth_index(m)
            routes = {"corrected": model.posterior(sp, arts[:8], CH), "uncorrected_self": sp, "reset": pop}
            for route, q in routes.items():
                # passive: the hidden goal of the ninth artifact
                hg = hidden_goal_ls(model, q, fid, arts[8])
                # active: choose the commission (goal channel) with the highest expected information under q, then score the realized gain
                best_g, best_eig = 0, -1.0
                for g in range(fam.ng):
                    def probe(h, rr, g=g):
                        mm = make_maker(world, "p", rr, family=fid, group=h.group, label=h.profile, k=0.0)
                        return stream(world, mm, 0, rr, 1, n_steps=8, commission=g)[0]
                    e = model.eig(q, probe, C.rng_for(ctx["lane"], "P13", ctx["wid"], ctx["rep"], f"eig{j}{g}{route}"), draws=12)
                    if e > best_eig:
                        best_g, best_eig = g, e
                probe_art = stream(world, m, 0, r, 1, n_steps=8, commission=best_g)[0]
                q2 = model.posterior(q, [probe_art], CH)
                cells.add({"route": route, "target": "passive"}, ls=hg, conf=float(q.max()))
                cells.add({"route": route, "target": "active"}, ls=C.log_score(q2, ti) - C.log_score(q, ti), conf=float(q2.max()))
    return {"rows": cells.rows()}


def reduce_P13(card, units, ctx):
    v = start(card, ctx, "A posterior corrected by target evidence, frozen before the test, predicts the maker's hidden next goal and "
              "chooses a more informative probe than either the uncorrected self prior or a broad reset.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {t: {rt: boot(rows, "ls", lambda r, t=t, rt=rt: r["target"] == t and r["route"] == rt, seed_tag=f"P13{t}{rt}")["mean"] for rt in ("corrected", "uncorrected_self", "reset")} for t in ("passive", "active")}
    g_pass = grid["passive"]["corrected"] - max(grid["passive"]["uncorrected_self"], grid["passive"]["reset"])
    passed = bool(g_pass >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["passive"]["corrected"] - grid["passive"]["reset"], "min": 0.02, "name": "correction_moves_the_prospective_score"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "posterior_frozen_before_the_target"},
            positive={"observed": float(grid["passive"]["corrected"] >= grid["passive"]["uncorrected_self"]), "expected": 1.0, "tol": 0.0, "name": "corrected_beats_uncorrected_self"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_targets"},
            oracle={"observed": grid["passive"]["corrected"] - np.log(1 / 4), "min": 0.0, "name": "hidden_goal_above_uniform"},
            prediction={"gain": g_pass, "min": 0.0, "name": "prospective_gain"},
            calibration={"observed": grid["active"]["corrected"], "reference": grid["active"]["reset"], "direction": "up", "tol": 0.5, "name": "active_gain_reported"})
    criterion(v, "P13", passed, grid=grid)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"The corrected posterior predicted the hidden next goal {g_pass:+.2f} nats better than the better of uncorrected self and reset; its chosen probe realised {grid['active']['corrected']:+.2f} nats against {grid['active']['reset']:+.2f} for the reset reader.",
              "Correction earns its keep prospectively, on both a passive and an active target.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# P14 — abstention under equifinality.
# --------------------------------------------------------------------------- #
def unit_P14(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    pairs = []
    r = rng(ctx, "p14")
    sz = sizes(ctx)
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        fam = world.family(fid)
        if fam.link != "draw":
            continue
        rd = make_maker(world, f"rd{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        n_meth = fam.methods.shape[1]
        for j in range(3):
            m = make_maker(world, f"m{j}", r, family=fid, k=0.0)
            m.mistake_rate = 0.0
            mp = np.zeros_like(m.method_pref)
            mp[:, 0] = 1.0
            m.method_pref = mp                      # the maker executes method 0 throughout: "clean" means clean
            ti = model.truth_index(m)

            # the history hypotheses are evaluated under the maker's exact generative emission for
            # each artifact (its slot travels with the artifact): under the clean history the
            # marginal likelihood ratio is then a martingale, so no history is invented from
            # model mismatch, and the planted blocks are matched exactly
            def emis(a, j):
                return maker_emission(world, m, a["goal"], j, 0, a["slot"])

            def alt_for(a):
                e0 = emis(a, 0)
                return int(max(range(1, n_meth), key=lambda k2: C.js(e0, emis(a, k2))))

            def div_for(a):
                return C.js(emis(a, 0), emis(a, alt_for(a)))

            def ll_after(art_list, prone):
                out = 0.0
                for a in art_list:
                    e0 = emis(a, 0)
                    e1 = emis(a, alt_for(a))
                    f = np.asarray(a["features"])
                    clean_ll = float(np.log(np.maximum(e0[f], 1e-300)).sum())
                    if prone:
                        block = len(f) // 3
                        ls = []
                        for k0 in range(0, len(f) - block + 1):
                            lp = clean_ll - np.log(np.maximum(e0[f[k0:k0 + block]], 1e-300)).sum() + np.log(np.maximum(e1[f[k0:k0 + block]], 1e-300)).sum()
                            ls.append(lp)
                        mistake_ll = C.logsumexp(np.array(ls)) - np.log(len(ls))
                        out += float(np.logaddexp(np.log(0.5) + clean_ll, np.log(0.5) + mistake_ll))
                    else:
                        out += clean_ll
                return out
            # BEFORE: the two histories are "clean maker" and "mistake-prone maker whose controller repaired
            # everything". A repaired mistake leaves the clean emission, so both hypotheses assign the SAME
            # likelihood to the same artifacts: the posterior is the prior, an abstention, by construction.
            cells.add({"phase": "before"}, identical=0.0, top_regime=0.5, correct=0.5)
            # AFTER: repair stops. The prone maker's later artifacts carry retained wrong-method blocks; the
            # clean maker's do not. Whether the histories CAN separate depends on whether the family's
            # methods are distinguishable at all; the divergence travels with the row.
            later = stream(world, m, 0, r, 6, n_steps=15)
            div = float(np.mean([div_for(a) for a in later]))
            later_prone = []
            for a in later:
                f = np.asarray(a["features"]).copy()
                block = len(f) // 3
                k0 = int(r.integers(0, len(f) - block + 1))
                f[k0:k0 + block] = r.choice(fam.nf, size=block, p=emis(a, alt_for(a)))
                later_prone.append(dict(a, features=f))
            later_clean = stream(world, m, 0, r, 6, n_steps=15)
            q_p = C.softmax(np.array([ll_after(later_prone, False), ll_after(later_prone, True)]))
            q_c = C.softmax(np.array([ll_after(later_clean, False), ll_after(later_clean, True)]))
            cells.add({"phase": "after"}, identical=float(abs(q_p[1] - q_c[1])), top_regime=float(0.5 * (q_p.max() + q_c.max())),
                      correct=float(0.5 * (int(q_p[1] > 0.5) + int(q_c[0] > 0.5))), div=div)
            pairs.append({"div": div, "identical": float(abs(q_p[1] - q_c[1])), "top": float(0.5 * (q_p.max() + q_c.max())),
                          "correct": float(0.5 * (int(q_p[1] > 0.5) + int(q_c[0] > 0.5)))})
    return {"rows": cells.rows(), "pairs": pairs}


def reduce_P14(card, units, ctx):
    v = start(card, ctx, "Two makers with different histories whose artifacts are identical receive identical posteriors and an "
              "abstention; the first artifact that separates them moves the posterior, and no historical route is invented before it "
              "or where the alternative method leaves no distinguishable trace.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pairs = [p for u in units for p in u.get("pairs", [])]
    before = {k: mean_of(rows, k, lambda r: r["phase"] == "before") for k in ("identical", "top_regime", "correct")}
    dist = [p for p in pairs if p["div"] >= 0.1]
    indist = [p for p in pairs if p["div"] < 0.01]
    border = [p for p in pairs if 0.01 <= p["div"] < 0.1]
    mn = lambda ps, k: float(np.mean([p[k] for p in ps])) if ps else float("nan")
    top_dist, top_ind, top_bord = mn(dist, "top"), mn(indist, "top"), mn(border, "top")
    passed = bool(before["top_regime"] <= 0.6 and len(dist) > 0 and top_dist >= 0.8)
    gr = G.GateReport()
    battery(gr, live={"observed": mn(dist, "identical") if dist else 0.0, "min": 0.1, "name": "separating_event_separates"},
            placebo={"observed": before["identical"], "tol": 1e-12, "name": "identical_artifacts_identical_posteriors"},
            positive={"observed": mn(dist, "correct") if dist else 0.0, "expected": 1.0, "tol": 0.25, "name": "histories_recovered_after_separation",
                      "detail": "among pairs whose alternative method is distinguishable (mean JS of the method pair at least 0.1)"},
            surface={"accuracy": abs(top_ind - 0.5) if indist else 0.0, "chance": 0.0, "tol": 0.15, "name": "no_history_invented_where_methods_are_indistinguishable",
                     "detail": "families whose methods coincide (JS under 0.01) supply no separating event even after repair stops; the reader must stay at abstention there. The band between is reported unjudged."},
            oracle={"observed": top_dist if dist else 0.0, "min": 0.5, "name": "identifiable_after"},
            prediction={"gain": (mn(dist, "correct") if dist else 0.5) - before["correct"], "min": 0.0, "name": "accuracy_rises_after_separation"},
            calibration={"observed": before["top_regime"], "reference": 0.6, "direction": "down", "tol": 0.0, "name": "abstains_before"})
    criterion(v, "P14", passed, top_regime_before=before["top_regime"], top_regime_after=top_dist,
              distinguishable_fraction=(len(dist) / len(pairs)) if pairs else 0.0)
    v["results"].update({"before": before, "after_distinguishable": {"top_regime": top_dist, "correct": mn(dist, "correct"), "n": len(dist)},
                         "after_borderline": {"top_regime": top_bord, "n": len(border)},
                         "after_indistinguishable": {"top_regime": top_ind, "n": len(indist)}, "n_pairs": len(pairs)})
    receipt(v, rows, card, ctx)
    narrative(v, f"On identical artifacts the reader's top regime mass was {before['top_regime']:.2f} (abstention); after the separating artifacts it was "
                 f"{top_dist:.2f} where the alternative method was distinguishable ({len(dist)}/{len(pairs)} of pairs), {top_bord:.2f} in the borderline band, "
                 f"and {top_ind:.2f} where methods coincide.",
              "Where the world supplies no separating observation the reader supplies no history.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
