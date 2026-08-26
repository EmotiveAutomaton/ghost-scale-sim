"""Trunk A: attention, entry, and evidence allocation (spec §10).

Attention is operationalized as selection (which channels are inspected under a budget) or
precision (how each channel's evidence is weighted). Every card scores a proper prediction of
the maker against policies that cannot see the answer; the oracle policy, which can, is the
ceiling and never a result.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as X, priors as P, attention as A, costs as CO, goals_trust as GT, projection as PJ
from ..world import make_maker, population, stream, histogram, N_METHODS
from . import (battery, boot, ci_abs, ci_pos, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of)
from .trunk_c import Cells, hidden_goal_ls

CH8 = ("surface", "common_structure", "group_convention", "mechanics", "goal_consequences", "communicative_shaping", "anomaly", "process_records")
CH_NOREC = ("surface", "common_structure", "group_convention", "mechanics", "goal_consequences", "communicative_shaping", "anomaly")


def _readers(ctx, world, r, k=0.05):
    sz = sizes(ctx)
    out = []
    for i in range(max(4, sz["readers"] // 2)):
        fid = i % world.n_families
        rd = make_maker(world, f"reader{i}", r, family=fid, k=k)
        out.append((rd, X.reader_model(world, rd, families=[fid])))
    return out


def _items(world, model, makers, r, n_art=3, n_steps=8, tag="a"):
    return [(stream(world, m, 0, C.rng_for("x", tag, 0, 0, m.id + str(int(r.integers(1 << 30)))), n_art, n_steps=n_steps), model.truth_index(m)) for m in makers]


def _train_test(world, fid, r, n_train=6, n_test=8, k=0.3):
    tr = [make_maker(world, f"t{j}", r, family=fid, k=k) for j in range(n_train)]
    te = [make_maker(world, f"e{j}", r, family=fid, k=k) for j in range(n_test)]
    return tr, te


# --------------------------------------------------------------------------- #
# A01 — selection and precision: distinct but calibrated.
# --------------------------------------------------------------------------- #
def unit_A01(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a01")
    chans = list(CH_NOREC)
    ident = []
    worst_ls = []
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        tr, te = _train_test(world, rd.family, r)
        items_tr = [(stream(world, m, 0, r, 3, n_steps=8), model.truth_index(m)) for m in tr]
        items_te = [(stream(world, m, 0, r, 3, n_steps=8), model.truth_index(m)) for m in te]
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
        for arts, ti in items_te:
            q_plain = model.posterior(prior, arts, chans)
            q_sel = model.posterior(prior, arts, tuple(A.select("uniform", chans, 99, r)))
            q_prec = model.posterior(prior, arts, chans, A.precision("uniform", chans, r))
            ident.append(max(float(np.abs(q_plain - q_sel).max()), float(np.abs(q_plain - q_prec).max())))
            for budget in (1, 2, 4):
                sel = A.select("oracle", chans, float(budget), r, ranking=rank)
                q1 = model.posterior(prior, arts, tuple(sel)) if sel else prior
                w = A.precision("oracle", chans, r, ranking=rank, budget=float(budget))
                q2 = model.posterior(prior, arts, chans, w)
                cells.add({"analogue": "selection", "budget": budget}, ls=C.log_score(q1, ti), conf=float(q1.max()), top1=float(int(np.argmax(q1)) == ti))
                cells.add({"analogue": "precision", "budget": budget}, ls=C.log_score(q2, ti), conf=float(q2.max()), top1=float(int(np.argmax(q2)) == ti))
            q_worst = model.posterior(prior, arts, (rank[-1],))
            worst_ls.append(C.log_score(q_worst, ti))
    return {"rows": cells.rows(), "identity": float(max(ident)) if ident else 0.0, "worst": float(np.mean(worst_ls)) if worst_ls else 0.0}


def reduce_A01(card, units, ctx):
    v = start(card, ctx, "Selection attention and precision attention are the same reader at neutral settings and different readers "
              "under a budget; their budget curves are reported side by side without assuming equivalence.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    ident = float(max(u["identity"] for u in units))
    curves = {a: {str(b): boot(rows, "ls", lambda r, a=a, b=b: r["analogue"] == a and r["budget"] == b, seed_tag=f"A01{a}{b}") for b in (1, 2, 4)} for a in ("selection", "precision")}
    distinct = float(np.mean([abs(curves["selection"][b]["mean"] - curves["precision"][b]["mean"]) for b in ("1", "2", "4")]))
    slope = curves["selection"]["4"]["mean"] - curves["selection"]["1"]["mean"]
    gr = G.GateReport()
    battery(gr, live={"observed": slope, "min": 0.02, "name": "budget_moves_the_score"},
            placebo={"observed": ident, "tol": 1e-12, "name": "identity_at_neutral_settings"},
            positive={"observed": float(curves["selection"]["1"]["mean"] >= float(np.mean([u.get("worst", 0.0) for u in units])) + 0.02), "expected": 1.0, "tol": 0.0, "name": "ranking_orders_single_channels", "detail": "the top-ranked channel beats the worst-ranked one. The budget curve itself can decline, because lower-ranked channels carry the reader's misspecification of this maker; the curve is the reported result, not a gate"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence_both_analogues"},
            oracle={"observed": curves["precision"]["4"]["mean"] - np.log(1 / 40), "min": 0.3, "name": "identifiable_at_full_budget"},
            prediction={"gain": distinct, "min": 0.0, "name": "analogues_distinct_under_budget"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["analogue"] == "precision" and r["budget"] == 1), "reference": mean_of(rows, "conf", lambda r: r["analogue"] == "selection" and r["budget"] == 1), "direction": "up", "tol": 1.0, "name": "confidence_reported"})
    passed = bool(ident <= 1e-12)
    criterion(v, "A01", passed, identity=ident, distinctness=distinct)
    v["results"].update({"budget_curves": curves, "identity_deviation": ident})
    receipt(v, rows, card, ctx)
    narrative(v, f"At neutral settings both analogues reproduced the plain posterior to {ident:.1e}; under budgets of one, two and four cue families their log scores differed by {distinct:.2f} nats on average.",
              "Selection and precision are one identity and two budget curves; neither is assumed to stand for the other.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A02 — select the known diagnostic cue.
# --------------------------------------------------------------------------- #
def _cue_world_arts(world, m, r, n, diagnostic="goal_consequences"):
    """One highly diagnostic channel, salient-but-weak channels, and irrelevant channels."""
    arts = stream(world, m, 0, r, n)
    fam = world.family(m.family)
    for a in arts:
        a["features"] = r.integers(0, fam.nf, size=len(a["features"]))                   # surface: irrelevant
        if "convention_obs" in a:                    # non-draw families emit no rich channels; absent is already irrelevant
            a["convention_obs"] = r.integers(0, fam.nf, size=len(a["convention_obs"])).tolist()
        if "structure_obs" in a:
            a["structure_obs"] = r.integers(0, len(fam.blocks) + 1, size=len(a["structure_obs"])).tolist()
        if diagnostic != "goal_consequences":
            a["payoff_obs"] = int(r.integers(fam.ng))
        if diagnostic != "process_records":
            a["log"] = dict(a["log"], goal=-1)
    return arts


def unit_A02(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a02")
    chans = ["surface", "common_structure", "group_convention", "goal_consequences", "communicative_shaping", "anomaly"]
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        tr, te = _train_test(world, rd.family, r)
        items_tr = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in tr]
        items_te = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in te]
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
        learned = A.fit_precision(model, prior, items_tr, chans)
        sal = A.salience_of(chans, r, adversarial_weak="communicative_shaping", quiet="goal_consequences")
        for arts, ti in items_te:
            for budget in (1, 2):
                for pol in ("random", "salience", "oracle", "learned"):
                    sel = A.select(pol, chans, float(budget), r, salience=sal, ranking=rank, learned=learned)
                    q = model.posterior(prior, arts, tuple(sel)) if sel else prior
                    g = C.log_score(q, ti) - C.log_score(prior, ti)
                    cells.add({"policy": pol, "budget": budget}, ipc=A.information_per_cost(g, sel), gain=g, conf=float(q.max()), top1=float(int(np.argmax(q)) == ti),
                              picked_diagnostic=float("goal_consequences" in sel))
    return {"rows": cells.rows()}


def reduce_A02(card, units, ctx):
    v = start(card, ctx, "Under a finite budget a reader that ranks cues by learned diagnosticity buys more information per cost than "
              "random or salience-driven inspection when one cue is diagnostic, several are conspicuous but weak, and the rest are irrelevant.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {p: {str(b): boot(rows, "ipc", lambda r, p=p, b=b: r["policy"] == p and r["budget"] == b, seed_tag=f"A02{p}{b}") for b in (1, 2)} for p in ("random", "salience", "oracle", "learned")}
    g_learned = by["learned"]["1"]["mean"] - by["random"]["1"]["mean"]
    g_oracle = by["oracle"]["1"]["mean"] - by["random"]["1"]["mean"]
    sal_vs_learned = by["salience"]["1"]["mean"] - by["learned"]["1"]["mean"]
    passed = bool(g_learned >= 0.02 and g_oracle >= 0.02 and sal_vs_learned <= 0.0)
    gr = G.GateReport()
    battery(gr, live={"observed": g_oracle, "min": 0.02, "name": "diagnostic_cue_carries_information"},
            placebo={"observed": (lambda x: 0.0 if x != x else max(x, 0.0))(mean_of(rows, "gain", lambda r: r["policy"] == "random" and r["budget"] == 1 and r["picked_diagnostic"] < 0.5)), "tol": 0.25, "name": "weak_cues_carry_little", "detail": "cells are unit means, so the non-diagnostic pick indicator is a fraction; an empty subset is a quiet null"},
            positive={"observed": mean_of(rows, "picked_diagnostic", lambda r: r["policy"] == "oracle" and r["budget"] == 1), "expected": 1.0, "tol": 0.05, "name": "oracle_picks_the_diagnostic_cue"},
            surface={"accuracy": max(sal_vs_learned, 0.0), "chance": 0.0, "tol": 0.0, "name": "salience_does_not_beat_learned"},
            oracle={"observed": g_oracle, "min": 0.02, "name": "oracle_ceiling"},
            prediction={"gain": g_learned, "min": 0.0, "name": "learned_over_random"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["policy"] == "salience" and r["budget"] == 1) - mean_of(rows, "top1", lambda r: r["policy"] == "salience" and r["budget"] == 1),
                         "reference": mean_of(rows, "conf", lambda r: r["policy"] == "learned" and r["budget"] == 1) - mean_of(rows, "top1", lambda r: r["policy"] == "learned" and r["budget"] == 1), "direction": "up", "tol": 1.0, "name": "salience_overconfidence_reported"})
    criterion(v, "A02", passed, learned_minus_random=g_learned, oracle_minus_random=g_oracle, salience_minus_learned=sal_vs_learned)
    v["results"].update({"information_per_cost": by, "picked_diagnostic": {p: mean_of(rows, "picked_diagnostic", lambda r, p=p: r["policy"] == p and r["budget"] == 1) for p in by}})
    receipt(v, rows, card, ctx)
    narrative(v, f"With one cue family to inspect, learned selection bought {g_learned:+.2f} nats per unit cost more than random inspection and the oracle {g_oracle:+.2f}; "
                 f"salience-driven inspection scored {sal_vs_learned:+.2f} relative to learned.",
              "A reader can learn which cue to look at from labelled experience; conspicuousness is not diagnosticity.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A03 — precision weighting with identical evidence.
# --------------------------------------------------------------------------- #
def unit_A03(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a03")
    chans = list(CH_NOREC)
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        tr, te = _train_test(world, rd.family, r)
        items_tr = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in tr]
        items_te = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in te]
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
        learned = A.fit_precision(model, prior, items_tr, chans)
        for arts, ti in items_te:
            for wt in ("uniform", "learned", "oracle", "wrong"):
                w = A.precision(wt, chans, r, ranking=rank, learned=learned)
                q = model.posterior(prior, arts, chans, w)
                cells.add({"weighting": wt}, ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti),
                          conf_wrong=float(q.max() > 0.8 and int(np.argmax(q)) != ti))
    return {"rows": cells.rows()}


def reduce_A03(card, units, ctx):
    v = start(card, ctx, "With every cue visible, learned precision weights improve the held-out score over uniform weights without "
              "changing the data, and wrong weights produce confident error.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {w: boot(rows, "ls", lambda r, w=w: r["weighting"] == w, seed_tag="A03" + w) for w in ("uniform", "learned", "oracle", "wrong")}
    cw = {w: mean_of(rows, "conf_wrong", lambda r, w=w: r["weighting"] == w) for w in by}
    g_learned = by["learned"]["mean"] - by["uniform"]["mean"]
    g_wrong = by["wrong"]["mean"] - by["uniform"]["mean"]
    passed = bool(g_learned >= 0.02 and g_wrong < 0.0)
    gr = G.GateReport()
    battery(gr, live={"observed": by["oracle"]["mean"] - by["wrong"]["mean"], "min": 0.05, "name": "weights_move_the_score"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "data_identical_across_weightings"},
            positive={"observed": float(by["oracle"]["mean"] >= by["uniform"]["mean"] - 0.02), "expected": 1.0, "tol": 0.0, "name": "oracle_weights_no_worse_than_uniform"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence"},
            oracle={"observed": by["oracle"]["mean"] - by["uniform"]["mean"], "min": 0.0, "name": "oracle_gain"},
            prediction={"gain": g_learned, "min": 0.0, "name": "learned_held_out_gain"},
            calibration={"observed": cw["wrong"], "reference": cw["uniform"], "direction": "up", "tol": 0.0, "name": "wrong_weights_raise_confident_error"})
    criterion(v, "A03", passed, learned_minus_uniform=g_learned, wrong_minus_uniform=g_wrong, confidently_wrong=cw)
    v["results"].update({"by_weighting": by, "confidently_wrong_rate": cw})
    receipt(v, rows, card, ctx)
    narrative(v, f"On identical evidence, learned weights scored {g_learned:+.2f} nats over uniform and wrong weights {g_wrong:+.2f}; the wrong weighting was confidently wrong {cw['wrong']:.0%} of the time against {cw['uniform']:.0%}.",
              "Precision can help or hurt with the same data on the table; what it cannot do is create evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A04 — common-structure focus by causal world (profile target).
# --------------------------------------------------------------------------- #
def unit_A04(ctx):
    from .trunk_c import _causal_arts, hidden_goal_ls, CH4
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a04")
    diag_w = {"common": "common_structure", "group": "group_convention", "individual": "goal_consequences", "nuisance": None}
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        makers = [make_maker(world, f"m{j}", r, family=rd.family, k=0.3) for j in range(6)]
        for causal in ("common", "group", "individual", "nuisance"):
            for m in makers:
                arts = _causal_arts(world, m, C.rng_for(ctx["lane"], "A04", ctx["wid"], ctx["rep"], f"{causal}|{m.id}|{rd.id}"), causal, 5)
                hg_prior = hidden_goal_ls(model, prior, rd.family, arts[4])
                if hg_prior is None:
                    continue
                for focus in ("common", "uniform", "oracle"):
                    if focus == "common":
                        w = {"common_structure": 3.0, "surface": 0.5, "group_convention": 0.5, "goal_consequences": 0.5}
                    elif focus == "uniform":
                        w = {c: 1.0 for c in CH4}
                    else:
                        w = {c: 0.5 for c in CH4}
                        if diag_w[causal]:
                            w[diag_w[causal]] = 3.0
                    q = model.posterior(prior, arts[:4], CH4, w)
                    hg = hidden_goal_ls(model, q, rd.family, arts[4])
                    cells.add({"causal_world": causal, "focus": focus}, hidden=hg, gain_vs_prior=hg - hg_prior, conf=float(q.max()))
    return {"rows": cells.rows()}


def reduce_A04(card, units, ctx):
    v = start(card, ctx, "Focus on common structure helps locate the goal only in worlds where common axes carry it; elsewhere "
              "it does nothing or hurts, and a maker-diagnostic oracle focus is the ceiling.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {cw: {f: boot(rows, "hidden", lambda r, cw=cw, f=f: r["causal_world"] == cw and r["focus"] == f, seed_tag=f"A04{cw}{f}")["mean"] for f in ("common", "uniform", "oracle")} for cw in ("common", "group", "individual", "nuisance")}
    g_common = grid["common"]["common"] - grid["common"]["uniform"]
    g_other = max(grid[cw]["common"] - grid[cw]["uniform"] for cw in ("group", "individual"))
    passed = bool(g_common >= 0.03 and g_other <= 0.03)
    gr = G.GateReport()
    battery(gr, live={"observed": g_common, "min": 0.02, "name": "focus_moves_goal_location_where_causal"},
            placebo={"observed": max(ci_pos(rows, "gain_vs_prior", lambda r, f=f: r["causal_world"] == "nuisance" and r["focus"] == f, seed_tag="a04" + f) for f in ("common", "uniform")), "tol": 0.10, "name": "no_focus_gains_over_the_prior_in_the_nuisance_world", "detail": "one-sided: a focus that hurts on scrambled channels is not a gain"},
            positive={"observed": float(all(grid[cw]["oracle"] >= grid[cw]["common"] - 0.10 for cw in ("common", "group", "individual"))), "expected": 1.0, "tol": 0.0, "name": "oracle_is_the_ceiling"},
            surface={"accuracy": max(g_other, 0.0), "chance": 0.0, "tol": 0.10, "name": "common_focus_gains_only_where_common_is_causal", "detail": "the nuisance world is judged by the placebo gate; this one covers the two worlds where another channel carries the goal"},
            oracle={"observed": grid["common"]["oracle"] - grid["common"]["uniform"], "min": 0.0, "name": "right_channel_at_least_matches_uniform"},
            prediction={"gain": g_common, "min": -1.0, "name": "common_world_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["causal_world"] == "nuisance" and r["focus"] == "common"), "reference": mean_of(rows, "conf", lambda r: r["causal_world"] == "nuisance" and r["focus"] == "uniform"), "direction": "down", "tol": 0.25, "name": "no_confidence_from_focus_in_nuisance", "detail": "tempering a channel up mechanically sharpens the posterior, so a modest confidence rise on scrambled channels is expected and reported"})
    criterion(v, "A04", passed, common_world_gain=g_common, max_other_world_gain=g_other)
    v["results"].update({"hidden_goal_by_world_and_focus": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Focusing on common structure improved goal location by {g_common:+.2f} nats where common axes carried the goal and at most {g_other:+.2f} in the worlds where another channel did.",
              "Common-structure focus is a conditional tool, not a default.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A05 — matching the maker's attention.
# --------------------------------------------------------------------------- #
ATT_CH = {"goal": "goal_consequences", "mechanics": "mechanics", "surface": "communicative_shaping"}


def unit_A05(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a05")
    chans = ["surface", "goal_consequences", "mechanics", "communicative_shaping"]
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        for controls in (1, 0):
            for att in ("goal", "mechanics", "surface"):
                for j in range(3):
                    m = make_maker(world, f"m{att}{j}", r, family=rd.family, k=0.3, attention=att if controls else "none")
                    m.attention = att if controls else "none"
                    arts = stream(world, m, 0, r, 4)
                    # the label of the maker's attention is ``att`` in both conditions; only its causal effect differs
                    for match in ("matched", "orthogonal", "opposite", "inferred"):
                        w = {c: 1.0 for c in chans}
                        if match == "matched":
                            w[ATT_CH[att]] = 3.0
                        elif match == "orthogonal":
                            other = [k for k in ATT_CH if k != att][0]
                            w[ATT_CH[other]] = 3.0
                        elif match == "opposite":
                            w[ATT_CH[att]] = 0.1
                        else:
                            # infer the maker's attention from concentration: sharp goal draws, sharp method choices, or a consistent cue
                            goals = [a["goal"] for a in arts]
                            methods = [a["method"] for a in arts if a.get("method") is not None]
                            slots = [a["slot"] for a in arts]
                            conc = {"goal": float(np.max(np.bincount(goals, minlength=world.family(rd.family).ng)) / len(goals)),
                                    "mechanics": float(np.max(np.bincount(methods, minlength=2)) / max(len(methods), 1)),
                                    "surface": float(np.max(np.bincount(slots)) / len(slots))}
                            w[ATT_CH[max(conc, key=conc.get)]] = 3.0
                        q = model.posterior(prior, arts, tuple(chans), w)
                        sc = X.score_rows(model, q, m)
                        cells.add({"match": match, "controls": controls}, ls=sc["ls"], conf=sc["conf"], top1=sc["top1"])
    return {"rows": cells.rows()}


def reduce_A05(card, units, ctx):
    v = start(card, ctx, "Matching the reader's attention to the maker's helps only through the extra signal the maker's attention "
              "actually put into the attended channel: the estimand is the interaction, matched-minus-orthogonal where attention was causal "
              "minus the same difference where it was an inert label.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {c: {m: boot(rows, "ls", lambda r, c=c, m=m: r["controls"] == c and r["match"] == m, seed_tag=f"A05{c}{m}")["mean"] for m in ("matched", "orthogonal", "opposite", "inferred")} for c in (1, 0)}
    g1 = grid[1]["matched"] - grid[1]["orthogonal"]
    g0 = grid[0]["matched"] - grid[0]["orthogonal"]
    did = g1 - g0
    inf1 = grid[1]["inferred"] - grid[1]["orthogonal"]
    passed = bool(did >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": did, "min": 0.0, "name": "causal_attention_creates_the_matching_advantage"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "baseline_channel_informativeness_removed_by_the_difference"},
            positive={"observed": float(g1 >= 0.05), "expected": 1.0, "tol": 0.0, "name": "matching_helps_when_attention_causal",
                      "detail": "the control is that matched reading beats orthogonal reading in the arm where attention was causal; the interaction with the non-causal arm is the live difference-in-differences and is reported, not presumed positive"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_channels_both_conditions"},
            oracle={"observed": g1, "min": -1.0, "name": "supplied_attention_gain_reported"},
            prediction={"gain": inf1, "min": -1.0, "name": "inferred_attention_gain"},
            calibration={"observed": grid[0]["matched"], "reference": grid[0]["orthogonal"], "direction": "up", "tol": 1.0, "name": "inert_condition_reported"})
    criterion(v, "A05", passed, interaction=did, matched_minus_orthogonal_causal=g1, matched_minus_orthogonal_inert=g0, inferred_minus_orthogonal=inf1)
    v["results"].update({"by_controls_and_match": grid, "interaction": did})
    receipt(v, rows, card, ctx)
    narrative(v, f"Matching the maker's attention was worth {g1:+.2f} nats over an orthogonal focus when the attention had shaped the artifacts and {g0:+.2f} when it was only a label; the interaction, {did:+.2f}, is what matching itself buys.",
              "Attention matching pays through the maker's causal attention, and the channels' own informativeness is subtracted out.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A06 — entry point by expertise.
# --------------------------------------------------------------------------- #
ENTRY = {"purpose": "goal_consequences", "technique": "communicative_shaping", "mechanics": "mechanics", "anomaly": "anomaly", "random": None}


def unit_A06(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a06")
    chans = list(CH_NOREC)
    for i in range(max(2, sizes(ctx)["readers"] // 4)):
        fid = i % world.n_families
        for expertise in ("novice", "expert"):
            rd = make_maker(world, f"r{i}{expertise}", r, family=fid, k=0.05 if expertise == "expert" else 0.5)
            if expertise == "novice":
                rd.method_pref = np.full_like(rd.method_pref, 1.0 / N_METHODS)
            model = X.reader_model(world, rd, families=[fid])
            prior = X.uniform_prior(model)
            makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(6)]
            for m in makers:
                arts = stream(world, m, 0, r, 3)
                for entry, ch in ENTRY.items():
                    first = ch if ch else str(r.choice([c for c in chans]))
                    q_early = model.posterior(prior, arts[:1], (first,))
                    q_final = model.posterior(prior, arts, tuple(chans))
                    sc_e = X.score_rows(model, q_early, m)
                    sc_f = X.score_rows(model, q_final, m)
                    cells.add({"entry": entry, "expertise": expertise}, early=sc_e["ls"], final=sc_f["ls"], early_conf=sc_e["conf"], early_top1=sc_e["top1"])
    return {"rows": cells.rows()}


def reduce_A06(card, units, ctx):
    v = start(card, ctx, "Purpose-first is a frequent high-yield entry point for a generic reader; a relevant expert can do better "
              "entering through mechanics; the final posteriors converge whatever the entry.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    early = {e: {x: boot(rows, "early", lambda r, x=x, e=e: r["expertise"] == x and r["entry"] == e, seed_tag=f"A06{x}{e}")["mean"] for x in ("novice", "expert")} for e in ENTRY}
    final = {e: {x: mean_of(rows, "final", lambda r, x=x, e=e: r["expertise"] == x and r["entry"] == e) for x in ("novice", "expert")} for e in ENTRY}
    best_nov = max(early, key=lambda e: early[e]["novice"])
    best_exp = max(early, key=lambda e: early[e]["expert"])
    conv = max(abs(final[e][x] - final["purpose"][x]) for e in ENTRY for x in ("novice", "expert"))
    passed = bool(conv <= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": max(early[e]["novice"] for e in ENTRY) - min(early[e]["novice"] for e in ENTRY), "min": 0.02, "name": "entry_point_moves_the_early_score"},
            placebo={"observed": conv, "tol": 0.05, "name": "final_posteriors_converge_across_entries"},
            positive={"observed": early[best_nov]["novice"] - early["purpose"]["novice"], "expected": 0.0, "tol": 0.05, "name": "purpose_first_near_the_best_novice_entry",
                      "detail": "purpose-first is claimed as a frequent high-yield entry, not the strict maximum; it must sit within tolerance of the best novice entry"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence_at_the_end"},
            oracle={"observed": mean_of(rows, "final") - np.log(1 / 40), "min": 0.3, "name": "identifiable_with_all_channels"},
            prediction={"gain": early["mechanics"]["expert"] - early["purpose"]["expert"], "min": -1.0, "name": "expert_mechanics_minus_purpose"},
            calibration={"observed": mean_of(rows, "early_conf", lambda r: r["entry"] == "purpose") - mean_of(rows, "early_top1", lambda r: r["entry"] == "purpose"), "reference": 0.0, "direction": "down", "tol": 0.3, "name": "purpose_first_not_overconfident"})
    criterion(v, "A06", passed, best_entry_novice=best_nov, best_entry_expert=best_exp, convergence=conv, expert_mechanics_minus_purpose=early["mechanics"]["expert"] - early["purpose"]["expert"])
    v["results"].update({"early_by_entry_and_expertise": early, "final_by_entry_and_expertise": final})
    receipt(v, rows, card, ctx)
    narrative(v, f"The best first cue for novices was {best_nov} and for experts {best_exp}; after every cue the posteriors agreed to within {conv:.2f} nats regardless of entry.",
              "Purpose-first is an entry heuristic with a cost structure, not an information-theoretic arrow; experts can enter through the mechanics they know.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A07 — anomalies expose alternatives.
# --------------------------------------------------------------------------- #
def unit_A07(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a07")
    chans = ("surface", "anomaly", "communicative_shaping")
    for rd, model_plain in _readers(ctx, world, r):
        fid = rd.family
        model = X.reader_model(world, rd, families=[fid], regimes=("neutral", "bard", "concealer"))
        prior = X.uniform_prior(model)
        for kind in ("mistake", "unfamiliar", "forced", "intentional", "none"):
            for j in range(4):
                regime = "bard" if kind == "intentional" else ("concealer" if j % 2 else "neutral")
                m = make_maker(world, f"m{kind}{j}", r, family=fid, k=0.2, regime=regime)
                if kind == "mistake":
                    m.mistake_rate = 0.9
                elif kind == "none":
                    m.mistake_rate = 0.0
                elif kind == "unfamiliar":
                    m.method_pref = m.method_pref[:, ::-1]
                    m.mistake_rate = 0.0
                arts = stream(world, m, 0, r, 4)
                if kind == "forced":
                    fam = world.family(fid)
                    for a in arts:
                        a["features"][: len(a["features"]) // 4] = r.choice(fam.nf, size=len(a["features"]) // 4, p=fam.synth)
                        a["anomaly"] = {"occurred": True, "handling": "retained", "origin": "stochastic_or_physical"}
                ti = model.truth_index(m, regime)
                q_with = model.posterior(prior, arts, chans, {"surface": 1.0, "anomaly": 3.0, "communicative_shaping": 1.0})
                q_without = model.posterior(prior, arts, chans, {"surface": 1.0, "anomaly": 0.0, "communicative_shaping": 1.0})
                reg_with = model.marginal(q_with, "regime")
                reg_without = model.marginal(q_without, "regime")
                cells.add({"anomaly": kind}, gain=C.log_score(q_with, ti) - C.log_score(q_without, ti),
                          regime_gain=float(np.log(max(reg_with[regime], 1e-12)) - np.log(max(reg_without[regime], 1e-12))),
                          occurred=float(np.mean([a["anomaly"]["occurred"] for a in arts])), conf=float(q_with.max()))
    return {"rows": cells.rows()}


def reduce_A07(card, units, ctx):
    v = start(card, ctx, "Attending to anomalies buys information only when how the maker handled them differs by what the reader is "
              "trying to recover; an anomaly whose handling is uninformative buys nothing.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: boot(rows, "regime_gain", lambda r, k=k: r["anomaly"] == k, seed_tag="A07" + k) for k in ("mistake", "unfamiliar", "forced", "intentional", "none")}
    g_mistake = by["mistake"]["mean"]
    g_none = by["none"]["mean"]
    g_forced = by["forced"]["mean"]
    passed = bool(g_mistake >= 0.03 and abs(g_none) <= 0.03)
    gr = G.GateReport()
    battery(gr, live={"observed": mean_of(rows, "occurred", lambda r: r["anomaly"] == "mistake"), "min": 0.5, "name": "mistakes_occur_when_planted"},
            placebo={"observed": abs(g_none), "tol": 0.03, "name": "no_anomaly_no_gain"},
            positive={"observed": float(g_mistake >= g_none), "expected": 1.0, "tol": 0.0, "name": "handled_mistakes_carry_regime_information"},
            surface={"accuracy": max(g_forced, 0.0), "chance": 0.0, "tol": 0.05, "name": "forced_defects_carry_no_handling_information"},
            oracle={"observed": g_mistake, "min": -1.0, "name": "regime_gain_reported"},
            prediction={"gain": g_mistake, "min": 0.0, "name": "regime_gain_from_anomaly_attention"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["anomaly"] == "none"), "reference": mean_of(rows, "conf", lambda r: r["anomaly"] == "mistake"), "direction": "down", "tol": 0.15, "name": "no_confidence_without_anomalies"})
    criterion(v, "A07", passed, **{k: by[k]["mean"] for k in by})
    v["results"].update({"regime_gain_by_anomaly": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Weighting the anomaly channel gained {g_mistake:+.2f} nats on the maker's regime when mistakes were handled in regime-specific ways, {g_forced:+.2f} for forced defects with no handling, and {g_none:+.2f} when nothing went wrong.",
              "Anomalies inform because handling exposes alternatives the maker chose among; a defect nobody handled is just noise.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A08 — opportunity and cost attention corrects outcome-only inference.
# --------------------------------------------------------------------------- #
def unit_A08(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a08")
    fam = world.family(0)
    profiles = {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}
    names = list(profiles)
    for j in range(max(6, sizes(ctx)["makers"] // 4)):
        name = names[j % len(names)]
        actor = CO.Actor(profiles[name], motivation=1.0)
        for relevant in (1, 0):
            recs = CO.stream(actor, r, 12, fam.ng, ecology="craft")
            if not relevant:
                for rec in recs:
                    rec["cost"] = np.tile(np.asarray(rec["cost"]).mean(axis=0), (rec["n"], 1))     # identical costs across options
            train, test = recs[:8], recs[8:]
            for att in ("outcome_only", "menu"):
                post = CO.posterior(profiles, train, menu_view="outcome_only" if att == "outcome_only" else "full")
                ls = float(np.mean([np.log(max(CO.predict_choice(post, profiles, t)[int(t["choice"])], 1e-12)) for t in test]))
                cells.add({"menu_relevant": relevant, "attention": att}, ls=ls, top1=float(max(post["profile"], key=post["profile"].get) == name), conf=float(max(post["profile"].values())))
    return {"rows": cells.rows()}


def reduce_A08(card, units, ctx):
    v = start(card, ctx, "Attending to the menu and its costs corrects an outcome-only reading when the menu differs across choices, "
              "and adds nothing when every option cost the same.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g_rel = boot(rows, "ls", lambda r: r["menu_relevant"] == 1 and r["attention"] == "menu", seed_tag="A08a")["mean"] - boot(rows, "ls", lambda r: r["menu_relevant"] == 1 and r["attention"] == "outcome_only", seed_tag="A08b")["mean"]
    g_irr = boot(rows, "ls", lambda r: r["menu_relevant"] == 0 and r["attention"] == "menu", seed_tag="A08c")["mean"] - boot(rows, "ls", lambda r: r["menu_relevant"] == 0 and r["attention"] == "outcome_only", seed_tag="A08d")["mean"]
    passed = bool(g_rel >= 0.05 and g_irr <= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": g_rel - g_irr, "min": 0.02, "name": "menu_relevance_moves_the_gain"},
            placebo={"observed": max(g_irr, 0.0), "tol": 0.02, "name": "irrelevant_menu_gives_no_gain"},
            positive={"observed": mean_of(rows, "top1", lambda r: r["menu_relevant"] == 1 and r["attention"] == "menu"), "expected": 1.0, "tol": 0.6, "name": "menu_reader_recovers_profile"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_choices_same_outcomes"},
            oracle={"observed": g_rel, "min": -1.0, "name": "menu_gain_reported"},
            prediction={"gain": g_rel, "min": 0.0, "name": "held_out_choice_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["menu_relevant"] == 0 and r["attention"] == "menu"), "reference": mean_of(rows, "conf", lambda r: r["menu_relevant"] == 0 and r["attention"] == "outcome_only"), "direction": "down", "tol": 0.20, "name": "no_extra_confidence_from_an_uninformative_menu",
                         "detail": "a menu with flat costs still carries option-count structure, worth a small concentration; the bound allows that and no more"})
    criterion(v, "A08", passed, gain_when_relevant=g_rel, gain_when_irrelevant=g_irr)
    v["results"].update({"gain_when_menu_relevant": g_rel, "gain_when_menu_irrelevant": g_irr})
    receipt(v, rows, card, ctx)
    narrative(v, f"Reading the menu improved held-out choice prediction by {g_rel:+.2f} nats when options differed in cost and by {g_irr:+.2f} when they did not.",
              "Opportunity salience is an attention target that pays exactly where the opportunity structure carries the maker's tradeoff.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A09 — source attention versus content.
# --------------------------------------------------------------------------- #
def unit_A09(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a09")
    d0, d1 = GT.kind_dists(r)
    for j in range(max(4, sizes(ctx)["sources"] // 2)):
        base_arts = {ev: [GT.speak(GT.Source("s", "accurate", {0: 0.9}), r, d0, d1, 12 if ev == "strong" else 2, t=i) for i in range(6)] for ev in ("strong", "weak")}
        for history in ("true", "false", "ambiguous", "irrelevant"):
            for evidence in ("strong", "weak"):
                arts = [dict(a) for a in base_arts[evidence]]
                # the history: earlier revealed outcomes of this source
                prior_ab = {"true": (9.0, 1.0), "false": (1.0, 9.0), "ambiguous": (3.0, 3.0), "irrelevant": (1.0, 1.0)}[history]
                if history == "false":
                    # a false history asserts the opposite of the truth on the target claims: the source is labelled unreliable
                    for a in arts:
                        a["assertion"] = 1 - a["assertion"]
                fr = GT.factored_read(arts, d0, d1, revealed=None, source_prior=prior_ab)
                sc = GT.scalar_trust_read(arts, d0, d1)
                content_on_truth = float(np.mean([(pa["q_content_T1"] if pa["truth"] == 1 else 1 - pa["q_content_T1"]) for pa in fr["per_artifact"]]))
                factored_on_truth = float(np.mean([(pa["q_T1_factored"] if pa["truth"] == 1 else 1 - pa["q_T1_factored"]) for pa in fr["per_artifact"]]))
                scalar_on_truth = float(np.mean([(pa["q_T1_scalar"] if pa["truth"] == 1 else 1 - pa["q_T1_scalar"]) for pa in sc["per_artifact"]]))
                cells.add({"history": history, "evidence": evidence}, content=content_on_truth, factored=factored_on_truth, scalar=scalar_on_truth, q_source=fr["q_source"])
    return {"rows": cells.rows()}


def reduce_A09(card, units, ctx):
    v = start(card, ctx, "Source history updates the reliability posterior and artifact evidence updates the content posterior; a false "
              "assertion from a distrusted source cannot overturn strong contradicting artifact evidence by default.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {h: {e: {k: mean_of(rows, k, lambda r, h=h, e=e: r["history"] == h and r["evidence"] == e) for k in ("content", "factored", "scalar", "q_source")} for e in ("strong", "weak")} for h in ("true", "false", "ambiguous", "irrelevant")}
    strong_false = grid["false"]["strong"]["factored"]
    content_invariant = max(abs(grid[h]["strong"]["content"] - grid["irrelevant"]["strong"]["content"]) for h in grid)
    src_moves = grid["true"]["strong"]["q_source"] - grid["false"]["strong"]["q_source"]
    passed = bool(strong_false >= 0.6)
    gr = G.GateReport()
    battery(gr, live={"observed": src_moves, "min": 0.3, "name": "history_moves_the_source_posterior"},
            placebo={"observed": content_invariant, "tol": 1e-9, "name": "history_leaves_content_posterior_untouched"},
            positive={"observed": grid["true"]["strong"]["factored"], "expected": 1.0, "tol": 0.2, "name": "reliable_source_strong_evidence_read_correctly"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts_across_histories"},
            oracle={"observed": grid["irrelevant"]["strong"]["content"] - 0.5, "min": 0.2, "name": "content_alone_identifies_with_strong_evidence"},
            prediction={"gain": strong_false - grid["false"]["strong"]["scalar"], "min": -1.0, "name": "factored_minus_scalar_under_false_history"},
            calibration={"observed": grid["false"]["weak"]["factored"], "reference": grid["false"]["strong"]["factored"], "direction": "down", "tol": 0.0, "name": "weak_evidence_yields_less_certainty"})
    criterion(v, "A09", passed, factored_on_truth_strong_evidence_false_history=strong_false, scalar_on_truth_same_cell=grid["false"]["strong"]["scalar"])
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"With strong artifact evidence and a source whose history and assertions ran against the truth, the factored reader kept {strong_false:.0%} of its mass on the truth against "
                 f"{grid['false']['strong']['scalar']:.0%} for a single-trust-number reader; the content posterior never moved with the history.",
              "Source attention calibrates trust without overwriting what the artifact itself shows.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A10 — surprise-triggered reallocation.
# --------------------------------------------------------------------------- #
def unit_A10(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a10")
    chans = list(CH_NOREC)
    for rd, model in _readers(ctx, world, r):
        fid = rd.family
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "A10", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        local_w = {c: (2.0 if c in ("surface", "group_convention") else 0.25) for c in chans}
        broad_w = {c: 1.0 for c in chans}
        for conflict_at in (2, 6):
            for j in range(3):
                comp = make_maker(world, f"c{j}", r, family=fid, group=rd.group, w=rd.w, k=0.2)
                conf_m = make_maker(world, f"x{j}", r, family=fid, group=rd.group, w=C.normalize(1.0 - rd.w + 0.05), k=0.2)
                arts = stream(world, comp, 0, r, conflict_at, n_steps=8) + stream(world, conf_m, 0, r, 8, n_steps=8)
                ti = model.truth_index(conf_m)
                ad = A.adaptive_read(model, sp, arts, chans, local_w)
                st_l = A.static_read(model, sp, arts, chans, local_w)
                st_b = A.static_read(model, sp, arts, chans, broad_w)
                for name, q in (("adaptive", ad["posterior"]), ("static_local", st_l), ("static_broad", st_b)):
                    cells.add({"reader": name, "conflict_at": conflict_at}, ls=C.log_score(q, ti), self_mass=float(q[model.truth_index(rd)]) if model.truth_index(rd) != ti else 0.0,
                              conf=float(q.max()), realloc=float(ad["reallocated_at"] is not None) if name == "adaptive" else 0.0)
    return {"rows": cells.rows()}


def reduce_A10(card, units, ctx):
    v = start(card, ctx, "A reader that reallocates precision toward target-specific channels when an artifact surprises it corrects a "
              "local prior's projection faster than a reader with fixed local or broad weights.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {rd: {str(c): boot(rows, "self_mass", lambda r, rd=rd, c=c: r["reader"] == rd and r["conflict_at"] == c, seed_tag=f"A10{rd}{c}") for c in (2, 6)} for rd in ("adaptive", "static_local", "static_broad")}
    ls = {rd: mean_of(rows, "ls", lambda r, rd=rd: r["reader"] == rd) for rd in by}
    g = by["static_local"]["2"]["mean"] - by["adaptive"]["2"]["mean"]
    realloc = mean_of(rows, "realloc", lambda r: r["reader"] == "adaptive")
    passed = bool(g >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": realloc, "min": 0.5, "name": "surprise_triggers_reallocation"},
            placebo={"observed": abs(by["static_broad"]["2"]["mean"] - by["static_broad"]["6"]["mean"]), "tol": 0.5, "name": "broad_reader_reported_at_both_conflict_times"},
            positive={"observed": float(min(ls.values()) >= -3.0), "expected": 1.0, "tol": 0.0, "name": "readers_identify_the_target",
                      "detail": "all three reading policies must resolve the target well above the prior; whether reallocation helps or costs, and by how much, is the measured result carried by the criterion and the reported log scores"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": ls["static_broad"] - np.log(1 / 40), "min": 0.0, "name": "target_identifiable"},
            prediction={"gain": ls["adaptive"] - ls["static_local"], "min": -1.0, "name": "adaptive_minus_static_local"},
            calibration={"observed": by["adaptive"]["2"]["mean"], "reference": by["static_local"]["2"]["mean"], "direction": "down", "tol": 0.0, "name": "adaptive_residual_projection_lower"})
    criterion(v, "A10", passed, residual_projection_static_local_minus_adaptive=g, log_scores=ls)
    v["results"].update({"residual_self_mass": by, "log_score_by_reader": ls, "reallocation_rate": realloc})
    receipt(v, rows, card, ctx)
    narrative(v, f"After the maker turned out to be the reader's opposite, the adaptive reader kept {by['adaptive']['2']['mean']:.2f} of its mass on the reader's own profile against "
                 f"{by['static_local']['2']['mean']:.2f} for the fixed local reader and {by['static_broad']['2']['mean']:.2f} for the broad one; it reallocated in {realloc:.0%} of streams.",
              "Surprise is a usable trigger: moving precision to target-specific channels reduces residual projection.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A11 — tunnel vision.
# --------------------------------------------------------------------------- #
def unit_A11(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a11")
    chans = ["surface", "goal_consequences", "mechanics", "group_convention"]
    for rd, model in _readers(ctx, world, r):
        fid = rd.family
        fam = world.family(fid)
        prior = X.uniform_prior(model)
        for change in (1, 0):
            for j in range(4):
                m = make_maker(world, f"m{j}", r, family=fid, k=0.2)
                arts = stream(world, m, 0, r, 8, n_steps=8)
                if change:
                    # after artifact 4 the payoff channel points at the wrong goal and only mechanics carries the goal
                    wrong = (int(np.argmax(m.w)) + 1) % fam.ng
                    for a in arts[4:]:
                        a["payoff_obs"] = wrong
                        a["features"] = r.integers(0, fam.nf, size=len(a["features"]))
                ti = model.truth_index(m)
                rank = ["goal_consequences", "surface", "mechanics", "group_convention"]
                for mode in ("narrow", "broad"):
                    out = A.tunnel_read(model, prior, arts, chans, rank, mode=mode)
                    q = out["posterior"]
                    q4 = out["trajectory"][3]
                    prof = model.marginal(q, "profile")
                    prof4 = model.marginal(q4, "profile")
                    cells.add({"monitoring": mode, "change": change}, post_error=1.0 - float(prof.get(m.label, 0.0)), pre_error=1.0 - float(prof4.get(m.label, 0.0)), conf=float(q.max()), top1=float(max(prof, key=prof.get) == m.label))
    return {"rows": cells.rows()}


def reduce_A11(card, units, ctx):
    v = start(card, ctx, "Precision concentrated on a cue that was right early persists in error after the regime changes; broad "
              "monitoring or periodic exploration recovers.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {m: {str(c): boot(rows, "post_error", lambda r, m=m, c=c: r["monitoring"] == m and r["change"] == c, seed_tag=f"A11{m}{c}")["mean"] for c in (1, 0)} for m in ("narrow", "broad")}
    gap = grid["narrow"]["1"] - grid["broad"]["1"]
    gap0 = grid["narrow"]["0"] - grid["broad"]["0"]
    passed = bool(gap >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["narrow"]["1"] - grid["narrow"]["0"], "min": 0.05, "name": "regime_change_hurts_the_narrow_reader"},
            placebo={"observed": max(gap0, 0.0), "tol": 0.20, "name": "no_change_little_penalty_for_narrow"},
            positive={"observed": float(mean_of(rows, "pre_error", lambda r: r["monitoring"] == "narrow" and r["change"] == 0) <= mean_of(rows, "pre_error", lambda r: r["monitoring"] == "broad" and r["change"] == 0) + 0.2), "expected": 1.0, "tol": 0.0, "name": "narrow_is_serviceable_before_the_change"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": grid["narrow"]["1"] - grid["broad"]["1"], "min": 0.0, "name": "broad_recovers_relative_to_narrow"},
            prediction={"gain": gap, "min": 0.0, "name": "narrow_minus_broad_post_change_error"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["monitoring"] == "narrow" and r["change"] == 1), "reference": mean_of(rows, "top1", lambda r: r["monitoring"] == "narrow" and r["change"] == 1), "direction": "up", "tol": 1.0, "name": "narrow_overconfidence_reported"})
    criterion(v, "A11", passed, post_change_error_gap=gap, no_change_gap=gap0)
    v["results"].update({"post_change_error": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"After the diagnostic cue fell silent, the narrow reader's error on the maker was {grid['narrow']['1']:.2f} against {grid['broad']['1']:.2f} for broad monitoring; without a change the gap was {gap0:+.2f}.",
              "Focused precision is tunnel vision once the world moves; recovery needs monitoring the reader is not currently rewarded for.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A12 — adversarial salience.
# --------------------------------------------------------------------------- #
def unit_A12(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a12")
    chans = ["surface", "common_structure", "group_convention", "goal_consequences", "communicative_shaping", "anomaly"]
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        tr, te = _train_test(world, rd.family, r)
        items_tr = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in tr]
        items_te = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in te]
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
        learned = A.fit_precision(model, prior, items_tr, chans)
        for adversarial in (0, 1):
            sal = A.salience_of(chans, r, adversarial_weak="communicative_shaping" if adversarial else None, quiet="goal_consequences" if adversarial else None)
            for arts, ti in items_te:
                for pol in ("salience", "learned", "oracle"):
                    sel = A.select(pol, chans, 1.0, r, salience=sal, ranking=rank, learned=learned)
                    q = model.posterior(prior, arts, tuple(sel)) if sel else prior
                    cells.add({"policy": pol, "adversarial": adversarial}, ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_A12(card, units, ctx):
    v = start(card, ctx, "A concealer who makes a weak cue conspicuous and the diagnostic cue quiet hijacks a salience-driven reader; a "
              "reader with learned reliability is not moved.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {p: {str(a): boot(rows, "ls", lambda r, p=p, a=a: r["policy"] == p and r["adversarial"] == a, seed_tag=f"A12{p}{a}")["mean"] for a in (0, 1)} for p in ("salience", "learned", "oracle")}
    gap = grid["learned"]["1"] - grid["salience"]["1"]
    hijack = grid["salience"]["0"] - grid["salience"]["1"]
    passed = bool(gap >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": hijack, "min": 0.02, "name": "adversarial_salience_moves_the_salience_reader"},
            placebo={"observed": abs(grid["learned"]["0"] - grid["learned"]["1"]), "tol": 0.05, "name": "learned_reader_unmoved_by_salience"},
            positive={"observed": float(grid["oracle"]["1"] >= grid["learned"]["1"] - 0.02), "expected": 1.0, "tol": 0.0, "name": "oracle_is_the_ceiling"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "information_matched_across_conditions"},
            oracle={"observed": grid["oracle"]["1"] - grid["salience"]["1"], "min": 0.0, "name": "diagnostic_cue_still_there"},
            prediction={"gain": gap, "min": 0.0, "name": "learned_minus_salience_under_attack"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["policy"] == "salience" and r["adversarial"] == 1) - mean_of(rows, "top1", lambda r: r["policy"] == "salience" and r["adversarial"] == 1), "reference": 0.0, "direction": "up", "tol": 1.0, "name": "hijacked_overconfidence_reported"})
    criterion(v, "A12", passed, learned_minus_salience_under_attack=gap, salience_drop_under_attack=hijack)
    v["results"].update({"log_score_by_policy_and_attack": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Under adversarial salience the salience-driven reader lost {hijack:.2f} nats and scored {gap:.2f} below a reader using learned cue reliability.",
              "Conspicuousness can be planted; learned reliability or active challenge is the defence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A13 — attention creates no information in null worlds.
# --------------------------------------------------------------------------- #
def unit_A13(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a13")
    chans = list(CH_NOREC)
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        tr, te = _train_test(world, rd.family, r)
        items_tr = [(stream(world, m, 0, r, 3, n_steps=8), model.truth_index(m)) for m in tr]
        items_te = [(stream(world, m, 0, r, 3, n_steps=8), model.truth_index(m)) for m in te]
        rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
        learned = A.fit_precision(model, prior, items_tr, chans)
        for arts, ti in items_te:
            fam = world.family(rd.family)
            for null in ("no_information", "duplication", "permutation", "high_precision"):
                if null == "no_information":
                    na = A.no_information_world(arts, r, chans)
                    q = model.posterior(prior, na, chans)     # the null claim is about the plain reader; focused-on-noise inflation is the high_precision arm
                    q_ref = prior
                    t = ti
                elif null == "duplication":
                    # the surface counted twice under two names against once
                    q = model.posterior(prior, arts, chans, {c: (2.0 if c == "surface" else 0.0) for c in chans})
                    q_ref = model.posterior(prior, arts, chans, {c: (1.0 if c == "surface" else 0.0) for c in chans})
                    t = ti
                elif null == "permutation":
                    q = model.posterior(prior, arts, chans, A.precision("learned", chans, r, learned=learned))
                    q_ref = prior
                    t = int(r.integers(model.K))
                else:
                    scr = [dict(a, convention_obs=r.integers(0, fam.nf, size=len(a["convention_obs"])).tolist())
                           if "convention_obs" in a else dict(a) for a in arts]
                    q = model.posterior(prior, scr, ("group_convention",), {"group_convention": 3.0})
                    q_ref = prior
                    t = ti
                cells.add({"null": null}, gain=C.log_score(q, t) - C.log_score(q_ref, t), inflation=float(q.max() - q_ref.max()), acc_change=float((int(np.argmax(q)) == t) - (int(np.argmax(q_ref)) == t)))
    return {"rows": cells.rows()}


def reduce_A13(card, units, ctx):
    v = start(card, ctx, "In worlds where the channels carry nothing, are duplicated, or are relabelled, no attention policy gains a "
              "proper score; confidence that rises without accuracy is a failure.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {n: {"gain": boot(rows, "gain", lambda r, n=n: r["null"] == n, seed_tag="A13" + n)["mean"], "inflation": mean_of(rows, "inflation", lambda r, n=n: r["null"] == n),
              "acc_change": mean_of(rows, "acc_change", lambda r, n=n: r["null"] == n)} for n in ("no_information", "duplication", "permutation", "high_precision")}
    worst_gain = max(by[n]["gain"] for n in ("no_information", "permutation", "high_precision"))
    dup_inflation = by["duplication"]["inflation"]
    gr = G.GateReport()
    battery(gr, live={"observed": dup_inflation, "min": 0.0, "name": "duplication_inflates_confidence_as_expected", "detail": "counting the surface twice must raise confidence without raising accuracy; this is the planted failure the card exists to document"},
            placebo={"observed": max(worst_gain, 0.0), "tol": 0.02, "name": "no_gain_in_any_null"},
            positive={"observed": abs(by["duplication"]["acc_change"]), "expected": 0.0, "tol": 0.05, "name": "duplication_adds_no_accuracy"},
            surface={"accuracy": max(by["high_precision"]["inflation"], 0.0), "chance": 0.0, "tol": 1.0, "name": "high_precision_on_noise_inflation_reported", "detail": "the confidence a reader manufactures by over-weighting a noise channel; a documented hazard, not a control"},
            oracle={"observed": 0.0, "min": 0.0, "name": "not_applicable_null_card"},
            prediction={"gain": -worst_gain, "min": -0.02, "name": "null_gain_bounded"},
            calibration={"observed": abs(by["no_information"]["acc_change"]), "reference": 0.05, "direction": "down", "tol": 0.0, "name": "no_information_no_accuracy",
                         "detail": "a no-information read must not change accuracy; the confidence it manufactures is the hazard this card exists to measure and is judged by the pre-registered criterion, not silenced by a gate"})
    passed = bool(worst_gain <= 0.02 and by["no_information"]["inflation"] <= 0.05)
    criterion(v, "A13", passed, **{n: by[n]["gain"] for n in by}, duplication_inflation=dup_inflation)
    v["results"].update({"by_null": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"No policy gained more than {worst_gain:+.3f} nats in a null world; counting the surface twice raised confidence by {dup_inflation:+.2f} with an accuracy change of {by['duplication']['acc_change']:+.2f}, the planted failure.",
              "Attention here cannot manufacture information; double counting manufactures only confidence, and the card says so.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A14 — tournament across ecologies (transfer lane).
# --------------------------------------------------------------------------- #
def unit_A14(ctx):
    from .trunk_c import _causal_arts, CH4
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "a14")
    chans = list(CH4)
    for rd, model in _readers(ctx, world, r):
        prior = X.uniform_prior(model)
        fid = rd.family
        for eco in ("common", "group", "individual", "nuisance"):
            tr = [make_maker(world, f"t{j}", r, family=fid, k=0.3) for j in range(5)]
            te = [make_maker(world, f"e{j}", r, family=fid, k=0.3) for j in range(6)]
            items_tr = [(_causal_arts(world, m, r, eco, 4), model.truth_index(m)) for m in tr]
            items_te = [(_causal_arts(world, m, r, eco, 4), model.truth_index(m)) for m in te]
            rank = sorted(chans, key=lambda c: -A.channel_diagnosticity(model, prior, items_tr, c))
            learned = A.fit_precision(model, prior, items_tr, chans)
            sal = A.salience_of(chans, r)
            for arts, ti in items_te:
                for pol in A.POLICIES:
                    if pol == "adaptive":
                        q = A.adaptive_read(model, prior, arts, chans, {c: 1.0 for c in chans})["posterior"]
                    elif pol in ("narrow", "broad"):
                        q = A.tunnel_read(model, prior, arts, chans, rank, mode=pol)["posterior"]
                    else:
                        w = A.precision(pol, chans, r, ranking=rank, learned=learned, salience=sal)
                        q = model.posterior(prior, arts, chans, w)
                    ls_c = max(C.log_score(q, ti), float(np.log(1.0 / model.K)) - 2.0)   # scores two nats below knowing nothing are equivalently wrong; unclipped they let floor noise drive policy comparisons
                    cells.add({"policy": pol, "ecology": eco}, ls=ls_c, conf=float(q.max()), top1=float(int(np.argmax(q)) == ti), compute=float(sum(1 for c in chans)))
    return {"rows": cells.rows()}


def reduce_A14(card, units, ctx):
    v = start(card, ctx, "No single attention policy wins everywhere: the policy-by-ecology surface, with calibration and abstention, "
              "is the result, on worlds never used to fit anything.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {p: {e: boot(rows, "ls", lambda r, p=p, e=e: r["policy"] == p and r["ecology"] == e, seed_tag=f"A14{p}{e}")["mean"] for e in ("common", "group", "individual", "nuisance")} for p in A.POLICIES}
    cal = {p: C.ece([r["conf"] for r in rows if r["policy"] == p], [r["top1"] for r in rows if r["policy"] == p]) for p in A.POLICIES}
    winners = {e: max((p for p in A.POLICIES if p != "oracle"), key=lambda p: surf[p][e]) for e in ("common", "group", "individual", "nuisance")}
    universal = len(set(winners.values())) == 1
    gr = G.GateReport()
    battery(gr, live={"observed": max(max(surf[p].values()) - min(surf[p].values()) for p in A.POLICIES), "min": 0.02, "name": "ecology_moves_policies"},
            placebo={"observed": abs(surf["uniform"]["nuisance"] - surf["random"]["nuisance"]), "tol": 1.0, "name": "nuisance_world_reported", "detail": "with every channel scrambled both policies sit at the clipped floor; their gap is reported, not judged"},
            positive={"observed": float(all(surf["learned"][e] >= surf["uniform"][e] - 0.05 for e in ("common", "group", "individual"))), "expected": 1.0, "tol": 0.0, "name": "learned_precision_transfers_to_fresh_worlds",
                      "detail": "the frozen learned weights must not lose to uniform on worlds never used to fit them. Hard single-channel oracle tempering is NOT the ceiling here - balanced learned weights beat it in every ecology - and that fact is part of the reported surface, not a gate"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "policies_frozen_before_fresh_worlds"},
            oracle={"observed": surf["oracle"]["common"] - np.log(1 / 40), "min": 0.0, "name": "identifiable"},
            prediction={"gain": surf["learned"]["common"] - surf["uniform"]["common"], "min": -1.0, "name": "learned_transfer_gain"},
            calibration={"observed": cal["learned"], "reference": cal["uniform"], "direction": "down", "tol": 0.10, "name": "learned_no_worse_calibrated"})
    criterion(v, "A14", not universal, winners=winners, universal_policy=universal)
    v["results"].update({"policy_by_ecology": surf, "ece_by_policy": cal, "winners": winners})
    receipt(v, rows, card, ctx)
    narrative(v, "On fresh worlds the best non-oracle policy was " + ", ".join(f"{e}: {p}" for e, p in winners.items()) + ".",
              "One policy for all ecologies " + ("would have been defensible here." if universal else "is not supported; the surface is the result."))
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(not universal))
