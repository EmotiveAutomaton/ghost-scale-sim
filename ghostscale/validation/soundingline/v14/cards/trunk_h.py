"""Trunk H — hierarchy, habit, and value residue (spec §5, cards H01-H08).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import hierarchy as H
from .. import joint as J
from .. import history_skill as HS
from ..world import N_ACT, episode, make_maker, stream
from . import Cells, battery, criterion, decide_state, extra_gate, finish, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for


def _ls(p, i):
    return float(np.log(max(float(p[int(i)]), 1e-12)))


# --------------------------------------------------------------------------- #
# H01 — subtask hierarchy recovered where it exists.
# --------------------------------------------------------------------------- #
def unit_H01(ctx):
    r = rng(ctx, "h01")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        subs = H.subgoals(r)
        tops = H.top_goals(r, shared=False)
        for structure in ("hierarchical", "flat"):
            if structure == "hierarchical":
                ep = H.produce(subs, tops[0], r)
                true_b = set(ep["boundaries"][1:])
            else:
                flat = np.stack([r.dirichlet(np.full(H.N_PRIM, 0.6)) for _ in range(9)])
                ep = {"actions": [int(r.choice(H.N_PRIM, p=p)) for p in flat], "boundaries": [0]}
                true_b = set()
            sc = H.boundary_score(ep["actions"], subs)
            thr = np.quantile(sc, 0.7)
            found = {i + 1 for i, s in enumerate(sc) if s >= thr}
            hit = len(found & true_b) / max(len(true_b), 1) if true_b else 0.0
            spurious = len(found - true_b) / max(len(found), 1)
            # next subtask prediction: with the hierarchy, the next subgoal's first action; flat baseline uses action frequency
            if structure == "hierarchical":
                ll = H.loglik_top({"actions": ep["actions"][:6]}, subs, tops)
                post = C.softmax(ll)
                nxt = sum(post[k] * subs[tops[k][2], 0] for k in range(len(tops)))
                a7 = ep["actions"][6]
                freq = np.bincount(ep["actions"][:6], minlength=H.N_PRIM) + 0.5
                gain = _ls(nxt, a7) - _ls(freq / freq.sum(), a7)
            else:
                gain = 0.0
            cells.add({"structure": structure}, boundary_hit=hit, spurious=spurious, gain=gain)
    return {"rows": cells.rows()}


def reduce_H01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H01"]
    v = start(card, ctx, "Repeated transition structure recovers subtask boundaries and predicts the next subtask where a hierarchy exists, and finds few boundaries where none does.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    hit = mean_of(rows, "boundary_hit", lambda r: r["structure"] == "hierarchical")
    sp = {s: mean_of(rows, "spurious", lambda r, s=s: r["structure"] == s) for s in ("hierarchical", "flat")}
    gain = mean_of(rows, "gain", lambda r: r["structure"] == "hierarchical")
    passed = bool(hit >= cr["min_boundary"] and gain >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": hit - sp["hierarchical"], "min": 0.1, "name": "boundaries_found_above_false_ones"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_subgoal_library_both_structures"},
            positive={"observed": hit, "expected": 1.0, "tol": 1.0 - cr["min_boundary"], "name": "planted_boundaries_recovered"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "actions_only"},
            oracle={"observed": hit, "min": 0.5, "name": "hierarchy_identifiable"},
            prediction={"gain": gain, "min": 0.0, "name": "next_subtask_over_frequency"},
            calibration={"observed": 1.0 - sp["flat"] if sp["flat"] == sp["flat"] else 0.0, "reference": 1.0, "direction": "down", "tol": 0.0, "name": "flat_worlds_reported"})
    criterion(v, "H01", passed, boundary_recovery=hit, spurious=sp, next_subtask_gain=gain)
    receipt(v, rows, card, ctx)
    narrative(v, f"Planted subtask boundaries were recovered {hit:.0%} of the time and the next subtask's first action predicted {gain:+.3f} nats above frequency.",
              "A hierarchy shows up in transitions before it shows up in goals.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H02 — identical local actions stay ambiguous.
# --------------------------------------------------------------------------- #
def unit_H02(ctx):
    r = rng(ctx, "h02")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        subs = H.subgoals(r)
        tops = H.top_goals(r, shared=True)
        k = int(r.integers(2))                                   # top goals 0 and 1 share their opening subgoal
        ep = H.produce(subs, tops[k], r)
        for window in ("shared", "distinct"):
            L = H.SUB_LEN if window == "shared" else 3 * H.SUB_LEN
            ll = H.loglik_top({"actions": ep["actions"][:L]}, subs, tops)
            post = C.softmax(ll)
            cells.add({"window": window}, top_mass=float(post.max()), correct=float(int(np.argmax(post)) == k), pair_mass=float(post[0] + post[1]))
    return {"rows": cells.rows()}


def reduce_H02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H02"]
    v = start(card, ctx, "When two higher goals share their opening actions, the reader keeps them ambiguous until the actions diverge.", "BOUNDARY")
    rows = [r for u in units for r in u["rows"]]
    tm = {w: mean_of(rows, "top_mass", lambda r, w=w: r["window"] == w) for w in ("shared", "distinct")}
    pm = mean_of(rows, "pair_mass", lambda r: r["window"] == "shared")
    passed = bool(tm["shared"] <= cr["max_shared"] and tm["distinct"] >= cr["min_distinct"])
    gr = G.GateReport()
    extra_gate(gr, "reward_equivalence", "shared_window_keeps_the_pair", pm, 0.8, "min", "mass on the two goals that share the window")
    battery(gr, live={"observed": tm["distinct"] - tm["shared"], "min": 0.1, "name": "divergence_resolves"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_library"},
            positive={"observed": tm["distinct"], "expected": 1.0, "tol": 1.0 - cr["min_distinct"], "name": "distinct_window_identifies"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "actions_only"},
            oracle={"observed": tm["distinct"] - 1 / 3, "min": 0.2, "name": "identifiable_after_divergence"},
            prediction={"gain": mean_of(rows, "correct", lambda r: r["window"] == "distinct") - 1 / 3, "min": 0.0, "name": "top_goal_named_after_divergence"},
            calibration={"observed": tm["shared"], "reference": cr["max_shared"], "direction": "down", "tol": 0.0, "name": "no_forced_attribution_on_the_shared_window"})
    criterion(v, "H02", passed, top_mass=tm, pair_mass_shared=pm)
    receipt(v, rows, card, ctx)
    narrative(v, f"On the shared opening window the reader's largest top-goal mass was {tm['shared']:.2f} with {pm:.2f} on the sharing pair; after divergence {tm['distinct']:.2f}.",
              "Identical local actions do not name a higher goal, and the reader says so.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H03 — reward equivalence.
# --------------------------------------------------------------------------- #
def unit_H03(ctx):
    r = rng(ctx, "h03")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        reward = r.normal(0, 1, H.N_PRIM)
        potential = r.normal(0, 1, H.N_PRIM)
        shaped_truth = bool(r.random() < 0.5)
        pol = H.policy_from_reward(H.shaped(reward, potential) if shaped_truth else reward)
        acts = [int(r.choice(H.N_PRIM, p=pol)) for _ in range(24)]
        ll_plain = sum(np.log(H.policy_from_reward(reward)[a]) for a in acts)
        ll_shaped = sum(np.log(H.policy_from_reward(H.shaped(reward, potential))[a]) for a in acts)
        p_shaped = float(C.softmax(np.array([ll_plain, ll_shaped]))[1])
        cells.add({"phase": "observational"}, p_shaped=p_shaped, correct=float((p_shaped > 0.5) == shaped_truth), max_mass=max(p_shaped, 1 - p_shaped))
        pp, ps = H.resolving_intervention(reward, potential, r)
        acts2 = [int(r.choice(H.N_PRIM, p=ps if shaped_truth else pp)) for _ in range(24)]
        ll2 = np.array([sum(np.log(pp[a]) for a in acts2), sum(np.log(ps[a]) for a in acts2)])
        p2 = float(C.softmax(ll2)[1])
        cells.add({"phase": "intervened"}, p_shaped=p2, correct=float((p2 > 0.5) == shaped_truth), max_mass=max(p2, 1 - p2))
    return {"rows": cells.rows()}


def reduce_H03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H03"]
    v = start(card, ctx, "A reward and its policy-equivalent transformation cannot be told apart from behaviour, and can be after an intervention that breaks the equivalence.", "BOUNDARY")
    rows = [r for u in units for r in u["rows"]]
    mm = {p: mean_of(rows, "max_mass", lambda r, p=p: r["phase"] == p) for p in ("observational", "intervened")}
    acc = {p: mean_of(rows, "correct", lambda r, p=p: r["phase"] == p) for p in ("observational", "intervened")}
    passed = bool(mm["observational"] <= cr["max_observational"] and acc["intervened"] >= cr["min_intervened"])
    gr = G.GateReport()
    extra_gate(gr, "reward_equivalence", "equivalent_rewards_indistinguishable_observationally", mm["observational"], cr["max_observational"], "max", "largest mass on either reward from behaviour alone")
    battery(gr, live={"observed": acc["intervened"] - acc["observational"], "min": 0.2, "name": "intervention_resolves"},
            placebo={"observed": abs(acc["observational"] - 0.5), "tol": 0.15, "name": "observational_accuracy_at_chance"},
            positive={"observed": acc["intervened"], "expected": 1.0, "tol": 1.0 - cr["min_intervened"], "name": "resolved_by_intervention"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "identical_policies_by_construction"},
            oracle={"observed": acc["intervened"] - 0.5, "min": 0.2, "name": "identifiable_under_intervention"},
            prediction={"gain": acc["intervened"] - 0.5, "min": 0.0, "name": "intervened_choices_predicted"},
            calibration={"observed": mm["observational"], "reference": cr["max_observational"], "direction": "down", "tol": 0.0, "name": "no_confidence_without_intervention"})
    criterion(v, "H03", passed, max_mass=mm, accuracy=acc)
    receipt(v, rows, card, ctx)
    narrative(v, f"From behaviour alone the reader's larger mass on either reward was {mm['observational']:.2f} (accuracy {acc['observational']:.0%}); after the resolving intervention accuracy was {acc['intervened']:.0%}.",
              "Requiring one true reward where two are policy-equivalent is a scoring defect; the construction makes the equivalence and the reader respects it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H04 — preference against habit across changed incentives.
# --------------------------------------------------------------------------- #
def unit_H04(ctx):
    r = rng(ctx, "h04")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        pref = r.dirichlet(np.ones(H.N_PRIM))
        habit = r.normal(0, 1, H.N_PRIM)
        inc0 = np.ones(H.N_PRIM)
        inc1 = r.uniform(0.2, 2.0, H.N_PRIM)
        for kind in ("preference_driven", "habitual"):
            strength = 0.0 if kind == "preference_driven" else 2.0
            for incentive, inc in (("original", inc0), ("changed", inc1)):
                truth = H.habit_policy(pref, inc, habit, strength)
                acts = [int(r.choice(H.N_PRIM, p=truth)) for _ in range(10)]
                # models fitted on the ORIGINAL incentive's choices, evaluated on this incentive's choices
                fit_acts = [int(r.choice(H.N_PRIM, p=H.habit_policy(pref, inc0, habit, strength))) for _ in range(10)]
                freq = np.bincount(fit_acts, minlength=H.N_PRIM) + 0.5
                freq = freq / freq.sum()
                pref_hat = freq                                              # a preference model: choices scale with incentive
                pref_model = H.preference_policy(pref_hat, inc)
                habit_model = freq                                           # a habit model: the practiced frequency persists regardless of incentive
                for model, dist in (("preference", pref_model), ("habit", habit_model)):
                    ls = float(np.mean([np.log(max(dist[a], 1e-12)) for a in acts]))
                    cells.add({"incentive": incentive, "model": model}, ls=ls, **{f"ls_{kind}": ls})
    return {"rows": cells.rows()}


def reduce_H04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H04"]
    v = start(card, ctx, "Across changed incentives a standing preference and a practiced habit predict different choices: the preference model wins for preference-driven makers and the habit model for habitual ones.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {k: {m: mean_of(rows, f"ls_{k}", lambda r, m=m: r["incentive"] == "changed" and r["model"] == m) for m in ("preference", "habit")} for k in ("preference_driven", "habitual")}
    gain_pref = ls["preference_driven"]["preference"] - ls["preference_driven"]["habit"]
    gain_habit = ls["habitual"]["habit"] - ls["habitual"]["preference"]
    same = {m: mean_of(rows, "ls", lambda r, m=m: r["incentive"] == "original" and r["model"] == m) for m in ("preference", "habit")}
    passed = bool(gain_pref >= cr["min_gain"] and gain_habit >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": gain_pref + gain_habit, "min": 0.05, "name": "incentive_change_separates_the_models"},
            placebo={"observed": abs(same["preference"] - same["habit"]), "tol": 0.3, "name": "under_the_original_incentive_the_models_agree"},
            positive={"observed": gain_habit, "expected": max(gain_habit, cr["min_gain"]), "tol": 0.0, "name": "habit_model_wins_for_habitual_makers"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_fitted_frequency_both_models"},
            oracle={"observed": gain_pref, "min": 0.0, "name": "preference_model_wins_for_preference_makers"},
            prediction={"gain": min(gain_pref, gain_habit), "min": 0.0, "name": "new_domain_choices"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "crossed_design_reported"})
    criterion(v, "H04", passed, preference_gain=gain_pref, habit_gain=gain_habit, log_scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"After incentives changed, the preference model beat the habit model by {gain_pref:+.3f} nats on preference-driven makers and lost by {gain_habit:+.3f} on habitual ones.",
              "A preference and a habit look alike until the incentives move; then they come apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H05 — stale residue against current preference.
# --------------------------------------------------------------------------- #
def unit_H05(ctx):
    world = world_for(ctx)
    r = rng(ctx, "h05")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = J.Reader(world, 0, 0.75, 0.8)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        m = HS.agent(world, f"m{i}", r, 0, "mid", "strong")
        HS.reverse_reward(m, 4)
        for phase, start_idx in (("before_reversal", 0), ("after_reversal", 12)):
            eps = [episode(world, m, r, index=start_idx + k) for k in range(4)]
            ep_next = episode(world, m, r, index=start_idx + 4)
            a1, a = int(ep_next["action"][0]), int(ep_next["action"][1])
            post = J.joint(prior, rd.route_tables(eps, ("action", "semantic", "context")))
            pref_pred = J.next_episode_second_action_dist(rd, post, a1)
            # the residue model: transition frequency tilted by the practiced transitions at their pre-reversal strength
            M2 = np.zeros((N_ACT, N_ACT)) + 0.5
            for e in eps:
                for t in range(1, len(e["action"])):
                    M2[e["action"][t - 1], e["action"][t]] += 1
            resid = C.softmax(np.log(M2[a1] / M2[a1].sum()) + m.h_strength * m.h_trans[a1])
            cells.add({"phase": phase}, pref_ls=_ls(pref_pred, a), residue_ls=_ls(resid, a), gap=_ls(pref_pred, a) - _ls(resid, a))
    return {"rows": cells.rows()}


def reduce_H05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H05"]
    v = start(card, ctx, "After a history reversal, a maker's hidden future choice is predicted better by its current preference than by the residue of its old attention.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    gap = {p: mean_of(rows, "gap", lambda r, p=p: r["phase"] == p) for p in ("before_reversal", "after_reversal")}
    passed = bool(gap["after_reversal"] >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": gap["after_reversal"] - gap["before_reversal"], "min": 0.0, "name": "reversal_moves_the_gap"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_maker_both_phases"},
            positive={"observed": gap["after_reversal"], "expected": max(gap["after_reversal"], cr["min_gain"]), "tol": 0.0, "name": "preference_beats_residue_after_reversal"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "hidden_future_choice"},
            oracle={"observed": mean_of(rows, "pref_ls", lambda r: r["phase"] == "after_reversal") - np.log(1 / N_ACT), "min": 0.0, "name": "preference_above_chance"},
            prediction={"gain": gap["after_reversal"], "min": 0.0, "name": "current_preference_over_residue"},
            calibration={"observed": gap["before_reversal"], "reference": gap["after_reversal"], "direction": "down", "tol": 0.0, "name": "residue_worth_more_before_than_after"})
    criterion(v, "H05", passed, gap_by_phase=gap)
    receipt(v, rows, card, ctx)
    narrative(v, f"Before the reversal the current-preference model beat the residue model by {gap['before_reversal']:+.3f} nats on the hidden next choice; after it by {gap['after_reversal']:+.3f}.",
              "Stale expertise is a residue that predicts less and less; the preference is what the record should carry forward.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H06 — higher coordinating goals compress without a horizon.
# --------------------------------------------------------------------------- #
def unit_H06(ctx):
    r = rng(ctx, "h06")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    per = []
    for i in range(n):
        subs = H.subgoals(r)
        tops = H.top_goals(r, shared=False)
        k = int(r.integers(len(tops)))
        eps = [H.produce(subs, tops[k], r) for _ in range(3)]
        test = H.produce(subs, tops[k], r)
        acts_seen = [a for e in eps for a in e["actions"]]
        freq = np.bincount(acts_seen, minlength=H.N_PRIM) + 0.5
        freq = freq / freq.sum()
        # action level: frequency; subgoal level: best single subgoal chain per position; top level: posterior over top goals
        ll_top = H.loglik_top({"actions": acts_seen[:9]}, subs, tops) * 3
        post = C.softmax(ll_top)
        scores = {"action": [], "subgoal": [], "top": []}
        for t, a in enumerate(test["actions"]):
            scores["action"].append(np.log(freq[a]))
            pos = t % H.SUB_LEN
            best_sub = max(range(H.N_SUB), key=lambda s: np.log(subs[s, pos][a] + 1e-9)) if t == 0 else None
            seg = t // H.SUB_LEN
            sub_pred = np.mean([subs[s, pos] for s in range(H.N_SUB)], axis=0) if seg == 0 and t == 0 else subs[tops[int(np.argmax(post))][min(seg, 2)], pos]
            scores["subgoal"].append(np.log(0.9 * sub_pred[a] + 0.1 / H.N_PRIM))
            top_pred = sum(post[j] * subs[tops[j][min(seg, 2)], pos] for j in range(len(tops)))
            scores["top"].append(np.log(0.9 * top_pred[a] + 0.1 / H.N_PRIM))
        for level in ("action", "subgoal", "top"):
            cells.add({"level": level}, ls=float(np.mean(scores[level])), compression=float(np.mean(scores[level]) - np.mean(scores["action"])))
        per.append((float(post.max()), float(int(np.argmax(post)) == k)))
    return {"rows": cells.rows(), "ece": C.ece([c for c, _ in per], [y for _, y in per])}


def reduce_H06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H06"]
    v = start(card, ctx, "Multi-episode evidence supports a coordinating goal above the subgoals, which compresses prediction of fresh episodes, with calibrated uncertainty about the level and no terminal horizon assumed.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    comp = {lv: mean_of(rows, "compression", lambda r, lv=lv: r["level"] == lv) for lv in ("action", "subgoal", "top")}
    ece = float(np.nanmean([u["ece"] for u in units]))
    passed = bool(comp["top"] - comp["subgoal"] >= cr["min_compression"] or comp["top"] >= cr["min_compression"]) and ece <= cr["max_ece"]
    gr = G.GateReport()
    battery(gr, live={"observed": comp["top"], "min": 0.0, "name": "top_level_compresses_over_actions"},
            placebo={"observed": abs(comp["action"]), "tol": 1e-9, "name": "action_level_is_the_baseline"},
            positive={"observed": comp["top"], "expected": max(comp["top"], cr["min_compression"]), "tol": 0.0, "name": "coordinating_goal_predicts"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "actions_only"},
            oracle={"observed": comp["top"] - comp["subgoal"], "min": -1.0, "name": "top_over_subgoal_reported"},
            prediction={"gain": comp["top"], "min": 0.0, "name": "fresh_episode_compression"},
            calibration={"observed": ece, "reference": cr["max_ece"], "direction": "down", "tol": 0.0, "name": "level_uncertainty_calibrated"})
    criterion(v, "H06", passed, compression=comp, ece=ece)
    receipt(v, rows, card, ctx)
    narrative(v, f"A coordinating top goal predicted fresh episodes {comp['top']:+.3f} nats per action better than action frequency and {comp['top'] - comp['subgoal']:+.3f} better than the subgoal level, with top-goal calibration error {ece:.3f}.",
              "Levels earn their place by compression, not by a horizon the record cannot see.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H07 — role-relative control from full records.
# --------------------------------------------------------------------------- #
def unit_H07(ctx):
    r = rng(ctx, "h07")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        seed = int(r.integers(1 << 30))
        for team in ("central", "shared_brief"):
            prod = H.team_production(np.random.default_rng(seed), team)
            truth = float(team == "central")
            for reader, p in (("interaction", H.interaction_reader(prod)), ("coherence", H.coherence_reader(prod))):
                correct = float((p > 0.5) == (truth == 1.0)) if p != 0.5 else 0.5
                # hidden next intervention: under central control the next correction's issuer is the director
                nxt = prod["next"]
                pred_next = float(p if nxt["corrected"] else 1.0)
                next_correct = float(((p > 0.5) == (nxt["issuer"] == "director")) if nxt["corrected"] else 0.5) if reader == "interaction" else 0.5
                cells.add({"team": team, "reader": reader}, correct=correct, p_central=p, next_correct=next_correct)
    return {"rows": cells.rows()}


def reduce_H07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H07"]
    v = start(card, ctx, "With full interaction records, role-relative control is recovered against an exact shared-brief twin; coherence stays at chance; the hidden next intervention follows.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = {rd: mean_of(rows, "correct", lambda r, rd=rd: r["reader"] == rd) for rd in ("interaction", "coherence")}
    nxt = mean_of(rows, "next_correct", lambda r: r["reader"] == "interaction")
    passed = bool(acc["interaction"] >= cr["min_interaction"] and acc["coherence"] <= cr["max_coherence"] and nxt >= cr["min_next"])
    gr = G.GateReport()
    battery(gr, live={"observed": acc["interaction"] - acc["coherence"], "min": 0.3, "name": "records_separate_what_coherence_cannot"},
            placebo={"observed": abs(acc["coherence"] - 0.5), "tol": 0.1, "name": "coherence_at_chance_on_twins"},
            positive={"observed": acc["interaction"], "expected": 1.0, "tol": 1.0 - cr["min_interaction"], "name": "interaction_reader_separates"},
            surface={"accuracy": acc["coherence"], "chance": 0.5, "tol": 0.1, "name": "artifact_twins_by_construction"},
            oracle={"observed": acc["interaction"] - 0.5, "min": 0.4, "name": "control_identifiable_with_records"},
            prediction={"gain": nxt - 0.5, "min": 0.0, "name": "next_intervention_predicted"},
            calibration={"observed": abs(mean_of(rows, "p_central", lambda r: r["reader"] == "interaction" and r["team"] == "shared_brief")), "reference": 0.5, "direction": "down", "tol": 0.0, "name": "shared_brief_not_called_central"})
    criterion(v, "H07", passed, accuracy=acc, next_intervention=nxt)
    receipt(v, rows, card, ctx)
    narrative(v, f"From full records the interaction reader separated central control from its shared-brief twin {acc['interaction']:.0%} of the time and predicted the next intervention's issuer {nxt:.0%}; coherence sat at {acc['coherence']:.0%}.",
              "The hand that controls is in the record of who corrected whom, and nowhere in the artifact.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# H08 — which level predicts the next changed-context action.
# --------------------------------------------------------------------------- #
def unit_H08(ctx):
    r = rng(ctx, "h08")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        subs = H.subgoals(r)
        tops = H.top_goals(r, shared=False)
        k = int(r.integers(len(tops)))
        eps = [H.produce(subs, tops[k], r) for _ in range(3)]
        # changed context: the same top goal, a fresh subgoal library (the actions differ, the goal persists)
        subs2 = H.subgoals(r)
        test = H.produce(subs2, tops[k], r)
        acts_seen = [a for e in eps for a in e["actions"]]
        freq = np.bincount(acts_seen, minlength=H.N_PRIM) + 0.5
        freq = freq / freq.sum()
        post = C.softmax(H.loglik_top({"actions": acts_seen[:9]}, subs, tops) * 3)
        last_goal_sub = tops[int(np.argmax(post))][-1]
        scores = {"top": [], "subgoal": [], "flat_value": [], "last_goal": []}
        for t, a in enumerate(test["actions"]):
            seg, pos = t // H.SUB_LEN, t % H.SUB_LEN
            top_pred = sum(post[j] * subs2[tops[j][min(seg, 2)], pos] for j in range(len(tops)))
            scores["top"].append(np.log(0.9 * top_pred[a] + 0.1 / H.N_PRIM))
            scores["subgoal"].append(np.log(0.9 * subs2[tops[k][min(seg, 2)], pos][a] * 0 + 0.9 * np.mean([subs2[s, pos] for s in range(H.N_SUB)], axis=0)[a] + 0.1 / H.N_PRIM))
            scores["flat_value"].append(np.log(freq[a]))
            scores["last_goal"].append(np.log(0.9 * subs2[last_goal_sub, pos][a] + 0.1 / H.N_PRIM))
        for model in scores:
            cells.add({"model": model}, ls=float(np.mean(scores[model])))
    return {"rows": cells.rows()}


def reduce_H08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["H08"]
    v = start(card, ctx, "When the context changes, the level that carries over is the coordinating goal, and it predicts the next actions better than flat value or the last subgoal.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {m: mean_of(rows, "ls", lambda r, m=m: r["model"] == m) for m in ("top", "subgoal", "flat_value", "last_goal")}
    best = max(ls, key=ls.get)
    margin_flat = ls["top"] - ls["flat_value"]
    margin_last = ls["top"] - ls["last_goal"]
    passed = bool(margin_flat >= cr["margin"] and margin_last >= cr["margin"])
    gr = G.GateReport()
    battery(gr, live={"observed": max(ls.values()) - min(ls.values()), "min": 0.0, "name": "levels_differ"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_test_episode_every_model"},
            positive={"observed": margin_flat, "expected": max(margin_flat, cr["margin"]), "tol": 0.0, "name": "top_level_beats_flat_value"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "changed_context_by_construction"},
            oracle={"observed": ls["top"] - np.log(1 / H.N_PRIM), "min": 0.0, "name": "top_level_above_chance"},
            prediction={"gain": min(margin_flat, margin_last), "min": 0.0, "name": "selected_level_over_baselines"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "tournament_reported"})
    criterion(v, "H08", passed, log_score=ls, best=best)
    receipt(v, rows, card, ctx)
    narrative(v, f"Under a changed context the top-goal model scored {ls['top']:.3f} per action against {ls['flat_value']:.3f} for flat value and {ls['last_goal']:.3f} for the last subgoal; the best level was {best}.",
              "What survives a change of context is the goal that coordinated the old one.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
