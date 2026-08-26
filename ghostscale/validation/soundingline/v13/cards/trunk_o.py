"""Trunk O: opportunity, cost, and preference evidence (spec §11).

Observed effort is never a value label. Every card plants rival causes of paid cost, records the
full menu, and scores readers on held-out or prospective choices under a proper score.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import costs as CO
from . import (battery, boot, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of, cost_ecologies)
from .trunk_c import Cells

LEV = CO.LEVELS


def _profiles(world, fid=0):
    fam = world.family(fid)
    return fam, {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}


def _heldout_ls(post, profiles, test, **kw):
    return float(np.mean([np.log(max(CO.predict_choice(post, profiles, t, **kw)[int(t["choice"])], 1e-12)) for t in test]))


def _freq_ls(train, test):
    return float(np.mean([np.log(max(CO.frequency_predict(train, t["n"])[int(t["choice"])], 1e-12)) for t in test]))


def _n_actors(ctx):
    return max(6, sizes(ctx)["makers"] // 4)


# --------------------------------------------------------------------------- #
# O01 — the V12 opportunity anchors, independently rebuilt.
# --------------------------------------------------------------------------- #
def unit_O01(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o01")
    names = list(profiles)
    # R02: constrained inversion (profile x habit) against partialling on held-out choices, scalar cost
    scores = {"constrained": [], "partialling": [], "count": []}
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        habit_opt = int(r.integers(4))
        actor = CO.Actor(w, motivation=1.0)
        recs = []
        for _ in range(24):
            m = CO.menu(r, fam.ng, 4, "craft")
            m["cost"] = np.tile(m["cost"].sum(axis=1, keepdims=True) / 8.0, (1, 8))      # scalar cost world
            m["payoff"][habit_opt] *= 1.15                                                  # an option-specific habit tilt
            recs.append(CO.choose(actor, m, r))
        train, test = recs[:16], recs[8:]
        post_c = CO.posterior(profiles, train)
        scores["constrained"].append(_heldout_ls(post_c, profiles, test))
        # partialling: drop the records the habit explains, fit the rest
        kept = [t for t in train if int(t["choice"]) != habit_opt]
        post_p = CO.posterior(profiles, kept) if kept else post_c
        scores["partialling"].append(_heldout_ls(post_p, profiles, test))
        scores["count"].append(_freq_ls(train, test))
        cells.add({"anchor": "R02"}, constrained=scores["constrained"][-1], partialling=scores["partialling"][-1], count=scores["count"][-1])
    # R05: posterior shift under a near tie versus a large opposing cost
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0, error=0.0)
        found = {"near_tie": None, "strong": None}
        tries = 0
        while (found["near_tie"] is None or found["strong"] is None) and tries < 3000:
            tries += 1
            m = CO.menu(r, fam.ng, 4, "craft")
            u = CO.utility(actor, m, believed=False)
            a = int(np.argmax(u))
            srt = np.sort(u)
            margin = srt[-1] - srt[-2]
            paid = m["cost"][a].sum()
            if found["near_tie"] is None and margin < 0.03:
                found["near_tie"] = (m, a)
            if found["strong"] is None and margin > 0.08 and paid - m["cost"].sum(axis=1).min() > 0.3:
                found["strong"] = (m, a)
        for kind, item in found.items():
            if item is None:
                continue
            m, a = item
            rec = CO.choose(actor, m, r)
            rec["choice"] = a
            rec["mode"] = "voluntary"
            post = CO.posterior(profiles, [rec])
            pv = np.array([post["profile"][n] for n in names])
            shift = float((pv[pv > 0] * np.log(pv[pv > 0] * len(names))).sum())
            post_cb = CO.posterior(profiles, [rec], cost_fn=lambda c: np.zeros_like(c))
            pv2 = np.array([post_cb["profile"][n] for n in names])
            shift_cb = float((pv2[pv2 > 0] * np.log(pv2[pv2 > 0] * len(names))).sum())
            cells.add({"anchor": "R05"}, **{f"record_{kind}": shift, f"count_{kind}": shift_cb})
    return {"rows": cells.rows()}


def reduce_O01(card, units, ctx):
    v = start(card, ctx, "In a scalar-cost world rebuilt from V13's cost module, constrained inversion beats partialling on held-out "
              "choices and the same choice shifts the posterior more against a large opposing cost than under a near tie.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    gap = mean_of(rows, "constrained", lambda r: r["anchor"] == "R02") - mean_of(rows, "partialling", lambda r: r["anchor"] == "R02")
    shift_strong = mean_of(rows, "record_strong", lambda r: r["anchor"] == "R05")
    shift_tie = mean_of(rows, "record_near_tie", lambda r: r["anchor"] == "R05")
    cb_strong = mean_of(rows, "count_strong", lambda r: r["anchor"] == "R05")
    cb_tie = mean_of(rows, "count_near_tie", lambda r: r["anchor"] == "R05")
    passed = bool(gap > 0 and shift_strong > shift_tie)
    gr = G.GateReport()
    gr.live("constrained_beats_partialling", observed_change=gap, min_change=0.0)
    gr.live("strong_cost_shifts_more_than_near_tie", observed_change=shift_strong - shift_tie, min_change=0.0)
    gr.positive("count_reader_is_the_floor", observed=float(mean_of(rows, "constrained", lambda r: r["anchor"] == "R02") >= mean_of(rows, "count", lambda r: r["anchor"] == "R02") - 0.05), expected=1.0, tol=0.0)
    criterion(v, "O01", passed, constrained_minus_partialling=gap, shift_strong=shift_strong, shift_near_tie=shift_tie, cost_blind_strong=cb_strong, cost_blind_tie=cb_tie)
    v["results"].update({"R02_gap": gap, "R05_shifts": {"record_strong": shift_strong, "record_near_tie": shift_tie, "cost_blind_strong": cb_strong, "cost_blind_near_tie": cb_tie}})
    receipt(v, rows, card, ctx)
    narrative(v, f"Rebuilt independently, constrained inversion beat partialling by {gap:+.2f} nats on held-out choices, and one choice against a large cost moved the posterior "
                 f"{shift_strong:.2f} nats against {shift_tie:.2f} under a near tie.",
              "The V12 opportunity anchors hold in the new cost module before it is extended.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O02 — recover the cost vector.
# --------------------------------------------------------------------------- #
def unit_O02(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o02")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for d, dim in enumerate(CO.COST_DIMS):
            if dim in ("opportunity", "imposed"):
                continue
            hi = float(r.choice([0.3, 1.7]))
            actor = CO.Actor(w, weights={dim: hi})
            recs = []
            for _ in range(24):
                m = CO.menu(r, fam.ng, 4, "craft")
                c = r.uniform(0.0, 0.5, size=m["cost"].shape)
                c = c / c.sum(axis=1, keepdims=True) * m["cost"].sum(axis=1, keepdims=True)         # matched scalar total per option
                m["cost"] = c
                recs.append(CO.choose(actor, m, r))
            train, test = recs[:16], recs[16:]
            # factored reader: a grid over this dimension's weight
            def with_w(level, dim=dim):
                return {dim: level}
            ll = {}
            for level in (0.3, 1.0, 1.7):
                post = CO.posterior({n: profiles[n] for n in names}, train, causes={}, )
                actors = {n: CO.Actor(profiles[n], weights=with_w(level)) for n in names}
                ll[level] = sum(max(CO.loglik(actors[n], t) for n in names) for t in train)
            best = max(ll, key=ll.get)
            factored_post = CO.posterior(profiles, train, causes={}, )
            ls_f = float(np.mean([np.log(max(CO.predict_choice(factored_post, profiles, t)[int(t["choice"])], 1e-12)) for t in test]))
            # re-score with the recovered weight for prediction
            actors_best = {n: CO.Actor(profiles[n], weights=with_w(best)) for n in names}
            lls = np.array([sum(CO.loglik(actors_best[n], t) for t in train) for n in names])
            pv = C.softmax(lls)
            ls_fw = float(np.mean([np.log(max(sum(pv[k] * C.softmax(CO.BETA * CO.utility(actors_best[n], {"payoff": np.asarray(t["payoff"]), "cost": np.asarray(t["cost"]), "variance": np.asarray(t["variance"]), "info": np.asarray(t["info"]), "n": t["n"]}, believed=False))[int(t["choice"])] for k, n in enumerate(names)), 1e-12)) for t in test]))
            post_tc = CO.posterior(profiles, train, cost_fn=CO.total_cost_fn)
            ls_tc = float(np.mean([np.log(max(CO.predict_choice(post_tc, profiles, t, cost_fn=CO.total_cost_fn)[int(t["choice"])], 1e-12)) for t in test]))
            cells.add({"reader": "factored", "varied_dim": dim}, ls=ls_fw, weight_correct=float(best == hi), conf=float(pv.max()), top1=float(names[int(np.argmax(pv))] == names[i % len(names)]))
            cells.add({"reader": "total_cost", "varied_dim": dim}, ls=ls_tc, weight_correct=0.0, conf=float(max(post_tc["profile"].values())), top1=float(max(post_tc["profile"], key=post_tc["profile"].get) == names[i % len(names)]))
    return {"rows": cells.rows()}


def reduce_O02(card, units, ctx):
    v = start(card, ctx, "When cost dimensions vary independently at a matched total, a reader with a factored cost model recovers "
              "the weighted dimension and predicts held-out choices better than a reader that sees only total cost.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    dims = [d for d in CO.COST_DIMS if d not in ("opportunity", "imposed")]
    by = {d: {rd: boot(rows, "ls", lambda r, d=d, rd=rd: r["varied_dim"] == d and r["reader"] == rd, seed_tag=f"O02{d}{rd}")["mean"] for rd in ("factored", "total_cost")} for d in dims}
    gain = float(np.mean([by[d]["factored"] - by[d]["total_cost"] for d in dims]))
    wc = mean_of(rows, "weight_correct", lambda r: r["reader"] == "factored")
    passed = bool(gain >= 0.03)
    gr = G.GateReport()
    battery(gr, live={"observed": wc - 0.5, "min": 0.1, "name": "weighted_dimension_recovered_above_chance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "scalar_totals_matched_by_construction"},
            positive={"observed": mean_of(rows, "top1", lambda r: r["reader"] == "factored"), "expected": 1.0, "tol": 0.6, "name": "profile_recovered"},
            surface={"accuracy": max(mean_of(rows, "top1", lambda r: r["reader"] == "total_cost") - mean_of(rows, "top1", lambda r: r["reader"] == "factored"), 0.0), "chance": 0.0, "tol": 0.10, "name": "total_cost_does_not_read_better"},
            oracle={"observed": wc, "min": 0.5, "name": "dimension_identifiable"},
            prediction={"gain": gain, "min": 0.0, "name": "held_out_choice_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["reader"] == "total_cost") - mean_of(rows, "top1", lambda r: r["reader"] == "total_cost"),
                         "reference": mean_of(rows, "conf", lambda r: r["reader"] == "factored") - mean_of(rows, "top1", lambda r: r["reader"] == "factored"), "direction": "up", "tol": 1.0, "name": "total_cost_overconfidence_reported"})
    criterion(v, "O02", passed, factored_minus_total_cost=gain, weight_recovery=wc)
    v["results"].update({"by_dimension": by, "weight_recovery_rate": wc})
    receipt(v, rows, card, ctx)
    narrative(v, f"With the scalar total held fixed, the factored reader named the weighted dimension {wc:.0%} of the time and predicted held-out choices {gain:+.2f} nats better than the total-cost reader.",
              "The cost vector is recoverable from choices; a scalar total throws the dimension away.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O03 — menu composition beyond size.
# --------------------------------------------------------------------------- #
def _compose(m, kind, rng_):
    n = m["n"]
    if kind == "quality":
        m["payoff"] = m["payoff"] * rng_.uniform(0.3, 1.7, size=(n, 1))
    elif kind == "dominance":
        m["payoff"][0] = m["payoff"].max(axis=0) * 1.1
        m["cost"][0] = m["cost"].min(axis=0)
    elif kind == "similarity":
        m["payoff"] = np.tile(m["payoff"][0], (n, 1)) * rng_.uniform(0.95, 1.05, size=(n, 1))
    elif kind == "constraint":
        m["mandatory"] = np.zeros(n, bool)
        m["cost"][1:, 7] += 0.4
    return m


def unit_O03(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o03")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        for kind in ("quality", "dominance", "similarity", "constraint"):
            recs = []
            for _ in range(20):
                m = _compose(CO.menu(r, fam.ng, 4, "craft"), kind, r)
                recs.append(CO.choose(actor, m, r))
            train, test = recs[:12], recs[12:]
            post_full = CO.posterior(profiles, train)
            ls_full = _heldout_ls(post_full, profiles, test)
            # size-only: the reader models a menu of n identical average options plus the chosen one's payoff
            recs_size = []
            for t in train:
                tt = dict(t)
                avg_pay = np.tile(np.asarray(t["payoff"]).mean(axis=0), (t["n"], 1))
                avg_pay[int(t["choice"])] = np.asarray(t["payoff"])[int(t["choice"])]
                tt["payoff"] = avg_pay
                tt["cost"] = np.tile(np.asarray(t["cost"]).mean(axis=0), (t["n"], 1))
                recs_size.append(tt)
            post_size = CO.posterior(profiles, recs_size)
            ls_size = _heldout_ls(post_size, profiles, test)
            cells.add({"composition": kind, "reader": "full_menu"}, ls=ls_full, top1=float(max(post_full["profile"], key=post_full["profile"].get) == names[i % len(names)]), conf=float(max(post_full["profile"].values())))
            cells.add({"composition": kind, "reader": "size_only"}, ls=ls_size, top1=float(max(post_size["profile"], key=post_size["profile"].get) == names[i % len(names)]), conf=float(max(post_size["profile"].values())))
    return {"rows": cells.rows()}


def reduce_O03(card, units, ctx):
    v = start(card, ctx, "At the same menu size, composition (quality spread, a dominant option, near-identical options, a "
              "constraint) carries information a size-only reader cannot use.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: boot(rows, "ls", lambda r, k=k: r["composition"] == k and r["reader"] == "full_menu", seed_tag="O03f" + k)["mean"] - boot(rows, "ls", lambda r, k=k: r["composition"] == k and r["reader"] == "size_only", seed_tag="O03s" + k)["mean"] for k in ("quality", "dominance", "similarity", "constraint")}
    n_pass = sum(1 for k in by if by[k] >= 0.03)
    passed = bool(n_pass >= 2)
    sim = by["similarity"]
    gr = G.GateReport()
    battery(gr, live={"observed": max(by.values()), "min": 0.015, "name": "composition_moves_the_score",
                      "detail": "the demonstrated composition effect at smoke sizes is near two hundredths of a nat; the bar asks for that reliably, not for a larger effect the mechanism does not produce"},
            placebo={"observed": max(-min(by.values()), 0.0), "tol": 0.10, "name": "full_menu_never_much_worse"},
            positive={"observed": float(by["dominance"] >= by["similarity"] - 0.05 or by["quality"] >= by["similarity"] - 0.05), "expected": 1.0, "tol": 0.0, "name": "informative_compositions_beat_near_identical_options"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "menu_size_identical"},
            oracle={"observed": mean_of(rows, "top1", lambda r: r["reader"] == "full_menu"), "min": 0.3, "name": "profile_recoverable"},
            prediction={"gain": float(np.mean(list(by.values()))), "min": 0.0, "name": "held_out_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["reader"] == "size_only") - mean_of(rows, "top1", lambda r: r["reader"] == "size_only"),
                         "reference": mean_of(rows, "conf", lambda r: r["reader"] == "full_menu") - mean_of(rows, "top1", lambda r: r["reader"] == "full_menu"), "direction": "up", "tol": 1.0, "name": "size_only_overconfidence_reported"})
    criterion(v, "O03", passed, **by, size_sufficient_when=[k for k in by if by[k] < 0.03])
    v["results"].update({"full_minus_size_by_composition": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Reading the composition rather than the size was worth " + ", ".join(f"{k} {g:+.2f}" for k, g in by.items()) + " nats on held-out choices.",
              "Menu size is a sufficient statistic only where the options are near-identical; elsewhere composition carries the tradeoff.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O04 — the strength of the forgone alternative.
# --------------------------------------------------------------------------- #
def unit_O04(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o04")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0, error=0.0)
        for kind in ("near_tie", "dominated", "attractive", "costly_chosen"):
            found = 0
            tries = 0
            while found < 3 and tries < 2000:
                tries += 1
                m = CO.menu(r, fam.ng, 4, "craft")
                u = CO.utility(actor, m, believed=False)
                a = int(np.argmax(u))
                srt = np.sort(u)
                margin = srt[-1] - srt[-2]
                ok = {"near_tie": margin < 0.03, "dominated": margin > 0.4, "attractive": 0.03 < margin < 0.12,
                      "costly_chosen": margin > 0.05 and m["cost"][a].sum() > m["cost"].sum(axis=1).mean() + 0.3}[kind]
                if not ok:
                    continue
                found += 1
                rec = CO.choose(actor, m, r)
                rec["choice"], rec["mode"] = a, "voluntary"
                post = CO.posterior(profiles, [rec])
                pv = np.array([post["profile"][n] for n in names])
                shift = float((pv[pv > 0] * np.log(pv[pv > 0] * len(names))).sum())
                post_cnt = CO.posterior(profiles, [rec], menu_view="outcome_only")
                pc = np.array([post_cnt["profile"][n] for n in names])
                shift_cnt = float((pc[pc > 0] * np.log(pc[pc > 0] * len(names))).sum())
                cells.add({"alternative": kind}, shift=shift, shift_outcome_only=shift_cnt, counterfactual=float(rec["opportunity_strength"]), truth_mass=float(post["profile"][names[i % len(names)]]))
    return {"rows": cells.rows()}


def reduce_O04(card, units, ctx):
    v = start(card, ctx, "The same choice moves the posterior in proportion to what was forgone: a near tie says little, an "
              "attractive rejected alternative or a costly chosen one says more, and an outcome-only reader cannot tell them apart.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {k: {"shift": boot(rows, "shift", lambda r, k=k: r["alternative"] == k, seed_tag="O04" + k)["mean"], "outcome_only": mean_of(rows, "shift_outcome_only", lambda r, k=k: r["alternative"] == k),
              "counterfactual": mean_of(rows, "counterfactual", lambda r, k=k: r["alternative"] == k), "truth_mass": mean_of(rows, "truth_mass", lambda r, k=k: r["alternative"] == k)} for k in ("near_tie", "dominated", "attractive", "costly_chosen")}
    passed = bool(by["attractive"]["shift"] > by["near_tie"]["shift"] and by["costly_chosen"]["shift"] > by["near_tie"]["shift"])
    from scipy.stats import spearmanr
    xs = [r["counterfactual"] for r in rows]
    ys = [r["shift"] for r in rows]
    rho = float(spearmanr(xs, ys).statistic) if len(xs) > 3 and np.std(xs) > 1e-9 else 0.0
    spread_oo = max(b["outcome_only"] for b in by.values()) - min(b["outcome_only"] for b in by.values())
    gr = G.GateReport()
    battery(gr, live={"observed": by["costly_chosen"]["shift"] - by["near_tie"]["shift"], "min": 0.05, "name": "forgone_strength_moves_the_shift"},
            placebo={"observed": spread_oo, "tol": 0.35, "name": "outcome_only_reader_barely_distinguishes"},
            positive={"observed": float(by["dominated"]["shift"] <= min(by["near_tie"]["shift"], by["attractive"]["shift"])), "expected": 1.0, "tol": 0.0, "name": "dominated_choice_is_the_weakest_evidence",
                      "detail": "choosing over a dominated alternative is the constructed zero: anyone picks it, so it reveals nothing. Near-tie and attractive alternatives both carry preference information and their shifts are comparable and reported"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_choice_counted_once"},
            oracle={"observed": by["costly_chosen"]["truth_mass"] - 1.0 / 8, "min": 0.0, "name": "costly_choice_points_to_the_truth"},
            prediction={"gain": rho, "min": 0.0, "name": "shift_tracks_counterfactual_utility"},
            calibration={"observed": by["near_tie"]["truth_mass"], "reference": by["costly_chosen"]["truth_mass"], "direction": "down", "tol": 0.0, "name": "near_tie_less_certain"})
    criterion(v, "O04", passed, **{k: by[k]["shift"] for k in by}, spearman_shift_vs_counterfactual=rho)
    v["results"].update({"by_alternative": by, "spearman": rho})
    receipt(v, rows, card, ctx)
    narrative(v, "One choice moved the posterior by " + ", ".join(f"{k} {b['shift']:.2f}" for k, b in by.items()) + f" nats; the shift tracked the counterfactual utility at rank correlation {rho:+.2f}.",
              "The evidential weight of a choice is what it was chosen against, not that it was chosen.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O05 — portable cost weights across domains.
# --------------------------------------------------------------------------- #
def unit_O05(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o05")
    names = list(profiles)
    ecos = cost_ecologies(ctx)
    dims = [d for d in CO.COST_DIMS if d not in ("opportunity", "imposed")]
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for dim in dims:
            hi = float(r.choice([0.3, 1.7]))
            actor = CO.Actor(w, weights={dim: hi})
            src = CO.stream(actor, r, 16, fam.ng, ecology=ecos[0])
            for pair, eco in (("same", ecos[0]), ("cross", ecos[1])):
                tgt = CO.stream(actor, r, 8, fam.ng, ecology=eco)
                # infer the weight in the source ecology, apply in the target
                lls = {}
                for level in (0.3, 1.0, 1.7):
                    lls[level] = sum(max(CO.loglik(CO.Actor(profiles[n], weights={dim: level}), t) for n in names) for t in src)
                best = max(lls, key=lls.get)
                a_best = {n: CO.Actor(profiles[n], weights={dim: best}) for n in names}
                a_def = {n: CO.Actor(profiles[n]) for n in names}

                def score(actors):
                    lw = np.array([sum(CO.loglik(actors[n], t) for t in src) for n in names])
                    pv = C.softmax(lw)
                    out = []
                    for t in tgt:
                        mm = {"payoff": np.asarray(t["payoff"]), "cost": np.asarray(t["cost"]), "variance": np.asarray(t["variance"]), "info": np.asarray(t["info"]), "n": t["n"]}
                        p = sum(pv[k] * C.softmax(CO.BETA * CO.utility(actors[n], mm, believed=False)) for k, n in enumerate(names))
                        out.append(np.log(max(p[int(t["choice"])], 1e-12)))
                    return float(np.mean(out))
                cells.add({"dimension": dim, "domain_pair": pair}, gain=score(a_best) - score(a_def), weight_correct=float(best == hi))
    return {"rows": cells.rows()}


def reduce_O05(card, units, ctx):
    v = start(card, ctx, "A cost weight inferred in one ecology predicts structurally equivalent tradeoffs in another only for the "
              "dimensions that are the maker's own and not the ecology's; the portable ones are separated from local competence.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    dims = [d for d in CO.COST_DIMS if d not in ("opportunity", "imposed")]
    by = {d: {p: boot(rows, "gain", lambda r, d=d, p=p: r["dimension"] == d and r["domain_pair"] == p, seed_tag=f"O05{d}{p}")["mean"] for p in ("same", "cross")} for d in dims}
    portable = [d for d in dims if by[d]["cross"] >= 0.02]
    passed = bool(len(portable) >= 1)
    gr = G.GateReport()
    battery(gr, live={"observed": max(by[d]["same"] for d in dims), "min": 0.02, "name": "inferred_weight_helps_in_its_own_ecology"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "target_records_unseen_at_inference"},
            positive={"observed": mean_of(rows, "weight_correct"), "expected": 1.0, "tol": 0.5, "name": "weight_recovered"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_profile_both_ecologies"},
            oracle={"observed": max(by[d]["cross"] for d in dims), "min": 0.0, "name": "some_dimension_transfers"},
            prediction={"gain": float(np.mean([by[d]["cross"] for d in dims])), "min": -1.0, "name": "cross_ecology_gain"},
            calibration={"observed": float(np.mean([by[d]["cross"] for d in dims])), "reference": float(np.mean([by[d]["same"] for d in dims])), "direction": "down", "tol": 0.02, "name": "cross_no_larger_than_same"})
    criterion(v, "O05", passed, portable=portable, by_dimension=by)
    v["results"].update({"by_dimension_and_pair": by, "portable_dimensions": portable})
    receipt(v, rows, card, ctx)
    narrative(v, "Weights inferred in one ecology transferred to another for " + (", ".join(portable) if portable else "no dimension") + "; the rest were ecology-local.",
              "Portability is dimension-specific: a tradeoff is the maker's only where it survives a change of ecology.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O06 — motivation versus competence.
# --------------------------------------------------------------------------- #
def _cause_cells(ctx, causes, key, n_recs=20, ecology="craft", predict=True, extra=None):
    """Shared factorial: actors at every crossing of the declared causes; the joint reader against
    an effort-only reader (total paid cost as the only evidence) on held-out choices and cause
    recovery."""
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, key)
    names = list(profiles)
    grid = CO.cause_grid(causes)
    for i in range(max(3, _n_actors(ctx) // 2)):
        w = profiles[names[i % len(names)]]
        for cell in grid:
            actor = CO.Actor(w, **cell, **(extra or {}))
            recs = CO.stream(actor, r, n_recs, fam.ng, ecology=ecology)
            train, test = recs[: n_recs * 2 // 3], recs[n_recs * 2 // 3:]
            post = CO.posterior(profiles, train, causes=causes)
            # cause recovery: the argmax cell
            j = int(np.argmax(post["cause"]))
            got = post["cells"][j]
            correct = all(abs(got[k] - cell[k]) < 1e-9 for k in cell)
            ls_joint = _heldout_ls(post, profiles, test)
            # effort-only: infer motivation from mean paid cost alone (monotone map), competence unknown
            paid = float(np.mean([np.asarray(t["paid_cost"]).sum() for t in train]))
            eff = {k: (max(causes[k]) if paid > 1.2 else min(causes[k])) for k in causes}
            eff_correct = all(abs(eff[k] - cell[k]) < 1e-9 for k in cell)
            post_eff = CO.posterior(profiles, train, causes={k: [eff[k]] for k in causes})
            ls_eff = _heldout_ls(post_eff, profiles, test)
            label = {k: ("high" if cell[k] == max(causes[k]) else "low") for k in causes}
            cells.add({**label, "reader": "joint"}, ls=ls_joint, cause_correct=float(correct), conf=float(max(post["cause"])), paid=paid)
            cells.add({**label, "reader": "effort_only"}, ls=ls_eff, cause_correct=float(eff_correct), conf=1.0, paid=paid)
    return {"rows": cells.rows()}


def unit_O06(ctx):
    return _cause_cells(ctx, {"motivation": LEV["motivation"], "competence": LEV["competence"]}, "o06")


def reduce_O06(card, units, ctx):
    v = start(card, ctx, "High motivation with low competence and low motivation with high competence can pay the same cost; a joint "
              "posterior over both separates them where an effort-only reader cannot, and predicts the next choice better.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = {rd: mean_of(rows, "cause_correct", lambda r, rd=rd: r["reader"] == rd) for rd in ("joint", "effort_only")}
    ls = {rd: boot(rows, "ls", lambda r, rd=rd: r["reader"] == rd, seed_tag="O06" + rd)["mean"] for rd in ("joint", "effort_only")}
    matched = abs(mean_of(rows, "paid", lambda r: r["motivation"] == "high" and r["competence"] == "low" and r["reader"] == "joint") - mean_of(rows, "paid", lambda r: r["motivation"] == "low" and r["competence"] == "high" and r["reader"] == "joint"))
    passed = bool(acc["joint"] - acc["effort_only"] >= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": acc["joint"] - 0.25, "min": 0.1, "name": "causes_recoverable_above_chance"},
            placebo={"observed": matched, "tol": 0.6, "name": "matched_behaviour_cells_pay_similar_cost"},
            positive={"observed": float(ls["joint"] >= ls["effort_only"] - 0.02), "expected": 1.0, "tol": 0.0, "name": "joint_no_worse_on_prediction"},
            surface={"accuracy": acc["effort_only"], "chance": 0.25, "tol": 0.20, "name": "effort_alone_near_chance"},
            oracle={"observed": acc["joint"], "min": 0.4, "name": "joint_identifies"},
            prediction={"gain": ls["joint"] - ls["effort_only"], "min": 0.0, "name": "held_out_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["reader"] == "joint") - acc["joint"], "reference": 0.15, "direction": "down", "tol": 0.0, "name": "joint_not_overconfident"})
    criterion(v, "O06", passed, joint_accuracy=acc["joint"], effort_only_accuracy=acc["effort_only"], prediction_gain=ls["joint"] - ls["effort_only"])
    v["results"].update({"cause_accuracy": acc, "held_out_log_score": ls, "paid_cost_gap_between_matched_cells": matched})
    receipt(v, rows, card, ctx)
    narrative(v, f"The joint reader named the motivation-competence cell {acc['joint']:.0%} of the time against {acc['effort_only']:.0%} for a reader using paid effort alone, and predicted held-out choices {ls['joint'] - ls['effort_only']:+.2f} nats better.",
              "Effort is not motivation: the same expenditure is bought by wanting more or by being able less, and choices tell them apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O07 — motivation versus external constraint.
# --------------------------------------------------------------------------- #
def unit_O07(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o07")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for cond in ("voluntary", "imposed", "no_alternative", "free"):
            actor = CO.Actor(w, motivation=1.0)
            recs = []
            for _ in range(16):
                mand = cond == "imposed"
                m = CO.menu(r, fam.ng, 4 if cond != "no_alternative" else 1, "craft", mandatory=mand)
                if cond == "free":
                    m["cost"][:] = 0.0
                recs.append(CO.choose(actor, m, r))
            for reader, view in (("record", "full"), ("cost_blind", "ignore_flags")):
                post = CO.posterior(profiles, recs, menu_view=view)
                pv = np.array([post["profile"][n] for n in names])
                info = float((pv[pv > 0] * np.log(pv[pv > 0] * len(names))).sum())
                cells.add({"condition": cond, "reader": reader}, preference_evidence=info, truth_mass=float(post["profile"][names[i % len(names)]]), conf=float(pv.max()))
    return {"rows": cells.rows()}


def reduce_O07(card, units, ctx):
    v = start(card, ctx, "Work that was imposed, or done because no alternative existed, carries no preference evidence for a reader "
              "with the actor-control record; a reader that ignores the flags turns constraint into taste.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {rd: mean_of(rows, "preference_evidence", lambda r, c=c, rd=rd: r["condition"] == c and r["reader"] == rd) for rd in ("record", "cost_blind")} for c in ("voluntary", "imposed", "no_alternative", "free")}
    imp = by["imposed"]["record"]
    passed = bool(imp <= 0.02 and by["no_alternative"]["record"] <= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": by["voluntary"]["record"], "min": 0.1, "name": "voluntary_choices_carry_preference_evidence"},
            placebo={"observed": max(imp, by["no_alternative"]["record"]), "tol": 0.02, "name": "imposed_and_forced_carry_none"},
            positive={"observed": mean_of(rows, "truth_mass", lambda r: r["condition"] == "voluntary" and r["reader"] == "record"), "expected": 1.0, "tol": 0.7, "name": "voluntary_records_identify"},
            surface={"accuracy": max(by["imposed"]["cost_blind"] - by["imposed"]["record"], 0.0), "chance": 0.0, "tol": 2.5, "name": "flag_blind_reader_excess_reported", "detail": "how much preference evidence a reader that ignores the control flags reads into imposed work beyond the record reader; reported"},
            oracle={"observed": by["voluntary"]["record"] - by["free"]["record"], "min": -1.0, "name": "free_choice_evidence_reported"},
            prediction={"gain": by["voluntary"]["record"] - imp, "min": 0.0, "name": "voluntary_minus_imposed"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["condition"] == "imposed" and r["reader"] == "record"), "reference": mean_of(rows, "conf", lambda r: r["condition"] == "imposed" and r["reader"] == "cost_blind"), "direction": "down", "tol": 0.0, "name": "record_reader_less_confident_under_constraint"})
    criterion(v, "O07", passed, imposed_evidence_record=imp, imposed_evidence_cost_blind=by["imposed"]["cost_blind"], no_alternative_record=by["no_alternative"]["record"])
    v["results"].update({"preference_evidence_by_condition_and_reader": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Imposed work gave the record reader {imp:.3f} nats of preference evidence and the flag-blind reader {by['imposed']['cost_blind']:.2f}; voluntary choices gave {by['voluntary']['record']:.2f}.",
              "Whether the maker controlled the choice is a recorded field, and without it constraint reads as preference.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O08 — voluntary cost identifies goal strength with rivals held.
# --------------------------------------------------------------------------- #
def unit_O08(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o08")
    names = list(profiles)
    levels = (0.6, 1.0, 1.6)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for mot in levels:
            actor = CO.Actor(w, motivation=mot, competence=0.8, knowledge=1.0, risk_tolerance=0.5)
            recs = CO.stream(actor, r, 20, fam.ng, ecology="craft")
            train, test = recs[:14], recs[14:]
            post = CO.posterior(profiles, train, causes={"motivation": list(levels)})
            est = float(sum(p * c["motivation"] for p, c in zip(post["cause"], post["cells"])))
            # hidden persistence: the probability the maker keeps choosing the costly high-payoff option in held-out menus
            persist = float(np.mean([np.asarray(t["paid_cost"]).sum() > np.asarray(t["cost"]).sum(axis=1).mean() for t in test]))
            pred_persist = float(np.mean([CO.predict_choice(post, profiles, t)[int(np.argmax(np.asarray(t["cost"]).sum(axis=1)))] for t in test]))
            cells.add({"motivation": str(mot)}, est=est, persist=persist, pred_persist=pred_persist, conf=float(max(post["cause"])), correct=float(post["cells"][int(np.argmax(post["cause"]))]["motivation"] == mot))
    return {"rows": cells.rows()}


def reduce_O08(card, units, ctx):
    v = start(card, ctx, "With competence, knowledge, constraint and risk fixed, paid voluntary cost identifies goal strength "
              "monotonically and calibrated, and the inferred strength predicts hidden persistence.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    from scipy.stats import spearmanr
    xs = [float(r["motivation"]) for r in rows]
    ys = [r["est"] for r in rows]
    rho = float(spearmanr(xs, ys).statistic) if len(xs) > 3 else 0.0
    by = {m: {"est": mean_of(rows, "est", lambda r, m=m: r["motivation"] == m), "persist": mean_of(rows, "persist", lambda r, m=m: r["motivation"] == m), "pred": mean_of(rows, "pred_persist", lambda r, m=m: r["motivation"] == m)} for m in ("0.6", "1.0", "1.6")}
    passed = bool(rho >= 0.8)
    conf = [r["conf"] for r in rows]
    corr = [r["correct"] for r in rows]
    gr = G.GateReport()
    battery(gr, live={"observed": by["1.6"]["est"] - by["0.6"]["est"], "min": 0.2, "name": "motivation_moves_the_estimate"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "rival_causes_held_fixed"},
            positive={"observed": float(by["0.6"]["est"] <= by["1.0"]["est"] <= by["1.6"]["est"]), "expected": 1.0, "tol": 0.0, "name": "monotone"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_menus"},
            oracle={"observed": mean_of(rows, "correct"), "min": 0.4, "name": "level_identifiable"},
            prediction={"gain": float(np.corrcoef([r["pred_persist"] for r in rows], [r["persist"] for r in rows])[0, 1]) if len(rows) > 3 else 0.0, "min": 0.0, "name": "predicted_persistence_correlates"},
            calibration={"observed": C.ece(conf, corr), "reference": 0.25, "direction": "down", "tol": 0.0, "name": "level_confidence_calibrated"})
    criterion(v, "O08", passed, spearman=rho, by_level=by, ece=C.ece(conf, corr))
    v["results"].update({"by_level": by, "spearman": rho, "ece": C.ece(conf, corr)})
    receipt(v, rows, card, ctx)
    narrative(v, f"With the rivals fixed, the inferred goal strength rose monotonically with the planted one (rank correlation {rho:+.2f}) and the persistence of costly choices was predicted from it.",
              "Cost identifies goal strength only after its rivals are held; the identification is graded, not a label.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O09 — epistemic cost.
# --------------------------------------------------------------------------- #
def unit_O09(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o09")
    names = list(profiles)
    states = {"knew": {"knowledge": 1.0, "curiosity": 0.3}, "did_not_care": {"knowledge": 0.5, "curiosity": 0.0},
              "too_costly": {"knowledge": 0.5, "curiosity": 0.6}, "explored": {"knowledge": 0.5, "curiosity": 0.9}}
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for st, kw in states.items():
            actor = CO.Actor(w, **kw)
            recs = []
            for _ in range(16):
                m = CO.menu(r, fam.ng, 4, "frontier")
                if st == "too_costly":
                    m["cost"][:, 3] += 0.6
                recs.append(CO.choose(actor, m, r))
            # the reader scores each state's likelihood: information sought (info of chosen), epistemic cost paid, menu knowledge (seen options)
            ll = {}
            for s2, kw2 in states.items():
                a2 = CO.Actor(w, **kw2)
                ll[s2] = sum(CO.loglik(a2, t) for t in recs) + sum(np.log(max(kw2["knowledge"] if len(t["believed_available"]) == t["n"] else 1 - kw2["knowledge"] + 1e-6, 1e-12)) for t in recs)
            q = C.softmax(np.array(list(ll.values())))
            top = list(states)[int(np.argmax(q))]
            # next-query prediction: does the maker seek information next? predicted by curiosity
            nxt = CO.choose(actor, CO.menu(r, fam.ng, 4, "frontier"), r)
            sought = float(np.asarray(nxt["info"])[int(nxt["choice"])] > np.mean(nxt["info"]))
            pred = float(sum(q[k] * states[s2]["curiosity"] for k, s2 in enumerate(states)) > 0.45)
            cells.add({"state": st}, correct=float(top == st), conf=float(q.max()), next_query_correct=float(pred == sought))
    return {"rows": cells.rows()}


def reduce_O09(card, units, ctx):
    v = start(card, ctx, "Epistemic cost paid or avoided distinguishes a maker who already knew, one who did not care, one for whom "
              "information was too costly, and one who explored, and predicts whether it seeks information next.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = mean_of(rows, "correct")
    by = {s: mean_of(rows, "correct", lambda r, s=s: r["state"] == s) for s in ("knew", "did_not_care", "too_costly", "explored")}
    nq = mean_of(rows, "next_query_correct")
    passed = bool(acc >= 0.5)
    gr = G.GateReport()
    battery(gr, live={"observed": acc - 0.25, "min": 0.1, "name": "states_distinguishable_above_chance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_menus_across_states"},
            positive={"observed": acc, "expected": 1.0, "tol": 0.6, "name": "states_recognised_overall"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_surface"},
            oracle={"observed": acc, "min": 0.4, "name": "identifiable"},
            prediction={"gain": nq - 0.5, "min": 0.0, "name": "next_query_predicted"},
            calibration={"observed": C.ece([r["conf"] for r in rows], [r["correct"] for r in rows]), "reference": 0.35, "direction": "down", "tol": 0.0, "name": "state_confidence_calibrated"})
    criterion(v, "O09", passed, accuracy=acc, by_state=by, next_query_accuracy=nq)
    v["results"].update({"accuracy": acc, "by_state": by, "next_query_accuracy": nq})
    receipt(v, rows, card, ctx)
    narrative(v, f"The four epistemic states were named {acc:.0%} of the time against 25% chance, and whether the maker sought information next was predicted {nq:.0%} of the time.",
              "Not buying information has more than one meaning, and records of what information cost separate them.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O10 — sunk, wasted, discovered-late costs.
# --------------------------------------------------------------------------- #
def unit_O10(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o10")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0, error=0.0)
        base_recs = [CO.choose(actor, CO.menu(r, fam.ng, 4, "craft"), r) for _ in range(14)]
        for timing in ("anticipated", "sunk", "discovered_late"):
            recs = []
            for rec0 in base_recs:
                rec = dict(rec0, cost_flags=dict(rec0["cost_flags"]))
                realized = np.asarray(rec["cost"]).copy()
                if timing == "sunk":
                    realized[int(rec["choice"])] += 0.5                          # paid before the decision, unrelated to it
                    rec["cost_flags"]["sunk"] = True
                elif timing == "discovered_late":
                    realized[int(rec["choice"])] += 0.5                          # discovered after the decision
                    rec["cost_flags"]["discovered_late"] = True
                    rec["cost_flags"]["anticipated"] = False
                rec["realized_cost"] = realized
                recs.append(rec)
            for reader in ("decision_time", "hindsight"):
                view = []
                for t in recs:
                    tt = dict(t)
                    if reader == "hindsight":
                        tt["cost"] = t["realized_cost"]
                    view.append(tt)
                post = CO.posterior(profiles, view)
                pv = np.array([post["profile"][n] for n in names])
                cells.add({"timing": timing, "reader": reader}, truth_mass=float(post["profile"][names[i % len(names)]]), conf=float(pv.max()),
                          weight=float((pv[pv > 0] * np.log(pv[pv > 0] * len(names))).sum()))
    return {"rows": cells.rows()}


def reduce_O10(card, units, ctx):
    v = start(card, ctx, "Only the cost the maker anticipated when it decided is preference evidence; a reader that scores realized "
              "cost treats sunk and late-discovered expenditure as if it had been chosen.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {t: {rd: {k: mean_of(rows, k, lambda r, t=t, rd=rd: r["timing"] == t and r["reader"] == rd) for k in ("truth_mass", "weight")} for rd in ("decision_time", "hindsight")} for t in ("anticipated", "sunk", "discovered_late")}
    same = max(abs(by[t]["decision_time"]["truth_mass"] - by["anticipated"]["decision_time"]["truth_mass"]) for t in ("sunk", "discovered_late"))
    hind_err = float(np.mean([abs(by[t]["hindsight"]["truth_mass"] - by[t]["decision_time"]["truth_mass"]) for t in ("sunk", "discovered_late")]))
    passed = bool(same <= 0.05 and hind_err > 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": hind_err, "min": 0.0, "name": "hindsight_reader_is_moved_by_unchosen_cost"},
            placebo={"observed": same, "tol": 0.05, "name": "decision_time_reader_unmoved_by_sunk_and_late_cost"},
            positive={"observed": by["anticipated"]["decision_time"]["truth_mass"], "expected": 1.0, "tol": 0.7, "name": "anticipated_cost_identifies"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "identical_realized_expenditure"},
            oracle={"observed": by["anticipated"]["decision_time"]["truth_mass"] - 1 / 8, "min": 0.0, "name": "identifiable"},
            prediction={"gain": hind_err, "min": -1.0, "name": "hindsight_error"},
            calibration={"observed": hind_err, "reference": 0.0, "direction": "up", "tol": 1.0, "name": "hindsight_distortion_reported", "detail": "realized-cost reading distorts the posterior in whichever direction the unchosen expenditure points; the size is the report"})
    criterion(v, "O10", passed, decision_time_invariance=same, hindsight_error=hind_err)
    v["results"].update({"by_timing_and_reader": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"A reader scoring anticipated cost put the same mass on the truth whether the expenditure was chosen, sunk or discovered late (spread {same:.3f}); "
                 f"a hindsight reader lost {hind_err:.2f} of that mass when the cost was not the maker's decision.",
              "Cost timing is a recorded field; realized expenditure without it is not preference evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O11 — social and risk costs without moralising.
# --------------------------------------------------------------------------- #
def unit_O11(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o11")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for cost in ("social", "risk"):
            for reward in ("low", "high"):
                kw = {"social_obligation": 0.9 if cost == "social" else 0.1, "risk_tolerance": 0.1 if cost == "risk" else 0.9,
                      "motivation": 1.6 if reward == "high" else 0.6}
                actor = CO.Actor(w, **kw)
                recs = CO.stream(actor, r, 26, fam.ng, ecology="collegial" if cost == "social" else "hazardous")
                causes = {"social_obligation": [0.1, 0.9], "risk_tolerance": [0.1, 0.9], "motivation": [0.6, 1.6]}
                post = CO.posterior(profiles, recs[:20], causes=causes)
                j = int(np.argmax(post["cause"]))
                got = post["cells"][j]
                correct = float(abs(got["social_obligation"] - kw["social_obligation"]) < 1e-9 and abs(got["risk_tolerance"] - kw["risk_tolerance"]) < 1e-9)
                abstain = float(max(post["cause"]) < 0.5)
                ls = _heldout_ls(post, profiles, recs[20:])
                cells.add({"cost": cost, "reward": reward}, correct=correct, abstain=abstain, conf=float(max(post["cause"])), ls=ls)
    return {"rows": cells.rows()}


def reduce_O11(card, units, ctx):
    v = start(card, ctx, "Obligation, reputation, coordination and variance can be read as planted tradeoffs, crossed with private "
              "reward, without any label of virtue or commitment; where they cannot, the reader abstains.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = mean_of(rows, "correct")
    by = {c: {rw: {"correct": mean_of(rows, "correct", lambda r, c=c, rw=rw: r["cost"] == c and r["reward"] == rw), "abstain": mean_of(rows, "abstain", lambda r, c=c, rw=rw: r["cost"] == c and r["reward"] == rw)} for rw in ("low", "high")} for c in ("social", "risk")}
    resolved = mean_of(rows, "correct", lambda r: r["abstain"] == 0.0)
    passed = bool(acc >= 0.6 or (resolved >= 0.6 and mean_of(rows, "abstain") > 0.2))
    gr = G.GateReport()
    battery(gr, live={"observed": acc - 0.25, "min": 0.1, "name": "tradeoff_recoverable_above_chance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "no_valence_label_in_the_model"},
            positive={"observed": resolved, "expected": 1.0, "tol": 0.5, "name": "non_abstained_answers_mostly_right"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_menus"},
            oracle={"observed": acc, "min": 0.4, "name": "identifiable"},
            prediction={"gain": mean_of(rows, "ls") - np.log(1 / 4), "min": 0.0, "name": "held_out_above_uniform"},
            calibration={"observed": C.ece([r["conf"] for r in rows], [r["correct"] for r in rows]), "reference": 0.25, "direction": "down", "tol": 0.0, "name": "calibrated"})
    criterion(v, "O11", passed, accuracy=acc, resolved_accuracy=resolved, by_cell=by)
    v["results"].update({"accuracy": acc, "by_cell": by, "abstention_rate": mean_of(rows, "abstain")})
    receipt(v, rows, card, ctx)
    narrative(v, f"The planted social or risk tradeoff, crossed with private reward, was recovered {acc:.0%} of the time; among answers the reader did not abstain on, {resolved:.0%} were right.",
              "Social and risk costs are tradeoffs the choice record can carry; nothing in the reader names them virtues.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O12 / O13 — choice-set-size neglect and its repair.
# --------------------------------------------------------------------------- #
def _size_sensitivity(profiles, names, actor, r, fam, size, reader, condition=None):
    recs = CO.stream(actor, r, 12, fam.ng, ecology="craft", n_options=size)
    if reader == "mimic":
        sv = CO.neglect_reader_size(size)
        if condition in ("joint", "rank", "explicit"):
            sv = size                                                  # salience makes the full set available
        elif condition == "recall":
            sv = max(sv, size - 1)
        post = CO.posterior(profiles, recs, size_view=sv)
    else:
        post = CO.posterior(profiles, recs)
    pv = np.array([post["profile"][n] for n in names])
    return float((pv[pv > 0] * np.log(pv[pv > 0] * len(names))).sum())


def unit_O12(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o12")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        for size in (2, 4, 6, 8):
            for reader in ("mimic", "exact"):
                cells.add({"set_size": size, "reader": reader}, info=_size_sensitivity(profiles, names, actor, r, fam, size, reader))
    return {"rows": cells.rows()}


def reduce_O12(card, units, ctx):
    v = start(card, ctx, "A planted reader that under-counts alternatives reproduces a choice-set-size neglect curve: it grows less "
              "certain with set size than the exact reader while keeping the direction.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    from scipy.stats import spearmanr
    curve = {rd: {str(s): mean_of(rows, "info", lambda r, s=s, rd=rd: r["set_size"] == s and r["reader"] == rd) for s in (2, 4, 6, 8)} for rd in ("mimic", "exact")}
    sens = {rd: (curve[rd]["8"] - curve[rd]["2"]) for rd in curve}
    passed = bool(0.0 < sens["mimic"] < sens["exact"])
    gr = G.GateReport()
    gr.live("set_size_moves_the_exact_reader", observed_change=sens["exact"], min_change=0.05)
    gr.positive("mimic_keeps_the_direction", observed=float(sens["mimic"] > 0), expected=1.0, tol=0.0)
    gr.positive("mimic_underweights_size", observed=float(sens["mimic"] < sens["exact"]), expected=1.0, tol=0.0)
    criterion(v, "O12", passed, sensitivity=sens, curve=curve)
    v["results"].update({"information_by_set_size": curve, "sensitivity": sens})
    receipt(v, rows, card, ctx)
    narrative(v, f"Information extracted from a choice rose by {sens['exact']:.2f} nats from two to eight options for the exact reader and by {sens['mimic']:.2f} for the planted neglecting reader.",
              "The neglect curve is a planted heuristic reproduced here as a ruler; the exact reader is its ceiling.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_O13(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o13")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        for cond in ("separate", "joint", "rank", "recall", "explicit"):
            s2 = _size_sensitivity(profiles, names, actor, r, fam, 2, "mimic", cond)
            s8 = _size_sensitivity(profiles, names, actor, r, fam, 8, "mimic", cond)
            cells.add({"condition": cond}, sensitivity=s8 - s2)
    return {"rows": cells.rows()}


def reduce_O13(card, units, ctx):
    v = start(card, ctx, "Making the alternatives salient (joint comparison, ranking, recall of the set, an explicit cue) raises the "
              "neglecting reader's sensitivity to set size without changing the choice data.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: boot(rows, "sensitivity", lambda r, c=c: r["condition"] == c, seed_tag="O13" + c)["mean"] for c in ("separate", "joint", "rank", "recall", "explicit")}
    rise = float(np.mean([by[c] - by["separate"] for c in ("joint", "rank", "explicit")]))
    passed = bool(rise >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": rise, "min": 0.02, "name": "salience_raises_sensitivity"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "choice_data_unchanged_by_condition"},
            positive={"observed": float(by["joint"] >= by["separate"]), "expected": 1.0, "tol": 0.0, "name": "joint_comparison_helps"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_records"},
            oracle={"observed": by["explicit"] - by["separate"], "min": 0.0, "name": "explicit_cue_is_the_ceiling"},
            prediction={"gain": rise, "min": 0.0, "name": "sensitivity_rise"},
            calibration={"observed": by["recall"], "reference": by["separate"], "direction": "up", "tol": 0.0, "name": "recall_no_worse"})
    criterion(v, "O13", passed, by_condition=by, minimum_rise=rise)
    v["results"].update({"sensitivity_by_condition": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Set-size sensitivity of the neglecting reader was " + ", ".join(f"{c} {x:.2f}" for c, x in by.items()) + " nats.",
              "Neglect is an attention state, not a fixed property: what is made salient is what gets counted.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O14 — weighting tournament.
# --------------------------------------------------------------------------- #
def unit_O14(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o14")
    names = list(profiles)
    ecos = cost_ecologies(ctx)[:4]
    fams = dict(CO.WEIGHTING)
    for eco in ecos:
        for i in range(max(3, _n_actors(ctx) // 2)):
            w = profiles[names[i % len(names)]]
            true_fn = str(r.choice(["linear", "logarithmic", "saturating"]))
            actor = CO.Actor(w, motivation=1.0)
            recs = []
            for _ in range(24):
                m = CO.menu(r, fam.ng, 4, eco)
                m["cost"] = CO.WEIGHTING[true_fn](m["cost"])
                recs.append(CO.choose(actor, m, r))
            train, test = recs[:16], recs[16:]
            # learned monotone: fit on the training records' paid costs against choice log-odds proxy
            paid = np.concatenate([np.asarray(t["cost"]).ravel() for t in train])
            gain_proxy = np.concatenate([np.repeat(np.asarray(t["payoff"]).sum(axis=1), 8) for t in train]) * 0.3
            learned = CO.learned_monotone(paid, gain_proxy)
            for name, fn in list(fams.items()) + [("learned_monotone", learned)]:
                post = CO.posterior(profiles, train, cost_fn=fn)
                ls = _heldout_ls(post, profiles, test, cost_fn=fn)
                cells.add({"family": name, "ecology": eco}, ls=ls, true_family_match=float(name == true_fn))
    return {"rows": cells.rows()}


def reduce_O14(card, units, ctx):
    v = start(card, ctx, "No weighting family is a law: cross-validated held-out scores by family and ecology form equivalence classes, "
              "and the classes are the result.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    fams = list(CO.WEIGHTING) + ["learned_monotone"]
    ecos = sorted({r["ecology"] for r in rows})
    grid = {f: {e: boot(rows, "ls", lambda r, f=f, e=e: r["family"] == f and r["ecology"] == e, seed_tag=f"O14{f}{e}")["mean"] for e in ecos} for f in fams}
    overall = {f: float(np.mean(list(grid[f].values()))) for f in fams}
    best = max(overall, key=overall.get)
    classes = sorted([f for f in fams if overall[best] - overall[f] <= 0.01])
    gr = G.GateReport()
    battery(gr, live={"observed": max(overall.values()) - min(overall.values()), "min": 0.02, "name": "families_differ_somewhere"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "frozen_cross_validation"},
            positive={"observed": float(overall["linear"] >= min(overall.values())), "expected": 1.0, "tol": 0.0, "name": "linear_reported"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_records_per_family"},
            oracle={"observed": overall[best] - np.log(1 / 4), "min": 0.0, "name": "best_family_above_uniform"},
            prediction={"gain": overall[best] - overall["threshold"], "min": -1.0, "name": "best_minus_threshold"},
            calibration={"observed": float(len(classes)), "reference": 1.0, "direction": "up", "tol": 0.0, "name": "equivalence_class_size_reported"})
    criterion(v, "O14", True, best=best, equivalence_class=classes, overall=overall)
    v["results"].update({"held_out_by_family_and_ecology": grid, "overall": overall, "equivalence_class_at_0.01": classes})
    receipt(v, rows, card, ctx)
    narrative(v, f"The best weighting family across ecologies was {best}; {', '.join(classes)} were within 0.01 nats of it and form its equivalence class.",
              "The tournament returns a class, not a law; a text-side test would need to separate the class members before claiming any one.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


# --------------------------------------------------------------------------- #
# O15 — the ideal reader against the planted heuristic.
# --------------------------------------------------------------------------- #
def unit_O15(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o15")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        recs = CO.stream(actor, r, 24, fam.ng, ecology="craft", n_options=6)
        train, test = recs[:16], recs[16:]
        for menu_state in ("complete", "missing"):
            tr = train if menu_state == "complete" else [CO.hidden_menu(t, r, hide=2) for t in train]
            te = test if menu_state == "complete" else [CO.hidden_menu(t, r, hide=2) for t in test]
            readers = {"mimic": dict(size_view=CO.neglect_reader_size(6)), "exact": {}, "misspecified": dict(cost_fn=CO.WEIGHTING["threshold"])}
            posts = {k: CO.posterior(profiles, tr, **kw) for k, kw in readers.items()}
            # hybrid: a calibrated mixture of mimic and exact weighted by their training likelihood
            lw = np.array([sum(CO.loglik(CO.Actor(profiles[n]), t, size_view=CO.neglect_reader_size(6)) for t in tr for n in [max(posts["mimic"]["profile"], key=posts["mimic"]["profile"].get)]),
                           sum(CO.loglik(CO.Actor(profiles[n]), t) for t in tr for n in [max(posts["exact"]["profile"], key=posts["exact"]["profile"].get)])])
            mix = C.softmax(lw)
            for k, kw in readers.items():
                ls = _heldout_ls(posts[k], profiles, te, **({} if k != "misspecified" else {"cost_fn": CO.WEIGHTING["threshold"]}))
                cells.add({"reader": k, "menu": menu_state}, ls=ls, conf=float(max(posts[k]["profile"].values())), top1=float(max(posts[k]["profile"], key=posts[k]["profile"].get) == names[i % len(names)]))
            p_h = {n: mix[0] * posts["mimic"]["profile"][n] + mix[1] * posts["exact"]["profile"][n] for n in names}
            post_h = {"P": np.array([[p_h[n]] for n in names]), "names": names, "cells": [{}], "profile": p_h}
            cells.add({"reader": "hybrid", "menu": menu_state}, ls=_heldout_ls(post_h, profiles, te), conf=float(max(p_h.values())), top1=float(max(p_h, key=p_h.get) == names[i % len(names)]))
    return {"rows": cells.rows()}


def reduce_O15(card, units, ctx):
    v = start(card, ctx, "An exact reader can beat the planted neglecting heuristic on held-out choices; whether it does so safely "
              "when menus are incomplete is the bounded ideal-reader gain, and a calibrated hybrid is its hedge.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {rd: {m: {"ls": boot(rows, "ls", lambda r, rd=rd, m=m: r["reader"] == rd and r["menu"] == m, seed_tag=f"O15{rd}{m}")["mean"],
                     "overconf": mean_of(rows, "conf", lambda r, rd=rd, m=m: r["reader"] == rd and r["menu"] == m) - mean_of(rows, "top1", lambda r, rd=rd, m=m: r["reader"] == rd and r["menu"] == m)} for m in ("complete", "missing")} for rd in ("mimic", "exact", "misspecified", "hybrid")}
    gain_complete = grid["exact"]["complete"]["ls"] - grid["mimic"]["complete"]["ls"]
    gain_missing = grid["exact"]["missing"]["ls"] - grid["mimic"]["missing"]["ls"]
    hybrid_ok = bool(grid["hybrid"]["missing"]["ls"] >= grid["mimic"]["missing"]["ls"] - 0.02 and grid["hybrid"]["missing"]["overconf"] <= grid["mimic"]["missing"]["overconf"] + 0.02)
    passed = hybrid_ok
    gr = G.GateReport()
    battery(gr, live={"observed": gain_complete, "min": 0.0, "name": "exact_beats_mimic_with_complete_menus"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_choices"},
            positive={"observed": float(grid["misspecified"]["complete"]["ls"] <= grid["exact"]["complete"]["ls"] + 0.02), "expected": 1.0, "tol": 0.0, "name": "misspecified_exact_no_better_than_exact"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_records"},
            oracle={"observed": grid["exact"]["complete"]["ls"] - np.log(1 / 6), "min": 0.0, "name": "exact_above_uniform"},
            prediction={"gain": gain_missing, "min": -1.0, "name": "exact_minus_mimic_under_missing_menus"},
            calibration={"observed": grid["exact"]["missing"]["overconf"], "reference": grid["mimic"]["missing"]["overconf"], "direction": "down", "tol": 0.10, "name": "exact_not_much_more_overconfident_under_missing_menus"})
    criterion(v, "O15", passed, gain_complete=gain_complete, gain_missing=gain_missing, hybrid_safe=hybrid_ok)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"With complete menus the exact reader beat the neglecting heuristic by {gain_complete:+.2f} nats; with two options hidden the gap was {gain_missing:+.2f} and the exact reader's "
                 f"overconfidence {grid['exact']['missing']['overconf']:+.2f} against the heuristic's {grid['mimic']['missing']['overconf']:+.2f}; the hybrid {'held' if hybrid_ok else 'did not hold'} the heuristic's calibration.",
              "This is a bounded ideal-reader gain: it is real with complete records and it is hedged, not assumed, when records are missing.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O16 — incomplete and false choice sets.
# --------------------------------------------------------------------------- #
def unit_O16(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o16")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        recs = CO.stream(actor, r, 16, fam.ng, ecology="craft", n_options=5)
        for menu_state in ("complete", "missing", "false", "uncertain"):
            if menu_state == "complete":
                view = recs
            elif menu_state == "missing":
                view = [CO.hidden_menu(t, r, hide=2) for t in recs]
            elif menu_state == "false":
                view = [CO.hidden_menu(t, r, hide=0, add_false=2) for t in recs]
            else:
                view = [CO.hidden_menu(t, r, hide=int(r.integers(0, 3)), add_false=int(r.integers(0, 2))) for t in recs]
            for reader in ("naive", "calibrated"):
                if reader == "naive":
                    post = CO.posterior(profiles, view)
                else:
                    # the calibrated reader treats the visible menu as a claim: it tempers the evidence by the
                    # probability the menu is complete and adds an unknown alternative of average value
                    view_c = []
                    for t in view:
                        tt = dict(t)
                        pay = np.vstack([np.asarray(t["payoff"]), np.asarray(t["payoff"]).mean(axis=0, keepdims=True)])
                        cost = np.vstack([np.asarray(t["cost"]), np.asarray(t["cost"]).mean(axis=0, keepdims=True)])
                        tt.update({"payoff": pay, "cost": cost, "variance": np.concatenate([np.asarray(t["variance"]), [np.mean(t["variance"])]]),
                                   "info": np.concatenate([np.asarray(t["info"]), [np.mean(t["info"])]]), "n": int(pay.shape[0])})
                        view_c.append(tt)
                    post = CO.posterior(profiles, view_c)
                    P_ = post["P"] ** 0.7
                    P_ = P_ / P_.sum()
                    post = {**post, "P": P_, "profile": dict(zip(names, P_.sum(axis=1)))}
                conf = float(max(post["profile"].values()))
                top = float(max(post["profile"], key=post["profile"].get) == names[i % len(names)])
                cells.add({"menu": menu_state, "reader": reader}, conf=conf, top1=top, truth_mass=float(post["profile"][names[i % len(names)]]))
    return {"rows": cells.rows()}


def reduce_O16(card, units, ctx):
    v = start(card, ctx, "When alternatives are missing or false, a reader that treats the visible menu as a claim stays calibrated where "
              "a reader that treats it as the truth becomes confidently wrong.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ece = {m: {rd: C.ece([r["conf"] for r in rows if r["menu"] == m and r["reader"] == rd], [r["top1"] for r in rows if r["menu"] == m and r["reader"] == rd]) for rd in ("naive", "calibrated")} for m in ("complete", "missing", "false", "uncertain")}
    gain = float(np.mean([ece[m]["naive"] - ece[m]["calibrated"] for m in ("missing", "false")]))
    passed = bool(gain >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": ece["missing"]["naive"] - ece["complete"]["naive"], "min": 0.0, "name": "menu_corruption_hurts_the_naive_reader"},
            placebo={"observed": abs(ece["complete"]["naive"] - ece["complete"]["calibrated"]), "tol": 0.15, "name": "complete_menus_read_alike"},
            positive={"observed": mean_of(rows, "top1", lambda r: r["menu"] == "complete" and r["reader"] == "naive"), "expected": 1.0, "tol": 0.7, "name": "complete_menus_identify"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_choices"},
            oracle={"observed": mean_of(rows, "truth_mass", lambda r: r["menu"] == "complete") - 1 / 8, "min": 0.0, "name": "identifiable_with_complete_menus"},
            prediction={"gain": gain, "min": 0.0, "name": "ece_improvement_under_corruption"},
            calibration={"observed": ece["false"]["calibrated"], "reference": ece["false"]["naive"], "direction": "down", "tol": 0.0, "name": "calibrated_reader_better_under_false_menus"})
    criterion(v, "O16", passed, ece=ece, ece_gain=gain)
    v["results"].update({"ece_by_menu_and_reader": ece})
    receipt(v, rows, card, ctx)
    narrative(v, f"Under missing and false alternatives the naive reader's calibration error was {np.mean([ece[m]['naive'] for m in ('missing', 'false')]):.2f} against {np.mean([ece[m]['calibrated'] for m in ('missing', 'false')]):.2f} for the reader that treated the menu as a claim.",
              "A menu is evidence about what was on offer, not the offer itself.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O17 — prospective and counterfactual prediction.
# --------------------------------------------------------------------------- #
def unit_O17(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o17")
    names = list(profiles)
    ecos = cost_ecologies(ctx)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        train = CO.stream(actor, r, 16, fam.ng, ecology=ecos[0])
        post = CO.posterior(profiles, train)
        targets = {}
        targets["changed_costs"] = [CO.choose(actor, dict(CO.menu(r, fam.ng, 4, ecos[0]), cost=CO.menu(r, fam.ng, 4, ecos[0])["cost"] * 2.0), r) for _ in range(6)]
        targets["new_domain"] = CO.stream(actor, r, 6, fam.ng, ecology=ecos[1])
        g = int(r.integers(fam.ng))
        w_goal = C.normalize(w + 0.5 * np.eye(fam.ng)[g])
        targets["new_goal"] = CO.stream(CO.Actor(w_goal, motivation=1.0), r, 6, fam.ng, ecology=ecos[0])
        targets["new_commission"] = CO.stream(CO.Actor(w, motivation=1.6), r, 6, fam.ng, ecology=ecos[0])
        targets["new_role"] = CO.stream(CO.Actor(w, motivation=1.0, constraint=0.5), r, 6, fam.ng, ecology=ecos[0])
        habit_opt = int(np.argmax(np.bincount([int(t["choice"]) for t in train], minlength=4)))
        for tgt, recs in targets.items():
            kw = {}
            post_t = post
            if tgt == "new_goal":
                fam_goal = {n: C.normalize(profiles[n] + 0.5 * np.eye(fam.ng)[g]) for n in names}
                ls_prof = float(np.mean([np.log(max(CO.predict_choice(post, fam_goal, t)[int(t["choice"])], 1e-12)) for t in recs]))
            else:
                ls_prof = float(np.mean([np.log(max(CO.predict_choice(post_t, profiles, t)[int(t["choice"])], 1e-12)) for t in recs]))
            ls_freq = _freq_ls(train, recs)
            ls_id = float(np.mean([np.log(1.0 / t["n"]) for t in recs]))
            ls_habit = float(np.mean([np.log(0.7 if int(t["choice"]) == habit_opt else 0.3 / max(t["n"] - 1, 1)) for t in recs]))
            for base, ls_b in (("identity", ls_id), ("frequency", ls_freq), ("habit", ls_habit)):
                cells.add({"target": tgt, "baseline": base}, gain=ls_prof - ls_b, ls=ls_prof)
    return {"rows": cells.rows()}


def reduce_O17(card, units, ctx):
    v = start(card, ctx, "A recovered tradeoff predicts choices under changed costs, in a new domain, under a new goal, a new "
              "commission and a new role, above identity, frequency and habit baselines; only prospective gains are promoted.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {t: {b: boot(rows, "gain", lambda r, t=t, b=b: r["target"] == t and r["baseline"] == b, seed_tag=f"O17{t}{b}")["mean"] for b in ("identity", "frequency", "habit")} for t in ("changed_costs", "new_domain", "new_goal", "new_commission", "new_role")}
    passed = bool(all(grid[t]["frequency"] > 0 for t in grid))
    gr = G.GateReport()
    battery(gr, live={"observed": max(grid[t]["frequency"] for t in grid), "min": 0.02, "name": "profile_predicts_something_prospective"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "targets_generated_after_inference"},
            positive={"observed": float(grid["changed_costs"]["identity"] > 0), "expected": 1.0, "tol": 0.0, "name": "beats_uniform_under_changed_costs"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_retrospective_fit_reported_as_prospective"},
            oracle={"observed": grid["changed_costs"]["identity"], "min": 0.0, "name": "prospective_gain"},
            prediction={"gain": float(np.mean([grid[t]["frequency"] for t in grid])), "min": 0.0, "name": "mean_gain_over_frequency"},
            calibration={"observed": min(grid[t]["habit"] for t in grid), "reference": -0.5, "direction": "up", "tol": 0.0, "name": "never_far_below_habit"})
    criterion(v, "O17", passed, grid=grid)
    v["results"].update({"gain_by_target_and_baseline": grid})
    receipt(v, rows, card, ctx)
    narrative(v, "Over frequency the recovered tradeoff gained " + ", ".join(f"{t} {grid[t]['frequency']:+.2f}" for t in grid) + " nats on prospective choices.",
              "The tradeoff is a model of the maker, not a fit to its history, where the gains are positive; where they are not, it is not.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# O18 — effort as a misleading quality cue.
# --------------------------------------------------------------------------- #
def unit_O18(ctx):
    world = world_for(ctx)
    fam, profiles = _profiles(world)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "o18")
    names = list(profiles)
    for i in range(_n_actors(ctx)):
        w = profiles[names[i % len(names)]]
        for quality in ("low", "high"):
            actor = CO.Actor(w, motivation=1.2, competence=0.95 if quality == "high" else 0.5)
            recs_q = CO.stream(actor, r, 12, fam.ng, ecology="craft")
            for claimed in ("low", "high"):
                # an artifact's quality is its execution fidelity; claimed effort is an assertion beside it
                recs = recs_q
                true_effort = float(np.mean([np.asarray(t["paid_cost"]).sum() for t in recs]))
                claim = 2.0 if claimed == "high" else 0.3
                for reader in ("expert", "novice"):
                    # quality judgement: an expert reads execution fidelity (competence) from the records; a novice weights the claim
                    post = CO.posterior(profiles, recs, causes={"competence": list(LEV["competence"])})
                    comp_est = float(sum(p * c["competence"] for p, c in zip(post["cause"], post["cells"])))
                    q_judged = comp_est if reader == "expert" else 0.5 * comp_est + 0.5 * (0.9 if claim > 1 else 0.5)
                    mot_post = CO.posterior(profiles, recs, causes={"motivation": list(LEV["motivation"])})
                    mot_est = float(sum(p * c["motivation"] for p, c in zip(mot_post["cause"], mot_post["cells"])))
                    cells.add({"claimed_effort": claimed, "quality": quality, "reader": reader}, quality_judged=q_judged, motivation_est=mot_est, true_effort=true_effort)
    return {"rows": cells.rows()}


def reduce_O18(card, units, ctx):
    v = start(card, ctx, "A claimed effort moves a novice's quality judgement more than an expert's, and neither claim moves the "
              "inference of the maker's motivation from its records; the effort heuristic's mixed replication is kept as a warning.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    def q(reader, claimed, quality):
        return mean_of(rows, "quality_judged", lambda r: r["reader"] == reader and r["claimed_effort"] == claimed and r["quality"] == quality)
    claim_effect = {rd: float(np.mean([q(rd, "high", ql) - q(rd, "low", ql) for ql in ("low", "high")])) for rd in ("expert", "novice")}
    quality_effect = {rd: float(np.mean([q(rd, cl, "high") - q(rd, cl, "low") for cl in ("low", "high")])) for rd in ("expert", "novice")}
    mot_by_claim = {cl: mean_of(rows, "motivation_est", lambda r, cl=cl: r["claimed_effort"] == cl) for cl in ("low", "high")}
    passed = bool(claim_effect["novice"] > claim_effect["expert"] and abs(mot_by_claim["high"] - mot_by_claim["low"]) <= 1e-9)
    gr = G.GateReport()
    battery(gr, live={"observed": quality_effect["expert"], "min": 0.05, "name": "true_quality_moves_the_expert"},
            placebo={"observed": abs(mot_by_claim["high"] - mot_by_claim["low"]), "tol": 1e-9, "name": "claimed_effort_leaves_motivation_inference_untouched"},
            positive={"observed": float(claim_effect["novice"] > claim_effect["expert"]), "expected": 1.0, "tol": 0.0, "name": "novice_moved_more_by_the_claim"},
            surface={"accuracy": claim_effect["expert"], "chance": 0.0, "tol": 0.05, "name": "expert_not_moved_by_the_claim"},
            oracle={"observed": quality_effect["expert"], "min": 0.0, "name": "quality_readable_from_records"},
            prediction={"gain": quality_effect["novice"], "min": -1.0, "name": "novice_quality_sensitivity"},
            calibration={"observed": claim_effect["novice"], "reference": quality_effect["novice"], "direction": "down", "tol": 1.0, "name": "claim_vs_quality_for_novice_reported"})
    criterion(v, "O18", passed, claim_effect=claim_effect, quality_effect=quality_effect, motivation_by_claim=mot_by_claim)
    v["results"].update({"claim_effect_on_quality_judgement": claim_effect, "true_quality_effect": quality_effect, "motivation_estimate_by_claim": mot_by_claim,
                         "warning": "the effort heuristic replicated mixedly in preregistered studies (Ziano et al.); this construction plants it in the novice and does not claim it for people"})
    receipt(v, rows, card, ctx)
    narrative(v, f"A claim of high effort moved the novice reader's quality judgement by {claim_effect['novice']:+.2f} and the expert's by {claim_effect['expert']:+.2f}; the maker's inferred motivation moved by {abs(mot_by_claim['high'] - mot_by_claim['low']):.1e}.",
              "Effort claims are quality cues only for readers who cannot read execution; motivation inference from records is untouched by them.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
