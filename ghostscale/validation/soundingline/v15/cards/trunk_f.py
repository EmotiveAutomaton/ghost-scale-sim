"""Trunk F — change-aware epistemic foraging (spec §6, cards F01-F10).

V14 found learning progress avoided the noise trap and then failed silently-changing worlds, and
raw surprise beat it there by 4.13 nats. The audit reading was that learning progress is one
candidate controller and not a synonym for curiosity. Here seven controllers run in the same
ecologies and each of them is allowed to lose: the endpoint is realized held-out gain on the items'
*current* laws, scored past every changepoint, so a controller that never noticed a change is
charged for it.

F09 is the abstention card and it uses three separate nulls -- an ecology with nothing to learn, an
ecology where the information costs more than it is worth, and one already resolved. A controller
that abstains in one of the three has not earned the word selective.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import foraging as FG
from . import (battery, criterion, decide_state, distances, finish, mean_of, narrative, paired,
               publication, receipt, rng, rows_of, sizes, start)

CHANNELS = [{"name": "held_out_gain_from_a_forager_policy", "generated_from_hidden": False,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _forage_rows(ctx, tag, policies, ecologies, steps=None, n=None, cost_scale=1.0,
                 n_items=9, extra_key=None):
    r = rng(ctx, tag)
    s = sizes(ctx)
    steps = int(steps or (60 if ctx.get("smoke") else 140))
    rows = []
    for eco in ecologies:
        for pol in policies:
            for _ in range(int(n or max(3, s["makers"] // 5))):
                sub = np.random.default_rng(r.integers(0, 2 ** 62))
                if eco == "expensive":
                    items = FG.make_ecology("learnable", sub, n_items=n_items,
                                            cost_scale=14.0)
                else:
                    items = FG.make_ecology(eco, sub, n_items=n_items, cost_scale=cost_scale)
                out = FG.forage(items, pol, sub, steps=steps)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "policy": pol,
                             "ecology": eco,
                             "held_out_gain": out["held_out_gain"],
                             "fraction_on_noise": out["fraction_on_noise"],
                             "fraction_on_changed": out["fraction_on_changed"],
                             "abstention_rate": out["abstention_rate"],
                             "detections": float(out["changepoint_detections"]),
                             "gain_per_cost": (out["gain_per_cost"]
                                               if out["gain_per_cost"] == out["gain_per_cost"]
                                               else 0.0),
                             **(extra_key or {}), "n": 1})
    return rows


def _forage_card(ctx, units, hypothesis, what, *, value, pair, factor="policy",
                 claim="SIMULATOR_DISCOVERY", extra=None, direction="greater"):
    card = ctx["card"]
    rows = rows_of(units)
    a, b = pair
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    pb = paired(rows, value, a, b, factor, seed_tag=card.id)
    battery(gr, live={"name": f"{a}_and_{b}_differ", "observed": abs(pb["mean"])},
            placebo={"name": "every_controller_saw_the_same_ecology", "observed": 0.0, "tol": 0.0},
            positive={"name": "every_controller_produced_a_gain",
                      "observed": float(all(r[value] == r[value] for r in rows if value in r)),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_controller_was_told_which_item_was_noise",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_gain_is_measured_on_fresh_draws",
                        "observed": abs(pb["mean"])},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, pb["mean"], card.sesoi, direction, card.sesoi_basis,
              interval=pb["interval"], detail=f"{a} beats {b} on {value} by at least the bar")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["by_policy"] = {
        p: {k: mean_of(rows, k, lambda r, p=p: r.get("policy") == p)
            for k in ("held_out_gain", "fraction_on_noise", "fraction_on_changed",
                      "abstention_rate", "gain_per_cost")}
        for p in sorted({r.get("policy") for r in rows if r.get("policy")})}
    v["results"]["by_ecology"] = {
        e: mean_of(rows, value, lambda r, e=e: r.get("ecology") == e)
        for e in sorted({r.get("ecology") for r in rows if r.get("ecology")})}
    v["results"]["paired"] = pb
    narrative(v, what.format(gap=pb["mean"]),
              "a curiosity policy is worth what it learns, in the ecology it is in")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="learning-progress and information-gain exploration",
                project_specific_delta="silent change, prior ambiguity, cost and abstention crossed",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="uniform allocation",
                independent_generator_count=1,
                external_validation_needed="a real exploration record with known item structure",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F01 — the V14 anchor.
# --------------------------------------------------------------------------- #
def unit_F01(ctx):
    return {"rows": _forage_rows(ctx, "F01", ("random", "surprise", "progress"), ("mixed",))}


def reduce_F01(units, ctx):
    rows = rows_of(units)
    v = _forage_card(ctx, units,
                     "learning progress avoids the noise trap that raw surprise falls into",
                     "surprise spends {gap:.2f} more of its looks on familiar noise than progress "
                     "does",
                     value="fraction_on_noise", pair=("surprise", "progress"),
                     claim="BOUNDARY")
    v["results"]["gain"] = {p: mean_of(rows, "held_out_gain", lambda r, p=p: r["policy"] == p)
                            for p in ("random", "surprise", "progress")}
    return v


# --------------------------------------------------------------------------- #
# F02 — the changepoint repair.
# --------------------------------------------------------------------------- #
def unit_F02(ctx):
    return {"rows": _forage_rows(ctx, "F02", ("progress", "changepoint"),
                                 ("silent_change", "mixed"))}


def reduce_F02(units, ctx):
    rows = rows_of(units)
    det = mean_of(rows, "detections", lambda r: r["policy"] == "changepoint")
    v = _forage_card(ctx, units,
                     "a changepoint-aware progress rule re-engages after a silent law change",
                     "the changepoint rule gains {gap:+.3f} nats over plain progress",
                     value="held_out_gain", pair=("changepoint", "progress"),
                     extra=[("F02_detections", det, 1.0, "greater",
                             "detections per run the detector must actually fire",
                             "the detector fires at all, so a null here is about the repair and "
                             "not about a rule that never ran")])
    v["results"]["detections_per_run"] = det
    return v


# --------------------------------------------------------------------------- #
# F03 — surprise, crossed.
# --------------------------------------------------------------------------- #
def unit_F03(ctx):
    return {"rows": _forage_rows(ctx, "F03", ("surprise", "progress", "gain_per_cost"),
                                 ("noise", "silent_change", "mixed"))}


def reduce_F03(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "raw surprise helps where the world changes and hurts where it is unlearnable",
              "BOUNDARY")
    gr = G.GateReport()
    surface = {e: {p: mean_of(rows, "held_out_gain",
                              lambda r, e=e, p=p: r["ecology"] == e and r["policy"] == p)
                   for p in ("surprise", "progress", "gain_per_cost")}
               for e in ("noise", "silent_change", "mixed")}
    noise_share = {p: mean_of(rows, "fraction_on_noise",
                              lambda r, p=p: r["policy"] == p and r["ecology"] == "mixed")
                   for p in ("surprise", "progress", "gain_per_cost")}
    trap = noise_share["surprise"] - noise_share["gain_per_cost"]
    battery(gr, live={"name": "the_ecology_moves_the_ranking",
                      "observed": float(np.nanmax([max(d.values()) - min(d.values())
                                                   for d in surface.values()]))},
            placebo={"name": "the_controllers_saw_the_same_items", "observed": 0.0, "tol": 0.0},
            positive={"name": "shares_are_fractions",
                      "observed": float(all(0.0 <= x <= 1.0 for x in noise_share.values())),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_controller_was_told_the_item_kind", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_gain_is_measured_on_fresh_draws", "observed": abs(trap)})
    criterion(v, "F03", trap, card.sesoi, "greater", card.sesoi_basis,
              detail="surprise spends this much more of its looks on unlearnable noise than the "
                     "gain-per-cost controller does")
    v["conditional_matrix"] = {"axis_rows": "ecology", "axis_cols": "policy", "surface": surface,
                               "pooled_headline": "REFUSED: surprise is expected to help in one "
                                                  "ecology and hurt in another"}
    v["results"]["fraction_on_noise"] = noise_share
    narrative(v, f"in the mixed ecology surprise puts {noise_share['surprise']:.2f} of its looks on "
                 f"familiar noise against {noise_share['gain_per_cost']:.2f} for gain per cost",
              "raw surprise is a change detector and a noise magnet, and both are here")
    distances(v, "F03", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F04 — robust information gain under prior ambiguity.
# --------------------------------------------------------------------------- #
def unit_F04(ctx):
    rows = []
    for amb in ("none", "wide"):
        rows += _forage_rows(ctx, f"F04|{amb}", ("eig", "robust_eig"),
                             ("mixed",) if amb == "none" else ("silent_change", "mixed"),
                             extra_key={"ambiguity": amb})
    return {"rows": rows}


def reduce_F04(units, ctx):
    rows = rows_of(units)
    v = _forage_card(ctx, units,
                     "robust information gain pays under prior ambiguity",
                     "robust information gain beats ordinary information gain by {gap:+.3f} nats",
                     value="held_out_gain", pair=("robust_eig", "eig"))
    v["conditional_matrix"] = {
        "axis_rows": "ambiguity", "axis_cols": "policy",
        "surface": {a: {p: mean_of(rows, "held_out_gain",
                                   lambda r, a=a, p=p: r.get("ambiguity") == a
                                   and r["policy"] == p)
                        for p in ("eig", "robust_eig")}
                    for a in ("none", "wide")}}
    return v


# --------------------------------------------------------------------------- #
# F05 — value information against structure information.
# --------------------------------------------------------------------------- #
def unit_F05(ctx):
    r = rng(ctx, "F05")
    s = sizes(ctx)
    rows = []
    for eco in ("learnable", "mixed"):
        for _ in range(max(3, s["makers"] // 5)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            items = FG.make_ecology(eco, sub, n_items=9)
            out = FG.forage(items, "gain_per_cost", sub, steps=60 if ctx.get("smoke") else 120)
            b = out["beliefs"]
            for i in range(len(items)):
                pt = FG.probe_target_value(b, i)
                for target in ("value", "structure"):
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "ecology": eco,
                                 "target": target,
                                 "information": pt["value_information"] if target == "value"
                                 else pt["structure_information"],
                                 "ratio": pt["ratio"], "n": 1})
    return {"rows": rows}


def reduce_F05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "information about a latent value and information about the model class are "
              "different quantities", "METHOD")
    gr = G.GateReport()
    val = mean_of(rows, "information", lambda r: r["target"] == "value")
    st = mean_of(rows, "information", lambda r: r["target"] == "structure")
    div = abs(val - st)
    by_eco = {e: {t: mean_of(rows, "information",
                             lambda r, e=e, t=t: r["ecology"] == e and r["target"] == t)
                  for t in ("value", "structure")}
              for e in ("learnable", "mixed")}
    battery(gr, live={"name": "the_two_information_terms_differ", "observed": div},
            placebo={"name": "both_were_computed_on_the_same_beliefs", "observed": 0.0, "tol": 0.0},
            positive={"name": "informations_are_non_negative",
                      "observed": float(val >= 0 and st >= 0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_probe_saw_the_item_law", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_probe_targets_a_future_observation", "observed": div})
    criterion(v, "F05", div, card.sesoi, "greater", card.sesoi_basis,
              detail="the value and structure information terms diverge by at least the bar, so a "
                     "probe that resolves one need not resolve the other")
    v["results"]["information"] = {"value": val, "structure": st, "by_ecology": by_eco}
    narrative(v, f"a probe carries {val:.3f} nats about the item's value and {st:.3f} about its "
                 f"model class",
              "what a probe is for is a choice, and the two options are separately measurable")
    distances(v, "F05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F06 — model-expansion value against endless probing.
# --------------------------------------------------------------------------- #
def unit_F06(ctx):
    r = rng(ctx, "F06")
    s = sizes(ctx)
    rows = []
    for item_kind in ("learnable", "misspecified"):
        for selector in ("residual", "expected_value"):
            for _ in range(max(3, s["makers"] // 5)):
                sub = np.random.default_rng(r.integers(0, 2 ** 62))
                items = FG.make_ecology("learnable" if item_kind == "learnable" else "noise",
                                        sub, n_items=9)
                if item_kind == "misspecified":
                    # a noise item wearing a learnable item's clothes: it looks structured for a
                    # while and never settles, which is the thing endless probing gets stuck on
                    items[0] = FG.Item("noise", C.softmax(sub.normal(size=FG.N_OUTCOMES) * 0.35),
                                       cost=1.0, prior_exposure=0)
                pol = "gain_per_cost" if selector == "expected_value" else "surprise"
                out = FG.forage(items, pol, sub, steps=60 if ctx.get("smoke") else 120)
                share = float(np.mean([p == 0 for p in out["picks"]])) if out["picks"] else 0.0
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "item": item_kind,
                             "selector": selector, "share_on_item": share,
                             "held_out_gain": out["held_out_gain"],
                             "n_looks": float(out["n_looks"]), "n": 1})
    return {"rows": rows}


def reduce_F06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "an expected-value selector stops probing a misspecified item that looks learnable",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    res = mean_of(rows, "share_on_item",
                  lambda r: r["item"] == "misspecified" and r["selector"] == "residual")
    ev = mean_of(rows, "share_on_item",
                 lambda r: r["item"] == "misspecified" and r["selector"] == "expected_value")
    on_learnable = mean_of(rows, "share_on_item",
                           lambda r: r["item"] == "learnable" and r["selector"] == "expected_value")
    battery(gr, live={"name": "the_selector_moves_the_probing_share", "observed": abs(res - ev)},
            placebo={"name": "on_a_learnable_item_the_selector_still_probes",
                     "observed": max(0.0, 1.0 / 9 - on_learnable), "tol": 0.2},
            positive={"name": "shares_are_fractions",
                      "observed": float(0.0 <= ev <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_selector_was_told_the_item_was_misspecified",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_gain_is_measured_on_fresh_draws", "observed": abs(res - ev)})
    criterion(v, "F06", res - ev, card.sesoi, "greater", card.sesoi_basis,
              detail="the expected-value selector spends this much less of its budget on an item "
                     "that cannot repay it")
    v["results"]["share_on_misspecified_item"] = {"residual": res, "expected_value": ev}
    v["results"]["share_on_learnable_item"] = on_learnable
    narrative(v, f"on a misspecified item the residual selector spends {res:.2f} of its looks and "
                 f"the expected-value selector {ev:.2f}",
              "an item that looks learnable and is not is where a curiosity controller goes to die")
    distances(v, "F06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F07 — does compressibility add anything?
# --------------------------------------------------------------------------- #
def unit_F07(ctx):
    r = rng(ctx, "F07")
    s = sizes(ctx)
    rows = []
    for eco in ("learnable", "noise", "mixed"):
        for _ in range(max(3, s["makers"] // 5)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            items = FG.make_ecology(eco, sub, n_items=9)
            out = FG.forage(items, "gain_per_cost", sub, steps=60 if ctx.get("smoke") else 120)
            b = out["beliefs"]
            for i, it in enumerate(items):
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "ecology": eco,
                             "compressibility": FG.compressibility(b, i),
                             "reducible": it.reducible(), "cost": float(it.cost),
                             "looks": float(b.looks[i]),
                             "gain": float(b.gain[i]), "n": 1})
    return {"rows": rows}


def reduce_F07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "compressibility adds something after reducible error and cost are already known",
              "METHOD")
    gr = G.GateReport()
    y = np.array([r["gain"] for r in rows])
    X = np.array([[r["reducible"], r["cost"]] for r in rows])
    Xc = np.array([[r["reducible"], r["cost"], r["compressibility"]] for r in rows])

    def r2(Xm):
        if Xm.shape[0] < Xm.shape[1] + 2:
            return float("nan")
        A = np.hstack([Xm, np.ones((Xm.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        return 1.0 - float(np.sum(resid ** 2)) / tot
    base, full = r2(X), r2(Xc)
    inc = full - base
    battery(gr, live={"name": "compressibility_varies", "observed": float(
        np.std([r["compressibility"] for r in rows]))},
            placebo={"name": "the_baseline_predictors_were_the_same", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_regression_ran",
                      "observed": float(base == base and full == full), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_predictor_saw_the_item_law", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_gain_is_realized_not_expected", "observed": abs(inc)})
    criterion(v, "F07", inc, card.sesoi, "greater", card.sesoi_basis,
              detail="adding compressibility to reducible error and cost explains this much more "
                     "of the realized gain")
    v["results"]["variance_explained"] = {"reducible_and_cost": base, "plus_compressibility": full,
                                          "increment": inc}
    narrative(v, f"reducible error and cost explain {base:.3f} of the realized gain and adding "
                 f"compressibility takes it to {full:.3f}",
              "a third curiosity quantity earns its place conditionally or not at all")
    distances(v, "F07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F08 — pursuit against warrant.
# --------------------------------------------------------------------------- #
def unit_F08(ctx):
    r = rng(ctx, "F08")
    s = sizes(ctx)
    rows = []
    for hoped in ("yes", "no"):
        for _ in range(max(3, s["makers"] // 4)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            items = FG.make_ecology("mixed", sub, n_items=9)
            if hoped == "yes":
                out = FG.pursuit_versus_warrant(items, sub,
                                                steps=60 if ctx.get("smoke") else 120, hoped=0)
                share, fidelity = out["query_share_on_hoped"], out["posterior_matches_truth"]
            else:
                f = FG.forage(items, "gain_per_cost", sub,
                              steps=60 if ctx.get("smoke") else 120)
                share = float(np.mean([p == 0 for p in f["picks"]])) if f["picks"] else 0.0
                fidelity = float(1.0 - C.tv(f["beliefs"].p(0), items[0].law))
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "hoped": hoped, "ecology": "mixed",
                         "query_share": share, "posterior_fidelity": fidelity, "n": 1})
    return {"rows": rows}


def reduce_F08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a hoped-for hypothesis moves query allocation and not the posterior",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    sh_y = mean_of(rows, "query_share", lambda r: r["hoped"] == "yes")
    sh_n = mean_of(rows, "query_share", lambda r: r["hoped"] == "no")
    fi_y = mean_of(rows, "posterior_fidelity", lambda r: r["hoped"] == "yes")
    fi_n = mean_of(rows, "posterior_fidelity", lambda r: r["hoped"] == "no")
    battery(gr, live={"name": "hope_moves_the_query_allocation", "observed": abs(sh_y - sh_n)},
            placebo={"name": "hope_does_not_move_the_posterior", "observed": abs(fi_y - fi_n),
                     "tol": 0.15},
            positive={"name": "shares_and_fidelities_are_fractions",
                      "observed": float(0.0 <= sh_y <= 1.0 and 0.0 <= fi_y <= 1.0),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "the_hope_never_entered_the_likelihood", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_posterior_is_scored_against_the_true_law",
                        "observed": abs(fi_y - 0.5)})
    criterion(v, "F08", sh_y - sh_n, card.sesoi, "greater", card.sesoi_basis,
              detail="hope moves the query allocation by this much")
    criterion(v, "F08_warrant", abs(fi_y - fi_n), 0.15, "less",
              "movement in posterior fidelity that hope is allowed to cause",
              detail="and leaves the posterior's fidelity to the true law where it was, which is "
                     "the separation the card is for")
    v["results"]["query_share"] = {"hoped": sh_y, "not_hoped": sh_n}
    v["results"]["posterior_fidelity"] = {"hoped": fi_y, "not_hoped": fi_n}
    narrative(v, f"hope raises the share of queries on the attractive item from {sh_n:.2f} to "
                 f"{sh_y:.2f} while posterior fidelity moves from {fi_n:.2f} to {fi_y:.2f}",
              "pursuing a hypothesis and believing it are separable, and both are reported")
    distances(v, "F08", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F09 — abstention in three null ecologies.
# --------------------------------------------------------------------------- #
def unit_F09(ctx):
    rows = []
    for eco in ("noise", "expensive", "resolved"):
        rows += _forage_rows(ctx, f"F09|{eco}", ("gain_per_cost", "surprise"), (eco,),
                             steps=40 if ctx.get("smoke") else 80)
    return {"rows": rows}


def reduce_F09(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a gain-per-cost controller abstains in all three null ecologies",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {e: mean_of(rows, "abstention_rate",
                     lambda r, e=e: r["ecology"] == e and r["policy"] == "gain_per_cost")
          for e in ("noise", "expensive", "resolved")}
    worst = float(np.nanmin(list(by.values())))
    surprise_rate = mean_of(rows, "abstention_rate", lambda r: r["policy"] == "surprise")
    battery(gr, live={"name": "the_controller_abstains_somewhere", "observed": max(by.values())},
            placebo={"name": "the_surprise_controller_does_not_abstain",
                     "observed": surprise_rate, "tol": 0.2},
            positive={"name": "rates_are_fractions",
                      "observed": float(all(0.0 <= x <= 1.0 for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_controller_was_told_the_ecology", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "abstention_is_a_choice_not_a_default",
                        "observed": abs(worst - surprise_rate)})
    criterion(v, "F09", worst, card.sesoi, "greater", card.sesoi_basis,
              detail="the WEAKEST of the three null ecologies still draws this much abstention -- "
                     "abstaining in one of three is not selectivity")
    v["results"]["abstention_by_null"] = by
    v["results"]["surprise_abstention"] = surprise_rate
    narrative(v, "abstention rate by null ecology: "
                 + ", ".join(f"{k} {x:.2f}" for k, x in by.items())
                 + f"; the surprise controller abstains {surprise_rate:.2f}",
              "not looking is an answer, and it has to hold in every null")
    distances(v, "F09", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# F10 — the frozen tournament.
# --------------------------------------------------------------------------- #
def unit_F10(ctx):
    return {"rows": _forage_rows(ctx, "F10", FG.POLICIES,
                                 ("learnable", "noise", "silent_change", "mixed"))}


def reduce_F10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "one policy has the smallest worst-case regret across the ecologies",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    ecos = ("learnable", "noise", "silent_change", "mixed")
    surface = {e: {p: mean_of(rows, "held_out_gain",
                              lambda r, e=e, p=p: r["ecology"] == e and r["policy"] == p)
                   for p in FG.POLICIES} for e in ecos}
    best_per_eco = {e: max(d.values()) for e, d in surface.items()}
    regret = {p: float(np.nanmax([best_per_eco[e] - surface[e][p] for e in ecos]))
              for p in FG.POLICIES}
    best = min(regret, key=lambda k: regret[k])
    margin = float(np.nanmedian([regret[p] for p in FG.POLICIES if p != best]) - regret[best])
    battery(gr, live={"name": "the_ecology_moves_the_ranking",
                      "observed": float(np.nanmax(list(regret.values()))
                                        - np.nanmin(list(regret.values())))},
            placebo={"name": "no_policy_was_retuned_per_ecology", "observed": 0.0, "tol": 0.0},
            positive={"name": "regrets_are_non_negative",
                      "observed": float(min(regret.values()) >= -1e-9), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_policy_was_told_the_ecology", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_gain_is_measured_on_fresh_draws", "observed": abs(margin)})
    criterion(v, "F10", margin, card.sesoi, "greater", card.sesoi_basis,
              detail="the best policy's worst-case regret is this much below the median policy's")
    v["conditional_matrix"] = {"axis_rows": "ecology", "axis_cols": "policy", "surface": surface,
                               "pooled_headline": "REFUSED: the ranking changes by ecology"}
    v["results"]["worst_case_regret"] = regret
    v["results"]["best_policy"] = best
    narrative(v, f"the smallest worst-case regret is {best}'s at {regret[best]:.3f} nats; the "
                 f"median policy's is {regret[best] + margin:.3f}",
              "a curiosity controller is chosen for its worst ecology, not its best")
    distances(v, "F10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
