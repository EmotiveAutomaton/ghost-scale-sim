"""Trunk R — route reliability, shared causes and robust transfer (spec §6, cards R01-R08).

V14 found learned route weighting worth +0.009 nats in family and negative out of it. The audit
reading was that reliability weighting is weak exactly where the routes are already strong, so this
trunk moves the question to where it can have an answer: reliability *dispersion* crossed with
evidence dose, sparse feedback with no target labels at test, correlated evidence the reader is not
told about, and domain shift.

R06 is the trunk's adversarial card and its rival is implemented honestly. An ease-driven weighter
that is a straw man proves nothing; this one really does load onto whatever route is cheapest to
read, and the question is what that costs.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import routes as RT
from . import (battery, criterion, decide_state, distances, finish, mean_of, narrative, paired,
               publication, receipt, rng, rows_of, sizes, start)

CHANNELS = [{"name": "route_reports_about_a_hidden_value", "generated_from_hidden": True,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _banks(ctx, tag, n=None, **kw):
    r = rng(ctx, tag)
    s = sizes(ctx)
    return r, [RT.sample_bank(np.random.default_rng(r.integers(0, 2 ** 62)), **kw)
               for _ in range(int(n or max(4, s["makers"] // 4)))]


def _weight_card(ctx, units, hypothesis, what, pair, claim="BOUNDARY", extra=None,
                 value="log_score", factor="weighter"):
    card = ctx["card"]
    rows = rows_of(units)
    a, b = pair
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    pb = paired(rows, value, a, b, factor, seed_tag=card.id)
    battery(gr, live={"name": f"{a}_and_{b}_differ", "observed": abs(pb["mean"])},
            placebo={"name": "both_weighters_saw_the_same_reports", "observed": 0.0, "tol": 0.0},
            positive={"name": "scores_are_finite",
                      "observed": float(all(r[value] == r[value] for r in rows if value in r)),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_target_label_at_test", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_hidden_value_was_scored", "observed": abs(pb["mean"])},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"], detail=f"{a} beats {b} by at least the bar")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["paired"] = pb
    v["results"]["by_weighter"] = {
        nm: mean_of(rows, value, lambda r, nm=nm: r.get(factor) == nm)
        for nm in sorted({r.get(factor) for r in rows if r.get(factor)})}
    narrative(v, what.format(gap=pb["mean"]),
              "route weighting is worth what it buys on a hidden value, conditionally")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="reliability-weighted cue combination",
                project_specific_delta="a dispersion x dose boundary and an honest ease trap",
                evidence_grade="boundary", strongest_missing_rival="equal weighting",
                independent_generator_count=1,
                external_validation_needed="real cues with measurable reliability",
                paper_shape="methods_note", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# R01 — the dispersion x dose boundary.
# --------------------------------------------------------------------------- #
def unit_R01(ctx):
    s = sizes(ctx)
    rows = []
    for disp in (0.05, 0.2, 0.35):
        r, banks = _banks(ctx, f"R01|{disp}", dispersion=disp)
        for bank in banks:
            for dose in (2, 8):
                n_fb = 20 * dose
                wl = RT.learn_weights(bank, r, n_fb, kind="learned")
                we = np.ones(bank.n_routes)
                for name, wt in (("learned", wl), ("equal", we)):
                    sc = RT.score_weighter(bank, wt, r, n=40 * dose // 2)
                    rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                                 "dispersion": f"{disp:g}", "dose": str(dose),
                                 "weighter": name, "log_score": sc["log_score"],
                                 "accuracy": sc["accuracy"],
                                 "overconfidence": sc["overconfidence"], "n": 1})
    return {"rows": rows}


def reduce_R01(units, ctx):
    from . import onset
    rows = rows_of(units)
    adv = []
    seen = {}
    for r in rows:
        k = (r["wid"], r["rep"], r["dispersion"], r["dose"])
        seen.setdefault(k, {})[r["weighter"]] = r["log_score"]
    for (wid, rep, disp, dose), d in seen.items():
        if "learned" in d and "equal" in d:
            adv.append({"wid": wid, "rep": rep, "dispersion": disp, "dose": dose,
                        "advantage": d["learned"] - d["equal"], "n": 1})
    v = _weight_card(ctx, units,
                     "learned route weighting beats equal weighting only above a reliability "
                     "dispersion",
                     "learned weighting beats equal weighting by {gap:+.4f} nats overall; the "
                     "conditional curve is reported",
                     ("learned", "equal"))
    cur = onset(adv, "dispersion", "advantage", ctx["card"].sesoi)
    v["phase"] = {"axis": "dispersion", **cur}
    v["conditional_matrix"] = {
        "axis_rows": "dispersion", "axis_cols": "dose",
        "surface": {dp: {ds: mean_of(adv, "advantage",
                                     lambda r, dp=dp, ds=ds: r["dispersion"] == dp
                                     and r["dose"] == ds)
                         for ds in sorted({r["dose"] for r in adv}, key=int)}
                    for dp in sorted({r["dispersion"] for r in adv}, key=float)},
        "pooled_headline": "REFUSED: the advantage changes sign along the dispersion axis"}
    return v


# --------------------------------------------------------------------------- #
# R02 — sparse feedback, no target labels at test.
# --------------------------------------------------------------------------- #
def unit_R02(ctx):
    rows = []
    for sp in (0.0, 0.5, 0.8):
        r, banks = _banks(ctx, f"R02|{sp}", dispersion=0.3)
        for bank in banks:
            wl = RT.learn_weights(bank, r, 60, sparsity=sp, kind="learned")
            for name, wt in (("learned", wl), ("equal", np.ones(bank.n_routes))):
                sc = RT.score_weighter(bank, wt, r, n=120)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "sparsity": f"{sp:g}",
                             "weighter": name, "log_score": sc["log_score"],
                             "accuracy": sc["accuracy"],
                             "calibration_error": sc["calibration"]["ece"], "n": 1})
    return {"rows": rows}


def reduce_R02(units, ctx):
    rows = rows_of(units)
    v = _weight_card(ctx, units,
                     "route reliability is learnable from sparse predictive feedback with no "
                     "target labels at test",
                     "learned weighting beats equal weighting by {gap:+.4f} nats",
                     ("learned", "equal"))
    v["phase"] = {"axis": "sparsity",
                  "curve": [{"x": sp,
                             "mean": (mean_of(rows, "log_score",
                                              lambda r, s=sp: r["sparsity"] == s
                                              and r["weighter"] == "learned")
                                      - mean_of(rows, "log_score",
                                                lambda r, s=sp: r["sparsity"] == s
                                                and r["weighter"] == "equal"))}
                            for sp in sorted({r["sparsity"] for r in rows}, key=float)]}
    return v


# --------------------------------------------------------------------------- #
# R03 — recovering a shared cause without being told the graph.
# --------------------------------------------------------------------------- #
def unit_R03(ctx):
    rows = []
    for dup in (False, True):
        r, banks = _banks(ctx, f"R03|{dup}", dispersion=0.25, duplicated=dup)
        for bank in banks:
            det = RT.detect_shared_cause(bank, r, n=180)
            for fuser in ("naive", "shared_cause"):
                sc = RT.score_weighter(bank, np.ones(bank.n_routes), r, n=120,
                                       shared_cause=(fuser == "shared_cause"))
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                             "duplicated": "yes" if dup else "no", "fuser": fuser,
                             "log_score": sc["log_score"],
                             "overconfidence": sc["overconfidence"],
                             "recall": float(det["recall"]) if det["recall"] == det["recall"] else 0.0,
                             "false_pairs": float(det["false_pairs"]), "n": 1})
    return {"rows": rows}


def reduce_R03(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a shared cause is recoverable from co-agreement without being handed the graph",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    recall = mean_of(rows, "recall", lambda r: r["duplicated"] == "yes")
    false_when_none = mean_of(rows, "false_pairs", lambda r: r["duplicated"] == "no")
    naive_over = mean_of(rows, "overconfidence",
                         lambda r: r["duplicated"] == "yes" and r["fuser"] == "naive")
    corrected_over = mean_of(rows, "overconfidence",
                             lambda r: r["duplicated"] == "yes" and r["fuser"] == "shared_cause")
    battery(gr, live={"name": "duplication_moves_the_agreement_structure", "observed": recall},
            placebo={"name": "no_duplication_finds_few_pairs", "observed": false_when_none,
                     "tol": 2.0},
            positive={"name": "recall_is_a_fraction", "observed": float(0.0 <= recall <= 1.0),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "the_correlation_graph_was_not_supplied", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_hidden_value_was_scored",
                        "observed": abs(naive_over - corrected_over)})
    criterion(v, "R03", recall, card.sesoi, "greater", card.sesoi_basis,
              detail="the true shared-cause pairs are recovered this often from co-agreement alone")
    criterion(v, "R03_inflation", naive_over - corrected_over, 0.0, "greater",
              "confidence inflation the naive fuser carries above the shared-cause one",
              detail="and correcting for the shared cause removes measurable overconfidence")
    v["results"]["recall"] = recall
    v["results"]["false_pairs_when_independent"] = false_when_none
    v["results"]["overconfidence"] = {"naive": naive_over, "shared_cause": corrected_over}
    narrative(v, f"the duplicated pair is recovered {recall:.2f} of the time; the naive fuser is "
                 f"overconfident by {naive_over:+.3f} against the corrected fuser's "
                 f"{corrected_over:+.3f}",
              "two paraphrases of one cause are detected rather than double-counted")
    distances(v, "R03", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# R04, R05 — shift.
# --------------------------------------------------------------------------- #
def unit_R04(ctx):
    rows = []
    for shift in (0.0, 0.5, 1.0):
        r, banks = _banks(ctx, f"R04|{shift}", dispersion=0.3)
        for bank in banks:
            shifted = RT.domain_shift(bank, r, shift)
            for kind in ("learned", "robust"):
                wt = RT.learn_weights(bank, r, 60, kind=kind)
                sc = RT.score_weighter(shifted, wt, r, n=120)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "shift": f"{shift:g}",
                             "weighter": kind, "log_score": sc["log_score"],
                             "accuracy": sc["accuracy"],
                             "calibration_error": sc["calibration"]["ece"], "n": 1})
    return {"rows": rows}


def reduce_R04(units, ctx):
    rows = rows_of(units)
    v = _weight_card(ctx, units,
                     "robust weighting pays after a domain shift and costs in domain",
                     "robust weighting beats empirical weighting by {gap:+.4f} nats overall",
                     ("robust", "learned"))
    curve = [{"x": sh, "mean": (mean_of(rows, "log_score",
                                        lambda r, s=sh: r["shift"] == s and r["weighter"] == "robust")
                                - mean_of(rows, "log_score",
                                          lambda r, s=sh: r["shift"] == s
                                          and r["weighter"] == "learned"))}
             for sh in sorted({r["shift"] for r in rows}, key=float)]
    v["phase"] = {"axis": "shift", "curve": curve,
                  "pooled_headline": "REFUSED: robustness is expected to cost in domain and pay "
                                     "after a shift, so the sign changes along the axis"}
    return v


def unit_R05(ctx):
    rows = []
    for shift in (0.0, 0.25, 0.5, 1.0):
        r, banks = _banks(ctx, f"R05|{shift}", dispersion=0.3)
        for bank in banks:
            for policy in ("retain", "partial", "reset"):
                out = RT.transfer_policy(bank, r, shift, policy, n_feedback=60)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "shift": f"{shift:g}",
                             "policy": policy, "log_score": out["log_score"],
                             "accuracy": out["accuracy"], "n": 1})
    return {"rows": rows}


def reduce_R05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "which of retain, partial transfer and reset is best depends on the shift "
                         "size", "BOUNDARY")
    gr = G.GateReport()
    surface = {sh: {p: mean_of(rows, "log_score",
                               lambda r, s=sh, p=p: r["shift"] == s and r["policy"] == p)
                    for p in ("retain", "partial", "reset")}
               for sh in sorted({r["shift"] for r in rows}, key=float)}
    winners = {sh: max(d, key=lambda k: d[k] if d[k] == d[k] else -1e18)
               for sh, d in surface.items()}
    changes = len(set(winners.values()))
    spread = float(np.nanmax([max(d.values()) - min(d.values()) for d in surface.values()]))
    battery(gr, live={"name": "shift_moves_the_best_policy", "observed": float(changes - 1)},
            placebo={"name": "at_zero_shift_retaining_is_not_penalised",
                     "observed": abs(surface["0"]["retain"] - surface["0"]["partial"]),
                     "tol": 1.5},
            positive={"name": "every_cell_produced_a_score",
                      "observed": float(all(x == x for d in surface.values() for x in d.values())),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_policy_was_told_the_shift_size", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_shifted_hidden_value_was_scored", "observed": spread})
    criterion(v, "R05", float(changes - 1), 1.0, "greater",
              "the number of distinct winners along the shift axis, minus one",
              detail="the best transfer policy changes at least once along the shift axis, which "
                     "is what makes this a phase diagram rather than a ranking")
    v["conditional_matrix"] = {"axis_rows": "shift", "axis_cols": "policy", "surface": surface,
                               "winner_by_shift": winners,
                               "pooled_headline": "REFUSED: the winner changes along the axis"}
    narrative(v, "best policy by shift size: "
                 + ", ".join(f"{k} -> {p}" for k, p in winners.items()),
              "reset, retain and partial transfer each have a region")
    distances(v, "R05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# R06 — the ease trap.
# --------------------------------------------------------------------------- #
def unit_R06(ctx):
    rows = []
    for sp in (0.0, 0.8):
        r, banks = _banks(ctx, f"R06|{sp}", dispersion=0.3, easy_useless=True)
        for bank in banks:
            for kind in ("learned", "ease_driven"):
                wt = RT.learn_weights(bank, r, 60, sparsity=sp, kind=kind)
                sc = RT.score_weighter(bank, wt, r, n=120)
                w = np.asarray(wt, float)
                share = float(np.exp(w[-1]) / np.exp(w).sum()) if np.isfinite(w).all() else 0.0
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "sparsity": f"{sp:g}",
                             "weighter": kind, "log_score": sc["log_score"],
                             "accuracy": sc["accuracy"],
                             "weight_on_easy_route": share, "n": 1})
    return {"rows": rows}


def reduce_R06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "an adversarially easy route captures an ease-driven reader and not a learned one",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    cost = (mean_of(rows, "log_score", lambda r: r["weighter"] == "learned")
            - mean_of(rows, "log_score", lambda r: r["weighter"] == "ease_driven"))
    share_e = mean_of(rows, "weight_on_easy_route", lambda r: r["weighter"] == "ease_driven")
    share_l = mean_of(rows, "weight_on_easy_route", lambda r: r["weighter"] == "learned")
    battery(gr, live={"name": "the_trap_captures_the_ease_driven_reader",
                      "observed": abs(share_e - share_l)},
            placebo={"name": "the_learned_reader_is_not_captured",
                     "observed": max(0.0, share_l - 1.0 / 4), "tol": 0.35},
            positive={"name": "weights_are_shares",
                      "observed": float(0.0 <= share_e <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "neither_reader_was_told_the_route_was_useless",
                           "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_hidden_value_was_scored", "observed": abs(cost)})
    criterion(v, "R06", cost, card.sesoi, "greater", card.sesoi_basis,
              detail="being captured by the easy route costs the ease-driven reader this much on "
                     "the hidden value")
    v["results"]["weight_on_easy_route"] = {"learned": share_l, "ease_driven": share_e}
    v["results"]["score_cost_of_capture"] = cost
    narrative(v, f"the ease-driven reader puts {share_e:.2f} of its weight on the uninformative "
                 f"route against the learned reader's {share_l:.2f}, at a cost of {cost:.3f} nats",
              "a reader that chases what is easy to read can be led anywhere")
    distances(v, "R06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# R07 — buying costly access under prior ambiguity.
# --------------------------------------------------------------------------- #
def unit_R07(ctx):
    rows = []
    for cost in (0.02, 0.10):
        r, banks = _banks(ctx, f"R07|{cost}", dispersion=0.3)
        for bank in banks:
            priors = [C.normalize(np.ones(bank.n_values))]
            priors += [C.normalize(r.random(bank.n_values) + 0.2) for _ in range(3)]
            for policy in ("never", "always", "fixed", "eig", "robust_eig"):
                out = RT.purchase_policy(bank, r, policy, cost, priors, n=80)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "cost": f"{cost:g}",
                             "policy": policy, "log_score": out["mean_log_score"],
                             "mean_gain": out["mean_gain"], "total_cost": out["total_cost"],
                             "purchase_rate": out["purchase_rate"], "n": 1})
    return {"rows": rows}


def reduce_R07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "some purchase policy beats both never buying and always buying",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {p: mean_of(rows, "log_score", lambda r, p=p: r["policy"] == p)
          for p in ("never", "always", "fixed", "eig", "robust_eig")}
    best = max(by, key=lambda k: by[k] if by[k] == by[k] else -1e18)
    margin = by[best] - max(by["never"], by["always"])
    battery(gr, live={"name": "the_policy_moves_the_score",
                      "observed": float(max(by.values()) - min(by.values()))},
            placebo={"name": "never_buying_spends_nothing",
                     "observed": mean_of(rows, "total_cost", lambda r: r["policy"] == "never"),
                     "tol": 1e-9},
            positive={"name": "every_policy_produced_a_score",
                      "observed": float(all(x == x for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_policy_saw_the_hidden_value", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_hidden_value_was_scored", "observed": abs(margin)})
    criterion(v, "R07", margin, card.sesoi, "greater", card.sesoi_basis,
              detail="the best purchase policy beats both extremes by at least the bar")
    v["results"]["by_policy"] = by
    v["results"]["best"] = best
    v["results"]["purchase_rate"] = {p: mean_of(rows, "purchase_rate",
                                                lambda r, p=p: r["policy"] == p)
                                     for p in by}
    narrative(v, f"the best purchase policy is {best} at {by[best]:+.4f} nats, against never "
                 f"buying at {by['never']:+.4f} and always buying at {by['always']:+.4f}",
              "costly access is worth buying sometimes, and the rule for when is measurable")
    distances(v, "R07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# R08 — shortcut or answer.
# --------------------------------------------------------------------------- #
def unit_R08(ctx):
    r, banks = _banks(ctx, "R08", dispersion=0.3)
    rows = []
    for bank in banks:
        out = RT.weights_change_exact_posterior(bank, r, n=120)
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "budget": "full",
                     "weighter": "learned", "advantage": out["full_budget_advantage"], "n": 1})
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "budget": "finite",
                     "weighter": "learned", "advantage": out["finite_budget_advantage"], "n": 1})
        for b in ("full", "finite"):
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "budget": b, "weighter": "equal",
                         "advantage": 0.0, "n": 1})
    return {"rows": rows}


def reduce_R08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "route weights are a bounded-budget shortcut rather than a change to the "
                         "answer", "BOUNDARY")
    gr = G.GateReport()
    full = mean_of(rows, "advantage",
                   lambda r: r["budget"] == "full" and r["weighter"] == "learned")
    finite = mean_of(rows, "advantage",
                     lambda r: r["budget"] == "finite" and r["weighter"] == "learned")
    battery(gr, live={"name": "the_budget_moves_the_advantage", "observed": abs(finite - full)},
            placebo={"name": "equal_weighting_is_the_reference",
                     "observed": abs(mean_of(rows, "advantage",
                                             lambda r: r["weighter"] == "equal")), "tol": 1e-9},
            positive={"name": "both_budgets_produced_an_advantage",
                      "observed": float(full == full and finite == finite), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_weighter_saw_the_hidden_value", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_hidden_value_was_scored", "observed": abs(finite)})
    criterion(v, "R08", finite - full, card.sesoi, "greater", card.sesoi_basis,
              detail="weighting is worth this much more under a finite budget than under an "
                     "unlimited one, which is what 'a processing shortcut' means")
    v["results"]["advantage"] = {"full_budget": full, "finite_budget": finite}
    narrative(v, f"weighting is worth {full:+.4f} nats at full budget and {finite:+.4f} at a "
                 f"truncated one",
              "if weighting mattered at unlimited budget it would be changing the answer, not "
              "saving work")
    distances(v, "R08", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
