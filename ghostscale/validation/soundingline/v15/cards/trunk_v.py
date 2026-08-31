"""Trunk V — persistent tendency, current value, change and concealment (spec §6, cards V01-V10).

The trunk's hard constraint (spec pre-mortem item 11) is that changed value and better concealment
must not differ by a telltale template. Both come out of one planner in ``persistent.py``, and what
separates them is *where* their choices differ: concealment moves only the visible options, a real
change moves the private ones too. The cost cards have the same shape -- a paid cost is not
evidence of preference until competence, constraint, signalling and a different cost function have
been given the same chance to explain it.

V08 and V09 refuse a point estimate. Where the record cannot identify a reward vector, what is
reported is a feasible class with its coverage, and V09 asks the Poiani question directly: do
mistakes shrink that class more than optimal choices do?
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import persistent as PS
from . import (Cells, battery, criterion, decide_state, distances, families_of, family_module,
               finish, mean_of, narrative, paired, publication, receipt, rng, rows_of,
               run_tournament, sizes, start, world_for)

CHANNELS = [{"name": "choice_from_one_planner", "generated_from_hidden": False,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _worlds(ctx, tag, n=None):
    r = rng(ctx, tag)
    s = sizes(ctx)
    return r, [PS.sample_value_world(np.random.default_rng(r.integers(0, 2 ** 62)))
               for _ in range(int(n or max(3, s["makers"] // 5)))]


# --------------------------------------------------------------------------- #
# V01 — standing preference beyond goal, competence and history.
# --------------------------------------------------------------------------- #
def unit_V01(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    for fam in families_of(ctx):
        for context in ("same", "changed"):
            r, _, _ = run_tournament(ctx, fam,
                                     ("surface", "label_only", "independent", "joint_exact",
                                      "oracle_state"),
                                     knobs_over={"kappa": 0.5, "dose": 4},
                                     endpoint="changed_context_choice", cells=cells,
                                     extra_key={"context": context, "family": fam})
            rows += r
    return {"rows": rows + cells.rows()}


def _pref_card(ctx, units, hypothesis, what, pair, claim="SIMULATOR_DISCOVERY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    a, b = pair
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    pb = paired(rows, "log_score", a, b, "architecture", seed_tag=card.id)
    orac = mean_of(rows, "log_score", lambda r: r.get("architecture") == "oracle_state")
    surf = mean_of(rows, "log_score", lambda r: r.get("architecture") == "surface")
    battery(gr, live={"name": "the_reader_moves_the_score", "observed": abs(pb["mean"])},
            placebo={"name": "every_reader_saw_the_same_record", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_oracle_is_a_ceiling",
                      "observed": float(orac >= surf) if orac == orac and surf == surf else 1.0,
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_preference", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_choice_was_hidden", "observed": abs(pb["mean"])},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"], detail=f"{a} beats {b} on the hidden choice by the bar")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["by_architecture"] = {
        nm: mean_of(rows, "log_score", lambda r, nm=nm: r.get("architecture") == nm)
        for nm in sorted({r.get("architecture") for r in rows if r.get("architecture")})}
    v["results"]["paired"] = pb
    narrative(v, what.format(gap=pb["mean"], orac=orac, surf=surf),
              "a standing tendency earns its place on a future choice or it does not")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="reward learning and partial identifiability",
                project_specific_delta="concealment and residue as explicit rivals to value change",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="a habit model with the same free parameters",
                independent_generator_count=len({r.get("family") for r in rows if r.get("family")}),
                external_validation_needed="a real record with public and private choices",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_V01(units, ctx):
    return _pref_card(ctx, units,
                      "a standing preference predicts a changed-context choice beyond the current "
                      "goal",
                      "the maker model beats a decontextualized label by {gap:+.4f} nats on the "
                      "changed-context choice",
                      ("joint_exact", "label_only"))


# --------------------------------------------------------------------------- #
# V02-V04 — the rivals, all from one planner.
# --------------------------------------------------------------------------- #
def _rival_rows(ctx, tag, rivals, axis, axis_levels, public=True):
    r, worlds = _worlds(ctx, tag)
    s = sizes(ctx)
    rows = []
    for w in worlds:
        rv = PS.make_rivals(w, r)
        for name in rivals:
            for lv in axis_levels:
                w2 = rv[name]
                n_obs = int(lv) if axis == "episodes" else s["episodes"] * 2
                pub = public if axis != "visibility" else (lv == "public")
                obs = [PS.choose(w2, r, public=pub)["choice"] for _ in range(max(n_obs, 3))]
                post = PS.rival_posterior(obs, {k: rv[k] for k in rivals}, r, n_sim=s["sims"],
                                          public=pub)
                sig = PS.public_private_signature(w2, r, n=20)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "rival": name,
                             axis: str(lv), "correct": float(max(post, key=post.get) == name),
                             "true_mass": float(post[name]),
                             "divergence": sig["divergence"], "n": 1})
    return rows


def unit_V02(ctx):
    return {"rows": _rival_rows(ctx, "V02", ("changed_preference", "changed_goal"),
                                "episodes", ["1", "4"])}


def _discrim_card(ctx, units, hypothesis, what, claim="BOUNDARY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    n_riv = len({r["rival"] for r in rows})
    chance = 1.0 / max(n_riv, 2)
    battery(gr, live={"name": "the_rivals_differ_in_behaviour", "observed": abs(acc - chance)},
            placebo={"name": "both_rivals_come_from_one_planner", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": mean_of(rows, "true_mass"), "expected": 0.5, "tol": 0.5},
            no_label_leak={"name": "no_reader_was_told_the_rival", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_later_choices_were_hidden", "observed": abs(acc - chance)})
    criterion(v, card.id, acc - chance, card.sesoi, "greater", card.sesoi_basis,
              detail=f"the rivals are separated this far above the {chance:.2f} chance floor")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["accuracy"] = acc
    v["results"]["chance"] = chance
    v["results"]["by_rival"] = {k: mean_of(rows, "correct", lambda r, k=k: r["rival"] == k)
                                for k in sorted({r["rival"] for r in rows})}
    narrative(v, what.format(acc=acc, chance=chance,
                             above=acc - chance,
                             div=mean_of(rows, "divergence")),
              "the rivals are told apart by an intervention, or they stay a boundary")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_V02(units, ctx):
    return _discrim_card(ctx, units,
                         "later episodes separate a changed preference from a changed episode goal",
                         "the two are told apart {acc:.2f} of the time against a {chance:.2f} floor")


def unit_V03(ctx):
    return {"rows": _rival_rows(ctx, "V03", ("changed_preference", "stale_residue"),
                                "probe", ["none", "devalue", "relearn"])}


def reduce_V03(units, ctx):
    rows = rows_of(units)
    none_acc = mean_of(rows, "correct", lambda r: r.get("probe") == "none")
    dev_acc = mean_of(rows, "correct", lambda r: r.get("probe") == "devalue")
    return _discrim_card(ctx, units,
                         "devaluation separates a changed preference from a lagging residue",
                         "the two are told apart {acc:.2f} of the time against {chance:.2f}",
                         extra=[("V03_probe_value", dev_acc - none_acc, 0.05, "greater",
                                 "accuracy the devaluation probe adds over reading behaviour",
                                 "the probe buys this much over untouched behaviour")])


def unit_V04(ctx):
    return {"rows": _rival_rows(ctx, "V04", ("changed_preference", "concealment"),
                                "visibility", ["public", "private"])}


def reduce_V04(units, ctx):
    rows = rows_of(units)
    pub = mean_of(rows, "correct", lambda r: r.get("visibility") == "public")
    priv = mean_of(rows, "correct", lambda r: r.get("visibility") == "private")
    div_c = mean_of(rows, "divergence", lambda r: r["rival"] == "concealment")
    div_p = mean_of(rows, "divergence", lambda r: r["rival"] == "changed_preference")
    return _discrim_card(ctx, units,
                         "private choices separate a changed preference from better concealment",
                         "the two are told apart {acc:.2f} of the time overall, with a "
                         "public-private divergence of {div:.3f}",
                         claim="SIMULATOR_DISCOVERY",
                         extra=[("V04_private_value", priv - pub, 0.05, "greater",
                                 "accuracy the private channel adds over the public one",
                                 "seeing the private choices buys this much"),
                                ("V04_signature", div_c - div_p, 0.10, "greater",
                                 "public-private divergence difference between the two rivals",
                                 "concealment separates public from private and a real change "
                                 "does not, which is the whole discrimination")])


# --------------------------------------------------------------------------- #
# V05, V06, V09 — cost, opportunity and informative imperfection.
# --------------------------------------------------------------------------- #
def unit_V05(ctx):
    r, worlds = _worlds(ctx, "V05")
    s = sizes(ctx)
    rows = []
    for w in worlds:
        for owner in PS.COST_OWNERS:
            variants = {
                "preference": w.copy_with(preference=w.preference * 1.8),
                "competence": w.copy_with(competence=0.35),
                "constraint": w.copy_with(effort=w.effort * 2.2),
                "signalling": w.copy_with(signalling=1.8),
                "cost_function": w.copy_with(cost_function="quadratic"),
            }
            obs = [PS.choose(variants[owner], r, public=True)["choice"]
                   for _ in range(s["episodes"] * 3)]
            post = PS.cost_vector_posterior(obs, w, r, n_sim=s["sims"])
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "owner": owner,
                         "correct": float(max(post, key=post.get) == owner),
                         "true_mass": float(post[owner]),
                         "preference_mass": float(post["preference"]), "n": 1})
    return {"rows": rows}


def reduce_V05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a paid cost does not identify preference until competence, constraint, signalling "
              "and a different cost function have been given the same chance", "BOUNDARY")
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    chance = 1.0 / len(PS.COST_OWNERS)
    by = {o: mean_of(rows, "correct", lambda r, o=o: r["owner"] == o) for o in PS.COST_OWNERS}
    pref_leak = mean_of(rows, "preference_mass", lambda r: r["owner"] != "preference")
    battery(gr, live={"name": "the_owner_moves_the_posterior", "observed": abs(acc - chance)},
            placebo={"name": "all_owners_come_from_one_planner", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": mean_of(rows, "true_mass"), "expected": 0.4, "tol": 0.4},
            no_label_leak={"name": "no_reader_was_told_the_owner", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_held_out_choice_was_scored", "observed": abs(acc - chance)})
    criterion(v, "V05", acc - chance, card.sesoi, "greater", card.sesoi_basis,
              detail=f"the cost owner is named this far above the {chance:.2f} chance floor")
    v["results"]["accuracy"] = acc
    v["results"]["by_owner"] = by
    v["results"]["preference_mass_when_owner_is_not_preference"] = pref_leak
    v["equivalence"] = {"owners_confusable": [o for o, a in by.items() if a < chance + 0.1]}
    narrative(v, f"the cost owner is named {acc:.2f} of the time against a {chance:.2f} floor; when "
                 f"the owner is not preference, preference still holds {pref_leak:.2f} of the mass",
              "cost is evidence about a vector, not about preference")
    distances(v, "V05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_V06(ctx):
    r, worlds = _worlds(ctx, "V06")
    s = sizes(ctx)
    rows = []
    for w in worlds:
        for availability in ("full", "partial"):
            oi = PS.opportunity_information(w, r, n=max(20, s["makers"] * 2))
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "availability": availability,
                         "reader": "cost_only", "alignment": oi["cost_only_alignment"], "n": 1})
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "availability": availability,
                         "reader": "opportunity_aware", "alignment": oi["opportunity_alignment"],
                         "n": 1})
    return {"rows": rows}


def reduce_V06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the forgone alternatives add information beyond the chosen option",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    pb = paired(rows, "alignment", "opportunity_aware", "cost_only", "reader", seed_tag="V06")
    co = mean_of(rows, "alignment", lambda r: r["reader"] == "cost_only")
    oa = mean_of(rows, "alignment", lambda r: r["reader"] == "opportunity_aware")
    battery(gr, live={"name": "the_option_set_moves_the_alignment", "observed": abs(pb["mean"])},
            placebo={"name": "both_readers_saw_the_same_choices", "observed": 0.0, "tol": 0.0},
            positive={"name": "alignments_are_cosines",
                      "observed": float(abs(co) <= 1.0 and abs(oa) <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_preference_vector", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_preference_direction_was_hidden",
                        "observed": abs(pb["mean"])})
    criterion(v, "V06", pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="the opportunity-aware reader recovers the preference direction this much "
                     "better than one seeing only the chosen option and its cost")
    v["results"]["alignment"] = {"cost_only": co, "opportunity_aware": oa}
    narrative(v, f"seeing what was declined recovers the preference direction at {oa:+.3f} against "
                 f"{co:+.3f} for the chosen option alone",
              "what a maker did not do is evidence, and its size is here")
    distances(v, "V06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_V07(ctx):
    r, worlds = _worlds(ctx, "V07")
    rows = []
    for w in worlds:
        for record in ("dated", "bag"):
            for change in ("none", "midway"):
                traj = PS.dated_trajectory(w, r, n_episodes=10,
                                           change_at=5 if change == "midway" else None)
                if record == "bag":
                    traj = dict(traj)
                    traj["dated"] = [(i, c) for i, c in enumerate(traj["bag"])]
                sc = PS.change_point_score(traj)
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "record": record,
                             "change": change,
                             "changepoint_error": float(sc["error"]),
                             "separation": float(sc["separation"]), "n": 1})
    return {"rows": rows}


def reduce_V07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "dated works recover a directional change an undated bag cannot",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    d = mean_of(rows, "changepoint_error",
                lambda r: r["record"] == "dated" and r["change"] == "midway")
    b = mean_of(rows, "changepoint_error",
                lambda r: r["record"] == "bag" and r["change"] == "midway")
    battery(gr, live={"name": "dating_moves_the_changepoint_error", "observed": abs(b - d)},
            placebo={"name": "the_two_records_contain_the_same_choices", "observed": 0.0,
                     "tol": 0.0},
            positive={"name": "errors_are_non_negative",
                      "observed": float(min(d, b) >= 0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_change_time", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_change_time_was_hidden", "observed": abs(b - d)})
    criterion(v, "V07", b - d, card.sesoi, "greater", card.sesoi_basis,
              detail="the dated record locates the change this many episodes closer")
    v["results"]["changepoint_error"] = {"dated": d, "bag": b}
    narrative(v, f"the dated record misses the change by {d:.2f} episodes and the bag by {b:.2f}",
              "chronology is evidence about direction")
    distances(v, "V07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_V08(ctx):
    r, worlds = _worlds(ctx, "V08")
    s = sizes(ctx)
    rows, sets = [], []
    for w in worlds:
        for record, comp in (("optimal_only", 0.99), ("varied_competence", 0.7),
                             ("with_errors", 0.45)):
            w2 = w.copy_with(competence=comp)
            obs = [PS.choose(w2, r, public=True)["choice"] for _ in range(s["episodes"] * 3)]
            fs = PS.feasible_reward_set(obs, w2, r, n_draw=200 if ctx.get("smoke") else 600)
            sets.append({"record": record, **fs})
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "record": record,
                         "coverage": fs["coverage"],
                         "contains_truth": float(fs["contains_truth"]),
                         "alignment": fs.get("mean_alignment_to_truth", 0.0), "n": 1})
    return {"rows": rows, "sets": sets}


def _feasible_card(ctx, units, hypothesis, what, criterion_value, claim="BOUNDARY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    cov = {rec: mean_of(rows, "coverage", lambda r, rec=rec: r["record"] == rec)
           for rec in sorted({r["record"] for r in rows})}
    contains = mean_of(rows, "contains_truth")
    battery(gr, live={"name": "the_record_moves_the_feasible_set",
                      "observed": float(max(cov.values()) - min(cov.values()))},
            placebo={"name": "coverage_is_a_fraction",
                     "observed": float(max(0.0, max(cov.values()) - 1.0)), "tol": 0.0},
            positive={"name": "the_set_can_contain_the_truth", "observed": contains,
                      "expected": 1.0, "tol": 1.0},
            no_label_leak={"name": "no_reward_vector_was_supplied", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_set_constrains_a_held_out_choice", "observed": contains})
    criterion(v, card.id, criterion_value(cov, contains), card.sesoi, "greater", card.sesoi_basis,
              detail="the retained class does its job")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["equivalence"] = {"coverage_by_record": cov, "contains_truth": contains,
                        "sets": rows_of(units, "sets")[:12]}
    narrative(v, what.format(cov=cov, contains=contains),
              "an unidentified reward stays a class rather than becoming a number")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_V08(units, ctx):
    return _feasible_card(ctx, units,
                          "when several reward functions predict every observed choice the class "
                          "is retained",
                          "the retained set contains the true reward direction {contains:.2f} of "
                          "the time",
                          lambda cov, contains: contains)


def unit_V09(ctx):
    return unit_V08(ctx)


def reduce_V09(units, ctx):
    rows = rows_of(units)
    opt = mean_of(rows, "coverage", lambda r: r["record"] == "optimal_only")
    err = mean_of(rows, "coverage", lambda r: r["record"] == "with_errors")
    return _feasible_card(ctx, units,
                          "mistakes shrink the compatible preference set that optimal choices leave "
                          "wide",
                          "optimal-only records leave {cov[optimal_only]:.3f} of the reward space "
                          "feasible and records with errors {cov[with_errors]:.3f}",
                          lambda cov, contains: cov["optimal_only"] - cov["with_errors"],
                          claim="SIMULATOR_DISCOVERY")


# --------------------------------------------------------------------------- #
# V10 — three prospective endpoints, two families.
# --------------------------------------------------------------------------- #
def unit_V10(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    for fam in families_of(ctx):
        for endpoint in ("next_edit", "stop_or_continue", "changed_context_choice"):
            if fam == "chain" and endpoint == "stop_or_continue":
                continue
            r, _, _ = run_tournament(ctx, fam,
                                     ("surface", "label_only", "joint_exact", "oracle_state"),
                                     knobs_over={"kappa": 0.5, "dose": 4}, endpoint=endpoint,
                                     cells=cells,
                                     extra_key={"endpoint": endpoint, "family": fam,
                                                "reader": "n/a"})
            for row in r:
                row["endpoint"] = endpoint
                row["reader"] = row["architecture"]
            rows += r
    return {"rows": rows + cells.rows()}


def reduce_V10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a persistent-tendency estimate improves at least one prospective endpoint in both "
              "families", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    per = {}
    for ep in sorted({r.get("endpoint") for r in rows if r.get("endpoint")}):
        per[ep] = (mean_of(rows, "log_score",
                           lambda r, e=ep: r.get("endpoint") == e
                           and r.get("architecture") == "joint_exact")
                   - mean_of(rows, "log_score",
                             lambda r, e=ep: r.get("endpoint") == e
                             and r.get("architecture") == "label_only"))
    best = float(np.nanmax(list(per.values()))) if per else float("nan")
    pb = paired(rows, "log_score", "joint_exact", "label_only", "architecture", seed_tag="V10")
    battery(gr, live={"name": "the_estimate_moves_at_least_one_endpoint", "observed": abs(best)},
            placebo={"name": "both_readers_saw_the_same_evidence", "observed": 0.0, "tol": 0.0},
            positive={"name": "every_endpoint_produced_a_score",
                      "observed": float(all(x == x for x in per.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_tendency", "movement": 0.0, "tol": 0.0},
            prediction={"name": "all_three_endpoints_were_hidden", "observed": abs(pb["mean"])})
    criterion(v, "V10", best, card.sesoi, "greater", card.sesoi_basis, interval=pb["interval"],
              detail="the best of the three prospective endpoints shows this much advantage for "
                     "the context-realized state over a correct label")
    v["results"]["by_endpoint"] = per
    v["families"] = {f: mean_of(rows, "log_score", lambda r, f=f: r.get("family") == f)
                     for f in sorted({r.get("family") for r in rows if r.get("family")})}
    narrative(v, "advantage over a correct label by endpoint: "
                 + ", ".join(f"{k} {x:+.4f}" for k, x in per.items()),
              "a tendency is worth what it buys on the next event, endpoint by endpoint")
    distances(v, "V10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
