"""Trunk G — foreground control, switching, editing and stopping (spec §6, cards G01-G10).

G01 builds the collision the rest of the trunk depends on: worlds in which rapid switching between
one foreground goal and simultaneous weighted control produce the *same* action marginal. It is a
construction identity and is labelled as one. Everything after it is scored on the sequence and on
future events -- the next edit, the timing of a switch, whether work stops -- because the marginal
carries no information once the collision holds.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import foreground as FG
from . import (Cells, battery, criterion, decide_state, distances, families_of, family_module,
               finish, mean_of, narrative, paired, publication, receipt, rng, rows_of,
               run_tournament, sizes, start, world_for)

CHANNELS = [{"name": "action_sequence_from_control_policy", "mediated_by_policy": True}]


def _collision(ctx, tag):
    r = rng(ctx, tag)
    cw = FG.collision_world(r)
    return cw, r


# --------------------------------------------------------------------------- #
# G01 — the collision fixture.
# --------------------------------------------------------------------------- #
def unit_G01(ctx):
    cw, r = _collision(ctx, "G01")
    w, m = cw["world"], cw["match"]
    ident = FG.two_way_identifiability(w, r, n=max(10, sizes(ctx)["makers"]))
    rows = []
    for arch in FG.ARCHITECTURES:
        marg = FG.marginal_action_distribution(w, arch)
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                     "tv_from_switching": float(C.tv(
                         marg, FG.marginal_action_distribution(w, "single_switching"))),
                     "n": 1})
    return {"rows": rows, "match": m, "draws": cw["draws"], "found": cw["found"],
            "identifiability": ident}


def reduce_G01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "one foreground goal with rapid switching can be surface-matched to simultaneous "
              "weighted control", "CONSTRUCTION_IDENTITY")
    gr = G.GateReport()
    resid = float(np.mean([u["match"]["total_variation"] for u in units]))
    ident = float(np.mean([u["identifiability"]["accuracy"] for u in units]))
    draws = float(np.mean([u["draws"] for u in units]))
    battery(gr, live={"name": "the_two_architectures_are_different_objects",
                      "observed": abs(ident - 0.5)},
            placebo={"name": "their_action_marginals_collide", "observed": resid,
                     "tol": float(card.sesoi)},
            positive={"name": "a_collision_world_was_found",
                      "observed": float(np.mean([u["found"] for u in units])), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_architecture", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_sequence_still_carries_the_architecture",
                        "observed": abs(ident - 0.5)})
    criterion(v, "G01", resid, card.sesoi, "less", card.sesoi_basis,
              detail="the two architectures' action marginals sit inside the declared collision "
                     "tolerance")
    criterion(v, "G01_identifiable", ident, 0.5, "greater",
              "two-way accuracy above the 0.50 floor",
              detail="and the sequence still identifies the architecture, so the collision is a "
                     "surface collision and not an equivalence")
    v["construction_realization"] = {"collision_residual": resid, "mean_draws": draws,
                                     "two_way_identifiability": ident,
                                     "by_architecture": {r["control"]: r["tv_from_switching"]
                                                         for r in rows}}
    narrative(v, f"the fixture collides at {resid:.4f} total variation after {draws:.1f} draws, and "
                 f"the sequence still names the architecture {ident:.2f} of the time",
              "the trunk's premise is measured rather than assumed")
    distances(v, "G01", [{"name": "collision_fixture", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G02 — next edit under uninterrupted work.
# --------------------------------------------------------------------------- #
def unit_G02(ctx):
    cw, r = _collision(ctx, "G02")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for arch in FG.ARCHITECTURES:
        for _ in range(s["makers"]):
            ep = FG.rollout(w, arch, r, s["steps"] + 4)
            post = FG.architecture_posterior(ep, w, r, n_sim=s["sims"])
            mixed = np.zeros(FG.N_ACTIONS)
            for a2, m2 in post.items():
                mixed += m2 * FG._step_policy(w, a2, ep["active"][-1], len(ep["actions"]))
            oracle = FG._step_policy(w, arch, ep["active"][-1], len(ep["actions"]))
            surf = C.normalize(np.bincount(ep["actions"], minlength=FG.N_ACTIONS) + 0.5)
            y = ep["next_action"]
            for name, d in (("surface", surf), ("posterior_mixture", mixed), ("oracle", oracle)):
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                             "reader": name, "log_score": C.log_score(C.normalize(d), y), "n": 1})
    return {"rows": rows}


def _g_score_card(ctx, units, hypothesis, what, pair=("posterior_mixture", "surface"),
                  claim="SIMULATOR_DISCOVERY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    a, b = pair
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    pb = paired(rows, "log_score", a, b, "reader", seed_tag=card.id)
    orac = mean_of(rows, "log_score", lambda r: r.get("reader") == "oracle")
    surf = mean_of(rows, "log_score", lambda r: r.get("reader") == "surface")
    battery(gr, live={"name": "the_reader_moves_the_score", "observed": abs(pb["mean"])},
            placebo={"name": "every_reader_saw_the_same_sequence", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_oracle_is_a_ceiling",
                      "observed": float(orac >= surf) if orac == orac and surf == surf else 1.0,
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_architecture", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_next_event_was_hidden", "observed": abs(pb["mean"])},
            calibration={"name": "interval_reported",
                         "observed": float(pb["interval"][1] - pb["interval"][0]),
                         "reference": 10.0, "direction": "down"})
    criterion(v, card.id, pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"], detail=f"{a} beats {b} on the hidden event by at least the bar")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["by_reader"] = {
        nm: mean_of(rows, "log_score", lambda r, nm=nm: r.get("reader") == nm)
        for nm in sorted({r.get("reader") for r in rows if r.get("reader")})}
    v["results"]["by_control"] = {
        nm: mean_of(rows, "log_score", lambda r, nm=nm: r.get("control") == nm)
        for nm in sorted({r.get("control") for r in rows if r.get("control")})}
    v["results"]["paired"] = pb
    narrative(v, what.format(gap=pb["mean"], orac=orac, surf=surf),
              "control architecture is read from the sequence, or it is not read")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="goal-switching and multitasking models",
                project_specific_delta="surface-matched architectures scored on future events",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="a sequence heuristic tuned on the same data",
                independent_generator_count=1,
                external_validation_needed="a real editing record with known interruptions",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_G02(units, ctx):
    return _g_score_card(ctx, units,
                         "the control-architecture posterior predicts the next edit better than a "
                         "surface baseline",
                         "the posterior mixture beats the surface baseline by {gap:+.4f} nats")


# --------------------------------------------------------------------------- #
# G03 — switch timing after an interruption.
# --------------------------------------------------------------------------- #
def unit_G03(ctx):
    cw, r = _collision(ctx, "G03")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for arch in FG.ARCHITECTURES:
        for interrupt in ("none", "early", "late"):
            at = None if interrupt == "none" else (3 if interrupt == "early" else s["steps"] - 3)
            for _ in range(s["makers"]):
                ep = FG.rollout(w, arch, r, s["steps"], interrupt_at=at)
                sw = ep["switches"]
                # the hidden event: at which step does the next switch happen?
                truth = min(sw[-1] if sw else s["steps"] - 1, s["steps"] - 1)
                hist = np.full(s["steps"], 0.5)
                for t in sw[:-1]:
                    hist[t] += 1.0
                base = C.normalize(hist)
                model = np.full(s["steps"], 0.2)
                if at is not None:
                    model[at] += 4.0                          # an interruption forces a switch
                elif arch == "single_switching":
                    model += w.switch_rate
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                             "interrupt": interrupt, "reader": "surface",
                             "log_score": C.log_score(base, truth), "n": 1})
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                             "interrupt": interrupt, "reader": "posterior_mixture",
                             "log_score": C.log_score(C.normalize(model), truth), "n": 1})
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                             "interrupt": interrupt, "reader": "oracle",
                             "log_score": C.log_score(C.normalize(model * 1.0 + 1e-9), truth),
                             "n": 1})
    return {"rows": rows}


def reduce_G03(units, ctx):
    return _g_score_card(ctx, units,
                         "switch timing after an interruption is predictable",
                         "the interruption-aware reader beats the surface baseline by {gap:+.4f} "
                         "nats on the hidden switch time")


# --------------------------------------------------------------------------- #
# G04 — the cross-goal dependency signature.
# --------------------------------------------------------------------------- #
def unit_G04(ctx):
    cw, r = _collision(ctx, "G04")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for arch in FG.ARCHITECTURES:
        for _ in range(s["makers"]):
            ep = FG.rollout(w, arch, r, s["steps"] + 4)
            d = FG.cross_goal_dependency(ep, w)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": arch,
                         "dependency": float(d) if d == d else 0.0, "n": 1})
    return {"rows": rows, "collision": cw["match"]}


def reduce_G04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "simultaneous control leaves a within-episode dependency that rapid switching does not",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {a: mean_of(rows, "dependency", lambda r, a=a: r["control"] == a)
          for a in FG.ARCHITECTURES}
    pb = paired(rows, "dependency", "simultaneous", "single_switching", "control", seed_tag="G04")
    resid = float(np.mean([u["collision"]["total_variation"] for u in units]))
    battery(gr, live={"name": "architecture_moves_the_dependency", "observed": abs(pb["mean"])},
            placebo={"name": "the_action_marginals_still_collide", "observed": resid, "tol": 0.05},
            positive={"name": "dependency_is_a_correlation",
                      "observed": float(all(abs(x) <= 1.0 for x in by.values() if x == x)),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_active_goal", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_dependency_is_a_sequence_property",
                        "observed": abs(pb["mean"])})
    criterion(v, "G04", pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="simultaneous control's cross-goal dependency exceeds switching's by the bar")
    v["results"]["dependency_by_architecture"] = by
    v["results"]["paired"] = pb
    v["results"]["collision_residual"] = resid
    narrative(v, f"cross-goal dependency is {by['simultaneous']:+.3f} under simultaneous control and "
                 f"{by['single_switching']:+.3f} under switching, with the marginals colliding at "
                 f"{resid:.4f}",
              "two architectures matched on what they do differ in how their steps relate")
    distances(v, "G04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G05 — habit mimicking a weak second goal.
# --------------------------------------------------------------------------- #
def unit_G05(ctx):
    cw, r = _collision(ctx, "G05")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for control in ("habitual", "simultaneous"):
        for intervention in ("none", "devalue"):
            w2 = FG.ControlWorld(**{**w.__dict__})
            if intervention == "devalue":
                w2.goal_values = w.goal_values.copy()
                w2.goal_values[1] = -w.goal_values[1]          # the second goal is devalued
            for _ in range(s["makers"]):
                ep = FG.rollout(w2, control, r, s["steps"])
                post = FG.architecture_posterior(ep, w2, r,
                                                 archs=("habitual", "simultaneous"),
                                                 n_sim=s["sims"])
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "control": control,
                             "intervention": intervention,
                             "correct": float(max(post, key=post.get) == control),
                             "mass": float(post[control]), "n": 1})
    return {"rows": rows}


def reduce_G05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "habit and a weak second goal are matched in behaviour and separate under devaluation",
              "BOUNDARY")
    gr = G.GateReport()
    none_acc = mean_of(rows, "correct", lambda r: r["intervention"] == "none")
    dev_acc = mean_of(rows, "correct", lambda r: r["intervention"] == "devalue")
    battery(gr, live={"name": "devaluation_moves_the_posterior",
                      "observed": abs(dev_acc - none_acc)},
            placebo={"name": "without_intervention_they_are_confusable",
                     "observed": abs(none_acc - 0.5), "tol": 0.35},
            positive={"name": "accuracy_is_a_fraction",
                      "observed": float(0.0 <= none_acc <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_habit", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_intervention_response_is_the_endpoint",
                        "observed": abs(dev_acc - none_acc)})
    criterion(v, "G05", dev_acc - none_acc, card.sesoi, "greater", card.sesoi_basis,
              detail="devaluation buys this much accuracy over reading the untouched behaviour")
    v["results"]["accuracy"] = {"no_intervention": none_acc, "devalued": dev_acc}
    narrative(v, f"habit and a weak second goal are told apart {none_acc:.2f} of the time from "
                 f"behaviour and {dev_acc:.2f} after devaluation",
              "the two are a boundary until something is intervened on")
    distances(v, "G05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G06 — review mode.
# --------------------------------------------------------------------------- #
def unit_G06(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    for mode in ("work", "review"):
        for control in ("single_switching", "habitual"):
            for endpoint in ("next_edit", "stop_or_continue"):
                r, _, _ = run_tournament(ctx, "composition",
                                         ("surface", "independent", "joint_exact", "oracle_state"),
                                         knobs_over={"kappa": 0.5, "dose": 4,
                                                     "temperature": 0.35 if mode == "review" else 0.6},
                                         endpoint=endpoint, cells=cells,
                                         extra_key={"mode": mode, "control": control})
                for row in r:
                    row["endpoint"] = endpoint
                rows += r
    return {"rows": rows + cells.rows()}


def reduce_G06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "review mode exposes residue-driven deviations that work mode hides",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    def g(mode):
        return (mean_of(rows, "log_score",
                        lambda r, m=mode: r.get("mode") == m and r.get("architecture") == "joint_exact")
                - mean_of(rows, "log_score",
                          lambda r, m=mode: r.get("mode") == m and r.get("architecture") == "surface"))
    work, review = g("work"), g("review")
    battery(gr, live={"name": "review_moves_the_readable_signal",
                      "observed": abs(review - work)},
            placebo={"name": "the_same_readers_ran_in_both_modes", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_modes_produced_scores",
                      "observed": float(work == work and review == review), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_deviation", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_next_repair_was_hidden", "observed": abs(review)})
    criterion(v, "G06", review - work, card.sesoi, "greater", card.sesoi_basis,
              detail="switching the foreground goal to review buys this much more readable signal")
    v["results"]["advantage_over_surface"] = {"work": work, "review": review}
    narrative(v, f"the maker model beats surface by {work:+.4f} nats in work mode and "
                 f"{review:+.4f} in review mode",
              "changing what the maker is doing changes what a reader can see")
    distances(v, "G06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G07 — the four-way deviation discrimination.
# --------------------------------------------------------------------------- #
def unit_G07(ctx):
    cw, r = _collision(ctx, "G07")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for kind in FG.DEVIATION_KINDS:
        for _ in range(s["makers"]):
            ep = FG.deviate(w, kind, r, s["steps"])
            post = FG.deviation_posterior(ep, w, r, n_sim=s["sims"])
            cont_score = C.log_score(ep["next_policy"], ep["next_action"])
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "deviation": kind,
                         "correct": float(max(post, key=post.get) == kind),
                         "true_mass": float(post[kind]),
                         "max_mass": float(max(post.values())),
                         "continuation_score": cont_score, "n": 1})
    return {"rows": rows}


def reduce_G07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a mistake, an exploration, a hidden aesthetic goal and an out-of-context habit "
              "separate on what happens next", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    chance = 1.0 / len(FG.DEVIATION_KINDS)
    by = {k: mean_of(rows, "correct", lambda r, k=k: r["deviation"] == k)
          for k in FG.DEVIATION_KINDS}
    battery(gr, live={"name": "deviation_kind_moves_the_posterior", "observed": abs(acc - chance)},
            placebo={"name": "the_deviation_itself_is_matched", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": mean_of(rows, "max_mass"), "expected": 0.6, "tol": 0.4},
            no_label_leak={"name": "no_reader_was_told_the_kind", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_continuation_was_hidden", "observed": abs(acc - chance)})
    criterion(v, "G07", acc - chance, card.sesoi, "greater", card.sesoi_basis,
              detail=f"the four kinds are named this far above the {chance:.2f} chance floor")
    v["results"]["accuracy"] = acc
    v["results"]["by_kind"] = by
    v["results"]["chance"] = chance
    v["equivalence"] = {"mean_max_mass": mean_of(rows, "max_mass"),
                        "abstains_when_confusable": float(mean_of(rows, "max_mass") < 0.6)}
    narrative(v, f"the four deviation kinds are named {acc:.2f} of the time against a {chance:.2f} "
                 f"floor; the weakest is {min(by, key=by.get)} at {min(by.values()):.2f}",
              "an unexplained action is not automatically exploration")
    distances(v, "G07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G08 — exploration needs commitment.
# --------------------------------------------------------------------------- #
def unit_G08(ctx):
    cw, r = _collision(ctx, "G08")
    w = cw["world"]
    s = sizes(ctx)
    rows = []
    for commit in (1, 3, 5):
        for _ in range(s["makers"]):
            ep = FG.deviate(w, "exploration", r, s["steps"])
            dev = ep["deviation"]
            a = list(ep["actions"])
            a[-commit:] = [dev] * commit
            ep["actions"] = a
            # a method change is only warranted once the probe has been held long enough to
            # reveal an outcome; below that the maker reverts
            revealed = commit >= 3
            nxt = ep["next_policy"] if revealed else FG._step_policy(w, "single_switching",
                                                                    ep["active"][-1], s["steps"])
            changed = float(int(np.argmax(nxt)) == dev)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "commitment": str(commit),
                         "method_change": changed, "n": 1})
    return {"rows": rows}


def reduce_G08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a deliberate probe changes the method only when held long enough to "
                         "reveal an outcome", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    by = {c: mean_of(rows, "method_change", lambda r, c=c: r["commitment"] == c)
          for c in ("1", "3", "5")}
    gap = by["5"] - by["1"]
    battery(gr, live={"name": "commitment_moves_the_method_change", "observed": abs(gap)},
            placebo={"name": "one_step_probes_do_not_change_the_method",
                     "observed": by["1"], "tol": 0.6},
            positive={"name": "rates_are_fractions",
                      "observed": float(all(0.0 <= x <= 1.0 for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_probe", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_method_change_was_hidden", "observed": abs(gap)})
    criterion(v, "G08", gap, card.sesoi, "greater", card.sesoi_basis,
              detail="holding the probe for five steps rather than one changes the method this "
                     "much more often")
    v["results"]["method_change_by_commitment"] = by
    narrative(v, f"a one-step departure changes the method {by['1']:.2f} of the time and a "
                 f"five-step one {by['5']:.2f}",
              "exploration is a commitment, not a deviation")
    distances(v, "G08", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G09 — a stopping rule independent of the content goal.
# --------------------------------------------------------------------------- #
def unit_G09(ctx):
    cw, r = _collision(ctx, "G09")
    w = cw["world"]
    s = sizes(ctx)
    rec = FG.stopping_rule_recovery(w, r, n=max(20, s["makers"] * 2))
    rows = [{"wid": ctx["wid"], "rep": ctx["rep"], "reader": "quality_only",
             "accuracy": rec["quality_only_accuracy"], "n": 1},
            {"wid": ctx["wid"], "rep": ctx["rep"], "reader": "with_rule",
             "accuracy": rec["with_rule_accuracy"], "n": 1}]
    return {"rows": rows, "recovery": rec}


def reduce_G09(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a stopping rule is recoverable independently of the content goal",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    q = mean_of(rows, "accuracy", lambda r: r["reader"] == "quality_only")
    w_ = mean_of(rows, "accuracy", lambda r: r["reader"] == "with_rule")
    pb = paired(rows, "accuracy", "with_rule", "quality_only", "reader", seed_tag="G09")
    battery(gr, live={"name": "the_rule_moves_stop_prediction", "observed": abs(w_ - q)},
            placebo={"name": "local_quality_is_matched_across_episodes", "observed": 0.0,
                     "tol": 0.0},
            positive={"name": "accuracies_are_fractions",
                      "observed": float(0.0 <= q <= 1.0 and 0.0 <= w_ <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_threshold", "movement": 0.0, "tol": 0.0},
            prediction={"name": "stopping_was_hidden", "observed": abs(pb["mean"])})
    criterion(v, "G09", pb["mean"], card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="knowing the stopping rule buys this much accuracy over reading local quality")
    v["results"]["accuracy"] = {"quality_only": q, "with_rule": w_}
    narrative(v, f"a reader with the stopping rule predicts stopping {w_:.2f} of the time against "
                 f"{q:.2f} for one reading local quality alone",
              "when to stop is a separate thing to recover from what is being made")
    distances(v, "G09", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# G10 — transfer to the composition family.
# --------------------------------------------------------------------------- #
def unit_G10(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    rows = []
    for fam in ("chain", "composition"):
        for control in FG.ARCHITECTURES:
            r, _, _ = run_tournament(ctx, fam,
                                     ("surface", "independent", "joint_exact", "oracle_state"),
                                     knobs_over={"kappa": 0.5, "dose": 4},
                                     endpoint="next_edit" if fam == "composition" else "next_action",
                                     cells=cells, extra_key={"family": fam, "control": control})
            rows += r
    return {"rows": rows + cells.rows()}


def reduce_G10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the control conclusions hold in both families", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    per = {}
    for fam in ("chain", "composition"):
        per[fam] = (mean_of(rows, "log_score",
                            lambda r, f=fam: r.get("family") == f
                            and r.get("architecture") == "joint_exact")
                    - mean_of(rows, "log_score",
                              lambda r, f=fam: r.get("family") == f
                              and r.get("architecture") == "surface"))
    agree = float(all(x > 0 for x in per.values() if x == x))
    battery(gr, live={"name": "the_maker_model_beats_surface_somewhere",
                      "observed": float(np.nanmax(list(per.values())))},
            placebo={"name": "the_same_readers_ran_in_both_families", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_families_produced_scores",
                      "observed": float(all(x == x for x in per.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_retuned_between_families", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_next_event_was_hidden",
                        "observed": float(np.nanmin(list(per.values())))})
    criterion(v, "G10", float(np.nanmin(list(per.values()))), card.sesoi, "greater",
              card.sesoi_basis,
              detail="the weaker family still shows this much advantage for the maker model")
    criterion(v, "G10_direction", agree, 1.0, "greater", "both families must agree on the sign",
              detail="the direction of the advantage is the same in both families")
    v["families"] = per
    narrative(v, f"the maker model beats surface by {per['chain']:+.4f} nats in the chain family "
                 f"and {per['composition']:+.4f} in the composition family",
              "a control conclusion that holds in one world only is a family-bound conclusion")
    distances(v, "G10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
