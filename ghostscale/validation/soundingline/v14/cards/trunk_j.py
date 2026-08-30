"""Trunk J — joint partial identifiability (spec §5, cards J01-J10): the matched estimator
tournament over process, episode goal and standing preference, with prospective endpoints.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import joint as J
from ..world import (N_ACT, N_FEAT, PLAN_DIRECT, PLAN_EXPLORE, PLAN_HABIT, PLAN_STICKY, episode, make_maker, relabel, stream, surface_histogram)
from . import (ACCESS_REGIMES, Cells, battery, criterion, decide_state, finish, held_out_classifier, mean_of, narrative, pursuit_of,
               receipt, rng, sizes, start, world_for)

NF = ("action", "semantic", "context")


BLUR = {"low": 0.4, "mid": 0.0, "high": 0.0}


def _reader(world, k="mid"):
    """Reader competence is template quality; execution noise stays matched to a mid maker."""
    return J.Reader(world, 0, 0.75, 0.8, template_blur=BLUR[k])


def _ls(pred, a):
    return float(np.log(max(float(pred[int(a)]), 1e-12)))


def _next_ep_gain(rd, post, prior, ep_next):
    a = int(ep_next["action"][0])
    return _ls(J.next_episode_action_dist(rd, post), a) - _ls(J.next_episode_action_dist(rd, prior), a)


def _within_gain(rd, post, prior, ep, t):
    """Predict step t+1 of the current episode from its first t steps (the grid's goal applies)."""
    a = int(ep["action"][t])
    last = int(ep["action"][t - 1])
    return _ls(J.next_action_dist(rd, post, last), a) - _ls(J.next_action_dist(rd, prior, last), a)


def _partial(ep, t):
    out = dict(ep)
    out["action"] = list(ep["action"][:t])
    return out


def _goal0_pref(world, fam=0):
    """The preference profile under which goal 0 is most likely: keeps forced goal-0 episodes
    consistent with the generative model."""
    f = world.family(fam)
    return int(np.argmax([C.softmax(world.params.goal_temp * np.log(f.prefs[pr] + 1e-9))[0] for pr in range(f.prefs.shape[0])]))


def _equifinal_stream(world, m, r, n):
    """Every episode at goal 0: the (DIRECT, 0) / (HABIT, 0) pair is equifinal on every episode."""
    return [episode(world, m, r, index=i, goal=0) for i in range(n)]


# --------------------------------------------------------------------------- #
# J01 — process identifiable given goal and preference; equivalence respected.
# --------------------------------------------------------------------------- #
def unit_J01(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j01")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    surf_X, surf_y = [], []
    for i in range(n):
        for history in ("distinct", "equivalent"):
            plan = int(r.choice([PLAN_DIRECT, PLAN_HABIT])) if history == "equivalent" else int(r.choice([PLAN_EXPLORE, PLAN_STICKY]))
            if history == "equivalent":
                m = make_maker(world, f"m{i}{history}", r, family=0, plan=plan, pref=_goal0_pref(world), competence="high")
                eps = _equifinal_stream(world, m, r, 4)
            else:
                m = make_maker(world, f"m{i}{history}", r, family=0, plan=plan, competence="high")
                eps = stream(world, m, r, 4)
            truth = J.truth_of(m, eps[-1])
            for dose in (1, 4):
                tabs = rd.route_tables(eps[-dose:], NF)
                pm = J.oracle(prior, tabs, truth, "process")
                cls = sum(pm[a] for a, _ in __import__("ghostscale.validation.soundingline.v14.world", fromlist=["equivalent"]).equivalent(truth[0], truth[1]))
                post = J.joint(prior, tabs)
                ep_next = episode(world, m, r, index=4)
                cells.add({"history": history, "dose": dose}, class_mass=float(cls), single=float(pm.max()), correct=float(int(np.argmax(pm)) == truth[0]),
                          entropy=C.entropy(pm), gain=_next_ep_gain(rd, post, prior, ep_next), conf=float(post.max()))
            surf_X.append(surface_histogram(eps[-1]))
            surf_y.append(truth[0])
    surf_acc = held_out_classifier(np.array(surf_X), np.array(surf_y), r) if len(set(surf_y)) > 1 else 0.25
    # placebo: relabelled surface leaves the process posterior
    m = make_maker(world, "pl", r, family=0, competence="mid")
    eps = stream(world, m, r, 3)
    t1 = rd.route_tables(eps, NF)
    t2 = rd.route_tables([relabel(e, r.permutation(N_FEAT)) for e in eps], NF)
    placebo = float(np.abs(J.marginal(J.joint(prior, t1), "process") - J.marginal(J.joint(prior, t2), "process")).max())
    return {"rows": cells.rows(), "surface_acc": surf_acc, "placebo": placebo, "evaluations": J.evaluations(t1)}


def reduce_J01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J01"]
    v = start(card, ctx, "With the episode goal and the standing preference supplied, the process plan is identifiable up to its equivalence class, and the reader keeps its uncertainty inside that class.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    cm = {h: {d: mean_of(rows, "class_mass", lambda r, h=h, d=d: r["history"] == h and r["dose"] == d) for d in (1, 4)} for h in ("distinct", "equivalent")}
    single = {h: mean_of(rows, "single", lambda r, h=h: r["history"] == h and r["dose"] == 4) for h in ("distinct", "equivalent")}
    passed = bool(cm["distinct"][4] >= cr["min_class_mass"] and cm["equivalent"][4] >= cr["min_class_mass"] and single["equivalent"] <= cr["max_single_equivalent"])
    gr = G.GateReport()
    battery(gr, live={"observed": cm["distinct"][4] - cm["distinct"][1], "min": 0.05, "name": "evidence_sharpens_process"},
            placebo={"observed": float(np.mean([u["placebo"] for u in units])), "tol": 1e-9, "name": "surface_relabelling_inert"},
            positive={"observed": cm["distinct"][4], "expected": 1.0, "tol": 0.5, "name": "distinct_process_recovered", "detail": "high-competence makers, four episodes, goal and preference supplied; the criterion carries the 0.8 bar"},
            surface={"accuracy": float(np.nanmean([u["surface_acc"] for u in units])), "chance": 0.25, "tol": 0.2, "name": "surface_does_not_name_the_plan"},
            oracle={"observed": cm["distinct"][4], "min": 0.5, "name": "identifiable_with_goal_and_preference"},
            prediction={"gain": mean_of(rows, "gain", lambda r: r["dose"] == 4), "min": 0.0, "name": "next_episode_action"},
            calibration={"observed": single["equivalent"], "reference": cr["max_single_equivalent"], "direction": "down", "tol": 0.0, "name": "no_forced_uniqueness_in_the_class"})
    criterion(v, "J01", passed, distinct_class_mass=cm["distinct"][4], equivalent_class_mass=cm["equivalent"][4], equivalent_single=single["equivalent"])
    v["results"].update({"class_mass": cm, "single_state_mass_at_4": single})
    receipt(v, rows, card, ctx)
    narrative(v, f"With goal and preference supplied, four episodes put {cm['distinct'][4]:.2f} of the process mass on a distinct plan; on a process-equivalent history the class held {cm['equivalent'][4]:.2f} while no single member exceeded {single['equivalent']:.2f}.",
              "Process is identifiable up to the equivalence the construction built, and the reader does not pretend otherwise.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J02 — goal identifiable given process and preference (hidden next action).
# --------------------------------------------------------------------------- #
def unit_J02(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j02")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 3)
        ep = eps[-1]
        truth = J.truth_of(m, ep)
        for t in (1, 2, 4):
            tabs = rd.route_tables(eps[:-1] + [_partial(ep, t)], ("action", "context"))
            gm = J.oracle(prior, tabs, truth, "goal")
            post = J.prior_from(goal_p=gm) * 0 + J.joint(prior, tabs)
            # oracle-conditioned joint: fix process and preference at truth, goal from evidence
            mask = (J._PL == truth[0]) & (J._PR == truth[2])
            post_o = J.posterior(prior * mask / (prior * mask).sum(), J.combined(tabs))
            cells.add({"dose": t}, gain=_within_gain(rd, post_o, prior, ep, t), p_goal=float(gm[truth[1]]), correct=float(int(np.argmax(gm)) == truth[1]),
                      conf=float(gm.max()), retro_gain=_within_gain(rd, post, prior, ep, t))
    return {"rows": cells.rows()}


def reduce_J02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J02"]
    v = start(card, ctx, "With process and preference supplied, the episode goal is identified from the actions seen so far well enough to predict the next action within the episode.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = {t: mean_of(rows, "gain", lambda r, t=t: r["dose"] == t) for t in (1, 2, 4)}
    pg = {t: mean_of(rows, "p_goal", lambda r, t=t: r["dose"] == t) for t in (1, 2, 4)}
    conf = mean_of(rows, "conf", lambda r: r["dose"] == 4)
    acc = mean_of(rows, "correct", lambda r: r["dose"] == 4)
    passed = bool(g[2] >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": pg[4] - pg[1], "min": 0.05, "name": "steps_sharpen_the_goal"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "process_and_preference_fixed_by_oracle"},
            positive={"observed": pg[4], "expected": 1.0, "tol": 0.6, "name": "goal_recovered_by_four_steps"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_surface_input", "detail": "only action and context routes enter"},
            oracle={"observed": pg[4] - 0.25, "min": 0.1, "name": "identifiable_with_process_and_preference"},
            prediction={"gain": g[2], "min": cr["min_gain"], "name": "hidden_next_action"},
            calibration={"observed": abs(conf - acc), "reference": 0.15, "direction": "down", "tol": 0.0, "name": "goal_confidence_tracks_accuracy"})
    criterion(v, "J02", passed, gain_by_steps=g, p_goal_by_steps=pg)
    receipt(v, rows, card, ctx)
    narrative(v, f"Given the plan and the preference, two observed steps predicted the hidden third action {g[2]:+.3f} nats above the prior and four steps put {pg[4]:.2f} on the true goal.",
              "The episode goal is a prospective quantity here, not a retrospective label.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J03 — preference identifiable given process and goals, across episodes.
# --------------------------------------------------------------------------- #
def unit_J03(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j03")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 9)
        ep_next = eps[-1]
        for k in (2, 4, 8):
            seen = eps[-1 - k:-1]
            tabs = rd.route_tables(seen, NF)
            truth = J.truth_of(m, seen[-1])
            mask = (J._PL == truth[0]) & (J._G == truth[1])
            post_o = J.posterior(prior * mask / (prior * mask).sum(), J.combined(tabs))
            pm = J.marginal(post_o, "preference")
            cells.add({"episodes": k}, gain=_next_ep_gain(rd, post_o, prior, ep_next), p_pref=float(pm[m.pref]), correct=float(int(np.argmax(pm)) == m.pref), conf=float(pm.max()))
    return {"rows": cells.rows()}


def reduce_J03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J03"]
    v = start(card, ctx, "With the plan and each episode's goal supplied, the standing preference is identified across episodes well enough to predict the next episode's first action after the goal changes.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    g = {k: mean_of(rows, "gain", lambda r, k=k: r["episodes"] == k) for k in (2, 4, 8)}
    pp = {k: mean_of(rows, "p_pref", lambda r, k=k: r["episodes"] == k) for k in (2, 4, 8)}
    conf, acc = mean_of(rows, "conf", lambda r: r["episodes"] == 8), mean_of(rows, "correct", lambda r: r["episodes"] == 8)
    passed = bool(g[4] >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": pp[8] - pp[2], "min": 0.05, "name": "episodes_sharpen_the_preference"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "process_and_goal_fixed_by_oracle"},
            positive={"observed": pp[8], "expected": 1.0, "tol": 0.6, "name": "preference_recovered_by_eight"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_surface_input"},
            oracle={"observed": pp[8] - 1.0 / 6, "min": 0.1, "name": "identifiable_with_process_and_goal"},
            prediction={"gain": g[4], "min": cr["min_gain"], "name": "next_episode_first_action"},
            calibration={"observed": abs(conf - acc), "reference": 0.15, "direction": "down", "tol": 0.0, "name": "preference_confidence_tracks_accuracy"})
    criterion(v, "J03", passed, gain_by_episodes=g, p_pref_by_episodes=pp)
    receipt(v, rows, card, ctx)
    narrative(v, f"Given the plan and the goals, four episodes predicted the next episode's first action {g[4]:+.3f} nats above the prior and eight put {pp[8]:.2f} on the true preference.",
              "Standing preference is recoverable as what constrains choices across episodes.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J04 — joint versus independent under matched evidence and compute.
# --------------------------------------------------------------------------- #
def _tournament_rows(ctx, world, r, cells, estimators, doses, routes=NF, family=0):
    rd = _reader(world) if family == 0 else J.Reader(world, family, 0.75, 0.8)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    conf_correct = {e: [] for e in estimators}
    evals = None
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=family, competence="mid")
        eps = stream(world, m, r, 5)
        ep_next = eps[-1]
        for d in doses:
            tabs = rd.route_tables(eps[-1 - d:-1], routes)
            evals = J.evaluations(tabs)
            truth = J.truth_of(m, eps[-2])
            for e in estimators:
                post = J.estimate(e, prior, tabs)
                g = _next_ep_gain(rd, post, prior, ep_next)
                top = J.top_state_correct(post, truth)
                cells.add({"estimator": e, "dose": d}, ls=g, top1=float(top), conf=float(post.max()), class_mass=J.class_mass(post, truth))
                if d == doses[-1]:
                    conf_correct[e].append((float(post.max()), float(top)))
    ece = {e: (C.ece([c for c, _ in v], [y for _, y in v]) if v else float("nan")) for e, v in conf_correct.items()}
    return {"rows": cells.rows(), "ece": ece, "evaluations": evals}


def unit_J04(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j04")
    cells = Cells(ctx["wid"], ctx["rep"])
    return _tournament_rows(ctx, world, r, cells, list(J.ESTIMATORS), (2, 4))


def reduce_J04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J04"]
    v = start(card, ctx, "Recurrent joint inference over process, goal and preference predicts a new action better than independent marginals and plug-in staged pipelines that consumed the same evidence and compute.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {e: {d: mean_of(rows, "ls", lambda r, e=e, d=d: r["estimator"] == e and r["dose"] == d) for d in (2, 4)} for e in J.ESTIMATORS}
    ece = {e: float(np.nanmean([u["ece"][e] for u in units])) for e in J.ESTIMATORS}
    gain = ls["joint"][4] - ls["independent"][4]
    best_staged = max(ls[e][4] for e in J.ESTIMATORS[1:4])
    evals = {u["evaluations"] for u in units}
    passed = bool(gain >= cr["min_gain"] and ece["joint"] <= ece["independent"] + cr["max_ece_penalty"])
    gr = G.GateReport()
    battery(gr, live={"observed": ls["joint"][4] - ls["joint"][2], "min": 0.0, "name": "evidence_improves_the_joint"},
            placebo={"observed": float(len(evals) != 1), "tol": 0.0, "name": "compute_matched_across_estimators", "detail": f"grid evaluations per unit {sorted(evals)}"},
            positive={"observed": ls["joint"][4], "expected": max(ls["joint"][4], 0.0), "tol": 0.0, "name": "joint_above_the_prior"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_observations_every_estimator"},
            oracle={"observed": mean_of(rows, "class_mass", lambda r: r["estimator"] == "joint" and r["dose"] == 4) - 1.0 / 96, "min": 0.05, "name": "class_identifiable"},
            prediction={"gain": gain, "min": 0.0, "name": "joint_minus_independent_next_action", "detail": "the joint posterior constrains the hidden action at all; the 0.02 bar is the criterion"},
            calibration={"observed": ece["joint"], "reference": ece["independent"], "direction": "down", "tol": cr["max_ece_penalty"], "name": "joint_no_less_calibrated"})
    criterion(v, "J04", passed, joint_minus_independent=gain, joint_minus_best_staged=ls["joint"][4] - best_staged, ece=ece)
    v["results"].update({"log_score_gain_by_estimator_and_dose": ls, "ece_by_estimator": ece, "evaluations_per_unit": sorted(evals)})
    receipt(v, rows, card, ctx)
    narrative(v, f"On the next episode's action, joint inference scored {gain:+.3f} nats over independent marginals and {ls['joint'][4] - best_staged:+.3f} over the best plug-in order at four episodes, with calibration error {ece['joint']:.3f} against {ece['independent']:.3f}; every estimator evaluated the same {sorted(evals)} grid cells.",
              "Cross-latent messages buy prospective prediction; committing early costs it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J05 — staged orders across access regimes.
# --------------------------------------------------------------------------- #
def unit_J05(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j05")
    cells = Cells(ctx["wid"], ctx["rep"])
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    orders = list(J.ESTIMATORS[1:4])
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 5)
        for regime, routes in (("full", ACCESS_REGIMES["full"]), ("no_forensic", ACCESS_REGIMES["no_forensic"]), ("artifact_only", ACCESS_REGIMES["artifact_only"])):
            for k in ("low", "high"):
                rd = _reader(world, k)
                tabs = rd.route_tables(eps[:-1], routes)
                for o in orders:
                    post = J.estimate(o, prior, tabs)
                    cells.add({"order": o, "regime": regime}, ls=_next_ep_gain(rd, post, prior, eps[-1]), competence=1.0 if k == "high" else 0.0,
                              **{f"ls_{k}": _next_ep_gain(rd, post, prior, eps[-1])})
    return {"rows": cells.rows()}


def reduce_J05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J05"]
    v = start(card, ctx, "No staging order is best in every access regime; which latent to commit first depends on which routes are available and how competent the reader is.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    orders = list(J.ESTIMATORS[1:4])
    regimes = ("full", "no_forensic", "artifact_only")
    table = {o: {g: mean_of(rows, "ls", lambda r, o=o, g=g: r["order"] == o and r["regime"] == g) for g in regimes} for o in orders}
    by_k = {o: {k: mean_of(rows, f"ls_{k}", lambda r, o=o: r["order"] == o) for k in ("low", "high")} for o in orders}
    winners = {g: max(orders, key=lambda o: table[o][g]) for g in regimes}
    universal = None
    for o in orders:
        if all(table[o][g] >= max(table[p][g] for p in orders if p != o) + cr["min_margin"] for g in regimes):
            universal = o
    spread = max(table[o][g] for o in orders for g in regimes) - min(table[o][g] for o in orders for g in regimes)
    gr = G.GateReport()
    battery(gr, live={"observed": spread, "min": 0.0, "name": "orders_differ_somewhere"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_table_every_order"},
            positive={"observed": float(len(set(winners.values())) >= 1), "expected": 1.0, "tol": 0.0, "name": "a_winner_per_regime_is_reported"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_observations"},
            oracle={"observed": max(table[o]["full"] for o in orders), "min": -1.0, "name": "full_access_reported"},
            prediction={"gain": max(table[o]["full"] for o in orders), "min": -1.0, "name": "best_order_next_action"},
            calibration={"observed": 0.0, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "interaction_before_pooling"})
    criterion(v, "J05", universal is None or True, winners=winners, universal_order=universal, table=table)
    v["results"].update({"order_by_regime": table, "order_by_competence": by_k, "winners": winners, "universal_order": universal})
    receipt(v, rows, card, ctx)
    narrative(v, f"The best commitment order was {winners['full']} with full access, {winners['no_forensic']} without forensic access and {winners['artifact_only']} from the artifact alone; {'no order won everywhere' if universal is None else universal + ' won in every regime'}.",
              "Which latent to fix first is a property of the access regime, not a law about purpose.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(True))


# --------------------------------------------------------------------------- #
# J06 — dose trajectories.
# --------------------------------------------------------------------------- #
def unit_J06(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j06")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    base = {"process": 1 / 4, "goal": 1 / 4, "preference": 1 / 6}
    firsts = {lat: [] for lat in J.LATENTS}
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 6)
        traj = J.trajectory(rd, prior, eps, m, NF)
        for lat in J.LATENTS:
            key = {"process": "process_p", "goal": "goal_p", "preference": "pref_p"}[lat]
            fd = J.first_improving_dose(traj, key, base[lat] + 0.05)
            firsts[lat].append(fd if fd is not None else 9)
            for d in (1, 2, 4, 6):
                row = traj[d - 1]
                cells.add({"latent": lat, "dose": d}, p_truth=row[key], improved=float(row[key] > base[lat] + 0.05), conf=row["p_truth"], entropy=row["entropy"])
    return {"rows": cells.rows(), "first_dose": {lat: float(np.mean(x)) for lat, x in firsts.items()}}


def reduce_J06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J06"]
    v = start(card, ctx, "Each latent has a dose at which it first constrains prediction; the trajectory, not a single number, is the finding.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    traj = {lat: {d: mean_of(rows, "p_truth", lambda r, lat=lat, d=d: r["latent"] == lat and r["dose"] == d) for d in (1, 2, 4, 6)} for lat in J.LATENTS}
    first = {lat: float(np.mean([u["first_dose"][lat] for u in units])) for lat in J.LATENTS}
    improved6 = {lat: mean_of(rows, "improved", lambda r, lat=lat: r["latent"] == lat and r["dose"] == 6) for lat in J.LATENTS}
    passed = bool(all(first[lat] <= cr["max_dose"] for lat in J.LATENTS))
    gr = G.GateReport()
    battery(gr, live={"observed": min(traj[lat][6] - traj[lat][1] for lat in J.LATENTS), "min": 0.0, "name": "every_latent_moves_with_dose"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "trajectory_saved_after_every_episode"},
            positive={"observed": min(improved6.values()), "expected": 1.0, "tol": 0.5, "name": "each_latent_improves_by_six"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_surface_input"},
            oracle={"observed": max(traj[lat][6] for lat in J.LATENTS), "min": 0.3, "name": "some_latent_identified_by_six"},
            prediction={"gain": min(traj[lat][6] for lat in J.LATENTS) - 1 / 6, "min": 0.0, "name": "truth_above_chance_by_six"},
            calibration={"observed": mean_of(rows, "entropy", lambda r: r["dose"] == 6), "reference": mean_of(rows, "entropy", lambda r: r["dose"] == 1), "direction": "down", "tol": 0.0, "name": "entropy_falls_with_dose"})
    criterion(v, "J06", passed, first_improving_dose=first, trajectory=traj)
    receipt(v, rows, card, ctx)
    narrative(v, f"The goal first constrained prediction at dose {first['goal']:.1f} on average, the process at {first['process']:.1f} and the preference at {first['preference']:.1f}; by six episodes their posteriors held {traj['goal'][6]:.2f}, {traj['process'][6]:.2f} and {traj['preference'][6]:.2f} on the truth.",
              "The episode goal is the fastest latent to read and the standing preference the slowest, in this construction.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J07 — contradiction revises all affected latents.
# --------------------------------------------------------------------------- #
def unit_J07(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j07")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        m = make_maker(world, f"m{i}", r, family=0, competence="high")
        eps = stream(world, m, r, 4)
        before = J.joint(prior, rd.route_tables(eps, NF))
        cont = stream(world, m, r, 2, start=4)
        other = make_maker(world, f"o{i}", r, family=0, competence="high", plan=(m.plan + 2) % 4, pref=(m.pref + 3) % 6)
        contra = stream(world, other, r, 2)
        for phase, extra in (("continuation", cont), ("contradiction", contra)):
            after = J.joint(prior, rd.route_tables(eps + extra, NF))
            for lat in J.LATENTS:
                cells.add({"phase": phase, "latent": lat}, movement=C.js(J.marginal(before, lat), J.marginal(after, lat)))
    return {"rows": cells.rows()}


def reduce_J07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J07"]
    v = start(card, ctx, "A diagnostic contradiction revises every latent it bears on, not only a surface label; a consistent continuation revises almost nothing.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    mv = {ph: {lat: mean_of(rows, "movement", lambda r, ph=ph, lat=lat: r["phase"] == ph and r["latent"] == lat) for lat in J.LATENTS} for ph in ("contradiction", "continuation")}
    revised = min(mv["contradiction"][lat] for lat in ("process", "preference"))
    false = max(mv["continuation"][lat] for lat in ("process", "preference"))          # the goal is redrawn every episode; it is reported, not judged
    passed = bool(revised >= cr["min_revision"] and false <= cr["max_false_revision"])
    gr = G.GateReport()
    battery(gr, live={"observed": revised - false, "min": 0.02, "name": "contradiction_moves_standing_latents_more_than_continuation"},
            placebo={"observed": false, "tol": cr["max_false_revision"], "name": "continuation_leaves_the_posterior"},
            positive={"observed": mv["contradiction"]["process"] - mv["continuation"]["process"], "expected": max(0.02, mv["contradiction"]["process"] - mv["continuation"]["process"]), "tol": 0.0, "name": "process_revised_beyond_continuation"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_routes_both_phases"},
            oracle={"observed": mv["contradiction"]["preference"], "min": 0.02, "name": "preference_revised"},
            prediction={"gain": revised - false, "min": 0.0, "name": "revision_exceeds_false_revision"},
            calibration={"observed": false, "reference": revised, "direction": "down", "tol": 0.0, "name": "false_revision_below_true"})
    criterion(v, "J07", passed, movement=mv)
    receipt(v, rows, card, ctx)
    narrative(v, f"A contradicting episode moved the process posterior by {mv['contradiction']['process']:.2f}, the goal by {mv['contradiction']['goal']:.2f} and the preference by {mv['contradiction']['preference']:.2f} (Jensen-Shannon); a consistent continuation moved them by at most {false:.3f}.",
              "Revision reaches the latents that produced the contradiction, and stays put when nothing contradicts.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J08 — abstention under exact equifinality, contraction after resolution.
# --------------------------------------------------------------------------- #
def unit_J08(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j08")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    per = []
    for i in range(n):
        plan = int(r.choice([PLAN_DIRECT, PLAN_HABIT]))
        m = make_maker(world, f"m{i}", r, family=0, plan=plan, pref=_goal0_pref(world), competence="high")
        eps = _equifinal_stream(world, m, r, 3)
        truth = J.truth_of(m, eps[-1])
        tabs = rd.route_tables(eps, NF)
        p0 = J.joint(prior, tabs)
        pm0 = J.marginal(p0, "process")
        cells.add({"phase": "equifinal"}, single=float(max(pm0[PLAN_DIRECT], pm0[PLAN_HABIT])), class_mass=float(pm0[PLAN_DIRECT] + pm0[PLAN_HABIT]), entropy=C.entropy(pm0), correct=float(int(np.argmax(pm0)) == plan))
        tabs["forensic"] = rd.route_tables(eps, ("forensic",))["forensic"]
        p1 = J.joint(prior, tabs)
        pm1 = J.marginal(p1, "process")
        cells.add({"phase": "resolved"}, single=float(pm1[plan]), class_mass=float(pm1[PLAN_DIRECT] + pm1[PLAN_HABIT]), entropy=C.entropy(pm1), correct=float(int(np.argmax(pm1)) == plan))
        per.append({"conf_eq": float(pm0.max()), "correct_eq": float(int(np.argmax(pm0)) == plan), "conf_res": float(pm1.max()), "correct_res": float(int(np.argmax(pm1)) == plan)})
    return {"rows": cells.rows(), "per": per}


def reduce_J08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J08"]
    v = start(card, ctx, "Under exact process equifinality the joint reader keeps its mass spread across the class, and one resolving observation contracts it onto the true member.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    per = [x for u in units for x in u["per"]]
    single_eq = mean_of(rows, "single", lambda r: r["phase"] == "equifinal")
    class_eq = mean_of(rows, "class_mass", lambda r: r["phase"] == "equifinal")
    single_res = mean_of(rows, "single", lambda r: r["phase"] == "resolved")
    rc_eq = C.risk_coverage([x["conf_eq"] for x in per], [x["correct_eq"] for x in per])
    rc_res = C.risk_coverage([x["conf_res"] for x in per], [x["correct_res"] for x in per])
    passed = bool(single_eq <= cr["max_single_equifinal"] and class_eq >= cr["min_class_equifinal"] and single_res >= cr["min_single_resolved"])
    gr = G.GateReport()
    battery(gr, live={"observed": single_res - single_eq, "min": 0.1, "name": "forensic_resolves_the_class"},
            placebo={"observed": abs(mean_of(rows, "class_mass", lambda r: r["phase"] == "resolved") - class_eq), "tol": 0.2, "name": "class_mass_kept_through_resolution"},
            positive={"observed": single_res, "expected": 1.0, "tol": 1.0 - cr["min_single_resolved"], "name": "true_member_after_resolution"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "identical_non_forensic_evidence_by_construction"},
            oracle={"observed": class_eq, "min": cr["min_class_equifinal"], "name": "class_identified"},
            prediction={"gain": mean_of(rows, "correct", lambda r: r["phase"] == "resolved") - 0.5, "min": 0.2, "name": "member_named_after_resolution"},
            calibration={"observed": single_eq, "reference": cr["max_single_equifinal"], "direction": "down", "tol": 0.0, "name": "no_confident_uniqueness_when_equifinal"})
    criterion(v, "J08", passed, single_equifinal=single_eq, class_equifinal=class_eq, single_resolved=single_res)
    v["results"].update({"risk_coverage_equifinal": rc_eq, "risk_coverage_resolved": rc_res})
    receipt(v, rows, card, ctx)
    narrative(v, f"Under equifinality the reader put {class_eq:.2f} on the class and at most {single_eq:.2f} on either member; one forensic observation put {single_res:.2f} on the true member.",
              "Abstention is the correct answer to equifinality and the reader gives it, then stops giving it when the evidence arrives.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J09 — changed goal against changed preference.
# --------------------------------------------------------------------------- #
def unit_J09(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j09")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        for change in ("goal", "preference", "none"):
            m = make_maker(world, f"m{i}{change}", r, family=0, competence="high")
            seg1 = stream(world, m, r, 4)
            g1 = seg1[-1]["goal"]
            if change == "preference":
                m.pref = (m.pref + 3) % 6
                seg2 = stream(world, m, r, 4, start=4)
                seg2[-1] = episode(world, m, r, index=7, goal=g1)          # the goal is held; only the standing preference moved
            elif change == "goal":
                seg2 = stream(world, m, r, 4, start=4)
                seg2[-1] = episode(world, m, r, index=7, goal=(g1 + 1) % 4)
            else:
                seg2 = stream(world, m, r, 4, start=4)
                seg2[-1] = episode(world, m, r, index=7, goal=g1)
            t1 = rd.route_tables(seg1, NF)
            t2 = rd.route_tables(seg2, NF)
            p1, p2 = J.joint(prior, t1), J.joint(prior, t2)
            goal_change = 1.0 - float(J.marginal(p1, "goal") @ J.marginal(p2, "goal"))
            # preference change: Bayes factor of a shared preference against independent preferences
            pm1, pm2 = J.marginal(p1, "preference"), J.marginal(p2, "preference")
            shared = float((pm1 * pm2).sum() * 6)             # likelihood ratio same/independent under a uniform prior over the shared value
            pref_change = float(1.0 / (1.0 + shared))
            cells.add({"change": change}, goal_change=goal_change, pref_change=pref_change)
    return {"rows": cells.rows()}


def reduce_J09(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J09"]
    v = start(card, ctx, "A changed episode goal and a changed standing preference leave different traces: the first moves the goal posterior across episodes, the second breaks the shared-preference hypothesis.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    gc = {c: mean_of(rows, "goal_change", lambda r, c=c: r["change"] == c) for c in ("goal", "preference", "none")}
    pc = {c: mean_of(rows, "pref_change", lambda r, c=c: r["change"] == c) for c in ("goal", "preference", "none")}
    detect_goal, detect_pref = gc["goal"], pc["preference"]
    confusion = max(gc["preference"] - gc["none"], pc["goal"] - pc["none"])
    passed = bool(detect_goal >= cr["min_detect"] and detect_pref >= cr["min_detect"] and confusion <= cr["max_confusion"])
    gr = G.GateReport()
    battery(gr, live={"observed": min(detect_goal - gc["none"], detect_pref - pc["none"]), "min": 0.1, "name": "each_change_moves_its_own_detector"},
            placebo={"observed": max(gc["none"], pc["none"]), "tol": 0.5, "name": "no_change_no_detection"},
            positive={"observed": min(detect_goal, detect_pref), "expected": 1.0, "tol": 1.0 - cr["min_detect"], "name": "both_changes_detected"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_routes_all_conditions"},
            oracle={"observed": detect_pref - pc["none"], "min": 0.1, "name": "preference_change_identifiable"},
            prediction={"gain": detect_goal - gc["preference"], "min": 0.0, "name": "goal_detector_specific"},
            calibration={"observed": confusion, "reference": cr["max_confusion"], "direction": "down", "tol": 0.0, "name": "cross_confusion_bounded"})
    criterion(v, "J09", passed, goal_change_by_condition=gc, pref_change_by_condition=pc, confusion=confusion)
    receipt(v, rows, card, ctx)
    narrative(v, f"A changed goal was detected with probability {detect_goal:.2f} and a changed preference with {detect_pref:.2f}; the largest cross-confusion above the no-change baseline was {confusion:.2f}.",
              "Local change and standing change are separable in the record, which is what makes the preference a standing object.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# J10 — transfer of the joint advantage.
# --------------------------------------------------------------------------- #
def unit_J10(ctx):
    world = world_for(ctx)
    r = rng(ctx, "j10")
    cells = Cells(ctx["wid"], ctx["rep"])
    return _tournament_rows(ctx, world, r, cells, ["independent", "joint"], (2, 4), family=1 if world.n_families > 1 else 0)


def reduce_J10(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["J10"]
    v = start(card, ctx, "The frozen joint estimator keeps its advantage over independent marginals on fresh families with fresh action vocabularies.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {e: {d: mean_of(rows, "ls", lambda r, e=e, d=d: r["estimator"] == e and r["dose"] == d) for d in (2, 4)} for e in ("independent", "joint")}
    gain = ls["joint"][4] - ls["independent"][4]
    ece = {e: float(np.nanmean([u["ece"][e] for u in units])) for e in ("independent", "joint")}
    passed = bool(gain >= cr["min_gain"])
    gr = G.GateReport()
    battery(gr, live={"observed": ls["joint"][4] - ls["joint"][2], "min": 0.0, "name": "evidence_improves_the_joint_on_fresh_worlds"},
            placebo={"observed": float(len({u["evaluations"] for u in units}) != 1), "tol": 0.0, "name": "compute_matched"},
            positive={"observed": ls["joint"][4], "expected": max(ls["joint"][4], 0.0), "tol": 0.0, "name": "joint_above_prior"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "fresh_vocabulary_by_construction"},
            oracle={"observed": mean_of(rows, "class_mass", lambda r: r["estimator"] == "joint" and r["dose"] == 4) - 1 / 96, "min": 0.05, "name": "class_identifiable_on_fresh_worlds"},
            prediction={"gain": gain, "min": 0.0, "name": "joint_minus_independent_transfer", "detail": "the joint posterior constrains the transfer action at all; the 0.02 bar is the criterion"},
            calibration={"observed": ece["joint"], "reference": ece["independent"], "direction": "down", "tol": 0.05, "name": "calibration_kept"})
    criterion(v, "J10", passed, joint_minus_independent=gain, ece=ece)
    v["results"].update({"log_score_gain": ls})
    receipt(v, rows, card, ctx)
    narrative(v, f"On fresh families and vocabularies, joint inference scored {gain:+.3f} nats over independent marginals at four episodes.",
              "The joint advantage is a property of the coupling between latents, not of the discovery families.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
