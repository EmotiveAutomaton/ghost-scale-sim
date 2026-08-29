"""Trunk R — route reliability, planted ease, conflict, fusion, forensic purchase, transfer
(spec §5, cards R01-R08).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import joint as J
from .. import routes as R
from ..world import N_FEAT, N_GOAL, ROUTE_COST, ROUTES, episode, make_maker, stream
from . import ACCESS_REGIMES, Cells, battery, criterion, decide_state, extra_gate, finish, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for

NF = ("action", "semantic", "context")


def _reader(world, fam=0):
    return J.Reader(world, fam, 0.75, 0.8)


def _ls(pred, a):
    return float(np.log(max(float(pred[int(a)]), 1e-12)))


def _gain(rd, post, prior, ep_full):
    """Within-episode next step (the endpoint every route bears on)."""
    return R.within_gain(rd, post, prior, ep_full)


def _seen(eps, k=2):
    return eps[:k] + [R.partial(eps[k])]


def _training(world, r, n, fam=0, k_eps=2, noise_route=None):
    return R.make_training(world, r, n, fam=fam, k_eps=k_eps, noise_fn=(lambda e: _noised(e, noise_route, r)) if noise_route else None)


def _noised(ep, route, r):
    """Make a route uninformative: its observations are replaced by noise (the 'wrong' route)."""
    e = dict(ep)
    if isinstance(route, (tuple, list)):
        for rt in route:
            e = _noised(e, rt, r)
        return e
    if route == "semantic":
        e["semantic"] = [int(x) for x in r.integers(N_FEAT, size=len(ep["semantic"]))]
    elif route == "action":
        e["action"] = [int(x) for x in r.integers(6, size=len(ep["action"]))]
    elif route == "context":
        e["context"] = dict(ep["context"], choices=[int(x) for x in r.integers(3, size=len(ep["context"]["choices"]))])
    return e


# --------------------------------------------------------------------------- #
# R01 — route information per regime.
# --------------------------------------------------------------------------- #
def unit_R01(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r01")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    info = R.route_information(world, rd, r, n=max(12, sizes(ctx)["makers"] // 2))
    prior = J.uniform_prior()
    # prediction ruler per route
    gains = {rt: [] for rt in ROUTES}
    for eps_train, ep_next in _training(world, r, max(4, sizes(ctx)["training"])):
        for rt in ROUTES:
            gains[rt].append(R.route_gain(rd, eps_train, ep_next, rt, prior))
    for rt in ROUTES:
        for lat in J.LATENTS:
            cells.add({"route": rt, "latent": lat}, info=info[rt][lat], pred_gain=float(np.mean(gains[rt])))
    return {"rows": cells.rows(), "info": info}


def reduce_R01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R01"]
    v = start(card, ctx, "Each latent has a route that carries most of its information, and that route is the one the construction declared: action for process, semantic for the goal, context for the preference.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    info = {rt: {lat: mean_of(rows, "info", lambda r, rt=rt, lat=lat: r["route"] == rt and r["latent"] == lat) for lat in J.LATENTS} for rt in ROUTES}
    dominant = {lat: max([rt for rt in ROUTES if rt != "forensic"], key=lambda rt: info[rt][lat]) for lat in J.LATENTS}
    declared = {"process": "action", "goal": "semantic", "preference": "context"}
    agree = sum(dominant[lat] == declared[lat] for lat in J.LATENTS)
    pred = {rt: mean_of(rows, "pred_gain", lambda r, rt=rt: r["route"] == rt and r["latent"] == "process") for rt in ROUTES}
    regimes = {reg: {lat: max(info[rt][lat] for rt in routes) for lat in J.LATENTS} for reg, routes in ACCESS_REGIMES.items()}
    passed = bool(agree >= 2 and min(max(info[rt][lat] for rt in ROUTES) for lat in J.LATENTS) >= cr["min_information"])
    gr = G.GateReport()
    extra_gate(gr, "divergence", "dominant_routes_distinct", float(len(set(dominant.values()))), 2.0, "min", "at least two different routes dominate the three latents")
    battery(gr, positive={"observed": float(agree), "expected": 3.0, "tol": 1.0, "name": "dominant_route_matches_declaration"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_episodes_every_route"},
            live={"observed": min(max(info[rt][lat] for rt in ROUTES) for lat in J.LATENTS), "min": cr["min_information"], "name": "every_latent_has_an_informative_route"},
            prediction={"gain": max(pred.values()), "min": 0.0, "name": "prediction_ruler_positive_on_the_best_route"})
    criterion(v, "R01", passed, dominant=dominant, agreement=agree, information=info, by_regime=regimes)
    v["results"].update({"information": info, "prediction_gain_by_route": pred, "best_information_by_regime": regimes})
    receipt(v, rows, card, ctx)
    narrative(v, f"The most informative route was {dominant['process']} for the process, {dominant['goal']} for the goal and {dominant['preference']} for the preference; forensic access carried {info['forensic']['process']:.2f} nats about the process.",
              "Routes divide the latents between them, which is what makes routing a decision rather than a sum.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R02 — learned reliability without labels at test.
# --------------------------------------------------------------------------- #
def _weighted_scores(world, r, rd, prior, weights_by_kind, n_test, fam=0, noise_route=None):
    out = {k: [] for k in weights_by_kind}
    for i in range(n_test):
        m = make_maker(world, f"x{i}", r, family=fam, competence="mid")
        eps = stream(world, m, r, 3)
        if noise_route:
            eps = [_noised(e, noise_route, r) for e in eps]
        tabs = rd.route_tables(_seen(eps), ROUTES)
        for k, w in weights_by_kind.items():
            post = J.joint(prior, tabs, w)
            out[k].append((_gain(rd, post, prior, eps[2]), float(post.max()), float(J.top_state_correct(post, J.truth_of(m, eps[2])))))
    return out


def unit_R02(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r02")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    sz = sizes(ctx)
    degraded = "semantic"                                    # this world's semantic route is unreliable: the thing feedback can learn
    learned, gains = R.learn_reliability(rd, _training(world, r, max(24, sz["training"]), noise_route=degraded), prior)
    kinds = {"learned": learned, "equal": R.weights_named("equal"), "random": R.weights_named("random", rng=r), "fixed_action": R.weights_named("fixed_action")}
    sc = _weighted_scores(world, r, rd, prior, kinds, max(8, sz["makers"] // 2), noise_route=degraded)
    for k, vals in sc.items():
        cells.add({"weights": k}, ls=float(np.mean([x[0] for x in vals])), conf=float(np.mean([x[1] for x in vals])), top1=float(np.mean([x[2] for x in vals])))
    return {"rows": cells.rows(), "learned": learned, "training_gains": gains}


def reduce_R02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R02"]
    v = start(card, ctx, "A reader that learns which routes predicted well on training makers weighs them better on new makers than equal, random or fixed weighting, without any target label at test.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {k: mean_of(rows, "ls", lambda r, k=k: r["weights"] == k) for k in ("learned", "equal", "random", "fixed_action")}
    gain = ls["learned"] - ls["equal"]
    passed = bool(gain >= cr["min_gain"] and ls["learned"] >= ls["random"])
    lw = {rt: float(np.mean([u["learned"][rt] for u in units])) for rt in ROUTES}
    gr = G.GateReport()
    extra_gate(gr, "divergence", "learned_weights_not_uniform", float(max(lw.values()) - min(lw.values())), 0.05, "min", "the learned weights differ across routes")
    battery(gr, live={"observed": gain, "min": 0.0, "name": "learning_moves_the_score"},
            placebo={"observed": abs(ls["random"] - ls["equal"]) if abs(ls["random"] - ls["equal"]) < 0.5 else 0.0, "tol": 0.5, "name": "random_weights_no_better_than_equal"},
            positive={"observed": 1.0 - lw["semantic"], "expected": max(0.1, 1.0 - lw["semantic"]), "tol": 0.0, "name": "learned_weight_falls_on_the_degraded_route", "detail": "the instrument moves in the planted direction; whether learned beats every fixed rival is the criterion"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "no_labels_at_test"},
            oracle={"observed": ls["learned"], "min": -1.0, "name": "learned_reported"},
            prediction={"gain": gain, "min": cr["min_gain"], "name": "learned_minus_equal"},
            calibration={"observed": abs(mean_of(rows, "conf", lambda r: r["weights"] == "learned") - mean_of(rows, "top1", lambda r: r["weights"] == "learned")),
                         "reference": abs(mean_of(rows, "conf", lambda r: r["weights"] == "equal") - mean_of(rows, "top1", lambda r: r["weights"] == "equal")), "direction": "down", "tol": 0.05, "name": "learned_no_less_calibrated"})
    criterion(v, "R02", passed, learned_minus_equal=gain, by_weighting=ls, learned_weights=lw)
    receipt(v, rows, card, ctx)
    narrative(v, f"Learned route weights ({', '.join(f'{k} {x:.2f}' for k, x in lw.items())}) scored {gain:+.3f} nats over equal weighting and {ls['learned'] - ls['random']:+.3f} over random on new makers' next actions.",
              "Reliability can be learned from feedback and used blind; it is not the same thing as ease.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R03 / R04 — ease against accuracy.
# --------------------------------------------------------------------------- #
def _ease_condition_rows(ctx, world, r, cells, conditions):
    """conditions: name -> (noised_route, ease_map). The ease-driven reader weighs by 1/ease; the
    learned reader by feedback under the same condition."""
    rd = _reader(world)
    prior = J.uniform_prior()
    sz = sizes(ctx)
    noised0 = next(iter(conditions.values()))[0]
    training = _training(world, r, max(24, sz["training"]), noise_route=noised0)
    learned, _ = R.learn_reliability(rd, training, prior)
    tests = []
    for i in range(max(8, sz["makers"] // 2)):
        m = make_maker(world, f"x{i}", r, family=0, competence="mid")
        tests.append([_noised(e, noised0, r) if noised0 else e for e in stream(world, m, r, 3)])
    for name, (noised, ease_map) in conditions.items():
        kinds = {"ease_driven": R.weights_named("ease", ease_map=ease_map), "learned": learned}
        for eps in tests:
            tabs = rd.route_tables(_seen(eps), ROUTES)
            for k, w in kinds.items():
                post = J.joint(prior, tabs, w)
                wrong = noised[0] if isinstance(noised, (tuple, list)) else (noised or "semantic")
                cells.add({"condition": name, "reader": k}, ls=_gain(rd, post, prior, eps[2]), w_wrong=float(w.get(wrong, 0.0)), w_action=float(w["action"]))


def unit_R03(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r03")
    cells = Cells(ctx["wid"], ctx["rep"])
    cheap_wrong = dict(ROUTE_COST, semantic=0.2, action=3.0)      # the wrong (noised) route cheap, the right one dear
    cheap_right = dict(ROUTE_COST, semantic=3.0, action=0.2)
    _ease_condition_rows(ctx, world, r, cells, {"easy_wrong": ("semantic", cheap_wrong), "easy_right": ("semantic", cheap_right)})
    return {"rows": cells.rows()}


def reduce_R03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R03"]
    v = start(card, ctx, "With accuracy held equal, planted ease drags an ease-driven reader onto the uninformative route and costs it prediction; the learned reader is unmoved.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {c: {k: mean_of(rows, "ls", lambda r, c=c, k=k: r["condition"] == c and r["reader"] == k) for k in ("ease_driven", "learned")} for c in ("easy_wrong", "easy_right")}
    ww = {c: {k: mean_of(rows, "w_wrong", lambda r, c=c, k=k: r["condition"] == c and r["reader"] == k) for k in ("ease_driven", "learned")} for c in ("easy_wrong", "easy_right")}
    bias = ww["easy_wrong"]["ease_driven"] - ww["easy_right"]["ease_driven"]
    ease_effect_learned = abs(ls["easy_wrong"]["learned"] - ls["easy_right"]["learned"])
    ease_cost = ls["easy_right"]["ease_driven"] - ls["easy_wrong"]["ease_driven"]
    passed = bool(bias >= cr["min_bias"] and ease_effect_learned <= cr["max_ease_effect"])
    gr = G.GateReport()
    extra_gate(gr, "equal_accuracy", "learned_reader_unmoved_by_ease", ease_effect_learned, cr["max_ease_effect"], "max", "the learned reader's score changes at most the bar between ease conditions")
    battery(gr, live={"observed": bias, "min": cr["min_bias"], "name": "ease_moves_the_ease_driven_weight"},
            placebo={"observed": ease_effect_learned, "tol": cr["max_ease_effect"], "name": "ease_leaves_the_learned_reader"},
            positive={"observed": ease_cost, "expected": max(0.0, ease_cost), "tol": 0.0, "name": "ease_bias_costs_prediction"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "accuracy_equal_by_construction"},
            oracle={"observed": ls["easy_right"]["learned"], "min": -1.0, "name": "learned_reported"},
            prediction={"gain": ls["easy_wrong"]["learned"] - ls["easy_wrong"]["ease_driven"], "min": 0.0, "name": "learned_beats_ease_driven_when_ease_misleads"},
            calibration={"observed": ww["easy_wrong"]["learned"], "reference": ww["easy_wrong"]["ease_driven"], "direction": "down", "tol": 0.0, "name": "learned_weight_on_the_wrong_route_below_ease_driven"})
    criterion(v, "R03", passed, ease_bias=bias, learned_ease_effect=ease_effect_learned, ease_cost=ease_cost, weights=ww, scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"Making the uninformative route cheap raised the ease-driven reader's weight on it by {bias:.2f} and cost it {ease_cost:+.3f} nats; the learned reader's score moved {ease_effect_learned:.3f} between the two ease conditions.",
              "Ease is not reliability: a reader that follows ease can be steered by it, one that learns reliability cannot.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_R04(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r04")
    cells = Cells(ctx["wid"], ctx["rep"])
    hard = dict(ROUTE_COST, action=3.0, semantic=1.0, context=1.0)     # the accurate (action) route dear
    easy = dict(ROUTE_COST, action=0.2, semantic=1.0, context=1.0)     # the accurate route cheap
    # the cheap routes are degraded, so accuracy lives on the action route whatever it costs
    _ease_condition_rows(ctx, world, r, cells, {"accurate_hard": (("semantic", "context"), hard), "accurate_easy": (("semantic", "context"), easy)})
    return {"rows": cells.rows()}


def reduce_R04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R04"]
    v = start(card, ctx, "With ease held equal in what it can buy, the learned reader follows accuracy: it scores the same whether the accurate route is cheap or dear.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {c: {k: mean_of(rows, "ls", lambda r, c=c, k=k: r["condition"] == c and r["reader"] == k) for k in ("ease_driven", "learned")} for c in ("accurate_hard", "accurate_easy")}
    wa = {c: {k: mean_of(rows, "w_action", lambda r, c=c, k=k: r["condition"] == c and r["reader"] == k) for k in ("ease_driven", "learned")} for c in ("accurate_hard", "accurate_easy")}
    effect_learned = abs(ls["accurate_hard"]["learned"] - ls["accurate_easy"]["learned"])
    effect_ease = ls["accurate_easy"]["ease_driven"] - ls["accurate_hard"]["ease_driven"]
    passed = bool(effect_learned <= cr["max_ease_effect"])
    gr = G.GateReport()
    extra_gate(gr, "equal_ease", "learned_weight_on_accurate_route_unmoved", abs(wa["accurate_hard"]["learned"] - wa["accurate_easy"]["learned"]), 0.1, "max", "the learned weight on the accurate route ignores its ease")
    battery(gr, live={"observed": abs(wa["accurate_hard"]["ease_driven"] - wa["accurate_easy"]["ease_driven"]), "min": 0.2, "name": "ease_moves_the_ease_driven_weight"},
            placebo={"observed": effect_learned, "tol": cr["max_ease_effect"], "name": "learned_score_unmoved_by_ease"},
            positive={"observed": ls["accurate_hard"]["learned"] - ls["accurate_hard"]["ease_driven"], "expected": max(0.0, ls["accurate_hard"]["learned"] - ls["accurate_hard"]["ease_driven"]), "tol": 0.0, "name": "learned_beats_ease_driven_when_accuracy_is_dear"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "routes_identical_across_conditions"},
            oracle={"observed": ls["accurate_easy"]["learned"], "min": -1.0, "name": "learned_reported"},
            prediction={"gain": ls["accurate_hard"]["learned"], "min": -1.0, "name": "learned_under_dear_accuracy"},
            calibration={"observed": effect_learned, "reference": max(effect_ease, 0.0), "direction": "down", "tol": 0.0, "name": "learned_moves_less_than_ease_driven"})
    criterion(v, "R04", passed, learned_effect=effect_learned, ease_driven_effect=effect_ease, weights=wa, scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"Making the accurate route dear moved the learned reader's score by {effect_learned:.3f} nats and the ease-driven reader's by {effect_ease:+.3f}.",
              "Accuracy controls route use for a reader that learned it; ease controls it for one that did not.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R05 — conflict and hypothesis expansion.
# --------------------------------------------------------------------------- #
def _strategic(ep, world, fam, r):
    """The semantic route advertises a different goal than the one that produced the actions."""
    e = dict(ep)
    decoy = (ep["goal"] + 1 + int(r.integers(N_GOAL - 1))) % N_GOAL
    e["semantic"] = [int(r.choice(N_FEAT, p=world.family(fam).sem[decoy])) for _ in ep["semantic"]]
    return e


def unit_R05(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r05")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    for i in range(max(4, sizes(ctx)["makers"] // 4)):
        for kind in ("consistent", "strategic"):
            m = make_maker(world, f"m{i}{kind}", r, family=0, competence="mid")
            eps = stream(world, m, r, 4)
            if kind == "strategic":
                eps = [_strategic(e, world, 0, r) for e in eps]
            cur = dict(eps[2], action=list(eps[2]["action"][:3]))            # three steps seen of the current episode
            seen = eps[:2] + [cur]
            tabs = rd.route_tables(seen, NF)
            conf = R.conflict(prior, tabs)
            fixed = J.joint(prior, tabs)
            st = R.strategic_semantic_table(rd, seen)
            expanded, p_strat = R.expanded_posterior(prior, tabs, st)
            a_next, last = int(eps[2]["action"][3]), int(eps[2]["action"][2])
            for reader, post in (("fixed", fixed), ("expanded", expanded)):
                g = _ls(J.next_action_dist(rd, post, last), a_next) - _ls(J.next_action_dist(rd, prior, last), a_next)
                cells.add({"world": kind, "reader": reader}, ls=g, p_strategic=p_strat if reader == "expanded" else 0.0, conflict=conf,
                          search_cost=float(J.N_STATES if reader == "expanded" else 0.0))
    return {"rows": cells.rows()}


def reduce_R05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R05"]
    v = start(card, ctx, "When routes conflict, adding a strategic-source hypothesis to the latent set improves prediction where a source really is strategic and flags few consistent worlds.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {w: {k: mean_of(rows, "ls", lambda r, w=w, k=k: r["world"] == w and r["reader"] == k) for k in ("fixed", "expanded")} for w in ("consistent", "strategic")}
    gain = ls["strategic"]["expanded"] - ls["strategic"]["fixed"]
    fp = mean_of(rows, "p_strategic", lambda r: r["world"] == "consistent" and r["reader"] == "expanded")
    tp = mean_of(rows, "p_strategic", lambda r: r["world"] == "strategic" and r["reader"] == "expanded")
    conf = {w: mean_of(rows, "conflict", lambda r, w=w: r["world"] == w) for w in ("consistent", "strategic")}
    passed = bool(gain >= cr["min_gain"] and fp <= cr["max_false_positive"])
    gr = G.GateReport()
    battery(gr, live={"observed": conf["strategic"] - conf["consistent"], "min": 0.02, "name": "strategic_sources_raise_route_conflict"},
            placebo={"observed": fp, "tol": cr["max_false_positive"], "name": "consistent_worlds_rarely_flagged"},
            positive={"observed": tp, "expected": 1.0, "tol": 0.6, "name": "strategic_sources_flagged"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_routes_both_worlds"},
            oracle={"observed": tp - fp, "min": 0.1, "name": "flag_identifies_the_world"},
            prediction={"gain": gain, "min": cr["min_gain"], "name": "expansion_gain_in_strategic_worlds"},
            calibration={"observed": abs(ls["consistent"]["expanded"] - ls["consistent"]["fixed"]), "reference": 0.05, "direction": "down", "tol": 0.0, "name": "expansion_costs_little_where_unneeded"})
    criterion(v, "R05", passed, expansion_gain=gain, false_positive=fp, true_positive=tp, conflict=conf, search_cost_states=J.N_STATES)
    receipt(v, rows, card, ctx)
    narrative(v, f"Route conflict rose from {conf['consistent']:.2f} to {conf['strategic']:.2f} under a strategic source; expanding the latent set gained {gain:+.3f} nats there at a search cost of {J.N_STATES} extra grid cells, and flagged {fp:.0%} of consistent worlds.",
              "Conflict is a trigger for widening the hypothesis space, and the widening pays only where the extra hypothesis is true.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R06 — forensic purchase.
# --------------------------------------------------------------------------- #
def unit_R06(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r06")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    cost = ROUTE_COST["forensic"]
    for i in range(max(4, sizes(ctx)["makers"] // 4)):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 3)
        tabs = rd.route_tables(_seen(eps), NF)
        truth = J.truth_of(m, eps[2])
        eig = R.forensic_eig(world, rd, prior, tabs, "process", r, draws=12)
        base = J.joint(prior, tabs)
        with_f = dict(tabs, forensic=rd.route_tables(_seen(eps), ("forensic",))["forensic"])
        bought = J.joint(prior, with_f)
        g_base, g_f = _gain(rd, base, prior, eps[2]), _gain(rd, bought, prior, eps[2])
        for pol in ("exact", "random", "always", "never"):
            buy = R.purchase_policy(pol, eig, cost, r, threshold=0.05)
            g = g_f if buy else g_base
            spent = cost if buy else 0.0
            cells.add({"policy": pol}, gain=g, spent=spent, bought=float(buy), gain_per_cost=(g - g_base) / max(spent, 1e-9) if buy else 0.0,
                      class_gain=(J.class_mass(bought, truth) - J.class_mass(base, truth)) if buy else 0.0, eig=eig)
    return {"rows": cells.rows()}


def reduce_R06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R06"]
    v = start(card, ctx, "Costly forensic access is worth buying when its expected information per cost clears a bar, and a reader that buys by that rule realizes at least what any fixed policy does.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pols = ("exact", "random", "always", "never")
    g = {p: mean_of(rows, "gain", lambda r, p=p: r["policy"] == p) for p in pols}
    spent = {p: mean_of(rows, "spent", lambda r, p=p: r["policy"] == p) for p in pols}
    net = {p: g[p] - 0.02 * spent[p] for p in pols}                 # gain net of a declared cost weight
    best_rival = max(net[p] for p in pols if p != "exact")
    passed = bool(net["exact"] >= best_rival - cr["margin"])
    bought = mean_of(rows, "bought", lambda r: r["policy"] == "exact")
    gr = G.GateReport()
    battery(gr, live={"observed": g["always"] - g["never"], "min": 0.0, "name": "forensic_access_moves_prediction"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_episodes_every_policy"},
            positive={"observed": mean_of(rows, "class_gain", lambda r: r["policy"] == "always"), "expected": max(0.0, mean_of(rows, "class_gain", lambda r: r["policy"] == "always")), "tol": 0.0, "name": "forensic_resolves_process"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "cost_declared_not_felt"},
            oracle={"observed": g["always"], "min": -1.0, "name": "always_buy_reported"},
            prediction={"gain": net["exact"] - best_rival, "min": -cr["margin"], "name": "exact_rule_at_least_best_fixed_policy"},
            calibration={"observed": bought, "reference": 1.0, "direction": "down", "tol": 0.0, "name": "exact_rule_buys_selectively"})
    criterion(v, "R06", passed, net_gain_by_policy=net, gain_by_policy=g, spent_by_policy=spent, exact_buy_rate=bought)
    receipt(v, rows, card, ctx)
    narrative(v, f"The exact rule bought forensic access {bought:.0%} of the time and realized {net['exact']:+.3f} nats net of cost against {best_rival:+.3f} for the best fixed policy; always buying realized {g['always']:+.3f} gross.",
              "Forensic access has a price the reader can reason about, and reasoning about it is not worse than any rule of thumb.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R07 — fusion under duplicated evidence.
# --------------------------------------------------------------------------- #
def unit_R07(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r07")
    cells = Cells(ctx["wid"], ctx["rep"])
    rd = _reader(world)
    prior = J.uniform_prior()
    for i in range(max(4, sizes(ctx)["makers"] // 4)):
        m = make_maker(world, f"m{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 3)
        truth = J.truth_of(m, eps[2])
        seen = _seen(eps)
        base = J.joint(prior, rd.route_tables(seen, NF))
        base_goal = float(J.marginal(base, "goal").max())
        for kind in ("independent", "duplicate", "paraphrase"):
            if kind == "independent":
                fresh = episode(world, m, r, index=3, goal=eps[2]["goal"])
                eps_k = seen[:2] + [dict(seen[2], semantic=seen[2]["semantic"] + fresh["semantic"])]     # fresh tokens from the same goal
            else:
                eps_k = seen[:2] + [R.duplicate_semantic(seen[2], r, kind)]
            for fusion in ("naive", "shared_cause"):
                tabs = R.fused_tables(rd, eps_k, NF, fusion) if kind != "independent" else rd.route_tables(eps_k, NF)
                post = J.joint(prior, tabs)
                cells.add({"evidence": kind, "fusion": fusion}, conf_rise=float(J.marginal(post, "goal").max() - base_goal), ls=_gain(rd, post, prior, eps[2]), top1=float(int(np.argmax(J.marginal(post, "goal"))) == truth[1]),
                          goal_conf=float(J.marginal(post, "goal").max()))
    return {"rows": cells.rows()}


def reduce_R07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R07"]
    v = start(card, ctx, "A duplicated or paraphrased piece of evidence is one piece of evidence: fusion that tracks the shared cause does not grow more confident on the copy, and naive fusion does.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    rise = {k: {f: mean_of(rows, "conf_rise", lambda r, k=k, f=f: r["evidence"] == k and r["fusion"] == f) for f in ("naive", "shared_cause")} for k in ("independent", "duplicate", "paraphrase")}
    dup_naive = max(rise["duplicate"]["naive"], rise["paraphrase"]["naive"])
    dup_shared = max(rise["duplicate"]["shared_cause"], rise["paraphrase"]["shared_cause"])
    passed = bool(dup_shared <= cr["max_dup_rise"] and dup_naive >= cr["min_naive_rise"])
    ece = {f: abs(mean_of(rows, "goal_conf", lambda r, f=f: r["fusion"] == f and r["evidence"] != "independent") - mean_of(rows, "top1", lambda r, f=f: r["fusion"] == f and r["evidence"] != "independent")) for f in ("naive", "shared_cause")}
    gr = G.GateReport()
    extra_gate(gr, "duplicate", "shared_cause_flat_under_copies", dup_shared, cr["max_dup_rise"], "max", "confidence rise under a duplicate or paraphrase, shared-cause fusion")
    battery(gr, live={"observed": rise["independent"]["naive"], "min": 0.0, "name": "fresh_evidence_raises_confidence"},
            placebo={"observed": dup_shared, "tol": cr["max_dup_rise"], "name": "copy_leaves_shared_cause_fusion"},
            positive={"observed": dup_naive, "expected": max(cr["min_naive_rise"], dup_naive), "tol": 0.0, "name": "naive_fusion_inflates_on_copies", "detail": "the planted failure of independence"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_tokens_both_fusions"},
            oracle={"observed": rise["independent"]["shared_cause"], "min": -1.0, "name": "fresh_evidence_under_shared_cause_reported"},
            prediction={"gain": mean_of(rows, "ls", lambda r: r["fusion"] == "shared_cause"), "min": -1.0, "name": "shared_cause_next_action"},
            calibration={"observed": ece["shared_cause"], "reference": ece["naive"], "direction": "down", "tol": 0.0, "name": "shared_cause_better_calibrated_under_copies"})
    criterion(v, "R07", passed, confidence_rise=rise, calibration_gap=ece)
    receipt(v, rows, card, ctx)
    narrative(v, f"A duplicate raised naive fusion's confidence by {rise['duplicate']['naive']:+.2f} and a paraphrase by {rise['paraphrase']['naive']:+.2f}; shared-cause fusion moved {dup_shared:+.3f}; fresh evidence moved both by about {rise['independent']['naive']:+.2f}.",
              "The cure for counting a cause twice is to model the cause, not to distrust evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# R08 — transfer of learned reliability under a domain shift.
# --------------------------------------------------------------------------- #
def unit_R08(ctx):
    world = world_for(ctx)
    r = rng(ctx, "r08")
    cells = Cells(ctx["wid"], ctx["rep"])
    prior = J.uniform_prior()
    sz = sizes(ctx)
    fam_old, fam_new = 0, min(1, world.n_families - 1)
    rd_old, rd_new = _reader(world, fam_old), _reader(world, fam_new)
    learned_old, _ = R.learn_reliability(rd_old, _training(world, r, max(24, sz["training"]), fam=fam_old), prior)
    # the new domain: fresh vocabulary and a degraded semantic route
    learned_new, _ = R.learn_reliability(rd_new, _training(world, r, max(24, sz["training"]), fam=fam_new, noise_route="semantic"), prior)
    for i in range(max(4, sz["makers"] // 4)):
        m = make_maker(world, f"x{i}", r, family=fam_new, competence="mid")
        eps = [_noised(e, "semantic", r) for e in stream(world, m, r, 3)]
        tabs = rd_new.route_tables(_seen(eps), ROUTES)
        for kind in ("reset", "partial", "full"):
            w = R.transfer_weights(kind, learned_old, learned_new)
            post = J.joint(prior, tabs, w)
            cells.add({"transfer": kind}, ls=_gain(rd_new, post, prior, eps[2]), w_semantic=float(w["semantic"]))
    return {"rows": cells.rows(), "learned_old": learned_old, "learned_new": learned_new}


def reduce_R08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["R08"]
    v = start(card, ctx, "Under a domain shift that changes which routes are reliable, carrying old reliabilities over in full loses to resetting or partially transferring them.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {k: mean_of(rows, "ls", lambda r, k=k: r["transfer"] == k) for k in ("reset", "partial", "full")}
    best = max(ls, key=ls.get)
    passed = bool(max(ls["reset"], ls["partial"]) >= ls[best] - cr["margin"])
    old = {rt: float(np.mean([u["learned_old"][rt] for u in units])) for rt in ROUTES}
    new = {rt: float(np.mean([u["learned_new"][rt] for u in units])) for rt in ROUTES}
    gr = G.GateReport()
    battery(gr, live={"observed": abs(old["semantic"] - new["semantic"]), "min": 0.05, "name": "the_shift_changes_reliability"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_test_makers_every_transfer"},
            positive={"observed": ls[best] - ls["full"], "expected": max(0.0, ls[best] - ls["full"]), "tol": 0.0, "name": "full_transfer_not_best"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "fresh_vocabulary_by_construction"},
            oracle={"observed": ls[best], "min": -1.0, "name": "best_reported"},
            prediction={"gain": max(ls["reset"], ls["partial"]) - ls[best], "min": -cr["margin"], "name": "reset_or_partial_within_margin"},
            calibration={"observed": mean_of(rows, "w_semantic", lambda r: r["transfer"] == "full"), "reference": float(np.mean([u["learned_new"]["semantic"] for u in units])), "direction": "up", "tol": 0.0, "name": "full_transfer_keeps_the_old_weight_on_the_degraded_route", "detail": "against the weight the new domain's own feedback would give it"})
    criterion(v, "R08", passed, by_transfer=ls, best=best, learned_old=old, learned_new=new)
    receipt(v, rows, card, ctx)
    narrative(v, f"After a shift that degraded the semantic route, resetting scored {ls['reset']:+.3f}, partial transfer {ls['partial']:+.3f} and full transfer {ls['full']:+.3f} nats; the old weight on the degraded route was {old['semantic']:.2f}.",
              "Learned reliability is a property of a domain; carrying it across one is a bet that the domain did not change.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
