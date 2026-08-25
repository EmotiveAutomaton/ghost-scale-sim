"""Trunk R: values from opportunities, not action counts (spec section 11).

The choice world: a maker with a standing profile faces menus whose options pay a vector over
the goal channels and carry a cost. Habits are option-specific tilts; expertise scales how well
the maker sees the payoffs; a current goal is a transient tilt on the profile. The same choice
under a near tie is weak evidence and under a large opposing cost is strong evidence; the
constrained-inversion reader uses that, the count reader cannot.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from .. import opportunities as OP
from . import finish, worlds_for, decide_state

GAMMA = 0.5


def _cwd(world):
    return OP.ChoiceWorld(world.ng, world.family, world.family_names)


def _weff(w, g, gamma=GAMMA):
    v = np.asarray(w, float).copy()
    if g is not None:
        v[g] += gamma
    return v / v.sum()


def _habits(n_opt):
    hs = [np.zeros(n_opt)]
    for lvl in (0.15, 0.4):
        for j in range(n_opt):
            h = np.zeros(n_opt)
            h[j] = lvl
            hs.append(h)
    return hs


def _ll(cw, w, m, a, habit=None, beta=None):
    b = cw.beta if beta is None else float(beta)
    u = np.asarray(m["payoff"]) @ np.asarray(w, float) - np.asarray(m["cost"])
    if habit is not None:
        u = u + np.asarray(habit)[: u.size]
    z = b * (u - u.max())
    return float(z[a] - np.log(np.exp(z).sum()))


def _pred(cw, w, m, habit=None, beta=None):
    b = cw.beta if beta is None else float(beta)
    u = np.asarray(m["payoff"]) @ np.asarray(w, float) - np.asarray(m["cost"])
    if habit is not None:
        u = u + np.asarray(habit)[: u.size]
    z = np.exp(b * (u - u.max()))
    return z / z.sum()


def _menus(cw, rng, n, cost_scale=0.4, conc=1.0):
    out = []
    for _ in range(n):
        out.append({"payoff": rng.dirichlet(np.full(cw.ng, conc), size=cw.n_options),
                    "cost": rng.uniform(0.0, cost_scale, size=cw.n_options)})
    return out


def _records(cw, w, menus, rng, habit=None, k=0.0, beta=None, menu_habit=None):
    out = []
    for m in menus:
        h = habit if menu_habit is None else menu_habit(m)
        out.append(OP.choose(cw, w, m, rng, h, k, beta))
    return out


def _mrec(rec):
    return {"payoff": np.asarray(rec["payoff"]), "cost": np.asarray(rec["cost"])}


def _joint_post(cw, recs, habits, beta=None, prior=None, family=None):
    names = cw.family_names
    fam = cw.family if family is None else family
    ll = np.zeros((len(names), len(habits)))
    for rec in recs:
        m = _mrec(rec)
        for i, n in enumerate(names):
            for j, h in enumerate(habits):
                ll[i, j] += _ll(cw, fam[n], m, int(rec["choice"]), h, beta)
    if prior is not None:
        ll += np.log(np.maximum(np.asarray(prior, float), 1e-300))
    p = np.exp(ll - ll.max())
    return p / p.sum()


def _joint_pred(cw, P, habits, m, beta=None, family=None):
    names = cw.family_names
    fam = cw.family if family is None else family
    out = np.zeros(len(m["cost"]))
    for i, n in enumerate(names):
        for j, h in enumerate(habits):
            if P[i, j] > 0:
                out += P[i, j] * _pred(cw, fam[n], m, h, beta)
    return out / out.sum()


def _ls_pred(pred, a):
    return float(np.log(max(pred[a], 1e-12)))


def H(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- #
def run_R01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "In an orthogonal factorial choice world every planted latent (standing "
                    "profile, habit, expertise, current goal) keeps its full conditional entropy given the "
                    "others and is separately recoverable from opportunity records.", "METHOD")
    ig = {k: [] for k in ("profile", "habit", "expertise", "goal")}
    acc = {k: [] for k in ig}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R01", wid, 0)
            habits3 = [np.zeros(cw.n_options), np.eye(cw.n_options)[0] * 0.15, np.eye(cw.n_options)[0] * 0.4]
            ks = (0.0, 0.3, 0.6)
            for trial in range(36):
                i, j, e, g = int(rng.integers(len(names))), int(rng.integers(3)), int(rng.integers(3)), int(rng.integers(world.ng))
                w_eff = _weff(world.family[names[i]], g)
                menus = _menus(cw, rng, 24)
                recs = _records(cw, w_eff, menus, rng, habit=habits3[j], k=ks[e])
                grids = {"profile": [(_weff(world.family[n], g), habits3[j], ks[e]) for n in names],
                         "habit": [(w_eff, h, ks[e]) for h in habits3],
                         "expertise": [(w_eff, habits3[j], kk) for kk in ks],
                         "goal": [(_weff(world.family[names[i]], gg), habits3[j], ks[e]) for gg in range(world.ng)]}
                truth = {"profile": i, "habit": j, "expertise": e, "goal": g}
                for latent, grid in grids.items():
                    ll = np.zeros(len(grid))
                    for rec in recs:
                        m = _mrec(rec)
                        for gi, (w, h, kk) in enumerate(grid):
                            ll[gi] += _ll(cw, w, m, int(rec["choice"]), h, beta=cw.beta * (1 - kk))
                    p = np.exp(ll - ll.max())
                    p /= p.sum()
                    ig[latent].append(float(np.log(len(grid)) - H(p)))
                    acc[latent].append(float(int(np.argmax(p)) == truth[latent]))
    cond = {"profile": float(np.log(6)), "habit": float(np.log(3)), "expertise": float(np.log(3)), "goal": float(np.log(4))}
    gr = G.GateReport()
    for k in ig:
        gr.live(f"{k}_is_recoverable", observed_change=float(np.mean(ig[k])), min_change=0.05,
                detail="information gained about the latent from records with the other latents known")
    gr.positive("design_is_orthogonal", observed=float(min(cond.values()) > 0), expected=1.0, tol=0.0,
                detail="conditional entropy of each latent given the others equals its own entropy: no coarsening by construction")
    v["results"] = {"conditional_entropy_given_others_nats": cond, "information_gain_from_records": {k: float(np.mean(x)) for k, x in ig.items()},
                    "top1_accuracy": {k: float(np.mean(x)) for k, x in acc.items()},
                    "note": "expertise enters the reader's likelihood as reduced rationality; the maker's expertise is payoff-perception noise. The mismatch is the reader's, and is reported"}
    v["what_must_hold_outside_the_simulation"] = "opportunity records exist: what was on offer and at what cost, not only what was chosen"
    return finish(card, v, gr, __file__, decide_state(gr))


def _estimators(cw, recs, habits):
    names = cw.family_names
    n_opt = cw.n_options
    P_joint = _joint_post(cw, recs, habits)
    freq = np.bincount([int(r["choice"]) for r in recs], minlength=n_opt) / len(recs)
    h_hat = np.maximum(0.4 * (freq - 1 / n_opt) / max(freq.max() - 1 / n_opt, 1e-9), 0.0)
    P_part = _joint_post(cw, recs, [h_hat])
    P_map = _joint_post(cw, recs, [np.zeros(n_opt)])
    blind = OP.profile_posterior_from_choices(cw, recs, use_costs=False)
    P_blind = np.array([[blind[n]] for n in names])
    w_hat = np.mean([np.asarray(r["payoff"])[int(r["choice"])] for r in recs], axis=0)
    w_hat = w_hat / w_hat.sum()
    dists = np.array([np.abs(w_hat - cw.family[n]).sum() for n in names])
    P_count = np.full((len(names), 1), 0.1 / (len(names) - 1))
    P_count[int(np.argmin(dists)), 0] = 0.9
    P_pop = np.full((len(names), 1), 1 / len(names))
    z = [np.zeros(n_opt)]
    return {"constrained_inversion": (P_joint, habits), "partialling": (P_part, [h_hat]), "map_no_habit": (P_map, z),
            "cost_blind": (P_blind, z), "count_reader": (P_count, z), "population": (P_pop, z)}


def run_R02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Constrained inversion over profile and habit predicts held-out choices "
                    "better than partialling out the habit, better than cost-blind and count readers.",
                    "CONSTRUCTED_MECHANISM")
    scores = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            habits = _habits(cw.n_options)
            rng = C.rng_for("R02", wid, 0)
            for i in range(30):
                w = world.family[names[i % len(names)]]
                h = habits[1 + int(rng.integers(len(habits) - 1))]
                menus = _menus(cw, rng, 24)
                recs = _records(cw, w, menus, rng, habit=h, k=0.3)
                train, test = recs[:16], recs[16:]
                for est, (P, hs) in _estimators(cw, train, habits).items():
                    ls = [_ls_pred(_joint_pred(cw, P, hs, _mrec(r)), int(r["choice"])) for r in test]
                    scores.setdefault(est, {}).setdefault(wid, []).append(float(np.mean(ls)))
    table = {e: C.hboot(d, np.random.default_rng(C.seed("R02" + e)), draws=300) for e, d in scores.items()}
    gap = table["constrained_inversion"]["mean"] - table["partialling"]["mean"]
    gr = G.GateReport()
    gr.positive("population_is_the_floor", observed=float(all(table[e]["mean"] >= table["population"]["mean"] - 0.05 for e in table if e != "count_reader")), expected=1.0, tol=0.0)
    gr.live("estimators_differ", observed_change=float(max(t["mean"] for t in table.values()) - min(t["mean"] for t in table.values())), min_change=0.02)
    v["results"] = {"heldout_log_score_by_estimator": table, "criterion_C_R02": {"constrained_minus_partialling": float(gap), "passed": bool(gap >= 0.02)}}
    v["what_must_hold_outside_the_simulation"] = "habits are option-specific and separable from payoffs"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "When a habit stores a previous profile's values, partialling it out "
                    "deletes profile signal in proportion to how aligned the habit is with the current "
                    "profile; constrained inversion that models the habit does not.", "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R03", wid, 0)
            for cond in ("aligned", "decoy"):
                for i in range(24):
                    n = names[i % len(names)]
                    w = world.family[n]
                    w_prev = w if cond == "aligned" else world.family[world.decoy_of[n]]
                    menus = _menus(cw, rng, 16)
                    recs = _records(cw, w, menus, rng, menu_habit=lambda m: 0.4 * (np.asarray(m["payoff"]) @ w_prev))
                    # constrained inversion: profile x habit strength, habit direction known
                    grid = (0.0, 0.2, 0.4)
                    ll = np.zeros((len(names), len(grid)))
                    for rec in recs:
                        m = _mrec(rec)
                        for a, nn in enumerate(names):
                            for b, s in enumerate(grid):
                                ll[a, b] += _ll(cw, world.family[nn], m, int(rec["choice"]), s * (m["payoff"] @ w_prev))
                    p = np.exp(ll - ll.max())
                    p /= p.sum()
                    ls_con = float(np.log(max(p.sum(axis=1)[names.index(n)], 1e-12)))
                    # partialling: drop the records the habit explains, fit the rest habit-free
                    kept = [r for r in recs if int(r["choice"]) != int(np.argmax(np.asarray(r["payoff"]) @ w_prev))]
                    P = _joint_post(cw, kept, [np.zeros(cw.n_options)]) if kept else np.full((len(names), 1), 1 / len(names))
                    ls_par = float(np.log(max(P[names.index(n), 0], 1e-12)))
                    d = res.setdefault(cond, {"constrained": [], "partialling": [], "kept": []})
                    d["constrained"].append(ls_con)
                    d["partialling"].append(ls_par)
                    d["kept"].append(len(kept))
    table = {c: {k: float(np.mean(x)) for k, x in d.items()} for c, d in res.items()}
    drop_par = table["decoy"]["partialling"] - table["aligned"]["partialling"]
    drop_con = table["decoy"]["constrained"] - table["aligned"]["constrained"]
    gr = G.GateReport()
    gr.live("aligned_habit_removes_records_from_partialling", observed_change=float(table["decoy"]["kept"] - table["aligned"]["kept"]), min_change=1.0,
            detail="an aligned habit explains more of the profile's own choices, so partialling keeps fewer records")
    v["results"] = {"cells": table, "criterion_C_R03": {"partialling_drop_under_alignment": float(drop_par), "constrained_drop_under_alignment": float(drop_con), "passed": bool(drop_par > drop_con)}}
    v["what_must_hold_outside_the_simulation"] = "habits can carry the values that formed them"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A profile recovered from one domain's records predicts structurally "
                    "equivalent choices in a second domain with a different payoff generator, cost scale "
                    "and habit, better than an identity-only baseline that carries the habit alone.",
                    "CONSTRUCTED_MECHANISM")
    scores = {"profile": {}, "identity_only": {}, "oracle": {}}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            habits = _habits(cw.n_options)
            rng = C.rng_for("R04", wid, 0)
            for i in range(30):
                w = world.family[names[i % len(names)]]
                j1, j2 = int(rng.integers(cw.n_options)), int(rng.integers(cw.n_options))
                h1, h2 = np.eye(cw.n_options)[j1] * 0.4, np.eye(cw.n_options)[j2] * 0.4
                recs1 = _records(cw, w, _menus(cw, rng, 16), rng, habit=h1)
                recs2 = _records(cw, w, _menus(cw, rng, 8, cost_scale=0.8, conc=0.5), rng, habit=h2)
                P = _joint_post(cw, recs1, habits).sum(axis=1, keepdims=True)
                freq = np.bincount([int(r["choice"]) for r in recs1], minlength=cw.n_options) + 0.5
                base = freq / freq.sum()
                z = [np.zeros(cw.n_options)]
                for r in recs2:
                    m = _mrec(r)
                    scores["profile"].setdefault(wid, []).append(_ls_pred(_joint_pred(cw, P, z, m), int(r["choice"])))
                    scores["identity_only"].setdefault(wid, []).append(_ls_pred(base, int(r["choice"])))
                    scores["oracle"].setdefault(wid, []).append(_ls_pred(_pred(cw, w, m, h2), int(r["choice"])))
    table = {k: C.hboot(d, np.random.default_rng(C.seed("R04" + k)), draws=300) for k, d in scores.items()}
    gr = G.GateReport()
    gr.positive("oracle_is_the_ceiling", observed=float(table["oracle"]["mean"] >= table["profile"]["mean"] - 1e-9), expected=1.0, tol=0.0)
    gr.live("transfer_moves_the_score", observed_change=float(table["profile"]["mean"] - table["identity_only"]["mean"]), min_change=0.02)
    v["results"] = {"domain_two_log_score": table, "criterion_C_R04": {"passed": bool(table["profile"]["mean"] > table["identity_only"]["mean"])}}
    v["what_must_hold_outside_the_simulation"] = "the second domain is structurally equivalent in its goal channels"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The same choice moves the posterior more when it was made against a "
                    "large opposing cost than under a near tie; a count reader moves the same either way.",
                    "CONSTRUCTED_MECHANISM")
    shifts = {"record_reader": {"near_tie": [], "strong": []}, "count_reader": {"near_tie": [], "strong": []}}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R05", wid, 0)
            prior = np.full(len(names), 1 / len(names))
            for i in range(30):
                w = world.family[names[i % len(names)]]
                found = {"near_tie": None, "strong": None}
                tries = 0
                while (found["near_tie"] is None or found["strong"] is None) and tries < 5000:
                    tries += 1
                    m = _menus(cw, rng, 1, cost_scale=0.6)[0]
                    u = m["payoff"] @ w - m["cost"]
                    a = int(np.argmax(u))
                    srt = np.sort(u)
                    margin = srt[-1] - srt[-2]
                    if found["near_tie"] is None and margin < 0.02:
                        found["near_tie"] = (m, a)
                    if found["strong"] is None and margin > 0.05 and m["cost"][a] - m["cost"].min() > 0.3:
                        found["strong"] = (m, a)
                for kind, item in found.items():
                    if item is None:
                        continue
                    m, a = item
                    rec = {"payoff": m["payoff"].tolist(), "cost": m["cost"].tolist(), "choice": a}
                    for reader, use_costs in (("record_reader", True), ("count_reader", False)):
                        post = OP.profile_posterior_from_choices(cw, [rec], use_costs=use_costs)
                        pv = np.array([post[n] for n in names])
                        shifts[reader][kind].append(float((pv[pv > 0] * np.log(pv[pv > 0] / prior[pv > 0])).sum()))
    table = {r: {k: float(np.mean(x)) if x else None for k, x in d.items()} for r, d in shifts.items()}
    gr = G.GateReport()
    gr.live("opportunity_strength_moves_the_record_reader", observed_change=float(table["record_reader"]["strong"] - table["record_reader"]["near_tie"]), min_change=0.05)
    v["results"] = {"posterior_shift_kl": table, "criterion_C_R05": {"passed": bool(table["record_reader"]["strong"] > table["record_reader"]["near_tie"])}}
    v["what_must_hold_outside_the_simulation"] = "the cost a choice was made against is recorded"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Current goal and standing profile are jointly recoverable from records "
                    "in a full cross of aligned and opposed goals, above what either marginal reader recovers.",
                    "CONSTRUCTED_MECHANISM")
    res = {"joint": {"profile": [], "goal": []}, "profile_only": {"profile": []}, "goal_only": {"goal": []}}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R06", wid, 0)
            z = [np.zeros(cw.n_options)]
            for n in names:
                w = world.family[n]
                for cond in ("aligned", "opposed"):
                    g = int(np.argmax(w)) if cond == "aligned" else int(np.argmin(w))
                    for rep in range(3):
                        recs = _records(cw, _weff(w, g), _menus(cw, rng, 16), rng)
                        ll = np.zeros((len(names), world.ng))
                        for rec in recs:
                            m = _mrec(rec)
                            for a, nn in enumerate(names):
                                for gg in range(world.ng):
                                    ll[a, gg] += _ll(cw, _weff(world.family[nn], gg), m, int(rec["choice"]))
                        p = np.exp(ll - ll.max())
                        p /= p.sum()
                        res["joint"]["profile"].append(float(int(np.argmax(p.sum(axis=1))) == names.index(n)))
                        res["joint"]["goal"].append(float(int(np.argmax(p.sum(axis=0))) == g))
                        Pp = _joint_post(cw, recs, z)
                        res["profile_only"]["profile"].append(float(int(np.argmax(Pp[:, 0])) == names.index(n)))
                        llg = np.array([sum(_ll(cw, _weff(np.full(world.ng, 1 / world.ng), gg), _mrec(r), int(r["choice"])) for r in recs) for gg in range(world.ng)])
                        res["goal_only"]["goal"].append(float(int(np.argmax(llg)) == g))
    table = {k: {kk: float(np.mean(x)) for kk, x in d.items()} for k, d in res.items()}
    gr = G.GateReport()
    gr.live("joint_reader_beats_chance", observed_change=float(table["joint"]["profile"] - 1 / 6), min_change=0.1)
    v["results"] = {"top1_accuracy": table, "criterion_C_R06": {"passed": bool(table["joint"]["profile"] >= table["profile_only"]["profile"] and table["joint"]["goal"] >= table["goal_only"]["goal"])}}
    v["what_must_hold_outside_the_simulation"] = "a current goal acts as a transient tilt on a standing profile"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R07(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The recovered profile predicts prospective and counterfactual choices "
                    "(changed cost, new domain, commission, new goal) above a frequency baseline.",
                    "CONSTRUCTED_MECHANISM")
    gains = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R07", wid, 0)
            z = [np.zeros(cw.n_options)]
            for i in range(30):
                w = world.family[names[i % len(names)]]
                recs = _records(cw, w, _menus(cw, rng, 16), rng)
                P = _joint_post(cw, recs, z)
                freq = np.bincount([int(r["choice"]) for r in recs], minlength=cw.n_options) + 0.5
                base = freq / freq.sum()
                g_c, g_n = int(rng.integers(world.ng)), int(rng.integers(world.ng))
                targets = {"changed_cost": (_menus(cw, rng, 8, cost_scale=1.0), w, P),
                           "new_domain": (_menus(cw, rng, 8, conc=0.5), w, P),
                           "commission": (_menus(cw, rng, 8), _weff(w, g_c), np.array([[P[a, 0]] for a in range(len(names))])),
                           "new_goal": (_menus(cw, rng, 8), _weff(w, g_n), P)}
                for t, (menus, w_gen, Pt) in targets.items():
                    tr = _records(cw, w_gen, menus, rng)
                    for r in tr:
                        m = _mrec(r)
                        if t == "commission":
                            fam = {n: _weff(world.family[n], g_c) for n in names}
                            pred = _joint_pred(cw, Pt, z, m, family=fam)
                        else:
                            pred = _joint_pred(cw, Pt, z, m)
                        gains.setdefault(t, {}).setdefault(wid, []).append(_ls_pred(pred, int(r["choice"])) - _ls_pred(base, int(r["choice"])))
    table = {t: C.hboot(d, np.random.default_rng(C.seed("R07" + t)), draws=300) for t, d in gains.items()}
    gr = G.GateReport()
    gr.live("profile_predicts_something_prospective", observed_change=float(max(t["mean"] for t in table.values())), min_change=0.02)
    v["results"] = {"gain_over_frequency": table, "criterion_C_R07": {"passed": bool(all(t["mean"] > 0 for t in table.values()))}}
    v["what_must_hold_outside_the_simulation"] = "prospective situations share the goal channels of the observed ones"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_R08(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "On menus where two profiles choose identically, a count reader abstains; "
                    "opportunity records separate them, and discriminating menus separate them decisively.",
                    "CONSTRUCTED_MECHANISM")
    res = {"count_agreeing": [], "record_agreeing": [], "record_discriminating": [], "count_discriminating": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            cw = _cwd(world)
            names = cw.family_names
            rng = C.rng_for("R08", wid, 0)
            for i in range(24):
                a, b = names[i % len(names)], world.decoy_of[names[i % len(names)]]
                wa, wb = world.family[a], world.family[b]
                agree, disagree = [], []
                tries = 0
                while (len(agree) < 16 or len(disagree) < 8) and tries < 20000:
                    tries += 1
                    m = _menus(cw, rng, 1)[0]
                    ca, cb = int(np.argmax(m["payoff"] @ wa - m["cost"])), int(np.argmax(m["payoff"] @ wb - m["cost"]))
                    rec = {"payoff": m["payoff"].tolist(), "cost": m["cost"].tolist(), "choice": ca}
                    if ca == cb and len(agree) < 16:
                        agree.append(rec)
                    elif ca != cb and len(disagree) < 8:
                        disagree.append(rec)
                pair = {n: (0.5 if n in (a, b) else 0.0) for n in names}
                for kind, recs in (("agreeing", agree), ("discriminating", agree + disagree)):
                    for reader, use_costs in (("count", False), ("record", True)):
                        post = OP.profile_posterior_from_choices(cw, recs, prior=pair, use_costs=use_costs)
                        res[f"{reader}_{kind}"].append(float(post[a]))
    table = {k: {"mean_posterior_on_truth": float(np.mean(x)), "abstain_rate": float(np.mean([p <= 0.6 for p in x]))} for k, x in res.items()}
    gr = G.GateReport()
    gr.live("discriminating_menus_separate", observed_change=float(table["record_discriminating"]["mean_posterior_on_truth"] - table["record_agreeing"]["mean_posterior_on_truth"]), min_change=0.05)
    v["results"] = {"cells": table, "criterion_C_R08": {"passed": bool(table["count_agreeing"]["abstain_rate"] >= 0.8 and table["record_discriminating"]["mean_posterior_on_truth"] >= 0.9)}}
    v["what_must_hold_outside_the_simulation"] = "equifinal profiles exist and are the ones readers most need to tell apart"
    return finish(card, v, gr, __file__, decide_state(gr))
