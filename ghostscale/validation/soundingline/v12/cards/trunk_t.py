"""Trunk T: what supplying one latent buys for another (spec section 12).

The exact joint over (profile, goal, realization slot) given an artifact's features is small
enough to enumerate, so every supply is an exact conditioning and every gain is a difference of
exact log scores. Supply types: none, true, shuffled, wrong, uncertain. Directionality is read
only from cells where both latents are off ceiling and off floor.
"""
from __future__ import annotations

import numpy as np

from .....generative_model import build_observer_signature
from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import realization
from . import finish, worlds_for, decide_state

AXES = ("profile", "goal", "slot")


def tables(world, template, tier="CREATOR", regime="neutral"):
    names = world.family_names
    K, ng, ns = len(names), world.ng, len(names)
    a = world.alpha[tier]
    L = np.zeros((ng, ns, world.nf))
    for g in range(ng):
        for s in range(ns):
            e = realization(world, template[g], g, s)
            e = a * e + (1 - a) * world.synth
            L[g, s] = e / e.sum()
    Pg = np.stack([world.family[n] for n in names])
    Ps = np.zeros((K, ns))
    for i, n in enumerate(names):
        if regime == "bard":
            Ps[i, world.cue_of[n]] = 1.0
        elif regime == "concealer":
            Ps[i, world.cue_of[world.decoy_of[n]]] = 1.0
        else:
            Ps[i] = 1.0 / ns
    prior = (1.0 / K) * Pg[:, :, None] * Ps[:, None, :]
    return L, prior


def sample(L, prior, rng, n_steps):
    idx = int(rng.choice(prior.size, p=prior.ravel()))
    k, g, s = np.unravel_index(idx, prior.shape)
    f = rng.choice(L.shape[2], size=int(n_steps), p=L[g, s])
    return int(k), int(g), int(s), f


def posterior(L, prior, f, supply=None):
    ll = np.log(np.maximum(L[:, :, f], 1e-300)).sum(axis=2)
    logp = np.log(np.maximum(prior, 1e-300)) + ll[None]
    for axis, dist in (supply or {}).items():
        d = np.log(np.maximum(np.asarray(dist, float), 1e-300))
        shape = [1, 1, 1]
        shape[axis] = d.size
        logp = logp + d.reshape(shape)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def marg(p, axis):
    other = tuple(i for i in range(3) if i != axis)
    return p.sum(axis=other)


def supply_dist(kind, truth, n, rng, shuffled_value=None):
    if kind == "none":
        return None
    d = np.zeros(n)
    if kind == "true":
        d[truth] = 1.0
    elif kind == "wrong":
        d[int(rng.choice([i for i in range(n) if i != truth]))] = 1.0
    elif kind == "shuffled":
        d[shuffled_value if shuffled_value is not None else int(rng.integers(n))] = 1.0
    elif kind == "uncertain":
        d[:] = 0.3 / (n - 1)
        d[truth] = 0.7
    return d


def H(p):
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def battery(world, L, prior, rng, n_samples=300, n_steps=8, kinds=("none", "true", "shuffled", "wrong", "uncertain")):
    """Log score of each target latent under each supply of each other latent."""
    sizes = prior.shape
    out = {}
    draws = [sample(L, prior, rng, n_steps) for _ in range(n_samples)]
    for i, (k, g, s, f) in enumerate(draws):
        truth = (k, g, s)
        other = draws[(i + 1) % len(draws)]
        for t in range(3):
            base = posterior(L, prior, f)
            out.setdefault((AXES[t], "none", "none"), []).append(float(np.log(max(marg(base, t)[truth[t]], 1e-12))))
            for sax in range(3):
                if sax == t:
                    continue
                for kind in kinds:
                    if kind == "none":
                        continue
                    d = supply_dist(kind, truth[sax], sizes[sax], rng, other[sax])
                    p = posterior(L, prior, f, {sax: d})
                    out.setdefault((AXES[t], AXES[sax], kind), []).append(float(np.log(max(marg(p, t)[truth[t]], 1e-12))))
    return {f"{t}|{s}|{k}": float(np.mean(v)) for (t, s, k), v in out.items()}


def run_T01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The exact conditional-information ledger of the generative construction: "
                    "mutual information among profile, goal, slot and surface, with deterministic edges flagged.",
                    "METHOD")
    ledgers = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            for regime in ("neutral", "bard"):
                L, prior = tables(world, world.sig, regime=regime)
                Pk, Pg, Ps = marg(prior, 0), marg(prior, 1), marg(prior, 2)
                Pkg, Pks, Pgs = prior.sum(axis=2), prior.sum(axis=1), prior.sum(axis=0)

                def mi(joint, a, b):
                    j = joint[joint > 0]
                    return float((j * np.log(j / (np.outer(a, b)[joint > 0]))).sum())
                led = {"I(profile;goal)": mi(Pkg, Pk, Pg), "I(profile;slot)": mi(Pks, Pk, Ps), "I(goal;slot)": mi(Pgs, Pg, Ps),
                       "H(slot|profile)": H(prior) - H(Pkg), "H(goal|profile)": H(Pkg) - H(Pk)}
                rng = C.rng_for("T01", wid, 0, regime)
                for n_steps in (2, 8, 24):
                    posts = {t: [] for t in range(3)}
                    for _ in range(300):
                        k, g, s, f = sample(L, prior, rng, n_steps)
                        p = posterior(L, prior, f)
                        for t in range(3):
                            posts[t].append(H(marg(p, t)))
                    for t in range(3):
                        led[f"I({AXES[t]};surface)@{n_steps}"] = float(H(marg(prior, t)) - np.mean(posts[t]))
                led["deterministic_edges"] = [e for e, val in (("profile->slot", led["H(slot|profile)"]),) if val < 1e-9]
                ledgers[f"world{wid}|{regime}"] = led
    gr = G.GateReport()
    neutral = [l for k, l in ledgers.items() if k.endswith("neutral")]
    bard = [l for k, l in ledgers.items() if k.endswith("bard")]
    gr.identity("slot_independent_of_profile_under_neutral", float(max(l["I(profile;slot)"] for l in neutral)), 0.0, tol=1e-9)
    gr.positive("slot_determined_by_profile_under_bard", observed=float(all("profile->slot" in l["deterministic_edges"] for l in bard)), expected=1.0, tol=0.0)
    gr.live("surface_carries_profile_information", observed_change=float(np.mean([l["I(profile;surface)@24"] for l in neutral])), min_change=0.1)
    v["results"] = {"ledgers": ledgers}
    v["what_must_hold_outside_the_simulation"] = "nothing; the ledger is a property of the construction"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_T02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The full supply matrix: what supplying each latent (true, shuffled, wrong, "
                    "uncertain) buys for the recovery of each other latent, in nats of log score.",
                    "CONSTRUCTED_MECHANISM")
    mats = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            # at the CREATOR tier with eight steps the goal is at ceiling and every goal-related cell is zero by
            # construction (found in the one-world smoke pass); the battery runs at CURATOR alpha with four steps
            L, prior = tables(world, world.sig, tier="CURATOR")
            mats[str(wid)] = battery(world, L, prior, C.rng_for("T02", wid, 0), n_steps=4)
    keys = list(next(iter(mats.values())))
    mean = {k: float(np.mean([m[k] for m in mats.values()])) for k in keys}
    gains = {k: mean[k] - mean[f"{k.split('|')[0]}|none|none"] for k in keys if not k.endswith("none|none")}
    gr = G.GateReport()
    shuffled_max = float(max(g for k, g in gains.items() if k.endswith("shuffled")))
    true_min = float(min(g for k, g in gains.items() if k.endswith("|true")))
    wrong_min = float(min(g for k, g in gains.items() if k.endswith("|wrong")))
    gr.positive("shuffled_supply_never_helps", observed=float(shuffled_max <= 0.02), expected=1.0, tol=0.0,
                detail="a shuffled supply is the null: it must not help beyond Monte-Carlo noise (it may hurt a great deal)")
    gr.positive("true_supply_never_hurts_on_average", observed=float(true_min >= -0.02), expected=1.0, tol=0.0,
                detail="conditioning on the truth cannot lower the expected log score of an exact posterior beyond Monte-Carlo noise")
    gr.live("wrong_supply_reaches_the_posterior", observed_change=-wrong_min, min_change=0.05,
            detail="a wrong value of a determinative latent must hurt; if nothing moves, the supply is not reaching the posterior")
    v["results"] = {"tier": "CURATOR", "n_steps": 4, "log_score_by_target_supply_kind": mean, "gain_over_no_supply": gains, "per_world": mats,
                    "shuffled_max_gain": shuffled_max, "true_min_gain": true_min, "wrong_min_gain": wrong_min}
    v["what_must_hold_outside_the_simulation"] = "the latents are the ones a reader could actually be supplied"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_T03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Supplying the standing profile improves recovery of the process (slot) "
                    "and supplying the process improves recovery of the goal, reported on one scale and "
                    "only from off-ceiling cells.", "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
          for tier in ("CREATOR", "POLISHED", "CURATOR", "GHOST"):
            for n_steps in (1, 2, 4, 8, 12):
                L, prior = tables(world, world.sig, tier=tier)
                b = battery(world, L, prior, C.rng_for("T03", wid, 0, tier + str(n_steps)), n_samples=200, n_steps=n_steps, kinds=("none", "true"))
                rows.append({"wid": wid, "tier": tier, "n_steps": n_steps,
                             "slot_base": b["slot|none|none"], "goal_base": b["goal|none|none"], "profile_base": b["profile|none|none"],
                             "profile_to_slot": b["slot|profile|true"] - b["slot|none|none"],
                             "slot_to_goal": b["goal|slot|true"] - b["goal|none|none"],
                             "goal_to_profile": b["profile|goal|true"] - b["profile|none|none"],
                             "profile_to_goal": b["goal|profile|true"] - b["goal|none|none"]})
    off = [r for r in rows if np.log(0.15) <= r["slot_base"] <= np.log(0.9) and np.log(0.15) <= r["goal_base"] <= np.log(0.9)]
    gr = G.GateReport()
    gr.live("off_ceiling_cells_exist", observed_change=float(len(off)), min_change=1.0)
    v["results"] = {"cells": rows, "off_ceiling": off,
                    "profile_to_process_gain": float(np.mean([r["profile_to_slot"] for r in off])) if off else None,
                    "process_to_goal_gain": float(np.mean([r["slot_to_goal"] for r in off])) if off else None}
    v["what_must_hold_outside_the_simulation"] = "a process latent exists between goal and surface"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_T04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Supplying the maker's mechanic (its own realization template) unlocks goal, "
                    "process and profile recovery relative to a generic template; related and wrong "
                    "mechanics are ordered between.", "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("T04", wid, 0)
            for rep in range(4):
                own = build_observer_signature(world.sig, 0.4, rng)
                mech = {"correct": own, "related": build_observer_signature(own, 0.2, rng),
                        "wrong": build_observer_signature(world.sig, 0.4, rng), "generic": world.sig}
                L_true, prior = tables(world, own)
                draws = [sample(L_true, prior, rng, 8) for _ in range(150)]
                for name, tmpl in mech.items():
                    L_read, _ = tables(world, tmpl)
                    for k, g, s, f in draws:
                        p = posterior(L_read, prior, f)
                        for t, truth in zip(range(3), (k, g, s)):
                            res.setdefault(f"{AXES[t]}|{name}", {}).setdefault(wid, []).append(float(np.log(max(marg(p, t)[truth], 1e-12))))
    table = {k: C.hboot(d, np.random.default_rng(C.seed("T04" + k)), draws=200) for k, d in res.items()}
    gains = {k: table[k]["mean"] - table[f"{k.split('|')[0]}|generic"]["mean"] for k in table if not k.endswith("generic")}
    gr = G.GateReport()
    gr.positive("correct_mechanic_beats_wrong_mechanic", observed=float(all(table[f"{a}|correct"]["mean"] >= table[f"{a}|wrong"]["mean"] for a in AXES)), expected=1.0, tol=0.0)
    gr.live("mechanic_matters", observed_change=float(max(abs(g) for g in gains.values())), min_change=0.02)
    v["results"] = {"log_score": table, "gain_over_generic": gains}
    v["what_must_hold_outside_the_simulation"] = "a maker's mechanic is a template a reader could be handed"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_T05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Apparent directionality (goal-from-profile gain minus profile-from-goal gain) "
                    "is reported only from cells where both latents are off ceiling and off floor, across "
                    "tier and evidence length.", "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            for tier in ("CREATOR", "POLISHED", "CURATOR"):
                for n_steps in (2, 4, 8, 12, 24):
                    L, prior = tables(world, world.sig, tier=tier)
                    b = battery(world, L, prior, C.rng_for("T05", wid, 0, tier + str(n_steps)), n_samples=150, n_steps=n_steps, kinds=("none", "true"))
                    rows.append({"wid": wid, "tier": tier, "n_steps": n_steps, "goal_base": b["goal|none|none"], "profile_base": b["profile|none|none"],
                                 "goal_from_profile": b["goal|profile|true"] - b["goal|none|none"],
                                 "profile_from_goal": b["profile|goal|true"] - b["profile|none|none"]})
    for r in rows:
        r["directionality"] = r["goal_from_profile"] - r["profile_from_goal"]
        r["off_ceiling"] = bool(np.log(0.15) <= r["goal_base"] <= np.log(0.9) and np.log(0.15) <= r["profile_base"] <= np.log(0.9))
    off = [r for r in rows if r["off_ceiling"]]
    gr = G.GateReport()
    gr.live("off_ceiling_cells_exist", observed_change=float(len(off)), min_change=1.0)
    v["results"] = {"cells": rows, "directionality_off_ceiling_mean": float(np.mean([r["directionality"] for r in off])) if off else None,
                    "directionality_all_cells_mean": float(np.mean([r["directionality"] for r in rows]))}
    v["what_must_hold_outside_the_simulation"] = "difficulty varies in the same way outside"
    return finish(card, v, gr, __file__, decide_state(gr))


TOPOLOGIES = {
    "chain_with_independent_slot": {"obs_equals_do_slot": True, "slot_helps_goal": True, "slot_helps_profile": True, "goal_helps_profile": True},
    "common_cause_profile": {"obs_equals_do_slot": False, "slot_helps_goal": True, "slot_helps_profile": True, "goal_helps_profile": True},
    "river": {"obs_equals_do_slot": False, "slot_helps_goal": True, "slot_helps_profile": True, "goal_helps_profile": True},
    "triangle_goal_to_slot": {"obs_equals_do_slot": False, "slot_helps_goal": True, "slot_helps_profile": True, "goal_helps_profile": True},
    "flat_factor_graph": {"obs_equals_do_slot": True, "slot_helps_goal": True, "slot_helps_profile": True, "goal_helps_profile": True},
    "isolated_slot": {"obs_equals_do_slot": True, "slot_helps_goal": False, "slot_helps_profile": False, "goal_helps_profile": True},
}


def run_T06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Held-out interventions (do versus observe on the slot) pick the topology "
                    "class the supply matrix cannot; where several topologies predict the same matrix the "
                    "equivalence class is reported and no topology is claimed.", "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            for regime in ("neutral", "bard"):
                L, prior = tables(world, world.sig, tier="CURATOR", regime=regime)     # off ceiling, as T02
                K, ng, ns = prior.shape
                prior_do = (marg(prior, 0)[:, None, None] * (prior.sum(axis=2) / np.maximum(marg(prior, 0)[:, None], 1e-300))[:, :, None] * np.full((1, 1, ns), 1 / ns))
                prior_do = prior_do / prior_do.sum()
                rng = C.rng_for("T06", wid, 0, regime)
                obs, do, sg, sp, gp = [], [], [], [], []
                for _ in range(200):
                    k, g, s, f = sample(L, prior, rng, 4)
                    base = posterior(L, prior, f)
                    d = supply_dist("true", s, ns, rng)
                    p_obs = posterior(L, prior, f, {2: d})
                    p_do = posterior(L, prior_do, f, {2: d})
                    lp = lambda p, t, tr: float(np.log(max(marg(p, t)[tr], 1e-12)))
                    obs.append(lp(p_obs, 0, k) - lp(base, 0, k))
                    do.append(lp(p_do, 0, k) - lp(posterior(L, prior_do, f), 0, k))
                    sg.append(lp(p_obs, 1, g) - lp(base, 1, g))
                    sp.append(lp(p_obs, 0, k) - lp(base, 0, k))
                    dg = supply_dist("true", g, ng, rng)
                    gp.append(lp(posterior(L, prior, f, {1: dg}), 0, k) - lp(base, 0, k))
                measured = {"obs_equals_do_slot": bool(abs(np.mean(obs) - np.mean(do)) < 0.02), "slot_helps_goal": bool(np.mean(sg) > 0.02),
                            "slot_helps_profile": bool(np.mean(sp) > 0.02), "goal_helps_profile": bool(np.mean(gp) > 0.02)}
                scores = {t: int(sum(pred[k] == measured[k] for k in measured)) for t, pred in TOPOLOGIES.items()}
                best = max(scores.values())
                res[f"world{wid}|{regime}"] = {"measured": measured, "obs_gain": float(np.mean(obs)), "do_gain": float(np.mean(do)),
                                               "scores": scores, "equivalence_class": sorted(t for t, s in scores.items() if s == best)}
    gr = G.GateReport()
    linked = ("common_cause_profile", "river", "triangle_goal_to_slot")
    neutral_ok = all(not any(tp in r["equivalence_class"] for tp in linked) and r["measured"]["obs_equals_do_slot"]
                     for k, r in res.items() if k.endswith("neutral"))
    bard_ok = all("chain_with_independent_slot" not in r["equivalence_class"] for k, r in res.items() if k.endswith("bard"))
    gr.positive("neutral_regime_rejects_profile_to_slot_edges", observed=float(neutral_ok), expected=1.0, tol=0.0,
                detail="under the neutral regime the slot is independent of the profile by construction: do equals observe, and no "
                       "topology with a profile-to-slot edge may survive. Whether the slot's own edge to the surface is strong enough "
                       "to separate an independent slot from an isolated one is a property of the cue strength, reported, not gated")
    gr.positive("bard_regime_rejects_the_independent_slot", observed=float(bard_ok), expected=1.0, tol=0.0,
                detail="under bard the slot is a deterministic function of the profile: observing it is not intervening on it")
    v["results"] = {"per_world": res, "note": "the flat factor graph and the chain-with-independent-slot are one equivalence class on every "
                    "statistic here; where the slot's edge to the surface is weak (small cue), the isolated-slot topology joins that class. No claim separates them"}
    v["what_must_hold_outside_the_simulation"] = "interventions on a process latent are possible"
    return finish(card, v, gr, __file__, decide_state(gr))
