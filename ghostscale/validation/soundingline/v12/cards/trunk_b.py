"""Trunk B: bard, neutral, concealer (spec section 9).

A regime is a relation between the tail cue a maker emits and its profile: a bard emits its own
cue, a neutral maker a random one, a concealer its decoy's. Pair mass, entropy, length and
effort are identical across regimes by construction, so nothing but inferential correspondence
differs. Readers carry an assumption about the regime; B02 crosses the two.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, stream, realization
from .. import exact as X, uptake as U
from . import finish, worlds_for, decide_state

REGIMES = ("bard", "neutral", "concealer")
ASSUMPTIONS = ("bard", "neutral", "concealer", "plain")


def _emit_slot(world, maker, g, slot, rng, n_steps):
    e = realization(world, maker.template[g], g, slot)
    a = world.alpha[maker.tier]
    e = a * e + (1 - a) * world.synth
    e = e / e.sum()
    return rng.choice(world.nf, size=int(n_steps), p=e)


def _hist(feats, nf):
    h = np.bincount(np.asarray(feats), minlength=nf).astype(float)
    return h / h.sum()


def _js(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log(a[s] / b[s])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def H(q):
    q = np.asarray(q, float)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def run_B01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The three regimes differ only in inferential correspondence: every "
                    "realization is matched on pair mass and entropy, population surfaces coincide, and "
                    "a cheap surface classifier cannot tell them apart.", "METHOD")
    ent_gap, mass_gap, hist_js, clf_acc = [], [], [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            names = world.family_names
            for n in names:
                for g in range(world.ng):
                    base = world.sig[g]
                    slots = {"bard": world.cue_of[n], "concealer": world.cue_of[world.decoy_of[n]],
                             "neutral": int(C.rng_for("B01", wid, 0, n + str(g)).integers(len(names)))}
                    reals = {r: realization(world, base, g, s) for r, s in slots.items()}
                    pair = np.argsort(base)[-2:]
                    ents = [float(-(x[x > 0] * np.log(x[x > 0])).sum()) for x in reals.values()]
                    masses = [float(x[pair].sum()) for x in reals.values()]
                    ent_gap.append(max(ents) - min(ents))
                    mass_gap.append(max(masses) - min(masses))
            rng = C.rng_for("B01", wid, 1)
            hists, labels = [], []
            pop_hist = {}
            for r in REGIMES:
                for m in population(world, 30, rng, regimes=(r,), prefix=r):
                    feats = np.concatenate([a["features"] for a in stream(world, m, 0, rng, 4)])
                    hists.append(_hist(feats, world.nf))
                    labels.append(r)
                    pop_hist.setdefault(r, []).append(hists[-1])
            pop = {r: np.mean(np.array(h), axis=0) for r, h in pop_hist.items()}
            hist_js.append(max(_js(pop[a], pop[b]) for a in REGIMES for b in REGIMES if a < b))
            Hs, L = np.array(hists), np.array(labels)
            idx = np.arange(len(Hs))
            train = (idx % 2 == 0)          # a held-out half, not leave-one-out: LOO centroids sit below chance by construction
            cents = {r: Hs[train & (L == r)].mean(axis=0) for r in REGIMES}
            correct = [min(cents, key=lambda r: _js(Hs[i], cents[r])) == L[i] for i in idx[~train]]
            clf_acc.append(float(np.mean(correct)))
    gr = G.GateReport()
    gr.identity("regime_entropy_matched", float(max(ent_gap)), 0.0, tol=1e-9)
    gr.identity("regime_pair_mass_matched", float(max(mass_gap)), 0.0, tol=1e-9)
    gr.placebo("population_surfaces_coincide", observed_max_deviation=float(max(hist_js)), tol=0.02,
               detail="the population histogram under each regime is the same distribution up to sampling")
    gr.positive("cheap_classifier_at_chance", observed=float(np.mean(clf_acc)), expected=1 / 3, tol=0.12,
                detail="a surface-only classifier without profile knowledge cannot separate regimes")
    v["results"] = {"max_entropy_gap": float(max(ent_gap)), "max_pair_mass_gap": float(max(mass_gap)),
                    "max_population_histogram_js": float(max(hist_js)), "surface_classifier_accuracy": float(np.mean(clf_acc))}
    v["what_must_hold_outside_the_simulation"] = "nothing; a construction check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_B02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The cooperative (bard) assumption gains on bards and costs on concealers "
                    "relative to a neutral reader; the plain reader, which ignores the cue, marks the "
                    "easier-likelihood rival.", "CONSTRUCTED_MECHANISM")
    cells = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("B02", wid, 0)
            for r in REGIMES:
                for m in population(world, 20, rng, regimes=(r,), k_choices=(0.0, 0.3), prefix=r):
                    arts = stream(world, m, 0, C.rng_for("B02", wid, 1, m.id), 12)
                    for a in ASSUMPTIONS:
                        cum = X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, a)
                        for n in (1, 4, 12):
                            cells.setdefault((r, a, n), {}).setdefault(wid, []).append(C.log_score(X.posterior(cum, n), m.profile))
    table = {f"{r}|{a}|n={n}": C.hboot(d, np.random.default_rng(C.seed(f"B02{r}{a}{n}")), draws=300) for (r, a, n), d in cells.items()}
    gain_bard = table["bard|bard|n=4"]["mean"] - table["bard|neutral|n=4"]["mean"]
    cost_conc = table["concealer|bard|n=4"]["mean"] - table["concealer|neutral|n=4"]["mean"]
    gr = G.GateReport()
    gr.positive("matched_assumption_is_best_on_its_own_regime",
                observed=float(all(table[f"{r}|{r}|n=12"]["mean"] >= table[f"{r}|{a}|n=12"]["mean"] - 1e-9 for r in REGIMES for a in REGIMES)),
                expected=1.0, tol=0.0,
                detail="a reader whose assumption matches the regime must score at least as well as mismatched readers at twelve artifacts")
    gr.live("assumption_moves_the_score", observed_change=float(abs(gain_bard) + abs(cost_conc)), min_change=0.02)
    v["results"] = {"cells": table, "criterion_C_B02": {"cooperative_gain_on_bards": float(gain_bard), "cooperative_cost_on_concealers": float(cost_conc),
                                                        "passed": bool(gain_bard > 0 and cost_conc < 0)}}
    v["what_must_hold_outside_the_simulation"] = "readers hold an assumption about the maker's regime at all"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_B03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A reader filtering jointly over profile and regime with a sticky "
                    "regime prior recovers a source's regime after it switches within twelve artifacts; "
                    "a reader keyed to source identity never does.", "CONSTRUCTED_MECHANISM")
    rec, rec_rival = [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            names = world.family_names
            rng = C.rng_for("B03", wid, 0)
            for (r0, r1) in (("bard", "concealer"), ("concealer", "bard"), ("neutral", "concealer"), ("bard", "neutral")):
                for i in range(6):
                    m = make_maker(world, f"src{i}", names[i % len(names)], rng, regime=r0)
                    arts = stream(world, m, 0, rng, 12)
                    m.regime = r1
                    arts += stream(world, m, 0, rng, 12)
                    inc = {}
                    for reg in REGIMES:
                        cum = X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, reg)
                        inc[reg] = {n: np.diff(np.concatenate([[0.0], cum[n]])) for n in names}
                    for sticky, store in ((0.1, rec), (0.0, rec_rival)):
                        T = np.full((3, 3), sticky / 2) + np.eye(3) * (1 - 1.5 * sticky)
                        joint = np.full((len(names), 3), 1.0 / (3 * len(names)))
                        when = None
                        for t in range(24):
                            joint = joint @ T
                            ll = np.array([[inc[reg][n][t] for reg in REGIMES] for n in names])
                            joint = joint * np.exp(ll - ll.max())
                            joint /= joint.sum()
                            if t >= 12 and when is None and joint[:, REGIMES.index(r1)].sum() >= 0.5:
                                when = t - 12 + 1
                        store.append(when if when is not None else 24)
    gr = G.GateReport()
    gr.live("sticky_reader_recovers_earlier_than_fixed", observed_change=float(np.mean(rec_rival) - np.mean(rec)), min_change=1.0,
            detail="a reader that keys the regime to source identity (no transition mass) flips only once the new regime's cumulative "
                   "evidence outweighs the old; the sticky reader must recover at least one artifact earlier on average")
    gr.live("switch_is_detectable", observed_change=float(24 - np.mean(rec)), min_change=1.0)
    v["results"] = {"recovery_time_mean": float(np.mean(rec)), "recovery_within_12_rate": float(np.mean([x <= 12 for x in rec])),
                    "source_identity_rival_recovery_time": float(np.mean(rec_rival)), "criterion_C_B03": {"passed": bool(np.mean([x <= 12 for x in rec]) >= 0.8)}}
    v["what_must_hold_outside_the_simulation"] = "a source's regime can change while its profile stays"
    return finish(card, v, gr, __file__, decide_state(gr))


def _joint_hyps(world, names):
    return [(n, r) for n in names for r in REGIMES]


def _joint_emissions(world, hyps, p):
    from .trunk_q import _cw
    E = np.zeros((len(hyps), world.nf))
    for i, (n, r) in enumerate(hyps):
        E[i] = X.reader_emission(world, world.sig, None, _cw(world.family[n], p, world.ng), 0, "CREATOR", r, n)
    return E


def _eig_regime(E, q, groups, n_steps, rng, draws=150):
    """EIG about the regime marginal from one commissioned artifact under emissions E[h]."""
    q = np.asarray(q, float)
    G_ = np.asarray(groups)

    def hreg(qq):
        return H(np.array([qq[G_ == g].sum() for g in range(3)]))
    h0 = hreg(q)
    posts = []
    for _ in range(draws):
        h = rng.choice(len(q), p=q)
        f = rng.choice(E.shape[1], size=n_steps, p=E[h])
        ll = np.log(np.maximum(E[:, f], 1e-300)).sum(axis=1) + np.log(np.maximum(q, 1e-300))
        qq = np.exp(ll - ll.max())
        posts.append(hreg(qq / qq.sum()))
    return float(h0 - np.mean(posts))


def run_B04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Commissioned challenges chosen for information about the regime "
                    "separate scaffolding (bard) from strategic shaping (concealer) better than "
                    "uncertainty sampling or random challenges.", "CONSTRUCTED_MECHANISM")
    from .trunk_q import probe_artifact, uncertainty_probe
    from .. import pymdp_reader as PR
    res = {}
    agree = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            names = world.family_names
            hyps = _joint_hyps(world, names)
            groups = [REGIMES.index(r) for _, r in hyps]
            P = world.ng + 1
            E = np.stack([_joint_emissions(world, hyps, p) for p in range(P)])
            rng = C.rng_for("B04", wid, 0)
            for r in REGIMES:
                for m in population(world, 6, rng, regimes=(r,), prefix=r):
                    for pol in ("regime_aware", "uncertainty", "random"):
                        prng = C.rng_for("B04", wid, 1, m.id + pol)
                        q = np.full(len(hyps), 1 / len(hyps))
                        for _ in range(4):
                            if pol == "regime_aware":
                                p = int(np.argmax([_eig_regime(E[pp], q, groups, 24, prng) for pp in range(P)]))
                            elif pol == "uncertainty":
                                p = uncertainty_probe(E, q)
                            else:
                                p = int(prng.integers(P))
                            a = probe_artifact(world, m, p, prng)
                            ll = np.log(np.maximum(q, 1e-300)) + np.log(np.maximum(E[p][:, a["features"]], 1e-300)).sum(axis=1)
                            q = np.exp(ll - ll.max())
                            q /= q.sum()
                        mg = np.array([q[np.asarray(groups) == g].sum() for g in range(3)])
                        res.setdefault(pol, {}).setdefault(wid, []).append(float(np.log(max(mg[REGIMES.index(r)], 1e-12))))
            ag = PR.build_reader(E, np.full(len(hyps), 1 / len(hyps)), probe_costs=np.zeros(P))
            ag.infer_states([0, 0])
            ag.infer_policies()
            pm = int(ag.sample_action()[1])
            ex = int(np.argmax([_eig_regime(E[pp], np.full(len(hyps), 1 / len(hyps)), groups, 1, C.rng_for("B04", wid, 2), 400) for pp in range(P)]))
            agree.append(float(pm == ex))
    table = {p: C.hboot(d, np.random.default_rng(C.seed("B04" + p)), draws=300) for p, d in res.items()}
    gr = G.GateReport()
    gr.live("challenges_carry_regime_information", observed_change=float(table["regime_aware"]["mean"] - np.log(1 / 3)), min_change=0.1,
            detail="after four challenges the regime log score must beat the uniform guess")
    v["results"] = {"regime_log_score_by_policy": table, "pymdp_agrees_with_exact_regime_eig": float(np.mean(agree)),
                    "criterion_C_B04": {"passed": bool(table["regime_aware"]["mean"] >= table["uncertainty"]["mean"])}}
    v["what_must_hold_outside_the_simulation"] = "a maker can be challenged with a commission"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_B05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Partial concealment has type-specific signatures: mimicry is unreadable "
                    "to the neutral reader, deflection and mixed cues are partly readable, omission shows "
                    "in the surface itself, and a bounded budget leaks with evidence.", "BOUNDARY")
    types = ("deflection", "omission", "mimicry", "mixed", "budget")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            names = world.family_names
            rng = C.rng_for("B05", wid, 0)
            for t in types:
                for i in range(12):
                    m = make_maker(world, f"{t}{i}", names[i % len(names)], rng, habit_strength=0.0)
                    own, dec = world.cue_of[m.profile], world.cue_of[world.decoy_of[m.profile]]
                    arts = []
                    for j in range(12):
                        g = int(rng.choice(world.ng, p=m.w))
                        if t == "deflection":
                            slot = int(rng.choice([s for s in range(len(names)) if s != own]))
                        elif t == "omission":
                            slot = None
                        elif t == "mimicry":
                            slot = dec
                        elif t == "mixed":
                            slot = own if j % 2 == 0 else dec
                        else:
                            slot = dec if j < 4 else int(rng.integers(len(names)))
                        arts.append({"features": _emit_slot(world, m, g, slot, rng, 24), "domain": 0, "goals": [g]})
                    for a in ("neutral", "concealer"):
                        cum = X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, a)
                        for n in (4, 12):
                            post = X.posterior(cum, n)
                            d = res.setdefault(f"{t}|{a}|n={n}", {"ls": [], "conf_wrong": []})
                            d["ls"].append(C.log_score(post, m.profile))
                            d["conf_wrong"].append(float(C.top1(post) != m.profile and max(post.values()) >= 0.8))
                    feats = np.concatenate([a["features"] for a in arts])
                    h = _hist(feats, world.nf)
                    res.setdefault(f"{t}|surface_entropy", {"ls": [], "conf_wrong": []})["ls"].append(float(-(h[h > 0] * np.log(h[h > 0])).sum()))
    table = {}
    for k, d in res.items():
        if k.endswith("surface_entropy"):
            table[k] = {"entropy": float(np.mean(d["ls"]))}
        else:
            table[k] = {"log_score": float(np.mean(d["ls"])), "confidently_wrong_rate": float(np.mean(d["conf_wrong"]))}
    gr = G.GateReport()
    gr.positive("mimicry_fools_the_neutral_reader", observed=float(table["mimicry|neutral|n=12"]["log_score"] < table["mixed|neutral|n=12"]["log_score"]), expected=1.0, tol=0.0,
                detail="full mimicry must be less readable than half mimicry to the neutral reader; the known ordering")
    gr.live("budget_leaks_with_evidence", observed_change=float(table["budget|neutral|n=12"]["log_score"] - table["budget|neutral|n=4"]["log_score"]), min_change=0.05)
    v["results"] = {"cells": table}
    v["what_must_hold_outside_the_simulation"] = "concealment types are distinguishable in kind, not only in degree"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_B06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "An accurate model of a concealer raises the reader's own payoff on a task "
                    "that depends on the maker's next move, while adopting the maker's inferred preference "
                    "would lower it: understanding is not cooperation.", "CONSTRUCTED_MECHANISM")
    own, coop = {"accurate": [], "naive": [], "none": []}, []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("B06", wid, 0)
            for m in population(world, 30, rng, regimes=("concealer",)):
                arts = stream(world, m, 0, rng, 8)
                posts = {"accurate": X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "concealer"), 8),
                         "naive": X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "bard"), 8)}
                for k, post in posts.items():
                    pred = X.predictive_next_goal(post, world.family)
                    protect = np.argsort(-pred)[:2]
                    own[k].append(float(m.w[protect].sum()))
                own["none"].append(2.0 / world.ng)
                reader_pref = world.family["uniform"]
                out = U.task(rng, world.ng)
                p0 = U.policy(reader_pref, out)
                p1 = U.policy(U.bridge(reader_pref, U.representation(posts["accurate"], world.family, "mean"), 1.0), out)
                coop.append(U.regret(p1, out, reader_pref) - U.regret(p0, out, reader_pref))
    gr = G.GateReport()
    gr.positive("random_protection_is_the_floor", observed=float(np.mean(own["accurate"]) >= np.mean(own["none"]) - 1e-9), expected=1.0, tol=0.0)
    gr.live("model_accuracy_changes_protection", observed_change=float(np.mean(own["accurate"]) - np.mean(own["naive"])), min_change=0.02)
    v["results"] = {"own_payoff_by_model": {k: float(np.mean(x)) for k, x in own.items()}, "own_task_regret_change_under_adoption": float(np.mean(coop)),
                    "criterion_C_B06": {"passed": bool(np.mean(own["accurate"]) >= np.mean(own["naive"]))}}
    v["what_must_hold_outside_the_simulation"] = "the reader's task depends on the maker's next move"
    return finish(card, v, gr, __file__, decide_state(gr))
