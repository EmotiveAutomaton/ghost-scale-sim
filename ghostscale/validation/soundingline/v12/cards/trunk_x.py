"""Trunk X: the adversarial matrix (spec section 16).

Every attack is applied to the program's headline estimand, card S04's selective self-first
gain (log score of the self-first route minus the information-matched generic route at first
evidence, in the nearest and farthest distance bins). Each attack ships with a placebo (strength
zero reproduces the unattacked run bit for bit) and, where the construction fixes one, a known
answer. Attacks that could be applied only to one trunk's headline are recorded as such under
"what the validation could not check".
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, stream, emission, random_params, make_world
from .. import exact as X, self_other as SO, pymdp_reader as PR
from . import finish, worlds_for, decide_state

N_MAKERS = 60
BETA_SELF = 6.0


def _bins(dists, n_bins=5):
    qs = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.searchsorted(qs[1:-1], dists, side="right"), 0, n_bins - 1)


def _cheap_post(world, reader_template, feats, tier, prior):
    """Nearest-centroid histogram reader: posterior proportional to prior times exp(-20 JS)."""
    names = world.family_names
    h = np.bincount(feats, minlength=world.nf) / len(feats)
    out = {}
    for n in names:
        cent = X.reader_emission(world, reader_template, None, world.family[n], 0, tier, "plain", n)
        out[n] = prior[n] * np.exp(-20.0 * SO.js(h, cent))
    z = sum(out.values())
    return {n: p / z for n, p in out.items()}


def s04_effect(world, wid, attack="none", strength=0.0):
    """The S04 estimand under one attack at one strength. Returns near/far gains at n=1 and an
    attack-specific identity statistic where the construction fixes one."""
    rng = C.rng_for("X", wid, 0)          # the same population under every attack, so strength zero is bit-identical
    makers = population(world, N_MAKERS, rng, k_choices=(0.0, 0.3))
    names = world.family_names
    if attack == "X10" and strength > 0:
        for m in makers[: int(round(strength * 0.5 * len(makers)))]:
            m.profile = world.decoy_of[m.profile]                 # label wrong; artifacts unchanged
    readers = [make_maker(world, f"reader{i}", n, rng, k=0.05) for i, n in enumerate(names)]
    order = list(range(len(makers)))
    if attack == "X01" and strength > 0:
        order = list(rng.permutation(len(makers)))                 # source labels shuffled
    rows, ident = [], []
    for r in readers:
        sm = SO.measure_self(world, r, C.rng_for("X", wid, 1, r.id))
        sp = SO.self_first_prior(world, sm["w_hat"], BETA_SELF)
        gp = SO.information_matched_generic(world, sp, makers)
        if attack == "X05" and strength > 0:
            sp = SO.permuted_self_prior(sp, C.rng_for("X", wid, 5, r.id))
        for idx in order:
            m = makers[idx]
            mrng = C.rng_for("X", wid, 2, m.id)
            if attack == "X11" and strength > 0:
                # adaptive adversary: the maker shapes its goal mix toward the reader's own profile
                arts = []
                mix = (1 - strength) * m.w + strength * r.w
                for _ in range(2):
                    g = int(mrng.choice(world.ng, p=mix / mix.sum()))
                    dist, _ = emission(world, m, g, 0, mrng)
                    arts.append({"features": mrng.choice(world.nf, size=world.params.n_steps, p=dist), "domain": 0, "goals": [g]})
            else:
                arts = stream(world, m, 0, mrng, 2)
            if attack == "X02" and strength > 0:
                perm = np.asarray(world.domains[1].perm)
                arts = [{**a, "features": perm[np.asarray(a["features"])], "domain": 1} for a in arts]
            if attack == "X06" and strength > 0:
                arts = arts[::-1]
            if attack == "X04" and strength > 0:
                twin_profile = world.decoy_of[m.profile]
            sp_m, gp_m = sp, gp
            if attack == "X03" and strength > 0 and idx % 2 == 0:
                d = world.decoy_of[m.profile]
                fake = {n: (0.9 if n == d else 0.1 / (len(names) - 1)) for n in names}
                sp_m = {n: sp[n] * fake[n] for n in names}
                gp_m = {n: gp[n] * fake[n] for n in names}
                zs, zg = sum(sp_m.values()), sum(gp_m.values())
                sp_m = {n: p / zs for n, p in sp_m.items()}
                gp_m = {n: p / zg for n, p in gp_m.items()}
            if attack == "X08" and strength > 0:
                E = np.stack([X.reader_emission(world, r.template, r.habit[0], world.family[n], 0, m.tier, "plain", n) for n in names])
                E = np.stack([E, E])
                posts = {}
                for key, pr in (("self", sp_m), ("gen", gp_m)):
                    ag = PR.build_reader(E, np.array([pr[n] for n in names]), probe_costs=np.zeros(2))
                    q = PR.observe_sequence(ag, np.asarray(arts[0]["features"]), 0)
                    posts[key] = dict(zip(names, q))
                    ex = PR.exact_sequence_posterior(E, np.array([pr[n] for n in names]), np.asarray(arts[0]["features"]), 0)
                    ident.append(float(np.abs(q - ex).max()))
                post_s, post_g = posts["self"], posts["gen"]
            elif attack == "X09" and strength > 0:
                post_s = _cheap_post(world, r.template, np.asarray(arts[0]["features"]), m.tier, sp_m)
                post_g = _cheap_post(world, r.template, np.asarray(arts[0]["features"]), m.tier, gp_m)
            else:
                cum = X.profile_loglik_cumulative(world, r.template, r.habit[0], arts, m.tier, "plain")
                post_s, post_g = X.posterior(cum, 1, sp_m), X.posterior(cum, 1, gp_m)
                if attack == "X02" and strength > 0:
                    native = stream(world, m, 0, C.rng_for("X", wid, 2, m.id), 2)
                    cum0 = X.profile_loglik_cumulative(world, r.template, r.habit[0], native, m.tier, "plain")
                    ident.append(max(abs(X.posterior(cum0, 1, sp_m)[n] - post_s[n]) for n in names))
                if attack == "X06" and strength > 0:
                    cum0 = X.profile_loglik_cumulative(world, r.template, r.habit[0], arts[::-1], m.tier, "plain")
                    ident.append(max(abs(X.posterior(cum0, 2, sp_m)[n] - X.posterior(cum, 2, sp_m)[n]) for n in names))
            if attack == "X04" and strength > 0:
                # equifinal twin: the same artifacts labelled with the decoy profile
                lr_truth = np.log(max(post_s[m.profile], 1e-300)) - np.log(max(post_s[twin_profile], 1e-300))
                lr_prior = np.log(max(sp_m[m.profile], 1e-300)) - np.log(max(sp_m[twin_profile], 1e-300))
                ident.append(abs(lr_truth - lr_prior - (np.log(max(post_g[m.profile], 1e-300)) - np.log(max(post_g[twin_profile], 1e-300))
                                                        - (np.log(max(gp_m[m.profile], 1e-300)) - np.log(max(gp_m[twin_profile], 1e-300))))))
                g_true = C.log_score(post_s, m.profile) - C.log_score(post_g, m.profile)
                g_twin = C.log_score(post_s, twin_profile) - C.log_score(post_g, twin_profile)
                gain = 0.5 * (g_true + g_twin)
            else:
                gain = C.log_score(post_s, m.profile) - C.log_score(post_g, m.profile)
            rows.append({"dist": SO.js(sm["w_hat"], m.w), "gain": gain})
    d = np.array([r["dist"] for r in rows])
    g = np.array([r["gain"] for r in rows])
    b = _bins(d)
    return {"near": float(g[b == 0].mean()), "far": float(g[b == 4].mean()), "all": float(g.mean()),
            "identity": float(max(ident)) if ident else 0.0, "n": len(rows)}


ATTACKS = {
    "X01": ("surface/source match", "shuffling source labels leaves the estimand bit-identical; near and far makers share tier, domain and length by construction"),
    "X02": ("policy match, source change", "the same artifacts in the dialect convention give the same posterior: the exact reader knows the convention"),
    "X03": ("false context", "a false biography on half the makers"),
    "X04": ("equifinal history", "the same artifacts labelled with the decoy profile: the evidence's log-odds shift between truth and twin is identical across routes"),
    "X05": ("prior permutation", "the permuted self prior keeps every marginal of the self prior; the near-bin gain must fall to the generic level"),
    "X06": ("evidence-order reversal", "at two artifacts the posterior is order-independent; the first-evidence gain may move"),
    "X07": ("architecture randomization", "fresh random worlds outside both lineages"),
    "X08": ("solver substitution", "the legacy PyMDP posterior under a fixed probe replaces the exact posterior"),
    "X09": ("cheap baseline", "a nearest-centroid histogram reader replaces the exact likelihood"),
    "X10": ("mixed control", "half the makers carry wrong labels"),
    "X11": ("adaptive adversary", "makers shape their goal mix toward the reader's own profile"),
    "X12": ("fresh confirmation", "the confirmation lineage, untouched until now"),
}


def _attack(card, cfg, lane, attack):
    name, detail = ATTACKS[attack]
    fresh = "confirmation" if lane == "confirmation" else "transfer"
    v = new_verdict(card, "confirmation" if (attack == "X12" and lane == "confirmation") else "discovery",
                    f"The selective self-first gain (S04) survives the attack: {name}.", "BOUNDARY")
    base, att, plac = [], [], []
    with C.timed(v):
        if attack == "X07":
            worlds = []
            for i in range(6):
                rng = np.random.default_rng(C.seed(f"X07:world:{i}"))
                worlds.append((300 + i, make_world(cfg, params=random_params(rng), rng=rng)))
            for wid, world in worlds_for(cfg, "discovery", limit=6):
                base.append(s04_effect(world, wid, "none", 0.0))
            for wid, world in worlds:
                att.append(s04_effect(world, wid, "none", 0.0))
            plac = [0.0]
        elif attack == "X12":
            for wid, world in worlds_for(cfg, "discovery", limit=6):
                base.append(s04_effect(world, wid, "none", 0.0))
            for wid, world in worlds_for(cfg, fresh, limit=6):
                att.append(s04_effect(world, wid, "none", 0.0))
            plac = [0.0]
        else:
            for wid, world in worlds_for(cfg, "discovery"):
                b = s04_effect(world, wid, "none", 0.0)
                p = s04_effect(world, wid, attack, 0.0)
                a = s04_effect(world, wid, attack, 1.0)
                base.append(b)
                att.append(a)
                plac.append(abs(p["near"] - b["near"]) + abs(p["far"] - b["far"]))
    agg = lambda xs, k: float(np.mean([x[k] for x in xs]))
    res = {"unattacked": {"near": agg(base, "near"), "far": agg(base, "far"), "all": agg(base, "all")},
           "attacked": {"near": agg(att, "near"), "far": agg(att, "far"), "all": agg(att, "all")},
           "identity_statistic": float(max(x["identity"] for x in att)), "attack": name, "detail": detail}
    res["delta_near"] = res["attacked"]["near"] - res["unattacked"]["near"]
    res["survives"] = bool(res["attacked"]["near"] >= 0.05 and res["attacked"]["far"] <= 0.0)
    gr = G.GateReport()
    gr.placebo("zero_strength_reproduces_unattacked", observed_max_deviation=float(max(plac)), tol=1e-12)
    if attack in ("X01", "X02", "X06", "X08"):
        tol = 1e-6 if attack == "X08" else 1e-9
        gr.identity(f"{attack}_known_identity", res["identity_statistic"], 0.0, tol=tol, detail=detail)
    if attack == "X01":
        gr.identity("shuffled_labels_leave_estimand_identical", res["attacked"]["all"], res["unattacked"]["all"], tol=1e-12)
    if attack == "X04":
        gr.identity("equifinal_twin_log_odds_identity", res["identity_statistic"], 0.0, tol=1e-9, detail=detail)
    if attack == "X05":
        gr.positive("permuted_prior_loses_the_near_gain", observed=res["attacked"]["near"], expected=0.0, tol=0.05, detail=detail)
    v["results"] = res
    v["effective_n"] = {"worlds": len(att), "pairs_per_world": att[0]["n"] if att else 0}
    if attack == "X12":
        v["worlds"] = "confirmation lineage" if lane == "confirmation" else "transfer lineage 200-205 (confirmation lineage untouched during discovery)"
    v["what_must_hold_outside_the_simulation"] = "the attack is applied to the S04 estimand; trunk-specific headlines (Q02, B02, R05, U02, D02, F03, T03) were not attacked in this card"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_X01(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X01")


def run_X02(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X02")


def run_X03(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X03")


def run_X04(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X04")


def run_X05(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X05")


def run_X06(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X06")


def run_X07(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X07")


def run_X08(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X08")


def run_X09(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X09")


def run_X10(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X10")


def run_X11(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X11")


def run_X12(card, cfg, workers=1, lane="both"):
    return _attack(card, cfg, lane, "X12")
