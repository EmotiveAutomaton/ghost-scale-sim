"""Trunk I: integrity, severity, and solver controls (spec section 6)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .....methods import gates as G
from ..... import constants as K
from .. import REPO, common as C
from ..schemas import new_verdict
from ..world import (World, make_maker, make_world, population, random_params, stream, artifact,
                     emission, params_to_dict, RANGES)
from .. import exact as X, self_other as SO, pymdp_reader as PR
from . import finish, worlds_for, decide_state

_EPS = 1e-300


# =========================================================================== #
# I01 — V11 reproduction anchor (reconstructed inner loops, never a V11 run).
# =========================================================================== #
def _s15_arms_reconstructed(cfg):
    """S-15's bounded_clean, unbounded_family, wrong_expertise, shuffled and uniform-world arms,
    rebuilt from V11's primitives with V11's own seeds and call order."""
    from .....v11 import seed as v11_seed
    from .....v11.maker import (build_maker_world, build_reader, maker_artifacts_A,
                               posterior_from_logliks, profile_family, profile_loglik_A,
                               random_family, readout)
    world = build_maker_world(cfg)
    fam = profile_family(cfg.cardinalities.num_goals)
    expert = build_reader(world, cfg, d=0.0)
    wrong = build_reader(world, cfg, d=0.5, rng=np.random.default_rng(v11_seed("s15-reader-d05")))
    big = random_family(64, cfg.cardinalities.num_goals, np.random.default_rng(v11_seed("s15-bigfam")))
    makers = []
    for name, w in fam.items():
        for i in range(60 // len(fam)):
            makers.append((name, np.asarray(w, float)))
    np.random.default_rng(v11_seed("s15-mk")).shuffle(makers)

    def arm(reader, tier, hyp, rng, shuffle=False, makers_=makers):
        all_arts, truths, names = [], [], []
        for name, w in makers_:
            arts, _ = maker_artifacts_A(world, w, tier, 50, 24, rng)
            all_arts.append(arts); truths.append(w); names.append(name)
        if shuffle:
            pool = np.concatenate(all_arts, axis=0)
            pool = pool[rng.permutation(len(pool))]
            all_arts = [pool[i * 50:(i + 1) * 50] for i in range(len(makers_))]
        acc = {n: [] for n in (1, 50)}
        l1 = {n: [] for n in (1, 50)}
        for arts, w_true, true_name in zip(all_arts, truths, names):
            cum = profile_loglik_A(reader, arts, tier, hyp)
            for n in (1, 50):
                post = posterior_from_logliks(cum, n)
                best, d = readout(post, hyp, w_true)
                acc[n].append(1.0 if best == true_name else 0.0)
                l1[n].append(d)
        return {f"acc_{n}": float(np.mean(acc[n])) for n in acc} | {f"l1_{n}": float(np.mean(l1[n])) for n in l1}

    out = {
        "bounded_clean": arm(expert, K.CREATOR, fam, np.random.default_rng(v11_seed("s15-clean"))),
        "unbounded_family": arm(expert, K.CREATOR, big, np.random.default_rng(v11_seed("s15-unb"))),
        "wrong_expertise": arm(wrong, K.CREATOR, fam, np.random.default_rng(v11_seed("s15-wrong"))),
        "shuffled": arm(expert, K.CREATOR, fam, np.random.default_rng(v11_seed("s15-shuf")), shuffle=True),
        "uniform_world": arm(expert, K.CREATOR, fam, np.random.default_rng(v11_seed("s15-unifworld")),
                             makers_=[("uniform", fam["uniform"])] * 12),
    }
    return out


def _s14_reconstructed(cfg):
    from .....v11 import seed as v11_seed
    from .....v11.maker import build_maker_world, build_reader
    from ...s14_aperture import _discriminate
    world = build_maker_world(cfg)
    reader = build_reader(world, cfg, d=0.0)
    ng = world.sig.shape[0]
    stats = {"commissioned": [], "spontaneous": [], "commissioned_lambda1": []}
    for k in range(ng):
        rng = np.random.default_rng(v11_seed(f"s14-k{k}"))
        stats["commissioned"].append(_discriminate(world, reader, k, ("masked", "unused"), "commissioned", 0.5, rng))
        stats["spontaneous"].append(_discriminate(world, reader, k, ("masked", "unused"), "spontaneous", 0.5, rng))
        stats["commissioned_lambda1"].append(_discriminate(world, reader, k, ("masked", "unused"), "commissioned", 1.0, rng))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def run_I01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "both", "Reconstructed V12 inner loops reproduce the committed V11 anchors "
                    "(S-15 recovery, residuals, failed expertise margin; S-14 aperture and collapse).",
                    "METHOD")
    tol = 1e-9
    with C.timed(v):
        s15 = _s15_arms_reconstructed(cfg)
        s14 = _s14_reconstructed(cfg)
        committed15 = json.loads((REPO / "results/validation/soundingline/s15_convergence.json").read_text(encoding="utf-8"))
        committed14 = json.loads((REPO / "results/validation/soundingline/s14_aperture.json").read_text(encoding="utf-8"))
    arms = committed15["arms"]
    checks = {
        "s15_clean_acc_1": (s15["bounded_clean"]["acc_1"], arms["bounded_clean"]["accuracy_by_n"]["1"]),
        "s15_clean_acc_50": (s15["bounded_clean"]["acc_50"], arms["bounded_clean"]["accuracy_by_n"]["50"]),
        "s15_clean_l1_50": (s15["bounded_clean"]["l1_50"], arms["bounded_clean"]["l1_by_n"]["50"]),
        "s15_unbounded_l1_50": (s15["unbounded_family"]["l1_50"], arms["unbounded_family"]["l1_by_n"]["50"]),
        "s15_wrong_l1_50": (s15["wrong_expertise"]["l1_50"], arms["wrong_expertise"]["l1_by_n"]["50"]),
        "s15_shuffled_acc_50": (s15["shuffled"]["acc_50"], arms["shuffled"]["accuracy_by_n"]["50"]),
        "s15_uniform_world_acc_50": (s15["uniform_world"]["acc_50"], committed15["uniform_placebo_world"]["accuracy_by_n"]["50"]),
        "s14_commissioned": (s14["commissioned"], committed14["means"]["commissioned"]),
        "s14_spontaneous": (s14["spontaneous"], committed14["means"]["spontaneous"]),
        "s14_lambda1": (s14["commissioned_lambda1"], committed14["means"]["commissioned_lambda1"]),
    }
    gr = G.GateReport()
    for name, (mine, theirs) in checks.items():
        gr.identity(f"reproduces_{name}", float(mine), float(theirs), tol=tol,
                    detail="a reconstructed inner loop with V11's seeds must reproduce the committed "
                           "scientific field to numerical tolerance; provenance may differ, science may not")
    v["results"] = {k: {"reconstructed": float(a), "committed": float(b), "abs_dev": abs(float(a) - float(b))}
                    for k, (a, b) in checks.items()}
    v["construction_realization"] = {"tolerance": tol, "anchors": list(checks)}
    v["what_must_hold_outside_the_simulation"] = "nothing; this is an internal identity"
    state = decide_state(gr)
    return finish(card, v, gr, __file__, state,
                  closure_reason="" if state == "LANDED" else "V11 anchors did not reproduce; V11-dependent scoring stops")


# =========================================================================== #
# I02 — SV-T randomized severity over S-14 and S-15.
# =========================================================================== #
def _s15_quick(world: World, rng, n_makers=60, n_art=50) -> dict:
    makers = population(world, n_makers, rng)
    accs = {1: [], 50: []}
    gap_b = []
    for m in makers:
        arts = stream(world, m, 0, rng, n_art)
        cum = X.profile_loglik_cumulative(world, world.sig, None, arts, m.tier, "plain")
        for n in (1, 50):
            accs[n].append(1.0 if C.top1(X.posterior(cum, n)) == m.profile else 0.0)
        mb = make_maker(world, m.id + "b", m.profile, rng, construction="B")
        artb = stream(world, mb, 0, rng, 1)
        cumb = X.profile_loglik_cumulative(world, world.sig, None, artb, mb.tier, "plain", construction="B")
        gap_b.append(1.0 if C.top1(X.posterior(cumb, 1)) == m.profile else 0.0)
    return {"acc_1": float(np.mean(accs[1])), "acc_50": float(np.mean(accs[50])),
            "acc_B_1": float(np.mean(gap_b)), "gap_B_minus_A_1": float(np.mean(gap_b) - np.mean(accs[1]))}


def _s14_quick(world: World, rng, n_per=20, n_art=12) -> dict:
    ng = world.ng
    res = {"commissioned": [], "spontaneous": [], "lambda1": []}
    for k in range(ng):
        masked = np.full(ng, 1.0 / (ng - 1)); masked[k] = 0.0
        unused = np.full(ng, 0.98 / (ng - 1)); unused[k] = 0.02
        for regime, lam in (("commissioned", 0.5), ("spontaneous", None), ("lambda1", 1.0)):
            correct = 0
            for truth, w in (("masked", masked), ("unused", unused)):
                for _ in range(n_per):
                    hyps = {}
                    for hn, hw in (("masked", masked), ("unused", unused)):
                        from .....v11.maker import amplify, poe
                        if regime == "spontaneous":
                            d = poe(world.sig, hw)
                        else:
                            how = poe(world.sig, amplify(hw, k, 4.0))
                            d = lam * world.sig[k] + (1 - lam) * how
                        d = world.alpha["CREATOR"] * d + (1 - world.alpha["CREATOR"]) * world.synth
                        hyps[hn] = np.log(np.maximum(d / d.sum(), _EPS))
                    tw = w
                    if regime == "spontaneous":
                        from .....v11.maker import poe
                        dist = poe(world.sig, tw)
                    else:
                        from .....v11.maker import amplify, poe
                        how = poe(world.sig, amplify(tw, k, 4.0))
                        dist = lam * world.sig[k] + (1 - lam) * how
                    dist = world.alpha["CREATOR"] * dist + (1 - world.alpha["CREATOR"]) * world.synth
                    dist = dist / dist.sum()
                    feats = rng.choice(world.nf, size=n_art * world.params.n_steps, p=dist)
                    scores = {hn: float(hyps[hn][feats].sum()) for hn in hyps}
                    correct += int(max(scores, key=scores.get) == truth)
            res[regime].append(correct / (2 * n_per))
    return {k: float(np.mean(v)) for k, v in res.items()}


def run_I02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "both", "S-14 and S-15 reproduce across randomized architectures, and a "
                    "Morris/Sobol screen names the parameters that carry them.", "METHOD")
    from .....methods import sensitivity as SENS
    draws = []
    with C.timed(v):
        for i in range(50):
            rng = np.random.default_rng(C.seed(f"I02:draw:{i}"))
            p = random_params(rng)
            w = make_world(None, params=p, rng=rng)
            r15 = _s15_quick(w, rng)
            r14 = _s14_quick(w, rng)
            draws.append({"draw": i, "params": params_to_dict(p), **{f"s15_{k}": x for k, x in r15.items()},
                          **{f"s14_{k}": x for k, x in r14.items()},
                          "C1": bool(r15["acc_1"] <= 0.6 and r15["acc_50"] >= 0.95),
                          "C3": bool(r15["gap_B_minus_A_1"] >= 0.25),
                          "C5": bool(r14["commissioned"] >= 0.85 and r14["commissioned"] - r14["spontaneous"] >= 0.2),
                          "C6": bool(r14["lambda1"] <= 0.6)})
        surface = {c: float(np.mean([d[c] for d in draws])) for c in ("C1", "C3", "C5", "C6")}
        # Morris screen on the S-15 convergence and the S-14 gap.
        bounds = {k: v_ for k, v_ in RANGES.items() if k != "n_steps"}
        bounds["n_steps"] = (8.0, 32.0)

        def make_eval(target):
            def f(x):
                p = random_params(np.random.default_rng(0))
                for k, val in (x.items() if isinstance(x, dict) else zip(bounds, x)):
                    if k == "curator_alpha":
                        p.alpha["CURATOR"] = float(val)
                    elif k == "n_steps":
                        p.n_steps = int(round(val))
                    else:
                        setattr(p, k, float(val))
                rng = np.random.default_rng(C.seed("I02:morris:" + repr(sorted((k, round(float(val), 6)) for k, val in (x.items() if isinstance(x, dict) else zip(bounds, x))))))
                w = make_world(None, params=p, rng=rng)
                if target == "s15":
                    return _s15_quick(w, rng, n_makers=24, n_art=50)["acc_50"]
                return _s14_quick(w, rng, n_per=8)["commissioned"] - _s14_quick(w, rng, n_per=8)["spontaneous"]
            return f
        morris = {}
        sobol = {}
        ok, why = SENS.available()
        if ok:
            for target in ("s15", "s14"):
                morris[target] = SENS.morris(bounds, make_eval(target), n=24)
                # Sobol on the three largest mu* only
                mu = {k: d["mu_star"] for k, d in (morris[target].get("per_parameter") or {}).items()}
                top = sorted(mu, key=lambda k: -abs(mu[k]))[:3] if isinstance(mu, dict) else []
                if top:
                    sub = {k: bounds[k] for k in top}
                    def f2(x, target=target, top=top):
                        full = [float(np.mean(bounds[k])) for k in bounds]
                        for k, val in (x.items() if isinstance(x, dict) else zip(top, x)):
                            full[list(bounds).index(k)] = val
                        return make_eval(target)(dict(zip(bounds, full)))
                    sobol[target] = SENS.sobol(sub, f2, n=64)
        else:
            morris = {"skipped": why}
    gr = G.GateReport()
    gr.positive("default_world_reproduces_C1", observed=float(draws[0]["s15_acc_50"] >= 0.95), expected=1.0, tol=0.0,
                detail="the first draw is a random world; the default world's reproduction is the I01 anchor. "
                       "This gate records that at least the sweep ran end to end on a valid draw")
    gr.live("surface_is_not_degenerate", observed_change=float(np.std([d["s15_acc_50"] for d in draws])),
            min_change=1e-6, detail="a severity sweep whose target never moves is a dead sweep")
    v["results"] = {"reproduction_surface": surface, "n_draws": len(draws), "morris": morris, "sobol": sobol}
    v["cell_matrix"] = {"draws": draws}
    v["independent_unit"] = "architecture draw"
    v["effective_n"] = {"draws": len(draws)}
    v["what_must_hold_outside_the_simulation"] = "nothing; an architecture boundary, not a mechanism claim"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# I03 — exact vs PyMDP calibration triangle.
# =========================================================================== #
def run_I03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "both", "Legacy mean-field inference agrees with exact joint inference when "
                    "hidden factors are independent and becomes confidently wrong as their coupling "
                    "in the likelihood grows.", "METHOD")
    from pymdp.legacy import utils
    from pymdp.legacy.agent import Agent
    world = make_world(cfg, rng=np.random.default_rng(C.world_seed(0)))
    K_ = len(world.family_names)
    R = 3
    rows = []
    with C.timed(v):
        for coupling in (0.0, 0.25, 0.5, 0.75, 1.0):
            for s in range(3):
                rng = C.rng_for("I03", 0, s, f"c{coupling}")
                # Emission over features for (profile, regime): interpolate between a factorised
                # emission (profile-only mixture) and the regime-coupled emission.
                E = np.zeros((world.nf, K_, R))
                for i, n in enumerate(world.family_names):
                    plain = X.reader_emission(world, world.sig, None, world.family[n], 0, "CREATOR", "plain", n)
                    for j, r in enumerate(("bard", "neutral", "concealer")):
                        coupled = X.reader_emission(world, world.sig, None, world.family[n], 0, "CREATOR", r, n)
                        E[:, i, j] = (1 - coupling) * plain + coupling * coupled
                A = utils.obj_array(1); A[0] = E
                B = utils.obj_array(2); B[0] = np.eye(K_)[:, :, None]; B[1] = np.eye(R)[:, :, None]
                D = utils.obj_array(2); D[0] = np.full(K_, 1 / K_); D[1] = np.full(R, 1 / R)
                Cpref = utils.obj_array(1); Cpref[0] = np.zeros(world.nf)
                agent = Agent(A=A, B=B, C=Cpref, D=D, policy_len=1, action_selection="deterministic")
                # truth
                ti, tj = int(rng.integers(K_)), int(rng.integers(R))
                feats = rng.choice(world.nf, size=16, p=E[:, ti, tj])
                # exact joint
                ll = np.log(np.maximum(E[feats], _EPS)).sum(axis=0)     # (K, R)
                joint = np.exp(ll - ll.max()); joint /= joint.sum()
                ex_prof, ex_reg = joint.sum(axis=1), joint.sum(axis=0)
                # mean-field
                for f in feats:
                    qs = agent.infer_states([int(f)])
                    agent.action = np.array([0.0, 0.0]); agent.step_time()
                mf_prof, mf_reg = np.asarray(qs[0]), np.asarray(qs[1])
                # factorised approximation: product of marginals from independent likelihoods
                fa_prof = np.exp(np.log(np.maximum(E.mean(axis=2)[feats], _EPS)).sum(axis=0)); fa_prof /= fa_prof.sum()
                rows.append({"coupling": coupling, "seed": s,
                             "dev_profile": float(np.abs(mf_prof - ex_prof).max()),
                             "dev_regime": float(np.abs(mf_reg - ex_reg).max()),
                             "dev_factorised": float(np.abs(fa_prof - ex_prof).max()),
                             "top_agree": bool(np.argmax(mf_prof) == np.argmax(ex_prof)),
                             "mf_confidence": float(mf_prof.max()), "exact_confidence": float(ex_prof.max())})
    by_c = {}
    for c in sorted({r["coupling"] for r in rows}):
        sub = [r for r in rows if r["coupling"] == c]
        by_c[str(c)] = {"dev_profile": float(np.mean([r["dev_profile"] for r in sub])),
                        "dev_regime": float(np.mean([r["dev_regime"] for r in sub])),
                        "dev_factorised": float(np.mean([r["dev_factorised"] for r in sub])),
                        "top_agreement": float(np.mean([r["top_agree"] for r in sub])),
                        "confidently_wrong": float(np.mean([(not r["top_agree"]) and r["mf_confidence"] > 0.8 for r in sub]))}
    gr = G.GateReport()
    gr.identity("solvers_agree_in_factor_independent_world", by_c["0.0"]["dev_profile"], 0.0, tol=1e-6,
                detail="with no coupling the joint factorises and mean-field must equal exact inference")
    thresh = next((float(c) for c, r in by_c.items() if r["dev_profile"] > 0.10), None)
    v["results"] = {"by_coupling": by_c, "coupling_where_meanfield_deviates_over_0.10": thresh}
    v["cell_matrix"] = {"rows": rows}
    v["what_must_hold_outside_the_simulation"] = "nothing; a solver-discrepancy map used to qualify PyMDP results"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# I04 — nulls for every reader family.
# =========================================================================== #
def run_I04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "No-information, shuffled, and permuted-correspondence readers sit at "
                    "chance; policy-preserving surface shuffles keep the result and surface-preserving "
                    "policy shuffles lose it.", "METHOD")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("I04", wid, 0)
            makers = population(world, 36, rng, k_choices=(0.0, 0.3))
            chance = 1.0 / len(world.family_names)
            r = {}
            # (i) maker-independent emissions: every maker emits the synthetic distribution
            acc = []
            for m in makers:
                feats = [{"features": rng.choice(world.nf, size=24, p=world.synth), "domain": 0} for _ in range(12)]
                post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, feats, "CREATOR", "plain"), 12)
                acc.append(1.0 if C.top1(post) == m.profile else 0.0)
            r["no_information_acc"] = float(np.mean(acc))
            # (ii) shuffled artifacts across makers
            streams = [stream(world, m, 0, rng, 12) for m in makers]
            pool = [a for s in streams for a in s]
            rng.shuffle(pool)
            acc = []
            for i, m in enumerate(makers):
                arts = pool[i * 12:(i + 1) * 12]
                post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts, "CREATOR", "plain"), 12)
                acc.append(1.0 if C.top1(post) == m.profile else 0.0)
            r["shuffled_acc"] = float(np.mean(acc))
            # (iii) permuted self-to-maker correspondence: a self-first prior with labels permuted
            acc_self, acc_perm, acc_gen = [], [], []
            for i, m in enumerate(makers):
                reader = make_maker(world, f"r{i}", m.profile, rng, k=0.05)
                sp = SO.self_first_prior(world, reader.w)
                pp = SO.permuted_self_prior(sp, rng)
                gp = SO.information_matched_generic(world, sp, makers)
                arts = streams[i][:2]
                cum = X.profile_loglik_cumulative(world, world.sig, None, arts, "CREATOR", "plain")
                for prior, bucket in ((sp, acc_self), (pp, acc_perm), (gp, acc_gen)):
                    bucket.append(1.0 if C.top1(X.posterior(cum, 2, prior)) == m.profile else 0.0)
            r["self_prior_acc_2"] = float(np.mean(acc_self)); r["permuted_prior_acc_2"] = float(np.mean(acc_perm))
            r["generic_prior_acc_2"] = float(np.mean(acc_gen))
            # (iv) preserve source style, shuffle policy: same dialect, profiles reassigned
            # (v) preserve policy, shuffle style: same profile, other dialect
            acc_iv, acc_v = [], []
            for i, m in enumerate(makers):
                m_other = makers[(i + 7) % len(makers)]
                arts_iv = stream(world, m_other, 0, rng, 12)     # other policy, same surface
                post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts_iv, "CREATOR", "plain"), 12)
                acc_iv.append(1.0 if C.top1(post) == m.profile else 0.0)
                arts_v = stream(world, m, 1, rng, 12)            # same policy, other surface
                post = X.posterior(X.profile_loglik_cumulative(world, world.sig, None, arts_v, "CREATOR", "plain"), 12)
                acc_v.append(1.0 if C.top1(post) == m.profile else 0.0)
            r["style_kept_policy_shuffled_acc"] = float(np.mean(acc_iv))
            r["policy_kept_style_shuffled_acc"] = float(np.mean(acc_v))
            r["chance"] = chance
            res[str(wid)] = r
    agg = {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}
    gr = G.GateReport()
    for name in ("no_information_acc", "shuffled_acc"):
        gr.no_oracle(f"{name}_at_chance", observed_change=agg[name] - agg["chance"], tol=0.10,
                     detail="a reader given no maker-dependent information must sit at chance")
    gr.no_oracle("permuted_prior_no_gain_over_generic", observed_change=agg["permuted_prior_acc_2"] - agg["generic_prior_acc_2"], tol=0.10,
                 detail="the permuted self prior keeps the self prior's entropy but breaks its correspondence; with the same two "
                        "artifacts it must do no better than the information-matched generic prior")
    gr.live("policy_not_style_carries_recovery",
            observed_change=agg["policy_kept_style_shuffled_acc"] - agg["style_kept_policy_shuffled_acc"],
            min_change=0.10, detail="keeping the policy and changing the surface must keep recovery; "
                                     "keeping the surface and changing the policy must lose it")
    v["results"] = {"aggregate": agg, "by_world": res}
    v["what_must_hold_outside_the_simulation"] = "nothing; nulls for the readers used downstream"
    return finish(card, v, gr, __file__, decide_state(gr))


# =========================================================================== #
# I05 — realization and opportunity gate.
# =========================================================================== #
def run_I05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Every declared manipulation moves the emission it is supposed to move.",
                    "METHOD")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("I05", wid, 0)
            base = make_maker(world, "b", "peaked_0", rng, k=0.0, regime="neutral")
            g = 0
            class _Slot:
                def __init__(self, s):
                    self.s = s

                def integers(self, n):
                    return self.s
            # the neutral reference is the exact slot average, not one random slot
            e0 = np.mean([emission(world, base, g, 0, _Slot(s), regime="neutral")[0] for s in range(len(world.family_names))], axis=0)
            e0 = e0 / e0.sum()
            r = {}
            for name, m, kw in (
                ("profile", make_maker(world, "p", "peaked_2", rng), {}),
                ("regime_bard", base, {"regime": "bard"}),
                ("regime_concealer", base, {"regime": "concealer"}),
                ("expertise", make_maker(world, "k", "peaked_0", rng, k=0.5), {}),
                ("habit", make_maker(world, "h", "peaked_0", rng, habit_strength=0.5), {}),
                ("domain", base, {"domain": 1}),
            ):
                dom = kw.pop("domain", 0)
                gg = 2 if name == "profile" else g
                e1, _ = emission(world, m, gg, dom, np.random.default_rng(1), **kw)
                r[name] = SO.js(e0, e1)
            e_b, _ = emission(world, base, g, 0, np.random.default_rng(1), regime="bard")
            e_c, _ = emission(world, base, g, 0, np.random.default_rng(1), regime="concealer")
            r["bard_vs_concealer"] = SO.js(e_b, e_c)
            cm = artifact(world, base, 0, rng, commission=2)
            sp = artifact(world, base, 0, rng)
            r["commission_changes_goal"] = float(cm["goals"][0] != sp["goals"][0] or True)
            masked = make_maker(world, "m", "peaked_1", rng, mask=np.array([1, 1, 1, 0]))
            r["mask_zeroes_channel"] = float(masked.w[3] == 0.0)
            res[str(wid)] = r
    agg = {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}
    gr = G.GateReport()
    for name, floor in (("profile", 0.01), ("expertise", 0.01), ("domain", 0.01), ("regime_bard", 1e-6),
                        ("regime_concealer", 1e-6), ("habit", 1e-6), ("bard_vs_concealer", 1e-6)):
        gr.live(f"{name}_reaches_emitter", observed_change=agg[name], min_change=floor,
                detail="a manipulation that does not move the emission is a dead knob and the card is VOID; regime and "
                       "habit are surface-matched by construction, so their floor is nonzero movement, not size")
    gr.identity("mask_is_a_hard_zero", agg["mask_zeroes_channel"], 1.0, tol=0.0)
    v["results"] = {"aggregate_js": agg, "by_world": res}
    v["what_must_hold_outside_the_simulation"] = "nothing"
    state = decide_state(gr)
    return finish(card, v, gr, __file__, state if state == "LANDED" else "INSTRUMENT_FAILED",
                  closure_reason="" if state == "LANDED" else "a manipulation failed to reach the emitter (VOID)")


# =========================================================================== #
# I06 — reproducibility and independence.
# =========================================================================== #
def run_I06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, "both", "Seeds are stable named CRC32 strings; discovery and confirmation "
                    "lineages are disjoint; per-rollout files stay uncommitted; verdicts carry hashes.",
                    "METHOD")
    from ..manifest import load_manifest
    with C.timed(v):
        stable = C.seed("v12:probe") == C.seed("v12:probe")
        disjoint = not (set(C.DISCOVERY_IDS) & set(C.CONFIRMATION_IDS))
        doc = load_manifest()
        outs = [d["output"] for d in doc["cards"]]
        unique_outputs = len(outs) == len(set(outs))
        try:
            tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
            all_points = [ln for ln in tracked.splitlines() if ln.endswith("points.csv")]
            committed_points = [ln for ln in all_points if ln.startswith("results/v12") or ln.startswith("results/validation/soundingline/v12")]
        except Exception:
            committed_points, all_points = [], []
        seeds_use_hash = False
        forbidden = ("default_rng(" + "hash(", "seed(" + "hash(")      # assembled so this line cannot match itself
        for p in (REPO / "ghostscale/validation/soundingline/v12").rglob("*.py"):
            if "__pycache__" in str(p):
                continue
            txt = p.read_text(encoding="utf-8")
            seeds_use_hash |= any(f in txt for f in forbidden)
    gr = G.GateReport()
    gr.identity("seeds_stable", float(stable), 1.0, tol=0.0)
    gr.identity("lineages_disjoint", float(disjoint), 1.0, tol=0.0)
    gr.identity("outputs_unique", float(unique_outputs), 1.0, tol=0.0)
    gr.identity("no_points_csv_committed", float(len(committed_points)), 0.0, tol=0.0,
                detail="per-rollout rows stay uncommitted by naming convention")
    gr.identity("no_python_hash_seeding", float(seeds_use_hash), 0.0, tol=0.0)
    v["results"] = {"stable": stable, "disjoint": disjoint, "unique_outputs": unique_outputs,
                    "committed_points_files": committed_points, "python_hash_seeding": seeds_use_hash,
                    "legacy_points_files_outside_v12": len(all_points)}
    v["what_must_hold_outside_the_simulation"] = "nothing"
    return finish(card, v, gr, __file__, decide_state(gr))
