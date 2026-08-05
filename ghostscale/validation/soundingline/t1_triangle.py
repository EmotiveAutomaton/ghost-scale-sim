"""T-1 — is empathy a chain or a triangle? Six directed edges, measured the same way.

THE CLAIM. Empathy is three coupled variational inference problems -- proximal goal, process,
values -- each bootstrapping the others. E36 measured goal -> process and called it the sharpest
forward prediction in the project. If the triangle is right, the other five edges exist and were
never asked about.

WHAT THIS MODULE DOES AND DOES NOT ANSWER, stated first because the substitution is the whole
result and burying it would be dishonest.

**THE VALUES VERTEX DOES NOT EXIST IN THIS MODEL.** ``v6_model.build_values_map`` looks like it,
is used live by E41, E55 and E56, and is asserted non-injective under null N26. It is still
``M[g % nv, g] = 1``: a DETERMINISTIC many-to-one projection of the goal, so ``H(values | goal)``
is exactly zero. Non-injectivity only rules out the values layer being the goal RENAMED; it does
not make it independent. Four of the six edges through such a vertex are decided by the arity of
a hard-coded matrix and would return clean, bootstrapped, entirely artifactual numbers. That is
E45's oracle failure wearing a different mask, and ``values_degeneracy`` measures it rather than
asserting it.

``beta`` is not a fallback either. It controls how much of the emission is attached to the goal
rather than to the mode family's goal-marginal -- it is GOAL LEGIBILITY, a weight on the very
edge being measured, not a third vertex.

**WHAT IS RUN INSTEAD: goal - process - depth.** ``mu`` is a genuine third latent. The reader
carries a posterior over it, ``depth_marginal_invariance`` argues it has a materially better
identification channel than beta ever had, and E30/E31/E43 are all about it. This answers the
structural question -- chain or triangle, symmetric or not, is there a dead edge -- on three
latents that are actually separate. It does not answer the values question and nothing here
should be quoted as if it did.

EVERY EDGE IS MEASURED THE SAME WAY, which E36 is not. E36 splits a rollout at the step the goal
posterior settles and compares process recovery either side: that is OBSERVATIONAL. Supplying a
vertex down a side channel is INTERVENTIONAL. Asking whether goal->process equals process->goal
with one arm observational and the other interventional would return the difference between two
methods as if it were an asymmetry of the triangle -- and asymmetry is the headline. So all six
run interventionally here and E36 is reproduced separately, by ``e36_reference``, as a different
measurement of a related thing.

CROSS-VERTEX COMPARISON IS MADE IN NATS, NOT IN FIDELITY. Depth has three values and goal has
four, so the same fidelity supplies different amounts of information. Every primary contrast is
run at matched channel mutual information; the fidelity sweep is kept only for the monotonicity
check.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ... import v6_model as V6
from ...config import Config
from ...v5_model import MU_LEVELS, mgs_index
from ...v6 import SEED_OFFSET
from . import sl_dir
from .common import build, process_gain, rollouts, usable
from . import t_common as T

VERTS = ("goal", "process", "depth")

#: Matched-information levels, in nats. Capacity is log(3) = 1.099 for depth and log(4) = 1.386
#: for goal and process, so 1.00 is the highest level every vertex can actually carry.
NAT_LEVELS = (0.0, 0.15, 0.35, 0.70, 1.00)

#: Raw fidelities, kept only for the dose-response monotonicity check within a single vertex.
FID_LEVELS = (0.40, 0.55, 0.70, 0.85, 0.99)

#: The cells the triangle is run in. beta is the headroom dial: at beta = 1.0 the reader gets the
#: goal right every single time, so every edge INTO goal is unmeasurable there -- a ceiling, not
#: a null, and reported as one.
CELLS = [(3, 1.0), (3, 0.25), (3, 0.10), (2, 0.25), (2, 0.10), (1, 1.0), (1, 0.25)]

N_TIMESTEPS = 24
FORCED_K = 24
PROV = K.CREATOR


def _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta, channels, n_obs, tag, seed):
    rows = []
    for i in range(int(n_obs)):
        g = int(i % ng)
        r = T.run_supplied_encounter(
            world, cfg_r, channels, mu, beta, g, N_TIMESTEPS, FORCED_K, n_mu, n_sub, ng,
            np.random.default_rng(seed * 31 + i), np.random.default_rng(seed * 7907 + i),
            provenance=PROV, donor_seed=seed * 6421 + i)
        r.pop("true_modes", None)
        r.update({"mu": mu, "beta": beta, "arm": tag, "i": i, "true_goal": g})
        rows.append(r)
    return rows


def values_degeneracy(ng: int, n_values: int = 2) -> dict:
    """Measure what the values layer actually carries, rather than asserting it.

    Three numbers settle it. ``H(values | goal)`` is zero if values are a function of the goal.
    ``I(goal; values)`` is capped at ``log(n_values)`` however many goals there are, so the
    values vertex can never carry more than the coarsening. And the residual -- what values
    would add ON TOP of the goal -- is exactly zero, which is the statement that three of the six
    edges through this vertex are not edges.
    """
    M = V6.build_values_map(ng, n_values=n_values)
    # p(v | g) is a column of M, and every column is a point mass by construction.
    col_ent = [float(metrics.within_observer_entropy(M[:, g])) for g in range(ng)]
    pg = np.full(ng, 1.0 / ng)
    pv = M @ pg
    h_v = float(metrics.within_observer_entropy(pv))
    return {
        "n_goals": int(ng), "n_values": int(n_values),
        "H_values_given_goal_nats": float(np.mean(col_ent)),
        "H_values_given_goal_max_over_goals": float(np.max(col_ent)),
        "H_values_nats": h_v,
        "I_goal_values_nats": float(h_v - np.mean(col_ent)),
        "capacity_of_values_vertex_nats": float(np.log(n_values)),
        "capacity_of_goal_vertex_nats": float(np.log(ng)),
        "residual_information_values_adds_over_goal_nats": float(np.mean(col_ent)),
        "verdict": ("VALUES_IS_A_DETERMINISTIC_COARSENING_OF_GOAL"
                    if float(np.max(col_ent)) < 1e-12 else "VALUES_IS_INDEPENDENT"),
        "what_this_means": (
            "H(values | goal) = 0 means the values vertex adds nothing to the goal. goal->values "
            "is a lookup that returns 1.0 by construction; values->goal returns exactly the "
            "1/n_values coarsening; values->process and process->values carry only what routes "
            "through the goal. Those four edges are properties of the matrix, not measurements. "
            "The values vertex has to be BUILT before T-1 can be asked as posed."),
    }


def e36_reference(cfg: Config, n_obs: int = 120) -> dict:
    """Reproduce E36's observational goal->process gain on the unmodified rollout.

    Rule 2 of this package: if you claim to use an experiment's rollouts, reproduce its number
    first. This does not feed the triangle -- it is here so the report can say how far the
    interventional edge sits from the observational one, and so that a reader can tell the two
    apart at a glance.
    """
    out = {}
    for mu in (1, 2, 3):
        gains, n_used, n_seen = [], 0, 0
        for rec in rollouts(cfg, mus=[mu], betas=[1.0], n_obs=n_obs,
                            n_timesteps=N_TIMESTEPS, forced_k=FORCED_K):
            n_seen += 1
            if not usable(rec):
                continue
            n_used += 1
            gains.append(process_gain(rec["enc"], int(rec["settled"]), int(rec["n_sub"])))
        out[str(mu)] = {"observational_process_gain": float(np.mean(gains)) if gains else float("nan"),
                        "n_usable": n_used, "n_seen": n_seen}
    return out


def run(cfg: Config, n_obs: int = 150, n_obs_robust: int | None = None) -> dict:
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = build(cfg)
    rng = np.random.default_rng(SEED_OFFSET + 91_100)
    cards = {v: T.card(v, n_mu, n_sub, ng) for v in VERTS}

    # ---- harness check 0: the merged state index order this whole module depends on -------
    vv = T.vertex_value_by_state("goal", n_mu, n_sub, ng)
    vs = T.vertex_value_by_state("process", n_mu, n_sub, ng)
    vd = T.vertex_value_by_state("depth", n_mu, n_sub, ng)
    index_ok = all(
        vv[mgs_index(mi, g, s, ng, n_sub)] == g and vs[mgs_index(mi, g, s, ng, n_sub)] == s
        and vd[mgs_index(mi, g, s, ng, n_sub)] == mi
        for mi in range(n_mu) for g in range(ng) for s in range(n_sub))
    assert index_ok, "merged (mu, goal, subgoal) index order disagrees with mgs_index"

    rows = []
    # ---- the six edges, at matched information ---------------------------------------------
    for (mu, beta) in CELLS:
        seed = 91_000 + mu * 100 + int(beta * 100)
        rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta, [], n_obs, "control", seed)
        for v in VERTS:
            for nats in NAT_LEVELS:
                if nats <= 0.0:
                    f = 1.0 / cards[v]
                elif nats >= np.log(cards[v]) - 1e-9:
                    continue
                else:
                    f = T.fidelity_for_nats(nats, cards[v])
                rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta,
                                   [(v, f, cards[v])], n_obs, f"supply:{v}@{nats:.2f}nats", seed)
            # THREE negative controls at the same fidelity as the honest 1-nat arm.
            f_hi = T.fidelity_for_nats(1.00, cards[v])
            for cname, cmode in (("rot", 1), ("random", "random"), ("swap", "swap")):
                rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta,
                                   [(v, f_hi, cards[v], cmode)], n_obs,
                                   f"neg:{cname}:{v}", seed)
        # dose-response in raw fidelity, one cell's worth per vertex
        for v in VERTS:
            for f in FID_LEVELS:
                if f <= 1.0 / cards[v]:
                    continue
                rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta,
                                   [(v, f, cards[v])], n_obs, f"fid:{v}@{f:.2f}", seed)
        # BUDGET-MATCHED ARMS. Goal and depth are one value per artifact and process is a new
        # value every step, so a channel at one nat per step saturates the first two and hands
        # twenty-four independent nats to the third. These arms hold the DELIVERED TOTAL equal
        # by dropping the duty cycle, so the asymmetry can be read at a matched budget rather
        # than at a matched rate.
        for v in VERTS:
            for duty in (1.0 / 24.0, 3.0 / 24.0, 8.0 / 24.0):
                f = T.fidelity_for_nats(1.00, cards[v])
                rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta,
                                   [(v, f, cards[v], 0, duty)], n_obs,
                                   f"duty:{v}@{duty:.4f}", seed)
        # pairs, each vertex at 0.70 nats so the pair carries twice a single arm
        for a, b in (("goal", "process"), ("goal", "depth"), ("process", "depth")):
            ch = [(a, T.fidelity_for_nats(0.70, cards[a]), cards[a]),
                  (b, T.fidelity_for_nats(0.70, cards[b]), cards[b])]
            rows += _cell_rows(world, cfg_r, n_mu, n_sub, ng, mu, beta, ch, n_obs,
                               f"pair:{a}+{b}", seed)

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t1_triangle.csv", index=False)

    # ---- edges ------------------------------------------------------------------------------
    def arm(mu, beta, tag):
        return df[(df.mu == mu) & (df.beta == beta) & (df.arm == tag)]

    edges, ceilings, placebo, lying, monotone, pairs = {}, {}, {}, {}, {}, {}
    for (mu, beta) in CELLS:
        key = f"mu{mu}_beta{beta}"
        ctrl = arm(mu, beta, "control")
        # ceiling flags: a measure with no headroom cannot show an edge into it
        ceilings[key] = {
            "goal_control_accuracy": float(ctrl.goal_acc.mean()),
            "goal_at_ceiling": bool(ctrl.goal_acc.mean() >= 0.995),
            "process_control_recovery": float(ctrl.process.mean()),
            "depth_control_recovery": float(ctrl.depth.mean()),
        }
        for src in VERTS:
            # placebo: zero nats must be exactly the control
            p = arm(mu, beta, f"supply:{src}@0.00nats")
            if len(p):
                placebo[f"{key}|{src}"] = {
                    tgt: float(np.max(np.abs(p[tgt].to_numpy() - ctrl[tgt].to_numpy())))
                    for tgt in VERTS}
            hi = arm(mu, beta, f"supply:{src}@1.00nats")
            negs = {c: arm(mu, beta, f"neg:{c}:{src}") for c in ("rot", "random", "swap")}
            for tgt in VERTS:
                if tgt == src:
                    continue
                if len(hi):
                    edges[f"{key}|{src}->{tgt}"] = T.boot_paired(
                        hi[tgt].to_numpy(), ctrl[tgt].to_numpy(), rng)
                for cname, a_ in negs.items():
                    if len(a_):
                        lying[f"{key}|{src}->{tgt}|{cname}"] = T.boot_paired(
                            a_[tgt].to_numpy(), ctrl[tgt].to_numpy(), rng)
            # monotonicity of the dose-response
            for tgt in VERTS:
                if tgt == src:
                    continue
                xs = []
                for f in FID_LEVELS:
                    a = arm(mu, beta, f"fid:{src}@{f:.2f}")
                    if len(a):
                        xs.append(float(a[tgt].mean()) - float(ctrl[tgt].mean()))
                if len(xs) >= 3:
                    d = np.diff(xs)
                    monotone[f"{key}|{src}->{tgt}"] = {
                        "series": [float(x) for x in xs],
                        "monotone_increasing": bool(np.all(d >= -1e-9)),
                        "fraction_non_decreasing": float(np.mean(d >= -1e-9)),
                        "spearman_like": float(np.corrcoef(np.arange(len(xs)), xs)[0, 1])
                        if np.std(xs) > 0 else float("nan"),
                    }
        for a, b in (("goal", "process"), ("goal", "depth"), ("process", "depth")):
            tgt = [v for v in VERTS if v not in (a, b)][0]
            pr = arm(mu, beta, f"pair:{a}+{b}")
            sa = arm(mu, beta, f"supply:{a}@0.70nats")
            sb = arm(mu, beta, f"supply:{b}@0.70nats")
            if len(pr) and len(sa) and len(sb):
                c = float(ctrl[tgt].mean())
                ga, gb = float(sa[tgt].mean()) - c, float(sb[tgt].mean()) - c
                both = T.boot_paired(pr[tgt].to_numpy(), ctrl[tgt].to_numpy(), rng)
                pairs[f"{key}|{a}+{b}->{tgt}"] = {
                    "gain_a": ga, "gain_b": gb, "sum_of_singles": ga + gb,
                    "gain_both": both["difference"], "interval_both": both["interval"],
                    "superadditive_excess": float(both["difference"] - (ga + gb)),
                    "is_superadditive": bool(both["difference"] > ga + gb),
                }

    # ---- N28 analogue: at mu = 1 there is no process to recover -----------------------------
    n28 = {}
    for beta in (1.0, 0.25):
        k = f"mu1_beta{beta}"
        ctrl = arm(1, beta, "control")
        if not len(ctrl):
            continue
        n28[k] = {
            "control_process_recovery": float(ctrl.process.mean()),
            "control_process_accuracy": float(ctrl.process_acc.mean()),
            "chance_accuracy": 1.0 / n_sub,
        }
        for src in ("goal", "depth"):
            hi = arm(1, beta, f"supply:{src}@1.00nats")
            if len(hi):
                n28[k][f"supplying_{src}_moves_process"] = T.boot_paired(
                    hi["process"].to_numpy(), ctrl["process"].to_numpy(), rng)

    # ---- budget-matched edges ---------------------------------------------------------------
    budget = {}
    for (mu, beta) in CELLS:
        key = f"mu{mu}_beta{beta}"
        ctrl = arm(mu, beta, "control")
        for src in VERTS:
            for duty in (1.0 / 24.0, 3.0 / 24.0, 8.0 / 24.0):
                a_ = arm(mu, beta, f"duty:{src}@{duty:.4f}")
                if not len(a_):
                    continue
                for tgt in VERTS:
                    if tgt == src:
                        continue
                    budget[f"{key}|{src}->{tgt}|duty{duty:.4f}"] = T.boot_paired(
                        a_[tgt].to_numpy(), ctrl[tgt].to_numpy(), rng)

    # ---- the mutual-information symmetry identity, used as a correctness check ---------------
    symmetry = {}
    for (mu, beta) in CELLS:
        key = f"mu{mu}_beta{beta}"
        a = edges.get(f"{key}|goal->depth", {}).get("difference")
        b = edges.get(f"{key}|depth->goal", {}).get("difference")
        if a is not None and b is not None:
            symmetry[key] = {
                "goal_to_depth": float(a), "depth_to_goal": float(b),
                "abs_difference": float(abs(a - b)),
                "note": (
                    "at one nat per step both of these vertices are fully determined for the "
                    "reader, so each edge is the conditional mutual information I(goal; depth | "
                    "data) -- which is SYMMETRIC. These two numbers agreeing to six decimals is "
                    "an identity the harness has to satisfy, and it is used here as a "
                    "correctness check rather than reported as a finding."),
            }

    verdict = {
        "test": "T-1 — is empathy three coupled inference problems, or a chain?",
        "for": "Sounding Line, the triangle claim; decides chain vs triangle",
        "SUBSTITUTION": (
            "the values vertex does not exist in this model and was NOT invented. The triangle "
            "run here is goal - process - DEPTH. See values_degeneracy for why, measured rather "
            "than asserted."),
        "values_degeneracy": values_degeneracy(ng),
        "method": {
            "supply": "extra observation modality at controlled fidelity; D identical in all arms",
            "matched_on": "channel mutual information in nats, not fidelity",
            "nat_levels": list(NAT_LEVELS),
            "cardinalities": cards,
            "n_per_arm": int(n_obs), "paired_across_arms": True,
            "cells": [{"mu": m, "beta": b} for (m, b) in CELLS],
            "initial_glance": False,
            "why_not_e36_directly": (
                "E36 is observational (split at the settling step); these are interventional. "
                "Mixing the two would return a method difference as an asymmetry."),
        },
        "e36_observational_reference": e36_reference(cfg, n_obs=n_obs),
        "edges_at_1_nat": edges,
        "ceilings": ceilings,
        "validity": {
            "placebo_max_abs_deviation": placebo,
            "negative_controls": lying,
            "dose_response": monotone,
            "n28_no_process_at_mu1": n28,
            "merged_index_order_ok": bool(index_ok),
        },
        "pairs_superadditivity": pairs,
        "budget_matched_edges": budget,
        "mutual_information_symmetry_check": symmetry,
    }
    (sl_dir() / "t1_triangle.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
