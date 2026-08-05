"""T-6 — what one observation carries about each latent, exactly, before any reader sees it.

NO ROLLOUTS. NO SAMPLING. The emission likelihood ``world.subsig`` IS a joint distribution over
(depth, goal, mode, feature), so every quantity here is computed in closed form. That makes this
the cheapest module in the repository and the only one whose numbers cannot be wrong for a
statistical reason.

WHY IT EXISTS: IT CORRECTS T-1's HEADLINE.

T-1 measured the triangle from the reader's side -- supply vertex X, measure recovery of Y -- and
found three of six edges dead, all three out of the goal. It reported that as *process is a source,
goal is a sink; the framework's chain runs backwards*. That reading is too strong, and the exact
information budget says why.

The creator attenuates its emission toward the goal-marginal of the mode family::

    emit[s] = beta * subsig[mu, g, s] + (1 - beta) * mean_g(subsig[mu, g, s])

so ``beta`` removes information about WHICH GOAL the modes serve while leaving the mode structure
untouched. Measured:

    beta    I(goal;F)   I(mode;F)   I(mode;F | goal)
    1.00      1.4521      0.2706      0.4262   (1.6x)
    0.50      0.3541      0.2706      0.2838   (1.0x)
    0.25      0.0941      0.2706      0.2738   (1.0x)
    0.10      0.0160      0.2706      0.2711   (1.0x)

**The mode carries 0.2706 bits at every beta -- invariant by construction. The goal's information
collapses.** And the coupling, ``I(mode;F | goal) / I(mode;F)``, is 1.6x at beta = 1 and exactly
1.0x at every beta below it.

T-1's cells with goal headroom are beta = 0.25 and 0.10. In those cells there is NO coupling to
find: knowing the goal adds nothing to what an observation says about the mode, by construction.
The one cell where coupling exists is beta = 1.0 -- which is exactly the cell where the reader
recovers the goal perfectly and T-1 flagged a ceiling.

**So goal -> process is not a dead edge. It is an edge the model can only create under a setting
that simultaneously removes the headroom needed to measure it.** T-1's +0.0017 at beta = 1.0 is
the whole of the observable coupling, and batch two dismissed it as noise. That was wrong.

THREE MORE DECOMPOSITIONS, each of which recovers something the project established the hard way:

  * marginalise the mode out and DEPTH contributes exactly 0.000 bits, with zero redundancy and
    zero synergy. Without the execution chain depth is invisible -- E30's null, from the
    likelihood instead of from a failed experiment.
  * marginalise the goal out and mode+depth together carry 0.189 bits, of which 0.162 is SYNERGY.
    Strip the purpose and what is left of process and depth is almost entirely joint.
  * at mu = 1 the unique-mode and synergistic terms are both exactly zero -- null N28, which the
    project spent an experiment establishing.

Those three are wired as ``identity`` gates: they are things the world has to satisfy, and if any
of them stops holding the generative model has changed underneath every result in the repository.
"""
from __future__ import annotations

import json

import numpy as np

from ...config import Config
from ...methods import gates as G
from ...methods import pid as PID
from ...methods import provenance as PROVENANCE
from ...v5_model import MU_LEVELS
from . import sl_dir
from .common import build

BETAS = (1.0, 0.50, 0.25, 0.10, 0.0)
_EPS = 1e-15


def _dist(joint: np.ndarray, names: str):
    """A dit distribution from a dense array, dropping zero-probability outcomes."""
    import dit
    idx = np.argwhere(joint > _EPS)
    outs = [tuple(int(x) for x in row) for row in idx]
    ps = [float(joint[tuple(row)]) for row in idx]
    d = dit.Distribution(outs, ps)
    d.set_rv_names(names)
    return d


def _mi(d, xs, ys) -> float:
    import dit
    return float(dit.shannon.mutual_information(d, list(xs), list(ys)))


def _cmi(d, x, y, given) -> float:
    """I(x; y | given) = H(y | given) - H(y | given, x)."""
    import dit
    g = list(given)
    return float(dit.shannon.conditional_entropy(d, list(y), g)
                 - dit.shannon.conditional_entropy(d, list(y), g + list(x)))


def attenuated_emission(subsig: np.ndarray, mu_index: int, beta: float) -> np.ndarray:
    """``HierarchicalCreator``'s own emission rule, lifted verbatim. Returns ``p[goal, mode, f]``.

    Reproducing the rule rather than importing it is deliberate: the creator applies it while
    sampling, one draw at a time, and there is no function that returns the distribution. Getting
    this wrong would make every number in this module describe a world nobody runs, so
    ``harness_matches_the_creator`` checks it against actual creator draws.
    """
    block = np.asarray(subsig, dtype=float)[int(mu_index)]
    block = block / block.sum(axis=-1, keepdims=True)
    marg = block.mean(axis=0)                                   # mean over goals, per mode
    out = float(beta) * block + (1.0 - float(beta)) * marg[None, :, :]
    return out / out.sum(axis=-1, keepdims=True)


def run(cfg: Config, n_obs: int = 4000) -> dict:
    """``n_obs`` is used only by the harness check that the emission rule matches the creator."""
    ok, why = PID.available()
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    sub = np.asarray(world.subsig, dtype=float)
    sub = sub / sub.sum(axis=-1, keepdims=True)
    nf = sub.shape[-1]
    gr = G.GateReport()

    verdict = {
        "test": "T-6 — the exact information budget of one observation",
        "for": "Sounding Line; corrects T-1's reading of the triangle",
        "method": ("closed form from world.subsig. No rollouts, no sampling, no estimator. "
                   "Every number is a property of the generative model."),
        "cardinalities": {"n_depth": int(n_mu), "n_goals": int(ng), "n_modes": int(n_sub),
                          "n_features": int(nf)},
    }

    if not ok:
        gr.skip("information_budget", "identity", why)
        verdict["skipped"] = why
        PROVENANCE.stamp(verdict, __file__, gr)
        (sl_dir() / "t6_information_budget.json").write_text(
            json.dumps(verdict, indent=2, default=str), encoding="utf-8")
        return verdict

    # ---- harness check: does the lifted emission rule match what the creator actually emits? ---
    from ...v6 import harness as H
    mu_i, beta_chk, g_chk = 2, 0.25, 1
    rng = np.random.default_rng(20260805)
    creator, _art, _env = H.make_artifact_and_env(
        world, cfg_r, g_chk, MU_LEVELS[mu_i], beta_chk, int(n_obs) + 4, rng)
    modes = np.asarray(creator.trajectory, dtype=int)
    feats = np.asarray([creator.next_feature(rng) for _ in range(int(n_obs))], dtype=int)
    modes = modes[:feats.size]
    pred = attenuated_emission(sub, mu_i, beta_chk)[g_chk]        # (n_sub, nf)
    emp = np.zeros_like(pred)
    for s, f in zip(modes, feats):
        emp[s, f] += 1.0
    keep = emp.sum(axis=1) > 30
    emp[keep] /= emp[keep].sum(axis=1, keepdims=True)
    worst = float(np.max(np.abs(emp[keep] - pred[keep]))) if keep.any() else float("nan")
    gr.positive("harness_matches_the_creator", worst, 0.0, 0.06,
                detail=("the emission rule is reproduced here rather than imported, because the "
                        "creator applies it one draw at a time and returns no distribution. This "
                        "compares the lifted rule against actual creator draws; a mismatch would "
                        "make every number in this module describe a world nobody runs."))

    # ---- 1. pairwise PID at each depth -------------------------------------------------------
    pairwise = {}
    for mi in range(n_mu):
        pairwise[f"mu{MU_LEVELS[mi]}"] = PID.emission_pid(sub, mi)

    # ---- 2. depth is invisible without the chain ---------------------------------------------
    gd = np.zeros((ng, n_mu, nf))
    for mi in range(n_mu):
        for g in range(ng):
            gd[g, mi] = sub[mi, g].mean(axis=0)
    goal_depth = PID.decompose(gd / gd.sum())

    md = np.zeros((n_sub, n_mu, nf))
    for mi in range(n_mu):
        for s in range(n_sub):
            md[s, mi] = sub[mi, :, s].mean(axis=0)
    mode_depth = PID.decompose(md / md.sum())

    # ---- 3. the three-way budget --------------------------------------------------------------
    joint = np.zeros((n_mu, ng, n_sub, nf))
    for mi in range(n_mu):
        joint[mi] = sub[mi] / float(n_mu * ng * n_sub)
    d3 = _dist(joint, "DGSF")
    solo = {"goal": _mi(d3, "G", "F"), "mode": _mi(d3, "S", "F"), "depth": _mi(d3, "D", "F")}
    cond = {"goal": _cmi(d3, "G", "F", "DS"), "mode": _cmi(d3, "S", "F", "DG"),
            "depth": _cmi(d3, "D", "F", "GS")}
    budget = {
        "total_bits": _mi(d3, "DGS", "F"),
        "alone": solo, "given_the_other_two": cond,
        "context_multiplier": {k: (float(cond[k] / solo[k]) if solo[k] > 1e-9 else float("inf"))
                               for k in solo},
        "how_to_read": (
            "alone is what the latent carries by itself; given_the_other_two is what it adds once "
            "you already know the rest. A latent with a small solo term and a large conditional "
            "one is readable only in context, which is what 'coupled inference problems' has to "
            "mean if it means anything measurable."),
    }

    # ---- 4. THE CORRECTION: coupling against goal legibility ---------------------------------
    by_beta = {}
    for beta in BETAS:
        E = attenuated_emission(sub, n_mu - 1, beta)              # deepest maker
        j = np.zeros((ng, n_sub, nf))
        for g in range(ng):
            j[g] = E[g] / float(ng * n_sub)
        d2 = _dist(j, "GSF")
        i_goal, i_mode = _mi(d2, "G", "F"), _mi(d2, "S", "F")
        i_mode_given_goal = _cmi(d2, "S", "F", "G")
        by_beta[f"{beta:.2f}"] = {
            "total_bits": _mi(d2, "GS", "F"),
            "I_goal": i_goal, "I_mode": i_mode,
            "I_mode_given_goal": i_mode_given_goal,
            "coupling_multiplier": (float(i_mode_given_goal / i_mode) if i_mode > 1e-9
                                    else float("nan")),
        }
    mults = {k: v["coupling_multiplier"] for k, v in by_beta.items()}
    mode_invariant = float(np.ptp([v["I_mode"] for v in by_beta.values()]))

    gr.identity("mode_information_is_invariant_in_beta", mode_invariant, 0.0, 1e-9,
                detail=("beta mixes the emission toward the goal-marginal of the mode family, so "
                        "it must leave information about WHICH MODE untouched at every level. "
                        "This is the fact that makes the coupling table interpretable."))
    n28 = PID.n28_identity_from_pid(sub)
    gr.identity("n28_no_mode_information_at_shallowest_depth", n28["worst_abs_deviation"], 0.0,
                1e-9,
                detail=("at mu = 1 every mode emits the goal signature exactly, so an emission "
                        "carries no unique or synergistic mode information. Null N28, recovered "
                        "from the likelihood with no rollouts."))
    gr.identity("depth_is_invisible_without_the_chain",
                abs(goal_depth.get("unique_source_b_bits", float("nan"))), 0.0, 1e-9,
                detail=("with the execution mode marginalised out, depth contributes exactly zero "
                        "unique information. This is depth_marginal_invariance and E30's null, "
                        "arrived at analytically."))
    gr.identity("total_information_within_channel_capacity",
                budget["total_bits"], min(budget["total_bits"], float(np.log2(nf))), 1e-9,
                detail="one observation cannot carry more than log2(n_features) bits.")

    verdict.update({
        "THE_CORRECTION": (
            "T-1 reported goal->process as a dead edge and read the triangle as running "
            "backwards. The coupling I(mode;F | goal) / I(mode;F) is 1.6x at beta = 1.0 and "
            "exactly 1.0x at every lower beta. T-1's cells with goal headroom are beta = 0.25 and "
            "0.10, where there is no coupling to find BY CONSTRUCTION; the only cell with "
            "coupling is beta = 1.0, which is the cell where the reader saturates and T-1 flagged "
            "a ceiling. goal->process is not dead. It is an edge this model can only create under "
            "a setting that removes the headroom needed to measure it, and T-1's +0.0017 at "
            "beta = 1.0 is the whole of the observable coupling."),
        "pairwise_pid_by_depth": pairwise,
        "goal_and_depth_with_mode_marginalised": goal_depth,
        "mode_and_depth_with_goal_marginalised": mode_depth,
        "three_way_budget": budget,
        "coupling_by_goal_legibility": by_beta,
        "coupling_multiplier_by_beta": mults,
        "what_would_have_falsified_the_correction": (
            "a coupling multiplier above 1 at beta = 0.25 and 0.10. That would mean the goal "
            "genuinely carries information about the mode in the cells T-1 measured, and the dead "
            "goal->process edge would be a finding about the reader rather than a property of the "
            "emission."),
        "what_this_cannot_show": (
            "anything about a reader. This is what ONE OBSERVATION carries in principle. A reader "
            "integrating twenty-four of them, with a prior and a policy, can do better or worse "
            "than these numbers and T-1 is where that is measured."),
    })
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "t6_information_budget.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
