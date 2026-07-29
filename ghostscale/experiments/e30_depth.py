"""E30 — mu, estimated model depth, replaces beta (V5 C1).

A NEW DESIGN, NOT A RE-RUN, and V5 decision 8 says to report it as one. The V5 spec asks for
"identical design to E28 with mu as the inferred quantity", and that phrasing is only
satisfiable if mu is a scalar on beta's axis — which is exactly the failure C1 exists to
prevent. Under route A the creator is a hierarchy walking a plan, so there is no version of
E28's stimulus to re-run. E28 and E30 are reported side by side as two constructs measured on
comparable designs, and E28 is not deleted.

-----------------------------------------------------------------------------------------
WHAT IS BEING ASKED. E28 found beta recoverable from content alone and found update magnitude
falling with it, but goal accuracy collapsed to 0.257 at the bottom of beta's range: low-beta
content carries no goal information, so the observer that failed to read it demonstrated
nothing kappa_p could not explain. The predicted category — *legible, competent, and unmoving*
— existed over only part of the range.

Depth is constructed so that the shallow end is NOT illegible. The execution modes average
exactly to sig[g] and the mode chains are doubly stochastic, so a mu = 1 artifact and a mu = 3
artifact have identical time-averaged feature histograms and are exactly as easy to read for
goal. If E30's accuracy holds at every depth while update magnitude still scales, the category
exists across the WHOLE range and mu is doing something neither existing gate can express.

If accuracy instead falls with depth, the construction has leaked legibility into mu, mu is
doing kappa_p's job under a new name, and ``DEPTH_IS_LEGIBILITY`` is the verdict. That branch
was written before the run and is the reason the welcome answer is not the only reachable one.

-----------------------------------------------------------------------------------------
TWO AXES.

* the DEPTH grid, mu in {1, 2, 3}: how many levels of the hierarchy reach the surface.
* the CONTRAST grid, delta in {0, ..., 1} at mu = 3: how distinct the execution modes are.
  delta is the one free knob in the depth construction, so rather than fix it and hope, E30
  sweeps it. At delta = 0 every mode IS sig[g] and a mu = 3 artifact is indistinguishable
  from a mu = 1 one, which makes the bottom of this axis a built-in negative control: the
  observer must fail to find depth there, and if it finds some anyway the recovery is an
  artifact of the hypothesis space rather than a reading of the content.

TWO ATTENTION ARMS, and the contrast between them is a result rather than a robustness check.

* ``sustained`` — DEEP for the whole artifact. THE PRIMARY ARM, and the pre-registered
  criteria are scored on it, because N21 (E30's gate) ran under sustained attention and a gate
  measured under one condition does not transfer to another.
* ``free`` — E28's design: forced DEEP for 10 steps, then the observer chooses. Depth lives in
  the ORDER of execution modes, so a reader who stops after ten steps has seen about one and a
  half blocks of three and a half. Whether depth survives a reader who is free to stop is a
  question about attention, not about the construct, and it is reported separately.

Everything is UNSIGNED, for E28's and E19's reason: with a provenance tag present the observer
could read the tier off the label, and E30 asks what CONTENT ALONE reveals about depth. The
label-to-depth evidence path is C2's, and E31 manipulates it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v5 as P5
from ..config import Config
from ..environment import Artifact
from ..n21_depth_not_effort import merged_config
from ..observer import rollout_observer
from ..v5_model import (MU_LEVELS, HierarchicalCreator, V5Environment, build_v5_world,
                        as_mgs, load_v5_config, make_v5_observer, marginal_goal, marginal_mu,
                        marginal_subgoal, recovered_mu)
from . import _common as C

ARMS = ("sustained", "free")
PRIMARY_ARM = "sustained"


def _e30_worker(payload):
    (cfg_raw, cell_index, arm, true_mu, delta, seed_rep, base_seed, n_obs, n_timesteps,
     forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    # The CONTRAST sweep deliberately drives mode separation below its floor — that end of the
    # axis is the negative control. So the floor is enforced on the full-contrast cells and
    # skipped on the sweep, explicitly, and every OTHER assertion still runs on every cell.
    # The default is read BEFORE the override, or the comparison is delta against itself.
    delta_default = float(cfg.get("v5.depth.delta", 1.0))
    enforce = float(delta) >= delta_default - 1e-12
    cfg.set("v5.depth.delta", float(delta))
    world = build_v5_world(cfg, enforce_mode_separation=enforce)
    n_mu, n_sub, ng = len(MU_LEVELS), world.n_subgoals, FN.NUM_REAL_GOALS
    cfg_rollout = merged_config(cfg, n_mu, n_sub)

    k = n_timesteps if arm == "sustained" else forced_k

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    true_goal = int(art_rng.integers(ng))
    creator = HierarchicalCreator(world, true_goal, int(true_mu), 1.0,
                                  n_positions=n_timesteps + 2, rng=art_rng)
    artifact = Artifact(provenance=K.CREATOR, goal=true_goal, declared_signal=K.UNSIGNED)
    env = V5Environment(cfg_rollout, world.gm,
                        np.random.default_rng(base_seed + cell_index),
                        honesty=1.0, signing_rate=0.0, creator=creator)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        creator.reset()
        agent = make_v5_observer(world, rng)
        res = rollout_observer(agent, artifact, env, cfg_rollout, rng,
                               n_timesteps=n_timesteps, force_deep_k=k,
                               initial_glance=True, early_stop=False)

        q = np.asarray(res.final_goal_posterior, dtype=float)
        g_post = marginal_goal(q, n_mu, ng, n_sub)
        g_prior = marginal_goal(np.asarray(res.goal_prior, dtype=float), n_mu, ng, n_sub)
        free = np.asarray(res.attention)[k:]
        engaged = float(np.mean(free == K.DEEP)) if free.size else float("nan")

        # UPDATE MAGNITUDE, with the engagement gate held OPEN, exactly as E28 did and for
        # E28's reason: gating psi on the free-phase decision would mix "how much did it
        # learn" with "how long did it keep looking", and those are reported as two columns
        # here precisely so they can be read apart. theta is open too; closing it is E31's.
        psi = metrics.psi_analogue(g_post, g_prior, float(cfg.signal_model.kappa),
                                   engaged=True)

        # SECONDARY, ADDED AFTER THE FIRST RUN, DECIDES NOTHING. V5 deviation 2; see
        # RESULTS_V5.md and build_verdict.
        #
        # psi_analogue is KL(goal posterior || goal prior), and depth is CONSTRUCTED so that
        # the goal is exactly as recoverable at every depth. The measure therefore has no
        # headroom here by design, and reporting only it would answer "does depth move the
        # reader" with a quantity that cannot vary with depth whatever the answer is.
        #
        # This is the same object one level up: how far the observer's belief about the
        # creator's whole intent STRUCTURE moved — goal and execution mode jointly, with mu
        # marginalised out so the manipulation is not scored as its own effect.
        #
        # ITS LIMITATION, STATED HERE RATHER THAN LEFT FOR A READER TO FIND. Part of what it
        # credits is the observer knowing which mode is active AT THE END, which is transient
        # state rather than durable uptake. It is a lower bound on nothing and an upper bound
        # on nothing; it is a second view, and the pre-registered measure still decides.
        gs_post = as_mgs(q, n_mu, ng, n_sub).sum(axis=0).reshape(-1)
        gs_prior = as_mgs(np.asarray(res.goal_prior, dtype=float),
                          n_mu, ng, n_sub).sum(axis=0).reshape(-1)
        psi_structure = metrics.psi_analogue(gs_post, gs_prior,
                                             float(cfg.signal_model.kappa), engaged=True)

        recs.append({
            "arm": arm, "true_mu": int(true_mu), "delta": float(delta),
            "seed_rep": seed_rep, "observer": i, "true_goal": true_goal,
            "recovered_mu": recovered_mu(q, n_mu, ng, n_sub),
            "mu_posterior_entropy": float(metrics.shannon_entropy(
                marginal_mu(q, n_mu, ng, n_sub))),
            "subgoal_entropy": float(metrics.shannon_entropy(
                marginal_subgoal(q, n_mu, ng, n_sub))),
            "correct": int(int(np.argmax(g_post)) == true_goal),
            "final_entropy": float(metrics.within_observer_entropy(g_post)),
            "psi_analogue": float(psi),
            "psi_structure": float(psi_structure),
            "engaged_fraction": engaged,
            "cum_deep": int(res.cum_deep),
            "separation_enforced": bool(enforce),
            "posterior": g_post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg_path = res_dir / "v5_preregistration.json"
    P5.write_preregistration_v5(cfg, prereg_path)
    P5.assert_prereg_locked_v5(prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e30.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e30.n_seeds", 20))
    forced_k = int(cfg.get("experiments.e30.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e30.n_timesteps", 24))
    delta_default = float(cfg.get("v5.depth.delta", 1.0))

    # The depth grid at full contrast, plus the contrast grid at full depth. The shared cell
    # (mu = 3, delta = default) is run once and belongs to both.
    cells = [(mu, delta_default) for mu in P5.E30_MU_GRID]
    cells += [(3, d) for d in P5.E30_DELTA_GRID if abs(d - delta_default) > 1e-12]

    payloads, ci = [], 0
    for arm in ARMS:
        for mu, delta in cells:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, arm, mu, delta, s, base_seed, n_obs,
                                 n_timesteps, forced_k))
            ci += 1
    recs = C.run_parallel(payloads, _e30_worker, workers)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "posterior"} for r in recs])
    df.to_csv(res_dir / "e30_points.csv", index=False)

    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e30_cell_stats.csv", index=False)

    verdict = build_verdict(stats, delta_default)
    (res_dir / "e30_verdict.json").write_text(json.dumps(verdict, indent=2, default=float),
                                              encoding="utf-8")
    if make_fig:
        make_figure(stats, verdict, delta_default, fig_dir / "e30_depth.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for (arm, mu, delta), sub in df.groupby(["arm", "true_mu", "delta"]):
        betweens = []
        for _, g in sub.groupby("seed_rep"):
            betweens.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["posterior"]]))
        rows.append({
            "arm": arm, "true_mu": int(mu), "delta": float(delta), "n": int(len(sub)),
            "recovered_mu": float(sub.recovered_mu.mean()),
            "recovered_mu_sd": float(sub.recovered_mu.std()),
            "mu_posterior_entropy": float(sub.mu_posterior_entropy.mean()),
            "subgoal_entropy": float(sub.subgoal_entropy.mean()),
            "accuracy": float(sub.correct.mean()),
            "psi_analogue": float(sub.psi_analogue.mean()),
            "psi_analogue_sd": float(sub.psi_analogue.std()),
            "psi_structure": float(sub.psi_structure.mean()),
            "psi_structure_sd": float(sub.psi_structure.std()),
            "final_entropy": float(sub.final_entropy.mean()),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "cum_deep": float(sub.cum_deep.mean()),
            "between_observer": float(np.mean(betweens)),
        })
    return pd.DataFrame(rows).sort_values(["arm", "true_mu", "delta"]).reset_index(drop=True)


def build_verdict(stats: pd.DataFrame, delta_default: float) -> dict:
    depth = stats[(stats.arm == PRIMARY_ARM)
                  & (np.abs(stats.delta - delta_default) < 1e-12)].sort_values("true_mu")
    verdict = P5.e30_verdict(
        mu_grid=depth.true_mu.tolist(),
        recovered_mu=depth.recovered_mu.tolist(),
        accuracy=depth.accuracy.tolist(),
        update_magnitude=depth.psi_analogue.tolist())
    verdict["experiment"] = "E30 — mu, estimated model depth, replaces beta"
    verdict["primary_arm"] = PRIMARY_ARM
    verdict["design_note"] = (
        "A NEW DESIGN, not a re-run of E28 (V5 decision 8). Under a hierarchical creator "
        "there is no version of E28's stimulus to re-run; the two are reported side by side "
        "as two constructs on comparable designs and E28 is not deleted.")

    # THE NEGATIVE CONTROL. At delta = 0 every execution mode is sig[g], so a mu = 3 artifact
    # is identical to a mu = 1 one and there is no depth to find. Recovery here measures what
    # the hypothesis space produces on its own.
    contrast = stats[(stats.arm == PRIMARY_ARM) & (stats.true_mu == 3)].sort_values("delta")
    zero = contrast[contrast.delta < 1e-12]
    full = contrast[np.abs(contrast.delta - delta_default) < 1e-12]
    verdict["negative_control_delta_zero"] = {
        "recovered_mu_at_delta_0": float(zero.recovered_mu.iloc[0]) if len(zero) else None,
        "recovered_mu_at_full_delta": float(full.recovered_mu.iloc[0]) if len(full) else None,
        "contrast_rho": P5.spearman(contrast.delta.tolist(), contrast.recovered_mu.tolist()),
        "note": ("at delta = 0 the mode family collapses onto sig[g], so mu = 3 content IS "
                 "mu = 1 content and any recovered depth is an artifact of the hypothesis "
                 "space rather than a reading of the artifact."),
    }

    # The attention contrast, reported as a result rather than as a robustness check.
    free = stats[(stats.arm == "free")
                 & (np.abs(stats.delta - delta_default) < 1e-12)].sort_values("true_mu")
    verdict["attention_arms"] = {
        "sustained_recovered_mu": depth.recovered_mu.tolist(),
        "free_recovered_mu": free.recovered_mu.tolist(),
        "free_engaged_fraction": free.engaged_fraction.tolist(),
        "free_recovery_rho": P5.spearman(free.true_mu.tolist(), free.recovered_mu.tolist()),
        "note": ("depth lives in the ORDER of execution modes, so a reader free to stop after "
                 "ten steps has seen about one and a half blocks of three and a half. This "
                 "contrast is a question about attention, not about the construct."),
    }
    # V5 DEVIATION 2 — A SECOND UPDATE MEASURE, ADDED AFTER THE RUN, DECIDING NOTHING.
    #
    # The pre-registered measure is psi_analogue, KL(goal posterior || goal prior), and it
    # still decides ``verdict``. The sweep exposed a headroom problem in it that the design
    # itself creates: depth is CONSTRUCTED so the goal is exactly as recoverable at every
    # depth — that is the whole content of "depth is correlational, not marginal" — so the
    # goal-KL cannot vary with depth whatever is true of the reader. Reporting only it would
    # answer "does depth move the reader" with a quantity incapable of answering.
    #
    # psi_structure is the same object one level up: how far the observer's belief about the
    # creator's whole intent structure moved, goal and execution mode jointly, with mu
    # marginalised out so the manipulation is not scored as its own effect.
    #
    # THIS CANNOT MANUFACTURE THE WELCOME ANSWER AND THAT WAS CHECKED BEFORE ADDING IT: at
    # delta = 0 the negative-control cell must show no structural update either, because there
    # is no structure to recover there. If psi_structure rises with depth at delta = 0 it is
    # counting the hypothesis space rather than the artifact, and the flag below says so.
    depth_struct = depth.psi_structure.tolist()
    zero_struct = float(zero.psi_structure.iloc[0]) if len(zero) else float("nan")
    full_struct = float(full.psi_structure.iloc[0]) if len(full) else float("nan")
    verdict["secondary_update_measure"] = {
        "deviation": "V5-2, added after the run; the pre-registered measure still decides",
        "psi_structure_by_depth": depth_struct,
        "structure_rho": P5.spearman(depth.true_mu.tolist(), depth_struct),
        "psi_analogue_by_depth": depth.psi_analogue.tolist(),
        "analogue_rho": verdict["update_rho"],
        "negative_control_psi_structure_at_delta_0": zero_struct,
        "psi_structure_at_full_delta": full_struct,
        "control_is_clean": bool(np.isfinite(zero_struct) and np.isfinite(full_struct)
                                 and zero_struct < full_struct),
        "why": ("psi_analogue has no headroom here BY DESIGN: depth is constructed so the "
                "goal is exactly as recoverable at every depth, so the goal-KL cannot vary "
                "with depth whatever is true of the reader. This measures whether the reader "
                "took on more of the creator's STRUCTURE, which is what C1 says depth "
                "supplies."),
        "limitation": ("part of what this credits is knowing which execution mode is active "
                       "at the END, which is transient state rather than durable uptake. It "
                       "is a second view, not a better one."),
    }

    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, delta_default: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    depth = stats[(stats.arm == PRIMARY_ARM)
                  & (np.abs(stats.delta - delta_default) < 1e-12)].sort_values("true_mu")
    free = stats[(stats.arm == "free")
                 & (np.abs(stats.delta - delta_default) < 1e-12)].sort_values("true_mu")
    contrast = stats[(stats.arm == PRIMARY_ARM) & (stats.true_mu == 3)].sort_values("delta")

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.8))

    ax = axes[0]
    ax.errorbar(depth.true_mu, depth.recovered_mu, yerr=depth.recovered_mu_sd,
                marker="o", capsize=3, label="reader who keeps looking")
    ax.errorbar(free.true_mu, free.recovered_mu, yerr=free.recovered_mu_sd,
                marker="s", capsize=3, ls="--", color="C7", label="reader free to stop")
    ax.set(xlabel="how many layers of thought went in", xticks=[1, 2, 3],
           ylabel="how much the reader thinks went in",
           title="Can a reader tell how much\nthinking is behind it?")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(depth.true_mu, depth.accuracy, marker="o", color="C2")
    ax.axhline(0.25, ls="--", color="firebrick", lw=1, label="guessing")
    ax.axhline(P5.E30_ACCURACY_FLOOR, ls=":", color="k", lw=1, label="counts as 'reads it'")
    ax.set(xlabel="how many layers of thought went in", xticks=[1, 2, 3], ylim=(0, 1.05),
           ylabel="share who got the purpose right",
           title="Shallow work is just as readable\n(that is the design)")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.errorbar(depth.true_mu, depth.psi_analogue, yerr=depth.psi_analogue_sd,
                marker="o", color="C3", capsize=3)
    ax.set(xlabel="how many layers of thought went in", xticks=[1, 2, 3],
           ylabel="how much the reader changes their mind",
           title="Does depth move the reader?")

    ax = axes[3]
    ax.errorbar(contrast.delta, contrast.recovered_mu, yerr=contrast.recovered_mu_sd,
                marker="o", color="C4", capsize=3)
    ax.axvline(0.0, ls=":", color="k", lw=1)
    ax.set(xlabel="how distinct the stages are", ylabel="how much the reader thinks went in",
           title="Negative control: at zero,\nthere is no depth to find")

    fig.suptitle(f"E30 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E30 — mu, estimated model depth, replaces beta")
    args = ap.parse_args()
    cfg = load_v5_config(quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
