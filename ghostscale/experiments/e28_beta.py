"""E28 — beta as inferred rationality (V4.5 §4).

THE CONCEPTUAL CHANGE. MaxEnt IRL's demonstrator likelihood is properly
P(tau|R) ~ exp(beta * R(tau)). The preprint's Appendix A.1 writes it with beta = 1, which
models the creator as perfectly rational, always. That elision is the framework's missing
dimension: it has a values axis (theta) and a legibility axis (kappa_p) and no COMPETENCE
axis at all, while the canonical trust literature it cites — Mayer/Davis/Schoorman, Lee &
See — puts competence first.

So beta stops being a constant and becomes a hidden quantity the observer INFERS: how hard
was the creator optimising, alongside what were they optimising for. It is a fourth hidden
state factor with a uniform prior, not a swept parameter, because the claim is about what an
observer can recover from a trajectory rather than about what an experimenter can set.

-----------------------------------------------------------------------------------------
PREDICTED (V4.5 §4). Goal accuracy stays high across the beta range while update magnitude
falls with beta. That is the missing aesthetic category and it does not currently have a name
in the framework: *legible, competent, and unmoving*. The observer reads the intent
correctly — it can see what you were going for — and correctly treats it as weak evidence
about what you actually value, because you were not trying very hard. Neither existing gate
can express that. kappa_p would take the legibility away with the update; theta would leave
the update intact right up until the recovered goal became unacceptable.

FALSIFIED (V4.5 §4). If update magnitude does not fall with beta while accuracy holds, beta
is not doing separable work and collapses into kappa_p or theta.

A THIRD OUTCOME, named in the pre-registration before the run because it is the one a
careless read would report as the prediction: if accuracy falls WITH the update, beta is
doing something, but not the predicted thing. Content generated at low beta genuinely carries
less goal information, so an observer that fails to read it has demonstrated nothing kappa_p
could not explain. The predicted result requires the DISSOCIATION.

At beta = 0 exactly, the content is the creator's policy marginal and carries literally zero
goal information, so no observer can be accurate there and the grid's bottom point cannot
satisfy the accuracy clause by construction. That is not a flaw in the design; it is the
boundary where beta and kappa_p become the same condition empirically, and locating it is
part of what E28 measures. The pre-registration's
BETA_IS_SEPARABLE_OVER_PART_OF_THE_RANGE outcome exists for it.

-----------------------------------------------------------------------------------------
THE REQUIRED CONSISTENCY CHECK (V4.5 §3.3). beta = 0 and EXPLORE are asserted to be the same
axis: EXPLORE is beta = 0 as a discrete hypothesis, beta is its continuous generalisation. So
at beta = 0, E28 must reproduce E19's exploratory-human cell. If it does not, the two are not
the same axis, and the implementation shortcut §3.3 recommends is wrong.

Compared against E19's ``human_exploratory / explore_on`` real-goal entropy, read from
``results/e19_cell_stats.csv`` rather than hardcoded. That cell is the DISCRETE-EXPLORE
observer's residual posterior over the four real goals, which is the object a continuous
beta = 0 has to reproduce. The comparison is like for like: both are four-real-goal posterior
entropies on human content produced by an exploratory policy.

Everything is UNSIGNED, for E19's reason: with a provenance tag present the observer could
read the tier off the label, and E28 is a question about what content alone reveals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import prereg_v4_5 as P45
from ..config import Config
from ..creators import HumanCreator
from ..environment import Artifact, Environment
from ..observer import rollout_observer
from ..v4_5_model import (F_BETA, build_v45_world, demonstrator_signature, load_v4_5_config,
                          make_v45_observer, recovered_beta)
from . import _common as C

BETA_GRID = P45.E28_BETA_GRID


def _e28_worker(payload):
    (cfg_raw, cell_index, true_beta, seed_rep, base_seed, n_obs, n_timesteps,
     forced_k) = payload
    from ..config import Config as _Config

    cfg = _Config(cfg_raw).copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    world = build_v45_world(cfg)

    # THE WORLD GENERATES AT true_beta, using the SAME function the observer's likelihood is
    # built from. Sharing it is deliberate: an observer whose demonstrator model and whose
    # world disagreed about what beta means would make the recovery measurement meaningless,
    # and the disagreement would be invisible in the output.
    world_sig = demonstrator_signature(world.sigs.sig_true, world.sigs.sig_explore, true_beta)
    creators = {g: HumanCreator(cfg, world_sig, g) for g in range(FN.NUM_REAL_GOALS)}
    env = Environment(cfg, world.gm, np.random.default_rng(base_seed + cell_index),
                      honesty=1.0, signing_rate=0.0, creator_bank=creators)

    art_rng = np.random.default_rng(C.observer_seed(base_seed, cell_index, seed_rep, 9_999))
    true_goal = int(art_rng.integers(FN.NUM_REAL_GOALS))
    artifact = Artifact(provenance=K.CREATOR, goal=true_goal, declared_signal=K.UNSIGNED)

    recs = []
    for i in range(n_obs):
        rng = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_v45_observer(world, rng)
        res = rollout_observer(agent, artifact, env, cfg, rng, n_timesteps=n_timesteps,
                               force_deep_k=forced_k, initial_glance=True,
                               extra_factors=(F_BETA,))

        post = np.asarray(res.final_goal_posterior, dtype=float)
        beta_post = np.asarray(res.final_extra_posterior[F_BETA], dtype=float)
        free = np.asarray(res.attention)[forced_k:]
        engaged = float(np.mean(free == K.DEEP)) if free.size else float("nan")

        # UPDATE MAGNITUDE. The engagement gate is held OPEN here on purpose: every condition
        # gets the same forced-DEEP inference attempt, so gating psi on the free-phase
        # engagement decision would mix "how much did it learn" with "how long did it keep
        # looking", and E28 reports those as two separate columns precisely so they can be
        # read apart. theta is open too (lam = 0); closing it is E29's manipulation.
        psi = metrics.psi_analogue(post, res.goal_prior, float(cfg.signal_model.kappa),
                                   engaged=True)

        recs.append({
            "true_beta": float(true_beta), "seed_rep": seed_rep, "observer": i,
            "true_goal": true_goal,
            "recovered_beta": recovered_beta(beta_post, world.beta_levels),
            "beta_posterior_entropy": float(metrics.shannon_entropy(beta_post)),
            "modal_beta_level": int(np.argmax(beta_post)),
            "modal_goal": int(np.argmax(post)),
            "correct": int(int(np.argmax(post)) == true_goal),
            "final_entropy": float(metrics.within_observer_entropy(post)),
            "psi_analogue": float(psi),
            "engaged_fraction": engaged,
            "cum_deep": int(res.cum_deep),
            "posterior": post.tolist(),
            "beta_posterior": beta_post.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    prereg_path = res_dir / "v4_5_preregistration.json"
    P45.write_preregistration_v4_5(cfg, prereg_path)
    P45.assert_prereg_locked_v4_5(prereg_path)

    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e28.n_observers", 200))
    n_seeds = int(cfg.get("experiments.e28.n_seeds", 20))
    forced_k = int(cfg.get("experiments.e28.forced_deep_k", 10))
    n_timesteps = int(cfg.get("experiments.e28.n_timesteps", 24))

    payloads, ci = [], 0
    for b in BETA_GRID:
        for s in range(n_seeds):
            payloads.append((cfg.raw, ci, float(b), s, base_seed, n_obs, n_timesteps,
                             forced_k))
        ci += 1
    recs = C.run_parallel(payloads, _e28_worker, workers)

    drop = {"posterior", "beta_posterior"}
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in drop} for r in recs])
    df.to_csv(res_dir / "e28_points.csv", index=False)

    stats = cell_stats(recs)
    stats.to_csv(res_dir / "e28_beta_stats.csv", index=False)

    verdict = build_verdict(stats, res_dir, world=build_v45_world(cfg))
    (res_dir / "e28_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_figure(stats, verdict, fig_dir / "e28_beta.png")
    return stats


def cell_stats(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    rows = []
    for b, sub in df.groupby("true_beta"):
        betweens = []
        for _, g in sub.groupby("seed_rep"):
            betweens.append(metrics.between_observer_entropy(
                [np.asarray(p, dtype=float) for p in g["posterior"]]))
        bp = np.asarray([np.asarray(x, dtype=float) for x in sub["beta_posterior"]])
        rows.append({
            "true_beta": float(b), "n": int(len(sub)),
            # Mass on the GOAL-AGNOSTIC hypothesis: the level beta = 0, which is the
            # continuous representation of V4's discrete EXPLORE. Directly comparable to
            # E19's ``explore_mass`` column, and that comparison is what V4.5 §3.3's
            # identification actually claims.
            "mass_on_beta_zero": float(bp[:, 0].mean()),
            "mass_on_beta_one": float(bp[:, -1].mean()),
            "modal_beta_is_zero": float(np.mean(bp.argmax(axis=1) == 0)),
            "modal_beta_is_one": float(np.mean(bp.argmax(axis=1) == bp.shape[1] - 1)),
            "recovered_beta": float(sub.recovered_beta.mean()),
            "recovered_beta_sd": float(sub.recovered_beta.std()),
            "beta_posterior_entropy": float(sub.beta_posterior_entropy.mean()),
            "accuracy": float(sub.correct.mean()),
            "psi_analogue": float(sub.psi_analogue.mean()),
            "psi_analogue_sd": float(sub.psi_analogue.std()),
            "final_entropy": float(sub.final_entropy.mean()),
            "final_entropy_sd": float(sub.final_entropy.std()),
            "engaged_fraction": float(sub.engaged_fraction.mean()),
            "between_observer": float(np.mean(betweens)),
        })
    return pd.DataFrame(rows).sort_values("true_beta").reset_index(drop=True)


def _e19_exploratory(res_dir: Path) -> dict:
    """E19's discrete-EXPLORE observer on exploratory human content.

    Read from the CSV rather than hardcoded so the consistency check tracks E19 if E19 is
    ever re-run, instead of comparing against a number that has quietly moved.
    """
    path = Path(res_dir) / "e19_cell_stats.csv"
    if not path.exists():
        return {"real_goal_entropy": float("nan"), "explore_mass": float("nan"),
                "modal_is_explore": float("nan")}
    df = pd.read_csv(path)
    sub = df[(df.content == "human_exploratory") & (df.arm == "explore_on")]
    if not len(sub):
        return {"real_goal_entropy": float("nan"), "explore_mass": float("nan"),
                "modal_is_explore": float("nan")}
    r = sub.iloc[0]
    return {"real_goal_entropy": float(r.real_goal_entropy),
            "explore_mass": float(r.explore_mass),
            "modal_is_explore": float(r.modal_is_explore)}


def beta_marginal_invariance(world) -> dict:
    """Is the goal-MARGINAL likelihood the same at every beta?

    It is, exactly, and it is the reason the beta = 0 end of the grid behaves the way it does.
    Under a flat goal posterior,

        mean_g [ b * sig[g] + (1 - b) * sig_EXPLORE ]
          = b * mean_g(sig[g]) + (1 - b) * sig_EXPLORE
          = b * sig_EXPLORE + (1 - b) * sig_EXPLORE
          = sig_EXPLORE                                   for EVERY b,

    because ``sig_EXPLORE`` IS ``mean_g(sig[g])`` by C2's construction. So beta carries no
    information at all in the goal marginal: every bit of evidence about it comes from the
    JOINT (beta, goal) coupling. Measured rather than asserted, because it is the mechanism
    the §3.3 consistency-check result has to be read against.
    """
    from ..v4_5_model import demonstrator_signature as _demo
    base = None
    worst = 0.0
    for b in world.beta_levels:
        marg = _demo(world.sigs.sig_true, world.sigs.sig_explore, float(b)).mean(axis=0)
        if base is None:
            base = marg
        worst = max(worst, float(np.max(np.abs(marg - base))))
    return {
        "max_abs_deviation_of_goal_marginal_across_beta": worst,
        "consequence": (
            "beta is unidentifiable in the goal marginal by construction, because "
            "sig_EXPLORE is mean_g(sig[g]). All evidence about beta comes from the joint "
            "(beta, goal) coupling, which is exactly what a finite trajectory biases: some "
            "goal always looks slightly favoured by chance, and a higher beta paired with "
            "that lucky goal explains the sample better than beta = 0 does."),
    }


def build_verdict(stats: pd.DataFrame, res_dir: Path, world=None) -> dict:
    e19 = _e19_exploratory(res_dir)
    beta0 = stats[stats.true_beta == 0.0]
    beta0_entropy = float(beta0.final_entropy.iloc[0]) if len(beta0) else float("nan")

    verdict = P45.e28_verdict(
        beta_grid=stats.true_beta.tolist(),
        recovered_beta=stats.recovered_beta.tolist(),
        accuracy=stats.accuracy.tolist(),
        update_magnitude=stats.psi_analogue.tolist(),
        beta0_entropy=beta0_entropy,
        e19_exploratory_entropy=e19["real_goal_entropy"])
    verdict["experiment"] = "E28 — beta as inferred rationality"
    verdict["recovered_beta_at_zero"] = (float(beta0.recovered_beta.iloc[0])
                                         if len(beta0) else float("nan"))

    # THE §3.3 CHECK IN ITS SUBSTANTIVE FORM, ADDED AFTER THE RUN AND LOGGED AS A DEVIATION.
    #
    # The pre-registered comparison is entropy-against-entropy, and it is a MISMATCH: E19's
    # ``real_goal_entropy`` is the entropy of an arbitrary residual left after EXPLORE is
    # marginalised out (V4 decision D4 says so in as many words), while E28's is a genuine
    # goal posterior. Two different objects.
    #
    # What §3.3 actually claims is that beta = 0 and EXPLORE are the same hypothesis. The
    # like-for-like quantity is therefore the belief the observer puts on the GOAL-AGNOSTIC
    # hypothesis: E19's ``explore_mass`` against E28's mass on the beta = 0 level. Both are
    # "how much did the reader believe the maker was not aiming at anything in particular",
    # on the same content, under the two representations being identified.
    #
    # THIS ADDITION CANNOT MANUFACTURE THE WELCOME ANSWER AND THAT WAS CHECKED BEFORE MAKING
    # IT: the check fails under BOTH operationalisations. The pre-registered criterion is
    # left in place and still decides ``beta0_recovers_e19_explore_cell``; this one is
    # reported beside it and changes the REASON, not the outcome.
    mass0 = float(beta0.mass_on_beta_zero.iloc[0]) if len(beta0) else float("nan")
    verdict["beta0_check_substantive_form"] = {
        "e19_explore_mass": e19["explore_mass"],
        "e19_modal_is_explore": e19["modal_is_explore"],
        "e28_mass_on_beta_zero": mass0,
        "e28_modal_beta_is_zero": (float(beta0.modal_beta_is_zero.iloc[0])
                                   if len(beta0) else float("nan")),
        "flat_baseline": 1.0 / len(stats) if len(stats) else float("nan"),
        "recovers": bool(np.isfinite(mass0) and np.isfinite(e19["explore_mass"])
                         and mass0 >= 0.5 * e19["explore_mass"]),
        "note": (
            "added after the run; see the deviation logged in RESULTS_V4_5.md. The "
            "pre-registered entropy comparison is retained and still decides the flag. The "
            "check fails under both forms, so this changes the reason reported and not the "
            "outcome."),
    }
    if world is not None:
        verdict["beta_marginal_invariance"] = beta_marginal_invariance(world)
    verdict["cell_table"] = stats.to_dict(orient="records")
    return verdict


def make_figure(stats: pd.DataFrame, verdict: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..figures import set_style
    set_style()

    b = stats.true_beta.values
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    ax = axes[0]
    ax.errorbar(b, stats.recovered_beta, yerr=stats.recovered_beta_sd, marker="o", capsize=3)
    ax.plot([0, 1], [0, 1], ls=":", color="k", lw=1, label="perfect recovery")
    ax.set(xlabel="how hard the maker was actually trying",
           ylabel="how hard the reader thinks they were trying",
           title="Can a reader tell effort\nfrom the work alone?")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(b, stats.accuracy, marker="o", color="C2", label="reads the purpose right")
    ax.axhline(0.25, ls="--", color="firebrick", lw=1, label="guessing")
    ax.axhline(P45.E28_ACCURACY_FLOOR, ls=":", color="k", lw=1, label="counts as 'reads it'")
    ax.set(xlabel="how hard the maker was actually trying", ylim=(0, 1.05),
           ylabel="share of readers who got the purpose right",
           title="Legibility")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.errorbar(b, stats.psi_analogue, yerr=stats.psi_analogue_sd, marker="o", color="C3")
    ax.set(xlabel="how hard the maker was actually trying",
           ylabel="how much the reader changes their mind",
           title="Does the reader take it to heart?")

    fig.suptitle(f"E28 — {verdict['verdict'].replace('_', ' ').lower()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E28 — beta as inferred rationality")
    args = ap.parse_args()
    cfg = load_v4_5_config(quick=args.quick)
    if args.seed is not None:
        cfg.set("run.base_seed", int(args.seed))
    out = Path(args.out) if args.out else None
    stats = run(cfg, out_dir=out, workers=args.workers or C.default_workers())
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
