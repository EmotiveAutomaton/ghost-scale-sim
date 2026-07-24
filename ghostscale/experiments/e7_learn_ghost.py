"""E7 — Can the GHOST column be learned without labels? (V2 spec §2). STAGE 3.

    "This is the experiment that restores the strong claim. V1 showed the signal is
     metabolically useful to someone who already knows. E7 asks whether you can come to know
     without it. State the result either way."

V1's aggregator arrived holding a correct A[0] GHOST column — it already knew what goalless
output looks like, so "does the provenance label add anything?" had been answered before the
experiment began. The Learner (C3/D1) does not have that column. It knows the shared
goal->feature family and must discover, from an unlabelled contaminated corpus, that some
sources are hollow.

Design: learner observers x signing_rate {0.0, 0.5, 1.0} x kappa, biased synthetic content,
contamination f.

Measures
  * ghost_col_err   — KL(learned A[0][:,GHOST,:,DEEP] || true synth), over exposure
  * creator_mi      — MI(features; goal) of the learned HUMAN columns
  * time_to_competence — first exposure checkpoint at which the learned GHOST column comes
                      within ``competence_kl`` nats of the true synthetic distribution
                      (never => n_artifacts)

NOTE ON time_to_competence, and an honest consequence of decision D1. The natural definition
— time until the learner's HUMAN columns reach a fraction of the oracle's MI — is degenerate
here, and measured as such: the D1-seeded learner starts at ~95% of oracle MI by construction,
so that clock reads zero in every condition. It measures the seeding, not the learning. What
the learner genuinely does not have at t=0 is the GHOST column, so competence is timed on
that instead. The human-column MI is still reported, as a DEGRADATION measure (does it fall?)
rather than an acquisition one.
"""

  PREDICTED  without labels the learner folds synthetic features into its model of human
             intent (human columns acquire synthetic mass and lose goal-discriminability),
             and the degradation is worse at high kappa. With honest labels it acquires a
             clean GHOST column quickly and the human columns stay sharp.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import CREATOR, GHOST, DEEP
from ..generative_model import (build_shared_model as _build_model, assert_preferences_zero,
                                build_observer_model, build_D)
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import rollout_observer, observer_sig_rng
from ..preregistration import POP_GOAL_DIST, allocate_creator_goals
from .. import learning as L, metrics
from ..figures import set_style
from . import _common as C


def _e7_worker(payload):
    (cfg_raw, cell_index, kappa, signing_rate, f, seed_rep, base_seed,
     n_creators, n_artifacts, n_observers, infer_steps, synth_seed, checkpoints) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                    # N7
    bank = build_creator_bank(cfg, gm)
    oracle_mi = metrics.mutual_information_features_goal(np.asarray(gm.A[0]), CREATOR, DEEP)
    competence_kl = float(cfg.get("experiments.e7.competence_kl", 1.0))

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)

    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=signing_rate,
                      creator_bank=bank)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals,
                             rng=world)

    recs = []
    for i in range(n_observers):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        om = build_observer_model(gm, cfg, 0.0)
        agent = L.make_learner_agent(om, build_D(cfg, r), cfg, kappa=kappa)
        t_competent = None
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + i * 7919 + j)
            rollout_observer(agent, art, env, cfg, rr, infer_steps,
                             force_deep_k=infer_steps, kappa=kappa,
                             early_stop=False, learn=True)
            n = j + 1
            if n in checkpoints:
                mi = L.human_column_mi(agent.A[0], CREATOR)
                gerr = L.ghost_column_error(agent.A[0], gm.noise_free_synth)
                recs.append({
                    "kappa": kappa, "signing_rate": signing_rate, "contamination": f,
                    "seed_rep": seed_rep, "observer": i, "exposure": n,
                    "ghost_col_err": gerr,
                    "creator_mi": mi, "oracle_mi": oracle_mi,
                    "mi_ratio": mi / oracle_mi if oracle_mi > 0 else np.nan,
                    "ghost_col_entropy": float(np.mean(
                        [L.column_entropy(agent.A[0], GHOST, g) for g in range(num_goals)])),
                })
                if t_competent is None and gerr <= competence_kl:
                    t_competent = n
        recs.append({
            "kappa": kappa, "signing_rate": signing_rate, "contamination": f,
            "seed_rep": seed_rep, "observer": i, "exposure": -1,   # -1 = final summary row
            "ghost_col_err": L.ghost_column_error(agent.A[0], gm.noise_free_synth),
            "creator_mi": L.human_column_mi(agent.A[0], CREATOR), "oracle_mi": oracle_mi,
            "mi_ratio": L.human_column_mi(agent.A[0], CREATOR) / oracle_mi,
            "ghost_col_entropy": float(np.mean(
                [L.column_entropy(agent.A[0], GHOST, g) for g in range(num_goals)])),
            "time_to_competence": t_competent if t_competent is not None else n_artifacts,
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e7
    n_creators = int(cfg.get("experiments.e7.n_creators", 50))
    n_artifacts = int(e.n_artifacts)
    n_observers = int(e.n_observers)
    infer_steps = int(cfg.get("experiments.e7.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e7.n_replications", 3))
    f_levels = list(cfg.get("experiments.e7.contamination_levels", [0.3, 0.6]))
    kappas = list(cfg.get("experiments.e7.kappa_levels", [0.1, 0.9]))
    signing = list(cfg.get("experiments.e7.signing_rate_levels", [0.0, 0.5, 1.0]))
    synth_seed = int(cfg.get("experiments.e7.synth_draw_seed", 1))
    n_ck = int(cfg.get("experiments.e7.n_checkpoints", 6))
    checkpoints = set(np.unique(np.linspace(n_artifacts / n_ck, n_artifacts, n_ck).astype(int)))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for kap in kappas:
        for sr in signing:
            for f in f_levels:
                for s in range(n_reps):
                    payloads.append((cfg.raw, ci, kap, sr, f, s, base_seed, n_creators,
                                     n_artifacts, n_observers, infer_steps, synth_seed,
                                     checkpoints))
                ci += 1
    recs = C.run_parallel(payloads, _e7_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e7_raw.csv", index=False)

    final = df[df.exposure == -1]
    agg = (final.groupby(["kappa", "signing_rate", "contamination"])
                .agg(ghost_col_err=("ghost_col_err", "mean"),
                     creator_mi=("creator_mi", "mean"),
                     mi_ratio=("mi_ratio", "mean"),
                     mi_ratio_sd=("mi_ratio", "std"),
                     ghost_col_entropy=("ghost_col_entropy", "mean"),
                     time_to_competence=("time_to_competence", "mean"))
                .reset_index())
    agg.to_csv(res_dir / "e7_summary.csv", index=False)

    if make_fig:
        make_e7_figure(df, agg, fig_dir / "e7_learn_ghost.png")
    return agg


def make_e7_figure(df, agg, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    traj = df[df.exposure > 0]

    ax = axes[0]
    for sr, sub in traj.groupby("signing_rate"):
        d = sub.groupby("exposure").ghost_col_err.mean()
        ax.plot(d.index, d.values, "-o", ms=4, label=f"signing_rate={sr}")
    ax.set(xlabel="artifacts seen", ylabel="KL(learned GHOST column || true synth) [nats]",
           title="Acquiring the GHOST column")
    ax.legend(fontsize=8)

    ax = axes[1]
    for sr, sub in traj.groupby("signing_rate"):
        d = sub.groupby("exposure").mi_ratio.mean()
        ax.plot(d.index, d.values, "-o", ms=4, label=f"signing_rate={sr}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.text(traj.exposure.min(), 1.0, " oracle", va="bottom", fontsize=8)
    ax.set(xlabel="artifacts seen",
           ylabel="MI(features;goal) of learned human columns / oracle",
           title="Do the human columns stay sharp,\nor absorb synthetic mass?")
    ax.legend(fontsize=8)

    ax = axes[2]
    piv = agg.pivot_table(index="signing_rate", columns="kappa", values="mi_ratio")
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)), [f"κ={c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"sign={i}" for i in piv.index])
    for a in range(piv.shape[0]):
        for b in range(piv.shape[1]):
            ax.text(b, a, f"{piv.values[a, b]:.2f}", ha="center", va="center", color="w")
    ax.set(title="Final MI ratio vs oracle")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("E7 — Can the GHOST column be learned without labels?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E7 — learning the GHOST column without labels")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    print("\nE7 — final learned-model state by condition:")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
