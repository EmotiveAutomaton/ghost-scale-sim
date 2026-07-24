"""E9 — Poisoning versus starvation (V2 spec §2). STAGE 4.

Two corruption channels, run independently and together:

  * POISONING ONLY — hallucinated goal posteriors (high kappa, dishonest signals) are written
    into pA updates. Engagement policy FROZEN so disuse cannot occur.
  * STARVATION ONLY — honest signals, so no fabrication, but disengagement means no pA
    updates on genuine content. pA learning enabled, engagement free.
  * BOTH — the default condition.

    "Measure: shape of the learned A (KL from the true A per column) versus FLATNESS of the
     learned A (entropy per column). Poisoning distorts shape. Starvation flattens. These are
     distinguishable signatures and the diagnostic must separate them."

    PREDICTED  starvation is the larger effect at realistic contamination levels, because
               disengagement is fast and total while fabrication requires a dishonest signal
               to be present and trusted.

The two arms map cleanly onto switches that already exist: ``freeze_engagement`` (force DEEP
for the whole budget, so disuse is impossible) and ``honesty`` (whether the signal can lie).
Nothing bespoke is introduced for E9 — which is what makes its two channels comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import PROVENANCE_NAMES, CREATOR, GHOST, DEEP
from ..generative_model import (build_shared_model as _build_model, assert_preferences_zero,
                                build_observer_model, build_D)
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import rollout_observer
from ..preregistration import POP_GOAL_DIST, allocate_creator_goals
from .. import learning as L, metrics
from ..figures import set_style
from . import _common as C

# arm -> (honesty, freeze_engagement, learn)
ARMS = {
    "poisoning_only": (0.5, True, True),    # can lie; disuse prevented
    "starvation_only": (1.0, False, True),  # cannot lie; disuse free to occur
    "both": (0.5, False, True),             # the default condition
    "control": (1.0, True, True),           # neither channel active
}


def _e9_worker(payload):
    (cfg_raw, cell_index, arm, kappa, f, seed_rep, base_seed,
     n_creators, n_artifacts, n_observers, infer_steps, synth_seed) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals
    honesty, freeze, learn = ARMS[arm]

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                     # N7
    bank = build_creator_bank(cfg, gm)

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)
    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=honesty, signing_rate=1.0,
                      creator_bank=bank)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals,
                             rng=world)

    recs = []
    for i in range(n_observers):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        om = build_observer_model(gm, cfg, 0.0)
        agent = L.make_learner_agent(om, build_D(cfg, r), cfg, kappa=kappa)
        n_deep = n_gen = 0
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + i * 7919 + j)
            res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                                   force_deep_k=(infer_steps if freeze else 0),
                                   kappa=kappa, early_stop=False, learn=learn)
            if art.provenance == CREATOR:
                n_gen += 1
                n_deep += res.cum_deep

        rep = L.report_learned_model(agent.A[0], gm.A[0], gm.noise_free_synth, cfg)
        rec = {"arm": arm, "kappa": kappa, "contamination": f, "seed_rep": seed_rep,
               "observer": i, "honesty": honesty, "engagement_frozen": freeze,
               "deep_per_genuine": n_deep / max(n_gen, 1)}
        rec.update(rep.flat())
        # The two diagnostic axes, averaged over the HUMAN tiers (what corruption costs you).
        rec["shape_kl_human"] = float(np.mean([rep.kl_by_tier[p] for p in range(3)]))
        rec["flatness_human"] = float(np.mean([rep.entropy_by_tier[p] for p in range(3)]))
        recs.append(rec)
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e9
    n_creators = int(cfg.get("experiments.e9.n_creators", 50))
    n_artifacts, n_observers = int(e.n_artifacts), int(e.n_observers)
    infer_steps = int(cfg.get("experiments.e9.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e9.n_replications", 3))
    f_levels = list(cfg.get("experiments.e9.contamination_levels", [0.3, 0.6]))
    kappa = float(cfg.get("experiments.e9.kappa", 0.9))
    synth_seed = int(cfg.get("experiments.e9.synth_draw_seed", 1))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for arm in ARMS:
        for f in f_levels:
            for s in range(n_reps):
                payloads.append((cfg.raw, ci, arm, kappa, f, s, base_seed, n_creators,
                                 n_artifacts, n_observers, infer_steps, synth_seed))
            ci += 1
    recs = C.run_parallel(payloads, _e9_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e9_raw.csv", index=False)

    agg = (df.groupby(["arm", "contamination"])
             .agg(shape_kl=("shape_kl_human", "mean"), shape_kl_sd=("shape_kl_human", "std"),
                  flatness=("flatness_human", "mean"), flatness_sd=("flatness_human", "std"),
                  creator_mi=("creator_mi", "mean"),
                  ghost_col_err=("ghost_col_err", "mean"),
                  deep_per_genuine=("deep_per_genuine", "mean"))
             .reset_index())
    agg.to_csv(res_dir / "e9_summary.csv", index=False)

    if make_fig:
        make_e9_figure(df, agg, cfg, fig_dir / "e9_poison_starve.png")
    return agg


def make_e9_figure(df, agg, cfg, path):
    """The whole point: the two channels must be SEPARABLE. Shape distortion on one axis,
    flatness on the other; if poisoning and starvation land in different quadrants the
    diagnostic works."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colours = {"poisoning_only": "firebrick", "starvation_only": "C0",
               "both": "purple", "control": "grey"}

    ax = axes[0]
    for arm, sub in agg.groupby("arm"):
        sub = sub.sort_values("contamination")
        ax.errorbar(sub.contamination, sub.shape_kl, yerr=sub.shape_kl_sd, fmt="-o",
                    color=colours.get(arm, "k"), capsize=3, label=arm)
    ax.set(xlabel="contamination f", ylabel="per-column KL from true A  [nats]",
           title="SHAPE distortion (poisoning signature)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for arm, sub in agg.groupby("arm"):
        sub = sub.sort_values("contamination")
        ax.errorbar(sub.contamination, sub.flatness, yerr=sub.flatness_sd, fmt="-o",
                    color=colours.get(arm, "k"), capsize=3, label=arm)
    ax.set(xlabel="contamination f", ylabel="per-column entropy of learned A  [nats]",
           title="FLATNESS (starvation signature)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for arm, sub in df.groupby("arm"):
        ax.scatter(sub.shape_kl_human, sub.flatness_human, s=18, alpha=0.6,
                   color=colours.get(arm, "k"), label=arm)
    ax.set(xlabel="shape distortion  (KL from true A)",
           ylabel="flatness  (entropy of learned A)",
           title="The diagnostic: do the two channels separate?")
    ax.legend(fontsize=8)

    fig.suptitle("E9 — Poisoning distorts shape; starvation flattens", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E9 — poisoning versus starvation")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    print("\nE9 — shape (KL) versus flatness (entropy) of the learned A:")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
