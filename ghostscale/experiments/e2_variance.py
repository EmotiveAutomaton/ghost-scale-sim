"""E2 — IRL convergence failure (maps to H2; the headline result).

Design (Spec §5): 2x2 factorial, true origin {CREATOR, GHOST} x declared signal
{SIG_CREATOR, SIG_GHOST}. Observers are forced to DEEP for the first k steps (default 10)
so every cell gets a comparable inference attempt, then are free.

Because between-observer disagreement is INTER-RATER RELIABILITY, every observer in a cell
views the SAME true goal (a common stimulus). Disagreement then reflects hallucination, not
different stimuli. Observers differ only in their heterogeneous priors and the random draws.

Measure:
  * between-observer disagreement = entropy of the modal-goal distribution across observers.
  * within-observer confidence   = mean posterior entropy over creator_goal per observer.
The DISSOCIATION between the two is the signature of hallucination:
  (true=GHOST, signal=SIG_CREATOR) -> LOW within + HIGH between = R_hallucinated
  (true=GHOST, signal=SIG_GHOST)   -> HIGH within = uncertainty correctly preserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..constants import (PROVENANCE_NAMES, SIGNAL_NAMES, CREATOR, GHOST,
                         SIG_CREATOR, SIG_GHOST)
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment, Artifact
from ..observer import make_observer, rollout_observer
from .. import metrics, figures
from . import _common as C

# 2x2 cells: (true provenance, declared signal).
CELLS = [(CREATOR, SIG_CREATOR), (CREATOR, SIG_GHOST),
         (GHOST, SIG_CREATOR), (GHOST, SIG_GHOST)]


def _e2_worker(payload):
    (cfg_raw, cell_index, provenance, signal, seed_rep, base_seed,
     n_obs, T, k, true_goal) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, kappa=float(cfg.signal_model.kappa))
    assert_preferences_zero(gm.C)  # N7
    bank = build_creator_bank(cfg, gm)
    world_rng = np.random.default_rng(base_seed * 31 + seed_rep)
    env = Environment(cfg, gm, rng_world=world_rng, creator_bank=bank)
    recs = []
    for i in range(n_obs):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_observer(gm, cfg, r)
        # Declared signal is a DESIGN factor here: construct the artifact directly.
        artifact = Artifact(provenance=provenance, goal=true_goal, declared_signal=signal)
        res = rollout_observer(agent, artifact, env, cfg, r, T, force_deep_k=k,
                               kappa=float(cfg.signal_model.kappa))
        recs.append({
            "true_provenance": PROVENANCE_NAMES[provenance],
            "declared_signal": SIGNAL_NAMES[signal],
            "seed_rep": seed_rep, "observer": i,
            "true_goal": true_goal, "modal_goal": res.modal_goal,
            "within_entropy": float(res.within_entropy[-1]),
            "final_posterior": res.final_goal_posterior.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs, T, n_seeds = int(cfg.run.n_observers), int(cfg.run.n_timesteps), int(cfg.run.n_seeds)
    k = int(cfg.experiments.e2.forced_deep_k)
    true_goal = int(cfg.get("experiments.e2.true_goal", 1))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads = [(cfg.raw, ci, prov, sig, s, base_seed, n_obs, T, k, true_goal)
                for ci, (prov, sig) in enumerate(CELLS) for s in range(n_seeds)]
    recs = C.run_parallel(payloads, _e2_worker, workers)

    # The posterior is KEPT (V4.5 A2). It was dropped here through V1-V4, which made Brier
    # score and expected calibration error uncomputable from the committed CSV: those are
    # functions of a predicted DISTRIBUTION, and a modal goal plus an entropy does not
    # determine one. Keeping it costs one wide column and makes the calibration analysis a
    # true zero-compute reanalysis from this run forward.
    points = pd.DataFrame(recs)
    points = points.sort_values(
        ["true_provenance", "declared_signal", "seed_rep", "observer"]).reset_index(drop=True)
    points.to_csv(res_dir / "e2_points.csv", index=False)

    # Cell statistics: within (mean per-observer entropy) and between (modal-goal entropy),
    # each computed per seed then averaged, with the dissociation gap.
    cell_stats = {}
    stat_rows = []
    for prov, sig in CELLS:
        tier, sname = PROVENANCE_NAMES[prov], SIGNAL_NAMES[sig]
        sub = [r for r in recs if r["true_provenance"] == tier and r["declared_signal"] == sname]
        withins, betweens = [], []
        for s in range(n_seeds):
            post = [np.asarray(r["final_posterior"]) for r in sub if r["seed_rep"] == s]
            withins.append(metrics.mean_within_observer_entropy(post))
            betweens.append(metrics.between_observer_entropy(post))
        cell_stats[(tier, sname)] = {"within": float(np.mean(withins)),
                                     "between": float(np.mean(betweens)),
                                     "within_sd": float(np.std(withins)),
                                     "between_sd": float(np.std(betweens))}
        stat_rows.append({"true_provenance": tier, "declared_signal": sname,
                          **cell_stats[(tier, sname)]})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(res_dir / "e2_cell_stats.csv", index=False)

    if make_fig:
        # Scatter uses one representative population (seed_rep 0) for legibility.
        pts0 = points[points.seed_rep == 0]
        figures.fig_e2(pts0, cell_stats, fig_dir / "e2_variance.png")
    return points


def main():
    ap = C.standard_argparser("E2 — IRL convergence failure (headline)")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    stats = pd.read_csv((Path(args.out) if args.out else C.RESULTS_DIR) / "e2_cell_stats.csv")
    print("E2 cell statistics (within = confidence, between = disagreement):")
    print(stats.round(3).to_string(index=False))
    hall = stats[(stats.true_provenance == "GHOST") & (stats.declared_signal == "SIG_CREATOR")]
    print("\nHEADLINE (GHOST, SIG_CREATOR): within=%.3f (confident) between=%.3f (disagree)"
          % (hall.within.iloc[0], hall.between.iloc[0]))


if __name__ == "__main__":
    main()
