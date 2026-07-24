"""E3 — Ghost Scale titration (maps to H3).

Design (Spec §5): full four-tier ladder x observer type {calibrated kappa=0.9,
naive kappa=0.1} x c_effort sweep. Each observer views one artifact from a mixed corpus
(uniform over the four tiers). Calibrated and naive observers view the SAME artifacts.

Measure: cumulative HIGH_COST observations (metabolic expenditure) across the corpus, and
accuracy of the final goal posterior against ground truth on CREATOR trials.

Prediction: calibrated observers spend measurably less total effort on a mixed corpus
WITHOUT losing goal-recovery accuracy on genuine human artifacts. If they DO lose accuracy,
that is a real cost and is reported, not suppressed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..constants import PROVENANCE_NAMES, CREATOR, POLISHED, CURATOR, GHOST
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import make_observer, rollout_observer
from .. import figures
from . import _common as C

TIERS = [CREATOR, POLISHED, CURATOR, GHOST]


def _e3_worker(payload):
    (cfg_raw, cell_index, kappa, obs_type, c_effort, seed_rep, base_seed, n_obs, T) = payload
    cfg = Config(cfg_raw)
    cfg.set("preferences.c_effort", c_effort)
    gm = _build_model(cfg, kappa=kappa)
    assert_preferences_zero(gm.C)
    bank = build_creator_bank(cfg, gm)
    world = np.random.default_rng(base_seed * 31 + seed_rep)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=1.0, creator_bank=bank)
    recs = []
    for i in range(n_obs):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        # Artifact identity is seeded independently of kappa so calibrated & naive see the
        # SAME corpus (same tier + goal per observer index).
        art_rng = np.random.default_rng(base_seed * 777 + seed_rep * 97 + i)
        tier = TIERS[int(art_rng.integers(len(TIERS)))]
        goal = int(art_rng.integers(cfg.cardinalities.num_goals))
        agent = make_observer(gm, cfg, r, kappa=kappa)
        artifact = env.make_artifact(provenance=tier, goal=goal, rng=r)
        res = rollout_observer(agent, artifact, env, cfg, r, T, kappa=kappa)
        recs.append({"obs_type": obs_type, "kappa": kappa, "c_effort": c_effort,
                     "seed_rep": seed_rep, "observer": i,
                     "tier": PROVENANCE_NAMES[tier], "true_goal": goal,
                     "modal_goal": res.modal_goal,
                     "correct": int(res.modal_goal == goal),
                     "high_cost": res.cum_high_cost})
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs, T, n_seeds = int(cfg.run.n_observers), int(cfg.run.n_timesteps), int(cfg.run.n_seeds)
    cal = float(cfg.experiments.e3.calibrated_kappa)
    naive = float(cfg.experiments.e3.naive_kappa)
    sweep = list(cfg.experiments.e3.c_effort_sweep)
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for obs_type, kappa in [("calibrated", cal), ("naive", naive)]:
        for ce in sweep:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, kappa, obs_type, ce, s, base_seed, n_obs, T))
            ci += 1
    recs = C.run_parallel(payloads, _e3_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e3_raw.csv", index=False)

    # Aggregate per (obs_type, c_effort): total metabolic expenditure + CREATOR accuracy.
    rows = []
    for (obs_type, ce), g in df.groupby(["obs_type", "c_effort"]):
        creator = g[g.tier == "CREATOR"]
        rows.append({"obs_type": obs_type, "c_effort": ce,
                     "mean_high_cost": float(g.high_cost.mean()),
                     "creator_accuracy": float(creator.correct.mean()) if len(creator) else np.nan,
                     "creator_final_correct_n": int(creator.correct.sum())})
    agg = pd.DataFrame(rows).sort_values(["obs_type", "c_effort"]).reset_index(drop=True)
    agg.to_csv(res_dir / "e3_summary.csv", index=False)

    if make_fig:
        figures.save_simple_lines(
            agg, x="c_effort", y="mean_high_cost", hue="obs_type",
            path=fig_dir / "e3_titration_effort.png",
            xlabel="c_effort", ylabel="mean HIGH_COST (metabolic expenditure)",
            title="E3 — Calibrated observers spend less effort on a mixed corpus (H3)")
        figures.save_simple_lines(
            agg, x="c_effort", y="creator_accuracy", hue="obs_type",
            path=fig_dir / "e3_titration_accuracy.png",
            xlabel="c_effort", ylabel="goal-recovery accuracy on CREATOR trials",
            title="E3 — ...without losing accuracy on genuine human artifacts")
    return agg


def main():
    ap = C.standard_argparser("E3 — Ghost Scale titration")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    print("E3 summary (metabolic expenditure vs CREATOR accuracy):")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
