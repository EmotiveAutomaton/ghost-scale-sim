"""E1 — The generative crash (maps to H1).

Design (Spec §5): 1x4 across true provenance. Signals honest, signing_rate=1.0, kappa=0.9.
Per observer we measure: timestep of first free SKIM, cumulative DEEP steps, the epistemic
component of EFE for the DEEP policy at each t, and posterior entropy over creator_goal.

Prediction: disengagement short/clustered for GHOST, long/absent for CREATOR, monotonic
across tiers; the epistemic value of DEEP about the goal collapses toward zero for GHOST as
a step function across tiers (the step-vs-gradual claim lives in the EFE decomposition).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..constants import PROVENANCE_NAMES, CREATOR, POLISHED, CURATOR, GHOST, DEEP
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import make_observer, rollout_observer
from .. import figures
from . import _common as C

TIERS = [CREATOR, POLISHED, CURATOR, GHOST]


def _e1_worker(payload):
    (cfg_raw, cell_index, provenance, seed_rep, base_seed, n_obs, T, record_efe) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, kappa=float(cfg.signal_model.kappa))
    assert_preferences_zero(gm.C)  # N7 at construction in every experiment
    bank = build_creator_bank(cfg, gm)
    world_rng = np.random.default_rng(base_seed * 31 + seed_rep)
    env = Environment(cfg, gm, rng_world=world_rng, honesty=1.0, signing_rate=1.0,
                      creator_bank=bank)
    recs = []
    for i in range(n_obs):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        true_goal = int(r.integers(cfg.cardinalities.num_goals))
        agent = make_observer(gm, cfg, r)
        artifact = env.make_artifact(provenance=provenance, goal=true_goal, rng=r)
        res = rollout_observer(agent, artifact, env, cfg, r, T,
                               record_efe=record_efe, kappa=float(cfg.signal_model.kappa))
        recs.append({
            "tier": PROVENANCE_NAMES[provenance],
            "seed_rep": seed_rep, "observer": i,
            "true_goal": true_goal, "modal_goal": res.modal_goal,
            "correct": int(res.modal_goal == true_goal),
            "first_skim_t": res.first_skim_t,
            "cum_deep": res.cum_deep,
            "final_goal_entropy": float(res.within_entropy[-1]),
            # time series (padded to T by early-stop)
            "attention": res.attention.tolist(),
            "goal_entropy": res.within_entropy.tolist(),
            "deep_epi_goal": res.efe.get("deep_epi_goal", np.zeros(T)).tolist(),
            "deep_epi_total": res.efe.get("deep_epi_total", np.zeros(T)).tolist(),
            "deep_prag": res.efe.get("deep_prag", np.zeros(T)).tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs, T, n_seeds = int(cfg.run.n_observers), int(cfg.run.n_timesteps), int(cfg.run.n_seeds)
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads = [(cfg.raw, ci, prov, s, base_seed, n_obs, T, True)
                for ci, prov in enumerate(TIERS) for s in range(n_seeds)]
    recs = C.run_parallel(payloads, _e1_worker, workers)

    # Per-observer summary (drop the time-series columns).
    ts_cols = ["attention", "goal_entropy", "deep_epi_goal", "deep_epi_total", "deep_prag"]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ts_cols} for r in recs])
    summary = summary.sort_values(["tier", "seed_rep", "observer"]).reset_index(drop=True)
    summary.to_csv(res_dir / "e1_summary.csv", index=False)

    # Aggregated time-series per (tier, t): survival + entropy + EFE terms.
    effort_gap = 2 * int(cfg.agent.policy_len) * float(cfg.preferences.c_effort)
    rows = []
    for name in PROVENANCE_NAMES:
        sub = [r for r in recs if r["tier"] == name]
        A = np.array([r["attention"] for r in sub])          # (n, T)
        GE = np.array([r["goal_entropy"] for r in sub])
        EG = np.array([r["deep_epi_goal"] for r in sub])
        ET = np.array([r["deep_epi_total"] for r in sub])
        for t in range(T):
            rows.append({"tier": name, "t": t,
                         "frac_deep": float(np.mean(A[:, t] == DEEP)),
                         "goal_entropy": float(np.mean(GE[:, t])),
                         "deep_epi_goal": float(np.mean(EG[:, t])),
                         "deep_epi_total": float(np.mean(ET[:, t])),
                         "effort_gap": effort_gap})
    series = pd.DataFrame(rows)
    series.to_csv(res_dir / "e1_timeseries.csv", index=False)

    if make_fig:
        figures.fig_e1(series, fig_dir / "e1_crash.png")
    return summary


def main():
    ap = C.standard_argparser("E1 — the generative crash")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    summary = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    # Console digest.
    g = summary.groupby("tier")
    digest = g.agg(cum_deep=("cum_deep", "mean"),
                   final_entropy=("final_goal_entropy", "mean"),
                   accuracy=("correct", "mean")).reindex(PROVENANCE_NAMES)
    print("E1 digest (by tier):")
    print(digest.round(3).to_string())


if __name__ == "__main__":
    main()
