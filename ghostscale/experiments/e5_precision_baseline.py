"""E5 — kappa is not gamma (the precision baseline). Spec §8.

gamma is a GLOBAL scalar on policy selection (softmax decisiveness). kappa is the precision
of a SINGLE likelihood mapping (the provenance channel A[1]). The claim: kappa produces
CONTENT-SELECTIVE engagement (sustain human, drop synthetic, same observer, same gamma),
whereas gamma can only produce INDISCRIMINATE shifts in decisiveness.

Arms (same mixed corpus, same observer otherwise):
  A: kappa swept 0->0.99, gamma fixed          -> selectivity should RISE with kappa
  B: kappa fixed 0,       gamma swept low->high -> selectivity stays NEAR ZERO (content-only)
  C: kappa fixed 0.9,     gamma swept low->high -> selectivity PRESERVED across gamma

Primary metric: selectivity = P(DEEP | true=CREATOR) - P(DEEP | true=GHOST); overall
engagement rate as a control. Arm B's selectivity is the CONTENT-ONLY baseline: with kappa=0
the observer can still read provenance from content (A[0] differs across tiers). The Ghost
Scale's contribution is the increment of arm A/C above that baseline AND the reduction in
DEEP steps needed to achieve it (reported as steps-to-disengage on GHOST).

Falsification: if arm B produces selectivity comparable to arm A, kappa is redundant with
gamma. Reported plainly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import CREATOR, GHOST
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import make_observer, rollout_observer
from ..figures import set_style
from . import _common as C


def _engagement(cfg, gm, bank, provenance, kappa, base_seed, cell_index, seed_rep,
                n_obs, T, early_window):
    """Return (early-window engagement, mean DEEP steps).

    Engagement is measured over the FIRST ``early_window`` free steps, BEFORE extraction
    efficiency confounds it: CREATOR resolves its goal in a step or two and then disengages,
    so an all-timesteps P(DEEP) understates human engagement and can even invert the
    selectivity sign. The decision that matters for content-selectivity is "does the observer
    choose to look deeply at all", which lives in the opening steps."""
    world = np.random.default_rng(base_seed * 31 + seed_rep)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=1.0, creator_bank=bank)
    from ..constants import DEEP
    early, deep_steps = [], []
    for i in range(n_obs):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_observer(gm, cfg, r, kappa=kappa)
        artifact = env.make_artifact(provenance=provenance, goal=1, rng=r)
        res = rollout_observer(agent, artifact, env, cfg, r, T, kappa=kappa)
        early.append(float(np.mean(res.attention[:early_window] == DEEP)))
        deep_steps.append(res.cum_deep)
    return float(np.mean(early)), float(np.mean(deep_steps))


def _e5_worker(payload):
    (cfg_raw, cell_index, arm, kappa, gamma, seed_rep, base_seed, n_obs, T) = payload
    cfg = Config(cfg_raw)
    cfg.set("agent.gamma", gamma)
    # gamma scales the softmax over policies and is INERT under deterministic (argmax) action
    # selection. To test whether gamma can mimic kappa we must let it act, so E5 uses stochastic
    # selection throughout (all three arms, same setting) — otherwise the gamma sweep is a no-op
    # and the kappa-vs-gamma comparison is vacuous. Documented in RESULTS.md.
    cfg.set("agent.action_selection", "stochastic")
    early_window = int(cfg.get("experiments.e5.early_window", 3))
    gm = _build_model(cfg, kappa=kappa)
    assert_preferences_zero(gm.C)
    bank = build_creator_bank(cfg, gm)
    fc_creator, ds_creator = _engagement(cfg, gm, bank, CREATOR, kappa, base_seed, cell_index, seed_rep, n_obs, T, early_window)
    fc_ghost, ds_ghost = _engagement(cfg, gm, bank, GHOST, kappa, base_seed, cell_index * 1009 + 1, seed_rep, n_obs, T, early_window)
    return [{"arm": arm, "kappa": kappa, "gamma": gamma, "seed_rep": seed_rep,
             "p_deep_creator": fc_creator, "p_deep_ghost": fc_ghost,
             "selectivity": fc_creator - fc_ghost,
             "engagement": 0.5 * (fc_creator + fc_ghost),
             "deep_steps_ghost": ds_ghost}]


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs, T, n_seeds = int(cfg.run.n_observers), int(cfg.run.n_timesteps), int(cfg.run.n_seeds)
    kap_sweep = list(cfg.experiments.e5.kappa_sweep)
    gam_sweep = list(cfg.experiments.e5.gamma_sweep)
    fixed_gamma = float(cfg.experiments.e5.fixed_gamma)
    arm_c_kappa = float(cfg.experiments.e5.arm_c_kappa)
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for kap in kap_sweep:      # Arm A: sweep kappa, fixed gamma
        for s in range(n_seeds):
            payloads.append((cfg.raw, ci, "A", kap, fixed_gamma, s, base_seed, n_obs, T))
        ci += 1
    for gam in gam_sweep:      # Arm B: kappa=0, sweep gamma
        for s in range(n_seeds):
            payloads.append((cfg.raw, ci, "B", 0.0, gam, s, base_seed, n_obs, T))
        ci += 1
    for gam in gam_sweep:      # Arm C: kappa=0.9, sweep gamma
        for s in range(n_seeds):
            payloads.append((cfg.raw, ci, "C", arm_c_kappa, gam, s, base_seed, n_obs, T))
        ci += 1
    recs = C.run_parallel(payloads, _e5_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e5_raw.csv", index=False)

    agg = (df.groupby(["arm", "kappa", "gamma"])
           .agg(selectivity=("selectivity", "mean"),
                engagement=("engagement", "mean"),
                deep_steps_ghost=("deep_steps_ghost", "mean")).reset_index())
    agg.to_csv(res_dir / "e5_summary.csv", index=False)

    if make_fig:
        make_e5_figure(agg, fig_dir / "e5_precision_baseline.png")
    return agg


def make_e5_figure(agg, path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    a = agg[agg.arm == "A"].sort_values("kappa")
    axes[0].plot(a.kappa, a.selectivity, "-o", color="C0",
                 label="looks closely only at human work")
    axes[0].plot(a.kappa, a.engagement, "--s", color="C1",
                 label="looks closely at anything")
    axes[0].set(xlabel="how much the reader trusts the label (kappa)",
                ylabel="share of the time",
                title="Trusting the label makes the reader picky",
                ylim=(-0.05, 1.05))
    axes[0].legend()
    b = agg[agg.arm == "B"].sort_values("gamma")
    c = agg[agg.arm == "C"].sort_values("gamma")
    axes[1].plot(b.gamma, b.selectivity, "-o", color="C3",
                 label="picky, with no label to go on")
    axes[1].plot(c.gamma, c.selectivity, "-o", color="C0",
                 label="picky, with a trusted label")
    axes[1].plot(b.gamma, b.engagement, "--s", color="C3", alpha=0.5,
                 label="looks closely at anything, no label")
    axes[1].set(xlabel="how decisive the reader is in general (gamma)", xscale="log",
                ylabel="share of the time",
                title="Being more decisive in general does not make it picky",
                ylim=(-0.05, 1.05))
    axes[1].legend(fontsize=8)
    fig.suptitle("E5 — Trusting a label is a different knob from being decisive "
                 "in general", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E5 — kappa is not gamma")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    for arm in ["A", "B", "C"]:
        sub = agg[agg.arm == arm]
        print(f"Arm {arm}: selectivity range [{sub.selectivity.min():.3f}, {sub.selectivity.max():.3f}]")
    content_only = agg[(agg.arm == "B")].selectivity.mean()
    kappa_full = agg[(agg.arm == "A")].selectivity.max()
    print(f"content-only baseline (arm B mean) = {content_only:.3f}; "
          f"Ghost Scale increment (arm A max - baseline) = {kappa_full - content_only:.3f}")


if __name__ == "__main__":
    main()
