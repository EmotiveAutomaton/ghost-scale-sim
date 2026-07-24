"""E6 — Corpus corruption and the alignment payload (maps to H6).

This is the experiment that makes the framework an alignment result rather than a perception
result: it is the CIRL loop in miniature. C_recovered is what you would hand a downstream
agent as its preference prior.

Design (Spec §5): a population of 50 creators with a known goal distribution C_true. An
aggregator observer watches N=2000 artifacts from a corpus with contamination fraction f of
UNSIGNED synthetic artifacts, and accumulates a posterior over the population goal
distribution, C_recovered. Per artifact the aggregator attempts goal extraction (forced
DEEP) and weights that artifact's goal posterior by its belief that the artifact is genuine,
(1 - P(GHOST)); this is how an honest provenance signal protects the payload.

Measure: KL(C_recovered || C_true) as a function of f, at kappa in {0.1,0.5,0.9} and under
signing_rate in {0.0,0.5,1.0}.

Prediction: recovery degrades with contamination; the degradation is strongly attenuated by
an honest signal at any kappa, and worsened by high kappa under low signing rate (the
trusting observer with no signal is the most corruptible). Reported as measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import CREATOR, GHOST, F_PROVENANCE, F_GOAL
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment, Artifact
from ..observer import make_observer, rollout_observer
from .. import metrics
from ..figures import set_style
from . import _common as C

# A fixed, non-uniform population goal distribution (the alignment payload to be recovered).
POP_GOAL_DIST = np.array([0.40, 0.30, 0.20, 0.10])


def _e6_worker(payload):
    (cfg_raw, cell_index, kappa, signing_rate, f, seed_rep, base_seed,
     n_creators, n_artifacts, infer_steps) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals
    gm = _build_model(cfg, kappa=kappa)
    assert_preferences_zero(gm.C)
    bank = build_creator_bank(cfg, gm)

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = pop_rng.choice(num_goals, size=n_creators, p=POP_GOAL_DIST[:num_goals])
    c_true = np.bincount(creator_goals, minlength=num_goals).astype(float)
    c_true /= c_true.sum()

    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=signing_rate,
                      creator_bank=bank)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals, rng=world)

    # One aggregator observer, reset per artifact.
    agent = make_observer(gm, cfg, np.random.default_rng(base_seed * 101 + seed_rep), kappa=kappa)
    accum = np.zeros(num_goals)
    accum_naive = np.zeros(num_goals)
    for j, art in enumerate(corpus):
        r = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + j)
        res = rollout_observer(agent, art, env, cfg, r, infer_steps,
                               force_deep_k=infer_steps, kappa=kappa, early_stop=False)
        q_goal = res.final_goal_posterior
        p_ghost = float(res.prov_posterior[-1][GHOST])
        accum += (1.0 - p_ghost) * q_goal          # down-weight suspected synthetics
        accum_naive += q_goal                        # naive aggregation (no down-weighting)

    c_recovered = accum / accum.sum() if accum.sum() > 0 else np.full(num_goals, 1.0 / num_goals)
    c_naive = accum_naive / accum_naive.sum()
    return [{"kappa": kappa, "signing_rate": signing_rate, "contamination": f,
             "seed_rep": seed_rep,
             "kl_recovered": metrics.kl_divergence(c_recovered, c_true),
             "kl_naive": metrics.kl_divergence(c_naive, c_true),
             "c_recovered": c_recovered.tolist(), "c_true": c_true.tolist()}]


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e6 = cfg.experiments.e6
    n_creators, n_artifacts = int(e6.n_creators), int(e6.n_artifacts)
    infer_steps = int(cfg.get("experiments.e6.infer_steps", 8))
    n_reps = int(cfg.get("experiments.e6.n_replications", min(int(cfg.run.n_seeds), 5)))
    f_sweep = list(e6.contamination_sweep)
    kappas = list(e6.kappa_levels)
    signing = list(e6.signing_rate_levels)
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for kap in kappas:
        for sr in signing:
            for f in f_sweep:
                for s in range(n_reps):
                    payloads.append((cfg.raw, ci, kap, sr, f, s, base_seed,
                                     n_creators, n_artifacts, infer_steps))
                ci += 1
    recs = C.run_parallel(payloads, _e6_worker, workers)
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("c_recovered", "c_true")}
                       for r in recs])
    df.to_csv(res_dir / "e6_raw.csv", index=False)

    agg = (df.groupby(["kappa", "signing_rate", "contamination"])
           .agg(kl_recovered=("kl_recovered", "mean"),
                kl_naive=("kl_naive", "mean")).reset_index())
    agg.to_csv(res_dir / "e6_summary.csv", index=False)

    if make_fig:
        make_e6_figure(agg, kappas, signing, fig_dir / "e6_corpus_corruption.png")
    return agg


def make_e6_figure(agg, kappas, signing, path):
    """Two panels. LEFT: the alignment result — provenance-aware (down-weighted) aggregation
    protects C_recovered under contamination while naive aggregation degrades (averaged over
    kappa/signing, which barely matter). RIGHT: shows that near-invariance — recovery KL is
    flat across signing_rate, because a DEEP-inspecting aggregator identifies synthetics from
    content, so the signal's value is metabolic (E3/E5), not recovery accuracy."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    naive = agg.groupby("contamination").kl_naive.mean()
    recov = agg.groupby("contamination").kl_recovered.mean()
    ax.plot(naive.index, naive.values, "-o", color="firebrick", lw=2,
            label="naive aggregation")
    ax.plot(recov.index, recov.values, "-o", color="C0", lw=2,
            label="provenance-weighted (Ghost Scale)")
    ax.set(xlabel="contamination fraction f",
           ylabel="KL(C_recovered || C_true)  [nats]",
           title="Provenance-awareness protects the alignment payload")
    ax.legend()

    ax = axes[1]
    for sr in signing:
        d = agg[agg.signing_rate == sr].groupby("contamination").kl_recovered.mean()
        ax.plot(d.index, d.values, "-o", ms=4, label=f"signing_rate={sr}")
    ax.set(xlabel="contamination fraction f",
           ylabel="KL(C_recovered || C_true)  [nats]",
           title="Recovery is ~invariant to signing rate\n(value of the signal is metabolic, not accuracy)")
    ax.legend(fontsize=8)

    fig.suptitle("E6 — Corpus corruption of the alignment payload (H6)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E6 — corpus corruption and the alignment payload")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    print("E6 — KL(C_recovered || C_true) by (kappa, signing_rate) at max contamination:")
    fmax = agg.contamination.max()
    piv = agg[agg.contamination == fmax].pivot(index="signing_rate", columns="kappa",
                                               values="kl_recovered")
    print(f"(at f={fmax})")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
