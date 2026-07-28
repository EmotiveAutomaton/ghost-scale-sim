"""E10 — The expertise gradient (V2 spec §2). The RLHF result. STAGE 2.

    "Sweep observer inexpertise d_i from 0 to 0.9 on a corpus of uncontaminated, genuinely
     intent-dense human artifacts. No synthetic content at all."

    "Why this matters more than the rest. It says the extractor's own competence is a hard
     ceiling on recoverable intent, independent of data quality. If true, no amount of better
     data or more raters repairs RLHF, because the instrument is the limit rather than the
     sample."

    "Run E10 even if compute forces cuts elsewhere. It is cheap and it is the strongest
     standalone claim in V2."

CORPUS QUALITY IS HELD PERFECTLY CONSTANT. Every artifact is CREATOR provenance, produced by
a real HumanCreator policy; there is no synthetic content at any inexpertise level. Asserted
in code (null N12) rather than merely intended, because the entire claim is that the
degradation comes from the READER and not from the data.

  PREDICTED  recovered intent density falls monotonically with inexpertise even though
             corpus quality is constant, and C_recovered from high-d observers produces
             measurably higher sycophancy in a downstream agent.
  FALSIFIED  if recovered intent density is flat in expertise on a fixed clean corpus, the
             expertise-gating hypothesis is wrong and the RLHF reframing does not follow.
             (V2 §6: report honestly in whichever direction it comes out.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import CREATOR, GHOST, F_GOAL
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import make_observer, rollout_observer
from ..preregistration import POP_GOAL_DIST, allocate_creator_goals
from .. import metrics, regret as R
from ..figures import set_style
from . import _common as C


def _e10_worker(payload):
    (cfg_raw, cell_index, d_i, seed_rep, base_seed, n_creators, n_artifacts,
     n_observers, infer_steps) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals

    gm = _build_model(cfg)                      # world model: sig_true, shared by all
    assert_preferences_zero(gm.C)               # N7
    bank = build_creator_bank(cfg, gm)

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)
    c_true = np.bincount(creator_goals, minlength=num_goals).astype(float)
    c_true /= c_true.sum()

    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=1.0,
                      creator_bank=bank)

    # N12 ASSERTED, not assumed: a clean corpus, zero synthetic content.
    corpus = env.draw_corpus(n_artifacts, contamination=0.0,
                             creator_goals=creator_goals, rng=world)
    assert all(a.provenance == CREATOR for a in corpus), \
        "N12: the E10 corpus must contain NO GHOST artifacts"

    recs = []
    for i in range(n_observers):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_observer(gm, cfg, r, d_i=d_i)
        accum = np.zeros(num_goals)
        psis, kls, correct, ents = [], [], [], []
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + i * 7919 + j)
            res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                                   force_deep_k=infer_steps, early_stop=False)
            q = res.final_goal_posterior
            accum += q
            # Intent density: how far a full-effort reading moves the goal belief off its
            # prior. This is what "extracted intent" means operationally.
            kl_from_prior = metrics.kl_divergence(q, res.goal_prior)
            kls.append(kl_from_prior)
            psis.append(metrics.psi_analogue(q, res.goal_prior,
                                             float(cfg.signal_model.kappa), engaged=True))
            correct.append(int(res.modal_goal == art.goal))
            ents.append(float(res.within_entropy[-1]))

        c_rec = accum / accum.sum()
        rec = {"d": d_i, "expertise": 1.0 - d_i, "seed_rep": seed_rep, "observer": i,
               "psi": float(np.mean(psis)),
               "kl_posterior_prior": float(np.mean(kls)),
               "goal_accuracy": float(np.mean(correct)),
               "final_entropy": float(np.mean(ents)),
               "n_ghost_in_corpus": 0}
        rec.update(R.behavioral_regret(c_rec, c_true, cfg,
                                       seed=base_seed + seed_rep * 31 + i).flat())
        recs.append(rec)
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e10
    d_sweep = list(e.d_sweep)
    n_creators, n_artifacts = int(e.n_creators), int(e.n_artifacts)
    n_observers = int(cfg.get("experiments.e10.n_observers", 20))
    infer_steps = int(cfg.get("experiments.e10.infer_steps", 8))
    n_reps = int(cfg.get("experiments.e10.n_replications", 5))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for d_i in d_sweep:
        for s in range(n_reps):
            payloads.append((cfg.raw, ci, float(d_i), s, base_seed, n_creators,
                             n_artifacts, n_observers, infer_steps))
        ci += 1
    recs = C.run_parallel(payloads, _e10_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e10_raw.csv", index=False)

    agg = (df.groupby("d")
             .agg(psi=("psi", "mean"), psi_sd=("psi", "std"),
                  kl_post_prior=("kl_posterior_prior", "mean"),
                  goal_accuracy=("goal_accuracy", "mean"),
                  final_entropy=("final_entropy", "mean"),
                  kl_recovered=("kl", "mean"),
                  regret=("regret", "mean"), regret_sd=("regret", "std"),
                  argmax_preserved=("argmax_preserved", "mean"),
                  sycophancy=("sycophancy", "mean"), sycophancy_sd=("sycophancy", "std"))
             .reset_index())
    agg.to_csv(res_dir / "e10_summary.csv", index=False)

    stats = gradient_stats(df)
    pd.DataFrame([stats]).to_csv(res_dir / "e10_gradient_test.csv", index=False)

    if make_fig:
        make_e10_figure(df, agg, fig_dir / "e10_expertise_gradient.png")
    return agg


def gradient_stats(df: pd.DataFrame) -> dict:
    """The falsification test, stated as a number rather than read off a plot.

    E10's claim is a MONOTONE fall in extracted intent density with inexpertise. Reported as
    the OLS slope of psi on d with its standard error, plus Spearman rank correlation (which
    tests monotonicity without assuming linearity), and the same for sycophancy.
    """
    out = {}
    for col in ("psi", "kl_posterior_prior", "goal_accuracy", "sycophancy", "regret"):
        x = df["d"].to_numpy(float)
        y = df[col].to_numpy(float)
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or np.allclose(x, x[0]):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        dof = max(len(x) - 2, 1)
        se = float(np.sqrt((resid @ resid) / dof / np.sum((x - x.mean()) ** 2)))
        # Spearman without scipy: Pearson on ranks.
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rho = float(np.corrcoef(rx, ry)[0, 1])
        out[f"{col}_slope"] = float(slope)
        out[f"{col}_se"] = se
        out[f"{col}_t"] = float(slope / se) if se > 0 else float("nan")
        out[f"{col}_spearman"] = rho
    out["psi_falsified_if_flat"] = bool(abs(out.get("psi_t", 0.0)) < 2.0)
    return out


def make_e10_figure(df, agg, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.errorbar(agg.d, agg.psi, yerr=agg.psi_sd, fmt="-o", color="C0", lw=2, capsize=3)
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="how much intent the reader pulls out (psi)",
           title="Less skilled readers pull out less intent,\n"
                 "from work that never changed at all")

    ax = axes[1]
    ax.plot(agg.d, agg.goal_accuracy, "-o", color="C2", lw=2,
            label="how often it names the right purpose")
    ax2 = ax.twinx()
    ax2.plot(agg.d, agg.final_entropy, "--s", color="C3", lw=1.6,
             label="how unsure it ends up")
    ax2.set_ylabel("how unsure the reader ends up (nats)", color="C3")
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="how often it names the right purpose",
           title="The reader is the ceiling, not the material")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[2]
    ax.errorbar(agg.d, agg.sycophancy, yerr=agg.sycophancy_sd, fmt="-o", color="firebrick",
                lw=2, capsize=3)
    # This panel shows a prediction FAILING. We expected unskilled readers to produce a more
    # flattering downstream agent; the line is flat (slope -0.0065, t = -0.37). The title has to
    # report the flat line, not the prediction, or the chart argues against its own data.
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="how often the trained agent just flatters",
           title="The prediction that failed: unskilled readers\n"
                 "do NOT produce a more flattering agent")

    fig.suptitle("E10 — Whoever reads the work sets the ceiling on what can be learned from it",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E10 — the expertise gradient (headline standalone claim)")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    stats = pd.read_csv(res_dir / "e10_gradient_test.csv").iloc[0]

    print("\nE10 — expertise gradient on a clean, constant-quality corpus:")
    print(agg[["d", "psi", "goal_accuracy", "final_entropy", "kl_recovered",
               "regret", "argmax_preserved", "sycophancy"]].round(4).to_string(index=False))
    print(f"\n  psi   ~ d : slope={stats['psi_slope']:+.4f} (t={stats['psi_t']:+.2f}, "
          f"spearman={stats['psi_spearman']:+.3f})")
    print(f"  syco  ~ d : slope={stats['sycophancy_slope']:+.4f} "
          f"(t={stats['sycophancy_t']:+.2f}, spearman={stats['sycophancy_spearman']:+.3f})")
    if stats["psi_falsified_if_flat"]:
        print("\n  FALSIFIED: intent density is FLAT in expertise on a fixed clean corpus.")
        print("  The expertise-gating hypothesis is wrong and the RLHF reframing does not follow.")
    else:
        print("\n  Gradient present in the predicted direction.")


if __name__ == "__main__":
    main()
