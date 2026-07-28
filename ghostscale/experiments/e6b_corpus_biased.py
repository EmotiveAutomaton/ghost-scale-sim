"""E6b — Corpus corruption with BIASED synthetic content (V2 spec §2). STAGE 1. RUN FIRST, ALONE.

    "Identical to V1 E6 except goal_symmetric: false and the oracle aggregator retained.
     Nothing else changes. This is the cheap decisive test of whether the V1 post-mortem
     was correct."

The V1 post-mortem's claim: symmetrizing the synthetic distribution turned contamination
into NOISE, which averages out across a large corpus and contributes almost nothing net.
Real generative output is a regression to the statistical mean — a systematic pull with a
DIRECTION, which accumulates linearly in sample size rather than cancelling. E6b tests
exactly that, changing one axis and nothing else.

  PREDICTED     naive-aggregator KL at f=0.8 between 0.15 and 0.6 nats (V1 measured 0.066).
                Provenance-weighted stays near flat.
  FALSIFIED IF  naive KL at f=0.8 stays BELOW 0.10. Then the noise-cancellation diagnosis is
                wrong, symmetrization was not what suppressed the V1 effect, and every
                downstream V2 experiment needs rethinking before it is run. Report loudly.

The bound is pre-registered and content-hashed BEFORE any inference runs; see
``ghostscale/preregistration.py``. Four synthetic draws stratified by favoured goal (D3a),
so the reported claim is "KL scales with the realized lean" rather than one draw's number.

The SYMMETRIC arm is retained here as the named control (null N9): running both arms in the
same experiment is what proves the new effect comes from the bias axis and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..constants import GHOST
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment
from ..observer import make_observer, rollout_observer
from ..preregistration import (POP_GOAL_DIST, write_preregistration, allocate_creator_goals,
                               assert_prereg_locked, contamination_bound)
from .. import metrics, regret as R
from ..figures import set_style
from . import _common as C

PREREG_NAME = "e6b_preregistration.json"


def _e6b_worker(payload):
    (cfg_raw, cell_index, kappa, signing_rate, f, seed_rep, base_seed,
     n_creators, n_artifacts, infer_steps, symmetric, synth_seed, favoured_k) = payload
    cfg = Config(cfg_raw)
    num_goals = cfg.cardinalities.num_goals

    gm = _build_model(cfg, kappa=kappa, goal_symmetric=symmetric,
                      synth_draw_seed=(None if symmetric else synth_seed))
    assert_preferences_zero(gm.C)                      # N7 at construction, every experiment
    bank = build_creator_bank(cfg, gm)

    pop_rng = np.random.default_rng(base_seed * 13 + seed_rep)
    creator_goals = allocate_creator_goals(n_creators, POP_GOAL_DIST[:num_goals], pop_rng)
    c_true = np.bincount(creator_goals, minlength=num_goals).astype(float)
    c_true /= c_true.sum()
    assert np.all(c_true > 0), "C_true must have full support for KL to be defined"

    world = np.random.default_rng(base_seed * 31 + seed_rep * 7)
    env = Environment(cfg, gm, rng_world=world, honesty=1.0, signing_rate=signing_rate,
                      creator_bank=bank)
    corpus = env.draw_corpus(n_artifacts, contamination=f, creator_goals=creator_goals,
                             rng=world)

    # ORACLE aggregator, retained from V1 unchanged: it holds a correct A[0]. E6b asks
    # whether biased contamination corrupts even an observer that already knows what
    # goalless output looks like. (E7 removes that knowledge.)
    agent = make_observer(gm, cfg, np.random.default_rng(base_seed * 101 + seed_rep),
                          kappa=kappa)
    accum = np.zeros(num_goals)
    accum_naive = np.zeros(num_goals)
    for j, art in enumerate(corpus):
        r = np.random.default_rng(base_seed * 90001 + seed_rep * 337 + j)
        res = rollout_observer(agent, art, env, cfg, r, infer_steps,
                               force_deep_k=infer_steps, kappa=kappa, early_stop=False)
        q_goal = res.final_goal_posterior
        p_ghost = float(res.prov_posterior[-1][GHOST])
        accum += (1.0 - p_ghost) * q_goal      # provenance-weighted (Ghost Scale)
        accum_naive += q_goal                  # naive (no down-weighting)

    c_rec = accum / accum.sum() if accum.sum() > 0 else np.full(num_goals, 1.0 / num_goals)
    c_naive = accum_naive / accum_naive.sum()

    rec = {"arm": "symmetric" if symmetric else "biased",
           "synth_seed": int(synth_seed), "favoured_k": int(favoured_k),
           "synth_lean": float(gm.synth_lean), "synth_k_realized": int(gm.synth_k),
           # Recorded because the between-arm contrast is CONFOUNDED by it: symmetrizing
           # redistributes mass across all four goal pairs and necessarily raises entropy
           # (V1's symmetric synth H=1.52 vs ~0.2-0.6 for the biased draws). The clean,
           # entropy-controlled test is the WITHIN-biased-arm k-dependence, where all four
           # draws are matched on lean and roughly matched on entropy by the D3a selection
           # rule, and only the favoured goal varies.
           "synth_entropy": float(metrics.shannon_entropy(gm.noise_free_synth)),
           "kappa": kappa, "signing_rate": signing_rate, "contamination": f,
           "seed_rep": seed_rep,
           "kl_recovered": metrics.kl_divergence(c_rec, c_true),
           "kl_naive": metrics.kl_divergence(c_naive, c_true),
           "bound": contamination_bound(c_true, f, int(gm.synth_k))}
    # C5 panel, reported alongside KL everywhere. Sycophancy only on the naive arm at the
    # top of the sweep keeps the cost down without losing the E11 scatter.
    heavy = bool(f >= 0.6)
    rec.update(R.behavioral_regret(c_naive, c_true, cfg, seed=seed_rep,
                                   with_sycophancy=heavy).flat("naive_"))
    rec.update(R.behavioral_regret(c_rec, c_true, cfg, seed=seed_rep,
                                  with_sycophancy=heavy).flat("weighted_"))
    return [rec]


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True,
        force_prereg: bool = False) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    # ---- PRE-REGISTRATION FIRST. No inference happens before this is on disk. ----
    prereg_path = res_dir / PREREG_NAME
    write_preregistration(cfg, prereg_path, force=force_prereg)
    prereg = assert_prereg_locked(prereg_path)
    draws = prereg["selected_draws"]

    e = cfg.experiments.e6b
    n_creators, n_artifacts = int(e.n_creators), int(e.n_artifacts)
    infer_steps = int(cfg.get("experiments.e6b.infer_steps", 8))
    n_reps = int(cfg.get("experiments.e6b.n_replications", 3))
    f_sweep = list(e.contamination_sweep)
    kappas = list(e.kappa_levels)
    signing = list(e.signing_rate_levels)

    payloads, ci = [], 0
    for kap in kappas:
        for sr in signing:
            for f in f_sweep:
                for s in range(n_reps):
                    for d in draws:                      # biased arm: one cell per draw
                        payloads.append((cfg.raw, ci, kap, sr, f, s, base_seed, n_creators,
                                         n_artifacts, infer_steps, False,
                                         int(d["seed"]), int(d["favoured_k"])))
                    # symmetric control arm (N9), one cell
                    payloads.append((cfg.raw, ci, kap, sr, f, s, base_seed, n_creators,
                                     n_artifacts, infer_steps, True, 0, -1))
                ci += 1

    recs = C.run_parallel(payloads, _e6b_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e6b_raw.csv", index=False)

    agg = (df.groupby(["arm", "favoured_k", "contamination"])
             .agg(kl_naive=("kl_naive", "mean"), kl_naive_sd=("kl_naive", "std"),
                  kl_recovered=("kl_recovered", "mean"),
                  bound=("bound", "mean"), synth_lean=("synth_lean", "mean"),
                  naive_regret=("naive_regret", "mean"),
                  naive_argmax=("naive_argmax_preserved", "mean"),
                  weighted_regret=("weighted_regret", "mean"))
             .reset_index())
    agg.to_csv(res_dir / "e6b_summary.csv", index=False)

    verdict = evaluate_gate(df, prereg)
    (res_dir / "e6b_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_e6b_figure(df, agg, prereg, fig_dir / "e6b_corpus_biased.png")
    return agg


def evaluate_gate(df: pd.DataFrame, prereg: dict) -> dict:
    """Score the run against the pre-registered prediction and falsification threshold.

    Written to disk as ``e6b_verdict.json`` so the stage-1 gate is a recorded artifact
    rather than a judgement call made after the fact.
    """
    f_star = float(prereg["prediction"]["at_f"])
    thresh = float(prereg["falsification"]["threshold"])
    lo, hi = (float(prereg["prediction"]["predicted_low"]),
              float(prereg["prediction"]["predicted_high"]))

    biased = df[(df.arm == "biased") & (np.isclose(df.contamination, f_star))]
    sym = df[(df.arm == "symmetric") & (np.isclose(df.contamination, f_star))]
    observed = float(biased.kl_naive.mean()) if len(biased) else float("nan")
    observed_sym = float(sym.kl_naive.mean()) if len(sym) else float("nan")

    per_k = {}
    for k, sub in biased.groupby("favoured_k"):
        per_k[int(k)] = {"kl_naive": float(sub.kl_naive.mean()),
                         "bound": float(sub.bound.mean()),
                         "lean": float(sub.synth_lean.mean()),
                         "below_bound": bool(sub.kl_naive.mean() < sub.bound.mean())}

    falsified = bool(observed < thresh)
    return {
        "at_f": f_star,
        "observed_naive_kl_biased": observed,
        "observed_naive_kl_symmetric_control": observed_sym,
        "v1_reference": float(prereg["prediction"]["v1_reference"]),
        "predicted_range": [lo, hi],
        "within_predicted_range": bool(lo <= observed <= hi),
        "falsification_threshold": thresh,
        "FALSIFIED": falsified,
        "per_favoured_k": per_k,
        "all_below_bound": all(v["below_bound"] for v in per_k.values()) if per_k else None,
        "verdict": ("FALSIFIED — stop. The noise-cancellation diagnosis is wrong and every "
                    "downstream V2 experiment needs rethinking before it is run."
                    if falsified else
                    "NOT falsified — the bias axis produces the predicted corruption; "
                    "downstream V2 stages may proceed."),
    }


def make_e6b_figure(df, agg, prereg, path):
    """LEFT: the decisive comparison — biased vs symmetric synthetic content for a naive
    aggregator, against V1's measured value and the falsification threshold.
    RIGHT: the D3a claim — corruption scales with which goal the synth leans toward."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for arm, colour, label in [("biased", "firebrick",
                                "machine content that leans one way"),
                               ("symmetric", "C0",
                                "machine content with no lean (control)")]:
        d = df[df.arm == arm].groupby("contamination").kl_naive.agg(["mean", "std"])
        ax.errorbar(d.index, d["mean"], yerr=d["std"], fmt="-o", color=colour, lw=2,
                    capsize=3, label=label)
    d = df[df.arm == "biased"].groupby("contamination").kl_recovered.mean()
    ax.plot(d.index, d.values, "--s", color="darkgreen", lw=1.6,
            label="leaning content, but weighted by who made it")
    thresh = float(prereg["falsification"]["threshold"])
    ax.axhline(thresh, color="k", ls=":", lw=1.2)
    ax.text(0.02, thresh, f" the line this had to cross to count: {thresh}",
            va="bottom", fontsize=8)
    ax.axhline(float(prereg["prediction"]["v1_reference"]), color="grey", ls="-.", lw=1.0)
    ax.text(0.02, float(prereg["prediction"]["v1_reference"]),
            " what the earlier version measured: 0.066", va="top", fontsize=8, color="grey")
    ax.set(xlabel="share of the corpus that is machine-made (f)",
           ylabel="error in what people are believed to want (nats)",
           title="A lean pushes the error past the pre-set line.\n"
                 "Content with no lean stays under it.")
    ax.legend(fontsize=8)

    ax = axes[1]
    b = agg[agg.arm == "biased"]
    for k, sub in b.groupby("favoured_k"):
        sub = sub.sort_values("contamination")
        ax.plot(sub.contamination, sub.kl_naive, "-o", ms=4,
                label=f"machine leans toward purpose {int(k) + 1}  "
                      f"(people want it {POP_GOAL_DIST[int(k)]:.0%} of the time)")
    for k, sub in b.groupby("favoured_k"):
        sub = sub.sort_values("contamination")
        ax.plot(sub.contamination, sub.bound, ":", lw=1, color="grey", alpha=0.6)
    ax.set(xlabel="share of the corpus that is machine-made (f)",
           ylabel="error in what people are believed to want (nats)",
           title="The harder the machine leans, the more the read is skewed\n"
                 "(dotted lines were predicted before the run)")
    ax.legend(fontsize=7)

    fig.suptitle("E6b — Machine content with a lean of its own does not average away",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E6b — corpus corruption, biased synthetic content (STAGE 1)")
    ap.add_argument("--prereg-only", action="store_true",
                    help="write the pre-registration and exit, running no inference")
    ap.add_argument("--force-prereg", action="store_true",
                    help="deliberately reset an existing pre-registration (record as a deviation)")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    res_dir, _ = C.ensure_dirs(args.out)

    if args.prereg_only:
        payload = write_preregistration(cfg, res_dir / PREREG_NAME, force=args.force_prereg)
        print(f"Pre-registration written to {res_dir / PREREG_NAME}")
        print(f"  content hash: {payload['content_hash']}")
        print(f"  scanned {payload['seed_scan']['n']} seeds; selection rule:")
        print(f"    {payload['selection_rule']}")
        for d in payload["selected_draws"]:
            print(f"  seed {d['seed']:>5}  favours G{d['favoured_k']}  lean={d['lean']:.3f}  "
                  f"H={d['entropy']:.3f}  bound(f=0.8)={d['bounds'].get('0.8', float('nan')):.3f}")
        return

    workers = 1 if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed,
        force_prereg=args.force_prereg)

    verdict = json.loads((res_dir / "e6b_verdict.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 72)
    print("E6b STAGE-1 GATE")
    print("=" * 72)
    print(f"  naive KL at f={verdict['at_f']}   biased arm      : {verdict['observed_naive_kl_biased']:.4f}")
    print(f"                       symmetric control: {verdict['observed_naive_kl_symmetric_control']:.4f}")
    print(f"  V1 reference                          : {verdict['v1_reference']:.4f}")
    print(f"  predicted range {verdict['predicted_range']}  -> within: {verdict['within_predicted_range']}")
    print(f"  falsification threshold {verdict['falsification_threshold']} -> FALSIFIED: {verdict['FALSIFIED']}")
    print("\n  per favoured goal k:")
    for k, v in sorted(verdict["per_favoured_k"].items()):
        print(f"    G{k}: KL={v['kl_naive']:.4f}  bound={v['bound']:.4f}  "
              f"lean={v['lean']:.3f}  below bound={v['below_bound']}")
    print(f"\n  {verdict['verdict']}")


if __name__ == "__main__":
    main()
