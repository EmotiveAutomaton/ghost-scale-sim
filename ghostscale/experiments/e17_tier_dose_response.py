"""E17 — Confident fabrication across all four Ghost Scale tiers.

E2 is the headline result, and it was run on a BINARY contrast: true origin
{CREATOR, GHOST} x declared signal {SIG_CREATOR, SIG_GHOST}. It established the
dissociation that signals hallucination — low within-observer entropy (confidence) with high
between-observer entropy (disagreement) when goalless content is passed off as human:

    CREATOR / SIG_CREATOR   within 0.000   between 0.000
    GHOST   / SIG_CREATOR   within 0.090   between 1.379   <- confident fabrication
    GHOST   / SIG_GHOST     within 1.293   between 1.378   <- uncertainty correctly preserved

A binary contrast shows the effect exists. It does not show that the effect is *graded by
opacity*, which is the claim the four-tier Ghost Scale actually makes. This runs the same
design across every tier, so the headline becomes a dose-response curve and the four-tier
design is validated rather than assumed.

-----------------------------------------------------------------------------------------
DESIGN. 4 true provenances x 2 declared signals:

  * ``claimed_human``  — declared SIG_CREATOR whatever the truth. The fabrication condition;
    for CREATOR this coincides with the truthful cell and is the zero-dose control.
  * ``truthful``       — declared signal matches the true tier. The honest-labelling arm, and
    the reference showing what correct disclosure buys at each opacity.

The x-axis is **alpha**, the intent-transmission coefficient, NOT the tier index: alpha is
{CREATOR 1.00, POLISHED 0.95, CURATOR 0.60, GHOST 0.05}, which is deliberately not evenly
spaced. Plotting against the tier index would manufacture a straight line out of an uneven
dose spacing, and POLISHED at 0.95 sits so close to CREATOR that the four tiers supply
roughly three distinguishable dose levels. Stated here so the curve is not over-read.

PREDICTED, if the Ghost Scale's central commitment holds: under ``claimed_human``,
between-observer disagreement rises monotonically as alpha falls, while within-observer
entropy stays LOW — observers grow more confident and less agreeing as the content they are
reading carries less intent. Under ``truthful``, within-observer entropy tracks alpha instead:
told the truth, the reader correctly becomes uncertain rather than confidently wrong.

FALSIFIED if disagreement under ``claimed_human`` is flat in alpha, or if it is driven by
within-observer entropy rising in step — that would be ordinary uncertainty rather than
fabrication, and the dissociation would not be graded.
-----------------------------------------------------------------------------------------
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
from ..constants import (PROVENANCE_NAMES, SIGNAL_NAMES, TIER_OPACITY, TRUTHFUL_SIGNAL,
                         SIG_CREATOR)
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..creators import build_creator_bank
from ..environment import Environment, Artifact
from ..observer import make_observer, rollout_observer
from .. import metrics
from ..figures import set_style
from . import _common as C

TIERS = [0, 1, 2, 3]                       # CREATOR, POLISHED, CURATOR, GHOST
LABELLINGS = ("claimed_human", "truthful")


def declared_signal_for(labelling: str, provenance: int) -> int:
    return SIG_CREATOR if labelling == "claimed_human" else TRUTHFUL_SIGNAL[provenance]


def _e17_worker(payload):
    (cfg_raw, cell_index, provenance, labelling, seed_rep, base_seed,
     n_obs, T, k, true_goal) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, kappa=float(cfg.signal_model.kappa))
    assert_preferences_zero(gm.C)                     # N7
    bank = build_creator_bank(cfg, gm)
    world_rng = np.random.default_rng(base_seed * 31 + seed_rep)
    env = Environment(cfg, gm, rng_world=world_rng, creator_bank=bank)
    signal = declared_signal_for(labelling, provenance)

    recs = []
    for i in range(n_obs):
        r = C.observer_rng(base_seed, cell_index, seed_rep, i)
        agent = make_observer(gm, cfg, r)
        # Declared signal is a DESIGN factor, exactly as in E2: build the artifact directly.
        artifact = Artifact(provenance=provenance, goal=true_goal, declared_signal=signal)
        res = rollout_observer(agent, artifact, env, cfg, r, T, force_deep_k=k,
                               kappa=float(cfg.signal_model.kappa))
        recs.append({
            "true_provenance": PROVENANCE_NAMES[provenance],
            "alpha": TIER_OPACITY[PROVENANCE_NAMES[provenance]],
            "labelling": labelling,
            "declared_signal": SIGNAL_NAMES[signal],
            "seed_rep": seed_rep, "observer": i,
            "true_goal": true_goal, "modal_goal": res.modal_goal,
            "correct": int(res.modal_goal == true_goal),
            "within_entropy": float(res.within_entropy[-1]),
            "final_posterior": res.final_goal_posterior.tolist(),
        })
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    n_obs = int(cfg.get("experiments.e17.n_observers", cfg.run.n_observers))
    T = int(cfg.run.n_timesteps)
    n_seeds = int(cfg.get("experiments.e17.n_seeds", cfg.run.n_seeds))
    k = int(cfg.get("experiments.e17.forced_deep_k", cfg.experiments.e2.forced_deep_k))
    true_goal = int(cfg.get("experiments.e17.true_goal", 1))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for prov in TIERS:
        for labelling in LABELLINGS:
            for s in range(n_seeds):
                payloads.append((cfg.raw, ci, prov, labelling, s, base_seed,
                                 n_obs, T, k, true_goal))
            ci += 1
    recs = C.run_parallel(payloads, _e17_worker, workers)

    points = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != "final_posterior"}
                           for r in recs])
    points.to_csv(res_dir / "e17_points.csv", index=False)

    rows = []
    for prov in TIERS:
        tier = PROVENANCE_NAMES[prov]
        for labelling in LABELLINGS:
            sub = [r for r in recs if r["true_provenance"] == tier
                   and r["labelling"] == labelling]
            withins, betweens, accs = [], [], []
            for s in range(n_seeds):
                post = [np.asarray(r["final_posterior"]) for r in sub if r["seed_rep"] == s]
                if not post:
                    continue
                withins.append(metrics.mean_within_observer_entropy(post))
                betweens.append(metrics.between_observer_entropy(post))
                accs.append(np.mean([r["correct"] for r in sub if r["seed_rep"] == s]))
            rows.append({
                "true_provenance": tier, "alpha": TIER_OPACITY[tier], "labelling": labelling,
                "within": float(np.mean(withins)), "within_sd": float(np.std(withins)),
                "between": float(np.mean(betweens)), "between_sd": float(np.std(betweens)),
                "accuracy": float(np.mean(accs)),
                # The dissociation gap IS the fabrication signature: disagreement not
                # accompanied by uncertainty.
                "fabrication_gap": float(np.mean(betweens) - np.mean(withins)),
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(res_dir / "e17_tier_stats.csv", index=False)

    verdict = analyse(stats)
    (res_dir / "e17_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if make_fig:
        make_figure(stats, verdict, fig_dir / "e17_tier_dose_response.png")
    return stats


def _trend(x, y, tol: float = 1e-3) -> dict:
    """Trend of y against alpha, reported two ways because ties matter here.

    Spearman is the natural monotonicity statistic, but it PENALISES TIES, and this design
    produces a structural tie: CREATOR (alpha 1.00) and POLISHED (alpha 0.95) are close enough
    that both sit at exactly 0.000 on several measures. Four tiers then supply three
    distinguishable dose levels, and a perfectly monotone result scores only about -0.8.

    So ``weakly_monotone`` is reported alongside: does the quantity ever move the WRONG way as
    opacity increases? That is the claim the Ghost Scale actually makes — it does not require
    every tier to be distinguishable from its neighbour, only that nothing reverses.

    ``tol`` is 1e-3 nats, which is a MEASUREMENT tolerance rather than a float one. The
    between-observer sd in this experiment is ~0.016 nats and the within-observer sd ~0.003,
    so a step of 1e-3 is below anything the design can resolve and cannot mask a real
    reversal. (The CREATOR->POLISHED step on the fabrication gap comes out at -2.3e-6 — four
    orders of magnitude below noise — and a float-epsilon tolerance would score that as a
    violation of monotonicity, which would be an artifact of arithmetic, not a finding.)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return {"slope": float("nan"), "spearman": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    order = np.argsort(-x)                       # decreasing alpha = increasing opacity
    y_ord = y[order]
    diffs = np.diff(y_ord)
    return {"slope": float(slope), "spearman": rho,
            "monotone": bool(abs(rho) > 0.99),
            "weakly_monotone_increasing_with_opacity": bool(np.all(diffs >= -tol)),
            "n_distinguishable_levels": int(len(np.unique(np.round(y, 3)))),
            "steps_with_opacity": [float(v) for v in diffs]}


def analyse(stats: pd.DataFrame) -> dict:
    out = {}
    for labelling in LABELLINGS:
        s = stats[stats.labelling == labelling].sort_values("alpha")
        out[labelling] = {
            "between_vs_alpha": _trend(s.alpha, s.between),
            "within_vs_alpha": _trend(s.alpha, s.within),
            "gap_vs_alpha": _trend(s.alpha, s.fabrication_gap),
            "by_tier": s[["true_provenance", "alpha", "within", "between",
                          "fabrication_gap", "accuracy"]].round(4).to_dict("records"),
        }
    ch = out["claimed_human"]
    # Weak monotonicity is the criterion, not Spearman: see _trend on the structural
    # CREATOR/POLISHED tie. Spearman is still reported for both quantities.
    graded = bool(ch["between_vs_alpha"]["weakly_monotone_increasing_with_opacity"]
                  and ch["between_vs_alpha"]["slope"] < 0)
    dissociated = bool(ch["gap_vs_alpha"]["weakly_monotone_increasing_with_opacity"]
                       and ch["gap_vs_alpha"]["slope"] < 0)
    return {
        "experiment": "E17 — confident fabrication across the four Ghost Scale tiers",
        "x_axis_note": ("plotted against alpha {1.00, 0.95, 0.60, 0.05}, not tier index: the "
                        "doses are unevenly spaced and POLISHED sits within 0.05 of CREATOR, "
                        "so four tiers supply roughly three distinguishable dose levels"),
        "arms": out,
        "dose_response_graded": graded,
        "dissociation_graded": dissociated,
        "verdict": (
            "confirmed: disagreement under a human claim rises monotonically as intent "
            "transmission falls, and the fabrication gap rises with it — the four-tier design "
            "is carrying real signal rather than relabelling a binary contrast"
            if graded and dissociated else
            "NOT confirmed as graded: the effect exists at the extremes but is not monotone in "
            "alpha across the intermediate tiers; report the binary contrast as the result and "
            "the four-tier gradation as unsupported"),
    }


def make_figure(stats, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colours = {"claimed_human": "firebrick", "truthful": "C0"}
    lab_name = {"claimed_human": "all of it labelled human-made",
                "truthful": "labelled honestly"}

    ax = axes[0]
    for labelling, s in stats.groupby("labelling"):
        s = s.sort_values("alpha")
        ax.errorbar(s.alpha, s.between, yerr=s.between_sd, marker="o", ms=5, capsize=3,
                    color=colours[labelling], label=lab_name[labelling])
    ax.invert_xaxis()
    ax.set(xlabel="how much of the maker's intent survives\n"
                  "(left = all of it, right = almost none)",
           ylabel="how much readers disagree with each other (nats)",
           title="The more hollow the work, the more readers disagree")
    ax.legend(fontsize=8)
    # CREATOR and POLISHED sit at alpha 1.00 and 0.95, close enough that their labels collide;
    # alternating the vertical offset separates them without moving the points.
    for i, (_, r) in enumerate(stats[stats.labelling == "claimed_human"]
                               .sort_values("alpha", ascending=False).iterrows()):
        # r["between"], not r.between: on a Series that attribute is pandas' between() method.
        ax.annotate(r["true_provenance"], (r["alpha"], r["between"]), fontsize=6,
                    xytext=(0, 16 if i % 2 else 6), textcoords="offset points", ha="center")

    ax = axes[1]
    for labelling, s in stats.groupby("labelling"):
        s = s.sort_values("alpha")
        ax.errorbar(s.alpha, s.within, yerr=s.within_sd, marker="s", ms=5, capsize=3,
                    color=colours[labelling], label=lab_name[labelling])
    ax.invert_xaxis()
    ax.set(xlabel="how much of the maker's intent survives\n"
                  "(left = all of it, right = almost none)",
           ylabel="how unsure each reader is (nats)",
           title="Told the truth, readers get unsure.\nTold it is human, they stay certain.")
    ax.legend(fontsize=8)

    ax = axes[2]
    for labelling, s in stats.groupby("labelling"):
        s = s.sort_values("alpha")
        ax.plot(s.alpha, s.fabrication_gap, "-o", ms=5, color=colours[labelling],
                label=lab_name[labelling])
    ax.axhline(0, color="k", lw=1)
    ax.invert_xaxis()
    ax.set(xlabel="how much of the maker's intent survives\n"
                  "(left = all of it, right = almost none)",
           ylabel="disagreement minus doubt (nats)",
           title="Making things up, measured:\nreaders who disagree and are sure anyway")
    ax.legend(fontsize=8)

    fig.suptitle("E17 — This is a dial, not a switch: partly-hollow work does "
                 "partly the damage", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E17 — confident fabrication across the four tiers")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    stats = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e17_verdict.json").read_text(encoding="utf-8"))
    print("\nE17 — within (confidence) and between (disagreement) by tier:")
    print(stats[["true_provenance", "alpha", "labelling", "within", "between",
                 "fabrication_gap", "accuracy"]].round(4).to_string(index=False))
    ch = v["arms"]["claimed_human"]
    print(f"\n  claimed_human: between ~ alpha slope {ch['between_vs_alpha']['slope']:+.4f}, "
          f"spearman {ch['between_vs_alpha']['spearman']:+.3f}")
    print(f"  claimed_human: gap     ~ alpha slope {ch['gap_vs_alpha']['slope']:+.4f}, "
          f"spearman {ch['gap_vs_alpha']['spearman']:+.3f}")
    print(f"\n  {v['verdict']}")


if __name__ == "__main__":
    main()
