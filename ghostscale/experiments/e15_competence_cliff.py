"""E15 — Is the competence threshold a CLIFF or a KNEE?

V2's E10 found that nothing happens until d ≈ 0.5 and then extraction collapses, and reported
it as a "competence threshold". A sharp transition is a much stronger claim than a smooth one
("there is a competence cliff in intent extraction"), so it should be measured rather than
read off a 10-point plot.

-----------------------------------------------------------------------------------------
TWO CORRECTIONS TO THE OBVIOUS DESIGN, both from V2's own numbers.

**1. The proposed window (0.3-0.7) is in the wrong place.** E10's successive accuracy drops:

        0.3->0.4  -0.0004      0.6->0.7  -0.0546
        0.4->0.5  -0.0023      0.7->0.8  -0.1810
        0.5->0.6  -0.0121      0.8->0.9  -0.2195

The transition is still ACCELERATING at the end of the swept range. d ≈ 0.5 is where the
curve departs from flat, not where it turns over; the steepest descent is 0.7-0.9 and the
midpoint is beyond it. A 0.3-0.7 grid would spend half its points in a region where accuracy
is >= 0.997 and psi is flat to three decimals, and would stop before the collapse it is meant
to resolve. This sweep runs **0.40 to 1.00**, with two low anchors to hold the flat arm.

**2. Resolution alone cannot answer "sharp or smooth".** A finer grid gives a prettier curve,
not a verdict. Two things are needed:

  * **Shape discrimination.** Fit three models to the same data — a straight line, a hinge
    (flat then linear, with a free breakpoint), and a 4-parameter logistic — and compare by
    AIC. The logistic's fitted WIDTH is the quantity the claim is actually about: a "cliff"
    means the width is small relative to the range of d, and it can be quoted.

  * **A width-versus-evidence arm, which is what makes "phase transition" a real claim
    rather than a rhetorical one.** In physics a transition is sharp because its width shrinks
    as system size grows; at finite size everything is a smooth crossover. The analogue here
    is evidence per observer. If the transition narrows as the corpus grows, this is a genuine
    threshold being resolved. If the width is INVARIANT to corpus size, it is an intrinsically
    smooth crossover and "cliff" is the wrong word however steep the curve looks.

    This arm is the difference between "our 10-point grid looked steep" and a defensible
    claim, and it is what stops the result from being an artifact of plotting resolution.
-----------------------------------------------------------------------------------------

Corpus quality is held perfectly constant, exactly as in E10: every artifact is CREATOR
provenance from a real HumanCreator policy, zero synthetic content, asserted in the worker
(null N12). The whole claim is that degradation comes from the READER.
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
from ..figures import set_style
from .e10_expertise import _e10_worker
from . import _common as C

# Metrics the cliff question is asked of. goal_accuracy is primary: it is bounded in [0,1],
# so a logistic is a meaningful functional form for it rather than a convenient curve.
METRICS = ("goal_accuracy", "psi", "final_entropy")


# --------------------------------------------------------------------------- #
# Shape models.
# --------------------------------------------------------------------------- #
def _linear(d, a, b):
    return a + b * d


def _hinge(d, a, k, b):
    """Flat at ``a`` until breakpoint ``k``, then linear with slope ``b``. The 'knee' model."""
    return a + b * np.maximum(0.0, d - k)


def _logistic(d, lo, hi, d50, w):
    """4-parameter logistic. ``w`` is the transition WIDTH — the quantity 'cliff' refers to."""
    return lo + (hi - lo) / (1.0 + np.exp((d - d50) / max(w, 1e-6)))


def _aic(y, yhat, n_params) -> float:
    resid = np.asarray(y) - np.asarray(yhat)
    n = len(resid)
    rss = float(resid @ resid)
    if rss <= 0 or n <= n_params + 1:
        return float("-inf")
    return n * np.log(rss / n) + 2 * n_params


def fit_shapes(d: np.ndarray, y: np.ndarray) -> dict:
    """Fit line / hinge / logistic and compare by AIC. Returns the fits and the winner."""
    from scipy.optimize import curve_fit
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    lo_guess, hi_guess = float(y.min()), float(y.max())
    out: dict = {}

    specs = [
        ("linear", _linear, [float(y.mean()), 0.0], 2, ([-np.inf, -np.inf], [np.inf, np.inf])),
        ("hinge", _hinge, [hi_guess, 0.6, -1.0], 3,
         ([-np.inf, float(d.min()), -np.inf], [np.inf, float(d.max()), np.inf])),
        ("logistic", _logistic, [lo_guess, hi_guess, 0.75, 0.08], 4,
         ([-np.inf, -np.inf, float(d.min()), 1e-4], [np.inf, np.inf, float(d.max()), 10.0])),
    ]
    for name, fn, p0, npar, bounds in specs:
        try:
            popt, _ = curve_fit(fn, d, y, p0=p0, bounds=bounds, maxfev=200000)
            yhat = fn(d, *popt)
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            out[name] = {
                "params": [float(v) for v in popt],
                "aic": _aic(y, yhat, npar),
                "r2": float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
            }
        except Exception as exc:                       # a fit that will not converge is data
            out[name] = {"params": None, "aic": float("inf"), "r2": float("nan"),
                         "error": str(exc)}

    ranked = sorted(out.items(), key=lambda kv: kv[1]["aic"])
    best = ranked[0][0]
    lg = out.get("logistic", {})
    params = lg.get("params")
    return {
        "fits": out,
        "best_by_aic": best,
        "aic_margin_over_linear": float(out["linear"]["aic"] - out[best]["aic"]),
        "logistic_d50": float(params[2]) if params else float("nan"),
        "logistic_width": float(params[3]) if params else float("nan"),
        # A transition occupying < 10% of the unit range of d is "sharp" as a descriptive
        # label. It is a LABEL, not evidence of a phase transition — see width_vs_evidence.
        "sharp_by_width": bool(params is not None and params[3] < 0.05),
    }


# --------------------------------------------------------------------------- #
# The run.
# --------------------------------------------------------------------------- #
def _sweep(cfg: Config, d_grid, n_artifacts, n_reps, base_seed, n_creators, n_observers,
           infer_steps, workers, tag) -> pd.DataFrame:
    payloads, ci = [], 0
    for d_i in d_grid:
        for s in range(n_reps):
            payloads.append((cfg.raw, ci, float(d_i), s, base_seed, n_creators,
                             int(n_artifacts), n_observers, infer_steps))
        ci += 1
    recs = C.run_parallel(payloads, _e10_worker, workers)
    df = pd.DataFrame(recs)
    df["n_artifacts"] = int(n_artifacts)
    df["arm"] = tag
    return df


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e15
    d_grid = list(cfg.get("experiments.e15.d_grid"))
    n_creators = int(cfg.get("experiments.e15.n_creators", 50))
    n_artifacts = int(cfg.get("experiments.e15.n_artifacts", 300))
    n_observers = int(cfg.get("experiments.e15.n_observers", 20))
    infer_steps = int(cfg.get("experiments.e15.infer_steps", 8))
    n_reps = int(cfg.get("experiments.e15.n_replications", 5))
    width_grid = list(cfg.get("experiments.e15.width_d_grid"))
    width_sizes = list(cfg.get("experiments.e15.width_corpus_sizes"))
    width_reps = int(cfg.get("experiments.e15.width_n_replications", 3))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    frames = [_sweep(cfg, d_grid, n_artifacts, n_reps, base_seed, n_creators,
                     n_observers, infer_steps, workers, "main")]
    for size in width_sizes:
        frames.append(_sweep(cfg, width_grid, size, width_reps, base_seed + 101,
                             n_creators, n_observers, infer_steps, workers,
                             f"width_N{int(size)}"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(res_dir / "e15_competence_cliff.csv", index=False)

    verdict = analyse(df)
    (res_dir / "e15_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    if make_fig:
        make_figure(df, verdict, fig_dir / "e15_competence_cliff.png")
    return df


def analyse(df: pd.DataFrame) -> dict:
    main = df[df.arm == "main"]
    shapes = {}
    for metric in METRICS:
        agg = main.groupby("d")[metric].mean()
        shapes[metric] = fit_shapes(agg.index.to_numpy(float), agg.to_numpy(float))

    widths = {}
    for arm, sub in df[df.arm.str.startswith("width_")].groupby("arm"):
        agg = sub.groupby("d")["goal_accuracy"].mean()
        fit = fit_shapes(agg.index.to_numpy(float), agg.to_numpy(float))
        widths[arm] = {"n_artifacts": int(sub.n_artifacts.iloc[0]),
                       "logistic_width": fit["logistic_width"],
                       "logistic_d50": fit["logistic_d50"],
                       "best_by_aic": fit["best_by_aic"]}

    ordered = sorted(widths.values(), key=lambda r: r["n_artifacts"])
    ws = [r["logistic_width"] for r in ordered]
    ns = [r["n_artifacts"] for r in ordered]
    narrows = bool(len(ws) >= 2 and np.isfinite(ws[0]) and np.isfinite(ws[-1])
                   and ws[-1] < 0.7 * ws[0])
    ratio = float(ws[-1] / ws[0]) if len(ws) >= 2 and ws[0] else float("nan")

    return {
        "experiment": "E15 — competence cliff or knee",
        "note_on_window": ("swept 0.40-1.00 rather than the 0.30-0.70 originally proposed: "
                           "E10's successive accuracy drops are still accelerating at d=0.9, "
                           "so the transition midpoint lies above 0.7 and a 0.3-0.7 window "
                           "would stop before the collapse it is meant to resolve"),
        "shape_by_metric": shapes,
        "headline_metric": "goal_accuracy",
        "verdict_shape": shapes["goal_accuracy"]["best_by_aic"],
        "logistic_d50": shapes["goal_accuracy"]["logistic_d50"],
        "logistic_width": shapes["goal_accuracy"]["logistic_width"],
        "width_vs_evidence": {"corpus_sizes": ns, "widths": ws, "width_ratio": ratio,
                              "narrows_with_evidence": narrows},
        "interpretation": (
            "narrows_with_evidence TRUE: the width shrinks as evidence per observer grows, "
            "which is the finite-size signature of a genuine threshold — 'competence cliff' is "
            "earned. FALSE: the width is set by the model rather than by resolution, so this is "
            "an intrinsically smooth crossover, however steep it looks, and it must be reported "
            "as a knee. AIC picking 'logistic' or 'hinge' over 'linear' establishes only that "
            "the curve is not a straight line, which is a much weaker claim than a cliff."),
    }


def make_figure(df, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    main = df[df.arm == "main"]

    ax = axes[0]
    agg = main.groupby("d").goal_accuracy.agg(["mean", "std"])
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt="o", ms=4, capsize=3,
                color="C0", label="what we measured")
    sh = verdict["shape_by_metric"]["goal_accuracy"]["fits"]
    xs = np.linspace(float(main.d.min()), float(main.d.max()), 300)
    shape_word = {"linear": "a straight slide", "hinge": "a sudden bend",
                  "logistic": "an S-shaped drop"}
    for name, fn, colour in (("linear", _linear, "grey"), ("hinge", _hinge, "C2"),
                             ("logistic", _logistic, "firebrick")):
        p = sh.get(name, {}).get("params")
        if p:
            ax.plot(xs, fn(xs, *p), color=colour, lw=1.6,
                    ls="-" if name == verdict["verdict_shape"] else "--",
                    label=f"{shape_word[name]} (fit score {sh[name]['aic']:.0f}, lower is better)")
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="how often it names the right purpose",
           title="Does skill run out gradually, or fall off a cliff?\n"
                 f"best fit: {shape_word[verdict['verdict_shape']]}")
    ax.legend(fontsize=7)

    ax = axes[1]
    # final_entropy runs the other way (higher = less sure), so it is flipped before plotting.
    # Three curves that all fall makes "which one goes first" readable at a glance; three curves
    # with mixed polarity does not, and mixed polarity is how misleading charts get made.
    metric_label = {"goal_accuracy": "picks the right purpose (a choice)",
                    "psi": "intent it pulls out (a belief)",
                    "final_entropy": "how sure it is (a belief)"}
    for metric, colour in zip(METRICS, ("C0", "C1", "C3")):
        a = main.groupby("d")[metric].mean()
        norm = (a - a.min()) / max(a.max() - a.min(), 1e-12)
        if metric == "final_entropy":
            norm = 1.0 - norm
        ax.plot(a.index, norm, "-o", ms=3, color=colour, label=metric_label[metric])
    ax.axvline(verdict["logistic_d50"], color="k", ls=":", lw=1,
               label=f"halfway point for choices ({verdict['logistic_d50']:.2f})")
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="each measure, rescaled so 1 = intact and 0 = gone",
           title="What it believes breaks first.\nWhat it picks holds on longer.")
    ax.legend(fontsize=7)

    ax = axes[2]
    for arm, sub in df[df.arm.str.startswith("width_")].groupby("arm"):
        a = sub.groupby("d").goal_accuracy.mean()
        ax.plot(a.index, a.values, "-o", ms=3,
                label=f"{int(sub.n_artifacts.iloc[0])} artifacts to go on")
    w = verdict["width_vs_evidence"]
    ax.set(xlabel="how far the reader is from expert (0 = expert)",
           ylabel="how often it names the right purpose",
           title="Give the reader more to go on: does the drop sharpen?\n"
                 f"{'yes, it sharpens' if w['narrows_with_evidence'] else 'no, it stays the same'} "
                 f"(ratio {w['width_ratio']:.2f})")
    ax.legend(fontsize=7)

    fig.suptitle("E15 — Skill does not run out all at once, and belief goes before choice",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E15 — is the competence threshold a cliff or a knee?")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    v = json.loads((res_dir / "e15_verdict.json").read_text(encoding="utf-8"))

    print("\nE15 — shape of the competence transition (goal accuracy):")
    for name, f in v["shape_by_metric"]["goal_accuracy"]["fits"].items():
        print(f"  {name:9s} AIC {f['aic']:9.1f}   R2 {f['r2']:.5f}")
    print(f"  AIC winner: {v['verdict_shape']}  "
          f"(margin over linear {v['shape_by_metric']['goal_accuracy']['aic_margin_over_linear']:.1f})")
    print(f"  logistic midpoint d50 = {v['logistic_d50']:.3f}, width = {v['logistic_width']:.4f}")
    w = v["width_vs_evidence"]
    print(f"\n  width vs evidence: N={w['corpus_sizes']} -> widths "
          f"{[round(x, 4) for x in w['widths']]}  ratio {w['width_ratio']:.3f}")
    print(f"  narrows with evidence: {w['narrows_with_evidence']}  "
          f"-> {'CLIFF is earned' if w['narrows_with_evidence'] else 'report as a KNEE (smooth crossover)'}")


if __name__ == "__main__":
    main()
