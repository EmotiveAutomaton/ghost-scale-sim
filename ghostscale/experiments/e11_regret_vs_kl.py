"""E11 — Regret versus KL (V2 spec §2). STAGE 6. Pure post-processing, no simulation.

    "Not a separate simulation. A cross-cutting analysis over E6b, E7, E8 outputs: scatter
     KL(C_recovered || C_true) against behavioural regret, coloured by argmax preservation.

     PREDICTED: the relationship is weak and the argmax-flip boundary explains far more
     variance than KL magnitude. If so, report that KL is the wrong harm metric and say so
     plainly."

The test is stated as variance explained: fit regret ~ KL alone, regret ~ argmax alone, and
regret ~ both, and compare R^2. "Explains far more variance" is then a number rather than an
impression of a scatter plot.
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
from . import _common as C

# (file, kl column, regret column, argmax column, label)
SOURCES = [
    ("e6b_raw.csv", "kl_naive", "naive_regret", "naive_argmax_preserved", "E6b naive"),
    ("e6b_raw.csv", "kl_recovered", "weighted_regret", "weighted_argmax_preserved",
     "E6b provenance-weighted"),
    ("e10_raw.csv", "kl", "regret", "argmax_preserved", "E10 expertise"),
    ("e8_raw.csv", "kl", "regret", "argmax_preserved", "E8 recursive"),
]


def collect(res_dir: Path) -> pd.DataFrame:
    rows = []
    for fname, kl_col, reg_col, am_col, label in SOURCES:
        path = res_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {kl_col, reg_col, am_col} <= set(df.columns):
            continue
        rows.append(pd.DataFrame({
            "source": label,
            "kl": df[kl_col].astype(float),
            "regret": df[reg_col].astype(float),
            "argmax_preserved": df[am_col].astype(float),
        }))
    if not rows:
        return pd.DataFrame(columns=["source", "kl", "regret", "argmax_preserved"])
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out.kl) & np.isfinite(out.regret)
               & np.isfinite(out.argmax_preserved)].reset_index(drop=True)


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS fit of y on X (intercept added here)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - float(resid @ resid) / ss_tot)


def analyse(df: pd.DataFrame) -> dict:
    y = df.regret.to_numpy(float)
    kl = df.kl.to_numpy(float)
    am = df.argmax_preserved.to_numpy(float)
    r2_kl = _r2(y, kl[:, None])
    r2_am = _r2(y, am[:, None])
    r2_both = _r2(y, np.column_stack([kl, am]))
    rho = float(np.corrcoef(pd.Series(kl).rank(), pd.Series(y).rank())[0, 1])
    out = {
        "n": int(len(df)),
        "pearson_kl_regret": float(np.corrcoef(kl, y)[0, 1]),
        "spearman_kl_regret": rho,
        "r2_kl_only": r2_kl,
        "r2_argmax_only": r2_am,
        "r2_both": r2_both,
        "argmax_explains_more": bool(r2_am > r2_kl),
        "variance_ratio_argmax_over_kl": (r2_am / r2_kl) if r2_kl > 0 else float("inf"),
        "mean_regret_argmax_kept": float(y[am >= 0.5].mean()) if (am >= 0.5).any() else np.nan,
        "mean_regret_argmax_flipped": float(y[am < 0.5].mean()) if (am < 0.5).any() else np.nan,
    }
    out["conclusion"] = (
        "The argmax-flip boundary explains more variance in harm than KL magnitude. KL is the "
        "wrong harm metric; report regret." if out["argmax_explains_more"] else
        "KL magnitude explains at least as much variance as the argmax-flip boundary. The "
        "prediction did NOT hold; KL is not obviously the wrong metric here. Reported as measured.")
    return out


def run(cfg: Config, out_dir: Path | None = None, make_fig: bool = True, **_) -> dict:
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    df = collect(res_dir)
    if df.empty:
        raise RuntimeError("E11 found no upstream CSVs. Run E6b/E10/E8 first.")
    df.to_csv(res_dir / "e11_points.csv", index=False)
    stats = analyse(df)
    (res_dir / "e11_analysis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if make_fig:
        make_e11_figure(df, stats, fig_dir / "e11_regret_vs_kl.png")
    return stats


def make_e11_figure(df, stats, path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    kept = df[df.argmax_preserved >= 0.5]
    flip = df[df.argmax_preserved < 0.5]
    ax.scatter(kept.kl, kept.regret, s=16, alpha=0.5, color="C0", label="argmax preserved")
    ax.scatter(flip.kl, flip.regret, s=16, alpha=0.6, color="firebrick", label="argmax FLIPPED")
    ax.set(xlabel="KL(C_recovered || C_true)  [nats]", ylabel="behavioural regret",
           title=f"KL vs harm  (Spearman {stats['spearman_kl_regret']:+.2f})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(["KL only", "argmax only", "both"],
           [stats["r2_kl_only"], stats["r2_argmax_only"], stats["r2_both"]],
           color=["grey", "firebrick", "C0"])
    ax.set(ylabel="variance in regret explained  (R²)",
           title="Which predicts harm?")
    for i, v in enumerate([stats["r2_kl_only"], stats["r2_argmax_only"], stats["r2_both"]]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.suptitle("E11 — Is KL the right harm metric?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E11 — regret versus KL (cross-cutting analysis)")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    stats = run(cfg, out_dir=args.out)
    print("\nE11 — regret versus KL, pooled over E6b / E10 / E8:")
    for k in ("n", "pearson_kl_regret", "spearman_kl_regret", "r2_kl_only",
              "r2_argmax_only", "r2_both", "mean_regret_argmax_kept",
              "mean_regret_argmax_flipped"):
        v = stats[k]
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"\n  {stats['conclusion']}")


if __name__ == "__main__":
    main()
