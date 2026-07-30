#!/usr/bin/env python
"""Four diagnostic figures, drawn from the diagnostics CSVs.

    python scripts/make_diagnostic_figures.py

Writes to ``figures/diagnostics/``. These are RESEARCH figures, not distribution slides: they exist so
a reader can check a claim rather than absorb one, so they carry axes, identity lines, thresholds and
grid points rather than one idea per page. The social palette rules in
``scripts/make_social_figures.py`` deliberately do not apply here, and the two scripts never write to
each other's directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO / "results" / "diagnostics"
OUT = REPO / "figures" / "diagnostics"


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {(OUT / name).relative_to(REPO)}")


# --------------------------------------------------------------------------- #
def fig_p1():
    """Four panels, recovered against true, identity line drawn. The spec's section 4 output."""
    path = DIAG / "p1_recovery" / "recovery.csv"
    v = _json(DIAG / "p1_recovery.json")
    if not path.exists():
        print("  SKIPPED p1_recovery.png: run_diagnostics.py --stage 2 has not run")
        return
    df = pd.read_csv(path)
    scores = {(s["parameter"], s.get("dataset")): s for s in (v or {}).get("scores", [])}

    panels = [("kappa", "honest labels throughout"),
              ("kappa", "machine work passed off as human"),
              ("omega", None), ("mu", None), ("theta (lambda)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0))
    for ax, (param, dataset) in zip(np.atleast_1d(axes), panels):
        sub = df[df.parameter == param]
        if dataset is not None:
            sub = sub[sub.dataset == dataset]
        if not len(sub):
            ax.set_axis_off()
            continue
        t = sub["true"].values
        m = sub["recovered_mean"].values
        e = sub["recovered_sem"].values
        lo, hi = float(min(t.min(), m.min())), float(max(t.max(), m.max()))
        pad = 0.05 * max(hi - lo, 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.2, color="0.5",
                label="perfect recovery")
        ax.errorbar(t, m, yerr=e, marker="o", capsize=3, lw=1.6, color="C0")
        s = scores.get((param, dataset)) or {}
        cls = s.get("classification", "?")
        rho = s.get("rank_correlation", float("nan"))
        slope = s.get("slope", float("nan"))
        usable = s.get("usable_range_fraction", float("nan"))
        title = param if dataset is None else f"{param}\n({dataset})"
        ax.set_title(f"{title}\n{cls}", fontsize=9, fontweight="bold")
        ax.set_xlabel("true value")
        ax.set_ylabel("recovered")
        ax.text(0.03, 0.97,
                ("rho " + ("n/a" if not np.isfinite(rho) else f"{rho:.2f}")
                 + f"\nslope {slope:.2f}\nusable {100 * usable:.0f}%"),
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7"))
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("P-1 — can a known parameter value be read back out of the model's own data?",
                 fontweight="bold")
    save(fig, "p1_recovery.png")


# --------------------------------------------------------------------------- #
def fig_p2():
    """Accuracy and uptake variance against difficulty, with the target band marked."""
    path = DIAG / "p2_difficulty" / "difficulty.csv"
    v = _json(DIAG / "p2_difficulty.json")
    if not path.exists():
        print("  SKIPPED p2_difficulty.png: run_diagnostics.py --stage 3 has not run")
        return
    df = pd.read_csv(path)
    band = (v or {}).get("criteria", {}).get("target_band", [0.55, 0.85])
    factor = (v or {}).get("criteria", {}).get("uptake_variance_factor", 1.25)
    base = float((v or {}).get("default_cell", {}).get("uptake_sd", np.nan))

    d = df[df.knob == "reader inexpertise"].sort_values("value")
    t = df[df.knob == "observations before the decision"].sort_values("value")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    ax.axhspan(band[0], band[1], color="0.85", label=f"target band {band[0]}-{band[1]}")
    ax.plot(d.value, d.accuracy, marker="o", label="reader inexpertise (added knob)")
    if len(t):
        ax2 = ax.twiny()
        ax2.plot(t.value, t.accuracy, marker="s", ls=":", color="C1",
                 label="observations before deciding")
        ax2.set_xlabel("observations before deciding", color="C1")
        ax2.tick_params(axis="x", colors="C1")
    dead = df[df.knob.str.startswith("confirmatory")]
    for _, r in dead.iterrows():
        ax.scatter([0.0], [r.accuracy], marker="x", s=90, color="firebrick", zorder=5)
    ax.set(xlabel="reader inexpertise", ylabel="reads the right purpose", ylim=(-0.03, 1.05),
           title="Is there a difficulty regime?\n(x marks the spec's two dead knobs, at ceiling)")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ax.plot(d.value, d.uptake_sd, marker="o", color="C2")
    if np.isfinite(base):
        ax.axhline(base, ls=":", color="k", lw=1, label="variance at the default")
        ax.axhline(factor * base, ls="--", color="firebrick", lw=1,
                   label=f"{factor}x the default (the requirement)")
    ax.set(xlabel="reader inexpertise", ylabel="spread of uptake across readers",
           title="Does uptake have room to move?")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    ax.plot(d.accuracy, d.uptake, marker="o", color="C3")
    for _, r in d.iterrows():
        if r.value in (0.0, 0.5, 0.8, 0.85, 0.9, 1.0):
            ax.annotate(f"d={r.value:g}", (r.accuracy, r.uptake), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set(xlabel="reads the right purpose", ylabel="uptake",
           title="Uptake against recovery\n(the same axis D-2 maps in full)")
    fig.suptitle("P-2 — the model's own difficulty axis", fontweight="bold")
    save(fig, "p2_difficulty.png")


# --------------------------------------------------------------------------- #
def fig_d2():
    """The uptake response curve, which is the figure the flat depth result needed."""
    path = DIAG / "d2_uptake" / "uptake_curve.csv"
    v = _json(DIAG / "d2_uptake.json")
    if not path.exists():
        print("  SKIPPED d2_uptake.png: run_diagnostics.py --stage 4 has not run")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    ax.plot(df.inexpertise, df.uptake_ungated, marker="o", label="uptake")
    ax.fill_between(df.inexpertise, df.uptake_ungated - df.uptake_ungated_sd,
                    df.uptake_ungated + df.uptake_ungated_sd, alpha=0.15)
    m = (v or {}).get("minimum_at_inexpertise")
    if m is not None:
        ax.axvline(m, ls="--", color="firebrick", lw=1.2, label=f"minimum at d = {m:g}")
    ax.set(xlabel="reader inexpertise", ylabel="uptake",
           title="Uptake is U-shaped, not a slope")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(df.accuracy, df.uptake_ungated, marker="o", color="C3")
    ax.set(xlabel="reads the right purpose", ylabel="uptake",
           title="The same curve against recovery quality\n(a flat regression on this returns a null)")

    ax = axes[2]
    ax.plot(df.inexpertise, df.uptake_of_the_correct, marker="o", label="readers who are right")
    ax.plot(df.inexpertise, df.uptake_of_the_confidently_wrong, marker="s", ls="--",
            label="readers who are confidently wrong")
    ax.set(xlabel="reader inexpertise", ylabel="uptake",
           title="Being wrong moves you almost as far as being right")
    ax.legend(fontsize=7)
    fig.suptitle("D-2 — the shape of the uptake measure", fontweight="bold")
    save(fig, "d2_uptake.png")


# --------------------------------------------------------------------------- #
def fig_d1():
    """The two evidence channels and the crossover between them."""
    v = _json(DIAG / "d1_channels.json")
    if not v:
        print("  SKIPPED d1_channels.png: run_diagnostics.py --stage 1 has not run")
        return
    curve = pd.DataFrame(v["kappa_curve_on_the_conflicting_cell"])
    traj = v.get("measured_trajectory", [])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.plot(curve.kappa, curve.label_llr, marker="o", label="the label, per glance")
    ax.plot(curve.kappa, -curve.content_llr, marker="s", ls="--",
            label="the content, per glance (sign flipped)")
    cross = v.get("crossover_kappa")
    if cross is not None and np.isfinite(cross):
        ax.axvline(cross, ls=":", color="firebrick", lw=1.4, label=f"crossover at {cross:.3f}")
    ax.axvline(v["default_kappa"], ls="-", color="0.5", lw=1, label="the default")
    ax.set(xlabel="trust in the label", ylabel="evidence per glance (nats)",
           title="Two channels, one race\n(machine work passed off as human)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for row in traj:
        ts = sorted(int(k) for k in row["believes_the_label_at_t"])
        ys = [row["believes_the_label_at_t"][str(t)] for t in ts]
        ax.plot(ts, ys, marker="o", label=f"trust {row['kappa']:g}")
    ax.axhline(0.5, ls=":", color="k", lw=1)
    ax.set(xlabel="glances", ylabel="believes the label", xscale="log", ylim=(-0.03, 1.05),
           title="And the run length decides who wins")
    ax.legend(fontsize=7)
    fig.suptitle("D-1 — what the reader's belief about provenance is made of", fontweight="bold")
    save(fig, "d1_channels.png")


def main():
    style()
    print("building diagnostic figures:")
    fig_d1()
    fig_p1()
    fig_p2()
    fig_d2()
    print("done. figures/ and figures/social/ untouched.")


if __name__ == "__main__":
    main()
