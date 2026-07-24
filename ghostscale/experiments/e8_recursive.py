"""E8 — Recursive degradation across generations (V2 spec §2). STAGE 5. Run last, overnight.

    G_max = 4, contamination f in {0.0, 0.3, 0.6}, signal in {absent, honest}. Learner
    observers. Measure the C4 panel per generation, plus behavioural regret.

    PREDICTED  monotone degradation with generation at f > 0, attenuated by an honest signal,
               and SUPERLINEAR — generation 3's damage exceeds three times generation 1's,
               because a degraded reader produces degraded artifacts which further degrade
               the next reader.

    "Report the trend and its significance, not an equilibrium. With G_max = 4 the honest
     claim is 'a significant per-generation effect in the predicted direction', and that is
     sufficient. Do not extrapolate the curve."

That instruction is implemented rather than merely quoted: this module reports per-generation
SLOPES with t statistics and a quadratic term for the superlinearity claim, and it produces
no fitted curve beyond the observed generations.

N11 GATES THIS EXPERIMENT. The f=0 arm is included in every run precisely so the null is
measured on the same code path, in the same conditions, as the effect. See
``ghostscale/generations.py`` for the four ways the loop could leak and what is done about
each, and ``tests/test_nulls_v2.py::test_N11_*`` for the gate itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import Config
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..preregistration import POP_GOAL_DIST
from ..generations import run_chain, chain_trend
from ..figures import set_style
from . import _common as C

SIGNAL_MODES = {"absent": 0.0, "honest": 1.0}     # -> signing_rate


def _e8_worker(payload):
    (cfg_raw, cell_index, f, signal_mode, seed_rep, base_seed, g_max, n_creators,
     n_artifacts, n_observers, infer_steps, d_i, synth_seed) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                    # N7

    results = run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
                        contamination=f, signing_rate=SIGNAL_MODES[signal_mode],
                        honesty=1.0, g_max=g_max, n_creators=n_creators,
                        n_artifacts=n_artifacts, n_observers=n_observers,
                        infer_steps=infer_steps, d_i=d_i,
                        base_seed=base_seed * 7919 + seed_rep)
    recs = []
    for r in results:
        rec = {"contamination": f, "signal": signal_mode, "seed_rep": seed_rep,
               "generation": r.generation, "mean_expertise": r.mean_expertise,
               "mi_genuine": r.mi_genuine, "kl_payload": r.kl_payload,
               "mean_deep_genuine": r.mean_deep_genuine}
        rec.update({k: v for k, v in r.panel.items() if k not in ("mi_genuine",
                                                                  "mean_deep_genuine")})
        recs.append(rec)
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e8
    g_max = int(e.g_max)
    assert g_max >= 3, "V2 spec §4: G_max may not be reduced below 3 — two generations " \
                       "cannot establish a trend"
    n_creators = int(cfg.get("experiments.e8.n_creators_next", 20))
    n_artifacts, n_observers = int(e.n_artifacts), int(e.n_observers)
    infer_steps = int(cfg.get("experiments.e8.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e8.n_replications", 3))
    f_levels = list(cfg.get("experiments.e8.contamination_levels", [0.0, 0.3, 0.6]))
    d_i = float(cfg.get("experiments.e8.d", 0.0))
    synth_seed = int(cfg.get("experiments.e8.synth_draw_seed", 1))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads, ci = [], 0
    for f in f_levels:
        for mode in SIGNAL_MODES:
            for s in range(n_reps):
                payloads.append((cfg.raw, ci, f, mode, s, base_seed, g_max, n_creators,
                                 n_artifacts, n_observers, infer_steps, d_i, synth_seed))
            ci += 1
    recs = C.run_parallel(payloads, _e8_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e8_raw.csv", index=False)

    agg = (df.groupby(["contamination", "signal", "generation"])
             .agg(kl_payload=("kl_payload", "mean"), kl_sd=("kl_payload", "std"),
                  mi_genuine=("mi_genuine", "mean"),
                  mean_deep_genuine=("mean_deep_genuine", "mean"),
                  regret=("regret", "mean"), regret_sd=("regret", "std"),
                  argmax_preserved=("argmax_preserved", "mean"),
                  sycophancy=("sycophancy", "mean"))
             .reset_index())
    agg.to_csv(res_dir / "e8_summary.csv", index=False)

    trends = trend_table(df)
    trends.to_csv(res_dir / "e8_trends.csv", index=False)

    if make_fig:
        make_e8_figure(agg, trends, fig_dir / "e8_recursive.png")
    return agg


def trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition per-generation slopes with t statistics. This — not an equilibrium and
    not an extrapolation — is what E8 reports (V2 spec §2)."""
    rows = []
    for (f, mode), sub in df.groupby(["contamination", "signal"]):
        for key in ("kl_payload", "mi_genuine", "regret", "mean_deep_genuine"):
            g = sub["generation"].to_numpy(float)
            y = sub[key].to_numpy(float)
            ok = np.isfinite(y)
            g, y = g[ok], y[ok]
            if len(np.unique(g)) < 3:
                continue
            slope, intercept = np.polyfit(g, y, 1)
            resid = y - (slope * g + intercept)
            dof = max(len(g) - 2, 1)
            se = float(np.sqrt((resid @ resid) / dof / np.sum((g - g.mean()) ** 2)))
            quad = float(np.polyfit(g, y, 2)[0])
            rows.append({"contamination": f, "signal": mode, "metric": key,
                         "slope": float(slope), "se": se,
                         "t": float(slope / se) if se > 0 else np.nan,
                         "quadratic_term": quad,
                         "significant": bool(se > 0 and abs(slope / se) > 2.0)})
    return pd.DataFrame(rows)


def make_e8_figure(agg, trends, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    styles = {"absent": "--", "honest": "-"}

    ax = axes[0]
    for (f, mode), sub in agg.groupby(["contamination", "signal"]):
        sub = sub.sort_values("generation")
        ax.errorbar(sub.generation, sub.kl_payload, yerr=sub.kl_sd, capsize=2,
                    ls=styles[mode], marker="o", ms=4,
                    label=f"f={f}, signal {mode}")
    ax.set(xlabel="generation", ylabel="KL(C_recovered || C_true) [nats]",
           title="Payload degradation across generations\n(f=0 is the N11 null)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for (f, mode), sub in agg.groupby(["contamination", "signal"]):
        sub = sub.sort_values("generation")
        ax.plot(sub.generation, sub.regret, ls=styles[mode], marker="s", ms=4,
                label=f"f={f}, {mode}")
    ax.set(xlabel="generation", ylabel="behavioural regret",
           title="Harm, measured as regret\n(the headline harm claim)")
    ax.legend(fontsize=7)

    ax = axes[2]
    t = trends[trends.metric == "kl_payload"]
    labels = [f"f={r.contamination}\n{r.signal}" for r in t.itertuples()]
    colours = ["firebrick" if r.significant else "grey" for r in t.itertuples()]
    ax.bar(range(len(t)), t.slope, yerr=t.se, color=colours, capsize=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(t)), labels, fontsize=7)
    ax.set(ylabel="KL slope per generation",
           title="Per-generation trend and significance\n(red = |t| > 2)")

    fig.suptitle("E8 — Recursive degradation (trend, not equilibrium; no extrapolation)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E8 — recursive degradation across generations")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = 1 if args.workers is None else args.workers
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    trends = pd.read_csv(res_dir / "e8_trends.csv")

    print("\nE8 — per generation:")
    print(agg.round(4).to_string(index=False))
    print("\nE8 — per-generation trends (the reported claim):")
    print(trends[trends.metric.isin(["kl_payload", "regret"])].round(4).to_string(index=False))

    n11 = trends[(trends.contamination == 0.0) & (trends.metric == "kl_payload")]
    if len(n11):
        bad = n11[n11.significant]
        print(f"\n  N11 (zero-contamination recursion): "
              f"{'FAILED — the loop itself is lossy' if len(bad) else 'passed — no significant f=0 trend'}")


if __name__ == "__main__":
    main()
