"""E12 — Leak versus sample size (V3 spec §1 C2). STAGE 1. Runs FIRST; E8 is gated on it.

C1 (population-averaged seeding) is a patch. E12 is what makes it a *diagnosis*: if the V2
leak is finite-sample estimation error, then at zero contamination with an honest signal —
the exact condition that failed N11 in V2 — the leak must SHRINK AS THE SAMPLE GROWS, and
shrink in the specific way finite-sample error shrinks.

Two sub-sweeps, because C1 and C2 are two different claims:

  * **N sweep** — ``artifacts_per_generation`` across the swept range, with C1 averaging and
    without (V2's single-observer seeding), at f = 0 with an honest signal. This is the
    finite-sample diagnosis, and N13 is computed from its without-averaging arm.

  * **M sweep** — observers per generation at fixed N, averaging on. This is C1's own claim.

WHY THE M SWEEP EXISTS, AND WHAT IT IS EXPECTED TO SHOW (decision D2). V3 §1 C1 predicts that
averaging over M observers makes the slope "an additional factor of ~1/M smaller at every N".
That is true of the observer-side term and FALSE of the rest, and the code says why: a
generation draws its corpus ONCE (``run_generation`` -> ``env.draw_corpus``) and all M
observers read that same corpus. Only the feature emissions are redrawn per observer. So the
corpus's own goal-composition and signal noise is common-mode across the population and
averaging cannot touch it. The 1/M gain therefore has a FLOOR that only larger N can lower.

    Predicted: the M sweep's slope falls with M and then flattens. Where it flattens is the
    corpus-noise floor at that N, and it is a real quantity worth measuring rather than an
    inconvenience — it is the reason C2's sample-size sweep is load-bearing rather than
    confirmatory decoration on C1.

FALSIFICATION (V3 §1 C2, verbatim in force): if the without-averaging slope does NOT shrink
with N, the leak is not finite-sample estimation error, the entire V3 diagnosis is wrong, and
E8 must not be run until the structural reason is found. That verdict is written to
``results/e12_threshold.json`` and ``run_all_v3.py`` refuses to proceed on it.

N13 (V3 §3, as amended by decision D8) is computed here rather than only in the test suite,
because the number it produces — the exponent of the leak's decline in N — is a REPORTED
RESULT, not merely a gate. See ``ghostscale/prereg_v3.py`` for why it tests the 1/N exponent
instead of monotonicity.
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
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..preregistration import POP_GOAL_DIST
from ..generations import run_chain, chain_trend
from ..figures import set_style
from .. import prereg_v3 as P3
from . import _common as C


def _e12_worker(payload):
    (cfg_raw, sweep, n_artifacts, n_observers, averaging, seed_rep, base_seed, g_max,
     n_creators, infer_steps, d_i, synth_seed, contamination, signing_rate) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                    # N7

    results = run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
                        contamination=contamination, signing_rate=signing_rate,
                        honesty=1.0, g_max=g_max, n_creators=n_creators,
                        n_artifacts=n_artifacts, n_observers=n_observers,
                        infer_steps=infer_steps, d_i=d_i,
                        base_seed=base_seed * 7919 + seed_rep,
                        population_average_seed=bool(averaging))
    recs = []
    for r in results:
        recs.append({"sweep": sweep, "n_artifacts": n_artifacts,
                     "m_observers": n_observers, "averaging": bool(averaging),
                     "seed_rep": seed_rep, "generation": r.generation,
                     "kl_payload": r.kl_payload,
                     "creator_col_kl": r.creator_col_kl,
                     "eff_sample_count": r.eff_sample_count,
                     "mi_genuine": r.mi_genuine,
                     "mean_deep_genuine": r.mean_deep_genuine})
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True) -> pd.DataFrame:
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e12
    g_max = int(e.g_max)
    n_sweep = list(cfg.get("experiments.e12.n_artifacts_sweep", [100, 300, 1000, 3000, 10000]))
    m_sweep = list(cfg.get("experiments.e12.m_observers_sweep", [1, 5, 20]))
    m_sweep_n = int(cfg.get("experiments.e12.m_sweep_n_artifacts", 1000))
    n_observers = int(e.n_observers)
    n_creators = int(cfg.get("experiments.e12.n_creators_next", 20))
    infer_steps = int(cfg.get("experiments.e12.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e12.n_replications", 3))
    d_i = float(cfg.get("experiments.e12.d", 0.0))
    synth_seed = int(cfg.get("experiments.e12.synth_draw_seed", 17))
    f = float(cfg.get("experiments.e12.contamination", 0.0))
    signing = float(cfg.get("experiments.e12.signing_rate", 1.0))
    res_dir, fig_dir = C.ensure_dirs(out_dir)

    payloads = []
    for n_art in n_sweep:                      # the N sweep, both arms
        for averaging in (True, False):
            for s in range(n_reps):
                payloads.append((cfg.raw, "N", n_art, n_observers, averaging, s, base_seed,
                                 g_max, n_creators, infer_steps, d_i, synth_seed, f, signing))
    for m in m_sweep:                          # the M sweep, averaging on, N fixed
        for s in range(n_reps):
            payloads.append((cfg.raw, "M", m_sweep_n, m, True, s, base_seed,
                             g_max, n_creators, infer_steps, d_i, synth_seed, f, signing))

    recs = C.run_parallel(payloads, _e12_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e12_leak_vs_samplesize.csv", index=False)

    slopes = slope_table(df)
    slopes.to_csv(res_dir / "e12_slopes.csv", index=False)
    verdict = build_verdict(slopes)
    (res_dir / "e12_threshold.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    if make_fig:
        make_e12_figure(df, slopes, verdict, fig_dir / "e12_leak_convergence.png")
    return slopes


def slope_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell per-generation slope of the payload KL, with the pre-registered N11 verdict.

    Replications are POOLED into one slope per cell rather than averaged across per-rep
    slopes, so the standard error reflects the replication spread — which is the quantity
    the N11 t statistic is supposed to be measured against.
    """
    rows = []
    for (sweep, n_art, m, averaging), sub in df.groupby(
            ["sweep", "n_artifacts", "m_observers", "averaging"]):
        for key in ("kl_payload", "creator_col_kl"):
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
            t = float(slope / se) if se > 0 else float("nan")
            row = {"sweep": sweep, "n_artifacts": int(n_art), "m_observers": int(m),
                   "averaging": bool(averaging), "metric": key,
                   "slope": float(slope), "se": se, "t": t,
                   "mean_eff_sample_count": float(sub["eff_sample_count"].mean()),
                   "n_points": int(len(g))}
            if key == "kl_payload":
                v = P3.n11_verdict(slope, t)
                row.update({"n11_passed": v["passed"], "n11_t_ok": v["t_ok"],
                            "n11_slope_ok": v["slope_ok"]})
            rows.append(row)
    return pd.DataFrame(rows)


def build_verdict(slopes: pd.DataFrame) -> dict:
    """N13 and the sample-size decision, computed under the pre-registered criteria."""
    n_arm = slopes[(slopes.sweep == "N") & (slopes.metric == "kl_payload")]
    without = n_arm[~n_arm.averaging].sort_values("n_artifacts")
    with_avg = n_arm[n_arm.averaging].sort_values("n_artifacts")

    fit = P3.loglog_slope_fit(without.n_artifacts, without.slope)
    n13 = P3.n13_verdict(fit)
    fit_avg = P3.loglog_slope_fit(with_avg.n_artifacts, with_avg.slope)

    threshold = P3.select_sample_size(with_avg.to_dict("records"))
    m_arm = slopes[(slopes.sweep == "M") & (slopes.metric == "kl_payload")].sort_values("m_observers")

    return {
        "experiment": "E12",
        "spec": "V3 §1 C2, §3 N13",
        "condition": "f = 0, honest signal — the condition that failed N11 in V2",
        "N13": n13,
        "loglog_fit_with_averaging": fit_avg,
        "slopes_without_averaging": without[["n_artifacts", "slope", "se", "t"]].to_dict("records"),
        "slopes_with_averaging": with_avg[["n_artifacts", "slope", "se", "t",
                                           "n11_passed"]].to_dict("records"),
        "m_sweep": m_arm[["m_observers", "n_artifacts", "slope", "se", "t"]].to_dict("records"),
        "sample_size_decision": threshold,
        "e8_may_run": bool(n13["passed"] and threshold["found"]),
        "if_e8_may_not_run": (
            "V3 §1 C2: if the without-averaging slope does not shrink with N, the leak is not "
            "finite-sample estimation error and the entire V3 diagnosis is wrong. Report this "
            "loudly; the loop is lossy for a structural reason C1 does not address, and E8 "
            "must not be run until that reason is found."),
    }


def make_e12_figure(df, slopes, verdict, path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    n_arm = slopes[(slopes.sweep == "N") & (slopes.metric == "kl_payload")]

    ax = axes[0]
    for averaging, sub in n_arm.groupby("averaging"):
        sub = sub.sort_values("n_artifacts")
        ax.errorbar(sub.n_artifacts, np.abs(sub.slope), yerr=sub.se, capsize=3, marker="o",
                    ls="-" if averaging else "--",
                    color="C0" if averaging else "firebrick",
                    label="averaged over many readers" if averaging else "one reader at a time")
    ceiling = P3.N11_SLOPE_CEILING
    ax.axhline(ceiling, color="k", lw=1, ls=":",
               label=f"the level this had to get under ({ceiling})")
    ax.set(xscale="log", yscale="log", xlabel="artifacts per generation",
           ylabel="error added per generation (nats)",
           title="If the drift were just too little data,\nthis line would fall. It does not.\n"
                 f"(measured slope {verdict['N13'].get('b', float('nan')):.2f}; predicted -1.00)")
    ax.legend(fontsize=7)

    ax = axes[1]
    m_arm = slopes[(slopes.sweep == "M") & (slopes.metric == "kl_payload")].sort_values("m_observers")
    if len(m_arm):
        ax.errorbar(m_arm.m_observers, np.abs(m_arm.slope), yerr=m_arm.se, capsize=3,
                    marker="s", color="C2")
    ax.set(xscale="log", yscale="log", xlabel="readers averaged together per generation",
           ylabel="error added per generation (nats)",
           title="Averaging over more readers does not help either\n"
                 "(they all read the same corpus, so the error is shared)")

    ax = axes[2]
    for (n_art, averaging), sub in df[df.sweep == "N"].groupby(["n_artifacts", "averaging"]):
        m = sub.groupby("generation").kl_payload.mean().sort_index()
        ax.plot(m.index, m.values, ls="-" if averaging else "--", marker="o", ms=3,
                alpha=0.85,
                label=f"{n_art} artifacts, {'averaged' if averaging else 'single reader'}")
    ax.set(xlabel="generation (each one trains on the last one's output)",
           ylabel="error in what people are believed to want (nats)",
           title="The chains themselves, with zero machine content\n"
                 "(a lossless loop would be a flat line)")
    ax.legend(fontsize=6, ncol=2)

    fig.suptitle("E12 — Is the drift just a shortage of data? "
                 "The test says no, and that killed our explanation",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E12 — leak versus per-generation sample size")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    slopes = run(cfg, out_dir=args.out, workers=workers, seed=args.seed)
    res_dir, _ = C.ensure_dirs(args.out)
    verdict = json.loads((res_dir / "e12_threshold.json").read_text(encoding="utf-8"))

    print("\nE12 — per-generation leak slope by sample size:")
    print(slopes[slopes.metric == "kl_payload"].round(6).to_string(index=False))
    n13 = verdict["N13"]
    print(f"\n  N13 (the finite-sample diagnosis): "
          f"log|slope| ~ log N  ->  b = {n13.get('b', float('nan')):.3f} "
          f"(t = {n13.get('t', float('nan')):.2f})")
    print(f"    gate  (b significantly < 0): {'PASSED' if n13['passed'] else 'REFUTED'}")
    print(f"    1/N band {n13['consistent_band']}: "
          f"{'consistent' if n13['consistent_with_one_over_N'] else 'INCONSISTENT — report it'}")
    dec = verdict["sample_size_decision"]
    if dec["found"]:
        print(f"\n  E8 sample size := {dec['n_artifacts']} artifacts/generation "
              f"(slope {dec['slope']:+.5f}, t {dec['t']:.2f})")
    else:
        print(f"\n  NO PASSING SAMPLE SIZE. {dec['reason']}")
    print(f"  E8 may run: {verdict['e8_may_run']}")


if __name__ == "__main__":
    main()
