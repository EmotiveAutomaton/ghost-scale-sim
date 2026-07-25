"""E8 — Recursive degradation across generations (V3 spec §2). STAGE 4. GATED.

    G_max = 8, contamination f in {0.0, 0.3, 0.6}, signal in {absent, honest}. Learner
    observers, C1 population-averaged seeding, per-generation sample size SET BY E12.

    PREDICTED  monotone degradation with generation at f > 0, attenuated by an honest signal,
               and SUPERLINEAR — generation 6's damage exceeds six times generation 1's,
               because a degraded reader produces degraded artifacts which further degrade
               the next reader.

    "Report the trend and its significance, not an equilibrium. Do not extrapolate the curve."

That instruction is implemented rather than merely quoted: this module reports per-generation
SLOPES with t statistics and a quadratic term for the superlinearity claim, and it produces
no fitted curve beyond the observed generations.

WHAT V3 CHANGES (spec §1, §2):

  * **C1** — generation g+1 is seeded from the POPULATION-AVERAGED learned CREATOR column.
    V2 seeded from one observer's, and that was the leak. See ``ghostscale/generations.py``.
  * **Sample size from E12** — never hardcoded (§1 C2). ``resolve_sample_size`` reads
    ``results/e12_threshold.json`` and REFUSES to run without it, so the dependency is wired
    rather than remembered.
  * **G_max 8** — V2's 4 could not distinguish a curve from a line, and superlinearity is a
    claim about the curve. §4 forbids dropping below 6.
  * **C3 two-channel tracking** — value divergence (primary) AND encoder divergence
    (secondary), plus the relationship between them.

N11 GATES THIS EXPERIMENT, at full E8 scale, under the pre-registered conjunctive criterion
(``prereg_v3.n11_verdict``). The f=0 arm is included in every run precisely so the null is
measured on the same code path, in the same conditions, as the effect.

-----------------------------------------------------------------------------------------
DECISION D6 — the C3 relationship is tested by PARTIAL correlation, not a pooled regression.

V3 §1 C3 says to "regress the decoding-ability drop on the payload KL across generations and
conditions". Both quantities trend with generation by construction, so a pooled regression
returns a strong relationship EVEN IF THE TWO CHANNELS ARE INDEPENDENT — it would confirm the
shared-mechanism claim automatically and could never refute it. A test that cannot fail is
not evidence, and §5's discipline ("non-replications reported rather than tuned") applies to
the analysis as much as to the code.

So three statistics are reported, and the pooled one is explicitly labelled as not evidence:

  1. **partial correlation** of encoder divergence with value divergence, controlling for
     generation — the actual test of the shared-representation claim;
  2. **within-generation correlation** across conditions and replications — the same test
     from the other direction, with generation held fixed by construction;
  3. **lag-1 cross-correlation** in both directions — the "lags and tracks rather than leads"
     claim, which the spec asserts but never operationalises.
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
from ..generative_model import build_shared_model as _build_model, assert_preferences_zero
from ..preregistration import POP_GOAL_DIST
from ..generations import run_chain, chain_trend
from ..figures import set_style
from .. import prereg_v3 as P3
from . import _common as C

SIGNAL_MODES = {"absent": 0.0, "honest": 1.0}     # -> signing_rate

# The encoder-divergence channel is read from the FORCED-DEEP probe decode: C3's claim is
# about the observer's MODEL drifting with its value model, and the free-engagement number
# additionally contains disengagement, which E9/E13 already own. Both are in the CSV.
ENCODER_KEY = "enc_forced_encoder_mi"


def resolve_sample_size(cfg: Config, res_dir: Path, require_e12: bool = True) -> tuple[int, dict]:
    """E8's per-generation sample size comes from E12, or E8 does not run (V3 §1 C2).

    "``E12``'s output **sets the sample size** used by the re-run N11 and E8. Wire this
     dependency explicitly; do not hardcode E8's sample size."

    Implemented as a refusal rather than a default, because a silent fallback to the config
    placeholder is exactly how a hardcoded sample size would come back.

    ``require_e12=False`` means the caller is deliberately opting OUT of the E12 wiring — a
    ``--quick`` smoke run, or ``restore_v2_e8.py`` reproducing a V2-parameter cell — and it
    takes the config value WITHOUT consulting the threshold file.

    That last clause is load-bearing and was a real bug: an earlier version consulted
    ``e12_threshold.json`` whenever it existed, regardless of ``require_e12``. A leftover
    threshold from a ``--quick`` E12 (120 artifacts) then silently overrode the caller's 300,
    and the V2 reproduction came out at slope +0.0067 instead of +0.0119 — a wrong-looking
    result with no error message anywhere. Opting out of the gate must mean opting out of it.
    """
    if not require_e12:
        return int(cfg.get("experiments.e8.n_artifacts", 300)), {
            "source": "config (E12 wiring explicitly bypassed; NOT reportable)"}
    path = Path(res_dir) / "e12_threshold.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. E8's per-generation sample size is set by E12 and may not "
            "be hardcoded (V3 §1 C2). Run E12 first:\n"
            "    python -m ghostscale.experiments.e12_leak_vs_samplesize")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    decision = verdict.get("sample_size_decision", {})
    if not verdict.get("e8_may_run") or not decision.get("found"):
        raise RuntimeError(
            "E12 did not clear E8 to run.\n"
            f"  N13 passed: {verdict.get('N13', {}).get('passed')}\n"
            f"  sample size found: {decision.get('found')} "
            f"({decision.get('reason', '')})\n" + verdict.get("if_e8_may_not_run", ""))
    return int(decision["n_artifacts"]), {"source": "E12", **decision}


def _e8_worker(payload):
    (cfg_raw, cell_index, f, signal_mode, seed_rep, base_seed, g_max, n_creators,
     n_artifacts, n_observers, infer_steps, d_i, synth_seed, n_probes, averaging) = payload
    cfg = Config(cfg_raw)
    gm = _build_model(cfg, goal_symmetric=False, synth_draw_seed=synth_seed)
    assert_preferences_zero(gm.C)                    # N7

    results = run_chain(cfg, gm, POP_GOAL_DIST[:cfg.cardinalities.num_goals],
                        contamination=f, signing_rate=SIGNAL_MODES[signal_mode],
                        honesty=1.0, g_max=g_max, n_creators=n_creators,
                        n_artifacts=n_artifacts, n_observers=n_observers,
                        infer_steps=infer_steps, d_i=d_i,
                        base_seed=base_seed * 7919 + seed_rep,
                        population_average_seed=bool(averaging),
                        n_probes=int(n_probes))
    recs = []
    for r in results:
        rec = {"contamination": f, "signal": signal_mode, "seed_rep": seed_rep,
               "generation": r.generation, "mean_expertise": r.mean_expertise,
               "mi_genuine": r.mi_genuine, "kl_payload": r.kl_payload,
               "mean_deep_genuine": r.mean_deep_genuine,
               "creator_col_kl": r.creator_col_kl,
               "eff_sample_count": r.eff_sample_count,
               "n_artifacts": n_artifacts, "n_observers": n_observers,
               "population_average_seed": bool(averaging)}
        rec.update({k: v for k, v in r.panel.items() if k not in ("mi_genuine",
                                                                  "mean_deep_genuine")})
        recs.append(rec)
    return recs


def run(cfg: Config, out_dir: Path | None = None, workers: int = 1,
        seed: int | None = None, make_fig: bool = True,
        require_e12: bool = True,
        only_contamination: float | None = None) -> pd.DataFrame:
    """``only_contamination`` restricts the run to one f level.

    Used by ``run_all_v3.py`` stage 3 to run E8's f = 0 arm ALONE at full scale, so the
    repaired N11 is evaluated at the scale of the experiment it gates without E8's f > 0 arms
    having been run first. Stage 4 then re-runs the whole grid; the f = 0 cells are seeded
    identically and reproduce the gated numbers exactly.
    """
    base_seed = int(cfg.run.base_seed if seed is None else seed)
    e = cfg.experiments.e8
    g_max = int(e.g_max)
    min_g_max = int(cfg.get("experiments.e8.min_g_max", 6))
    assert g_max >= min_g_max, (
        f"V3 spec §4: G_max may not be reduced below {min_g_max} — the superlinearity claim "
        f"needs the generations to distinguish a curve from a line (got {g_max})")
    n_creators = int(cfg.get("experiments.e8.n_creators_next", 20))
    n_observers = int(e.n_observers)
    n_probes = int(cfg.get("experiments.e8.n_probes", 150))
    infer_steps = int(cfg.get("experiments.e8.infer_steps", 6))
    n_reps = int(cfg.get("experiments.e8.n_replications", 3))
    f_levels = list(cfg.get("experiments.e8.contamination_levels", [0.0, 0.3, 0.6]))
    d_i = float(cfg.get("experiments.e8.d", 0.0))
    synth_seed = int(cfg.get("experiments.e8.synth_draw_seed", 1))
    averaging = bool(cfg.get("experiments.e8.population_average_seed", True))
    res_dir, fig_dir = C.ensure_dirs(out_dir)
    n_artifacts, size_source = resolve_sample_size(cfg, res_dir, require_e12=require_e12)

    payloads, ci = [], 0
    for f in f_levels:
        # ci still advances over the skipped levels, so a restricted run's cells keep the same
        # observer seeds as the full run and its numbers reproduce exactly (run_all_v3 stage 3).
        if only_contamination is not None and float(f) != float(only_contamination):
            ci += 1
            continue
        for mode in SIGNAL_MODES:
            for s in range(n_reps):
                payloads.append((cfg.raw, ci, f, mode, s, base_seed, g_max, n_creators,
                                 n_artifacts, n_observers, infer_steps, d_i, synth_seed,
                                 n_probes, averaging))
            ci += 1
    recs = C.run_parallel(payloads, _e8_worker, workers)
    df = pd.DataFrame(recs)
    df.to_csv(res_dir / "e8_raw.csv", index=False)

    channels = channel_analysis(df)
    channels["sample_size_source"] = size_source
    (res_dir / "e8_channels.json").write_text(json.dumps(channels, indent=2), encoding="utf-8")

    agg_spec = dict(kl_payload=("kl_payload", "mean"), kl_sd=("kl_payload", "std"),
                    mi_genuine=("mi_genuine", "mean"),
                    mean_deep_genuine=("mean_deep_genuine", "mean"),
                    creator_col_kl=("creator_col_kl", "mean"),
                    regret=("regret", "mean"), regret_sd=("regret", "std"),
                    argmax_preserved=("argmax_preserved", "mean"),
                    sycophancy=("sycophancy", "mean"))
    if ENCODER_KEY in df.columns:
        agg_spec.update({"encoder_mi": (ENCODER_KEY, "mean"),
                         "encoder_mi_sd": (ENCODER_KEY, "std"),
                         "encoder_mi_free": ("enc_free_encoder_mi", "mean"),
                         "encoder_soft": ("enc_forced_encoder_soft", "mean")})
    agg = (df.groupby(["contamination", "signal", "generation"])
             .agg(**agg_spec).reset_index())
    agg.to_csv(res_dir / "e8_summary.csv", index=False)

    trends = trend_table(df)
    trends.to_csv(res_dir / "e8_trends.csv", index=False)

    if make_fig:
        make_e8_figure(agg, trends, channels, fig_dir / "e8_recursive.png")
    return agg


# --------------------------------------------------------------------------- #
# C3 — the two channels and the relationship between them (decision D6).
# --------------------------------------------------------------------------- #
def _encoder_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Encoder divergence = the DROP in probe-decoding MI from this chain's generation 0.

    Expressed as a drop, per chain, so that it is on the same footing as value divergence
    (a divergence from a fixed reference) and so that a chain's starting decoding ability —
    which varies with the replication's draw — is not smeared into the effect.
    """
    if ENCODER_KEY not in df.columns:
        return pd.DataFrame()
    keys = ["contamination", "signal", "seed_rep"]
    base = (df[df.generation == 0].set_index(keys)[ENCODER_KEY]
            .rename("encoder_mi_gen0"))
    out = df.join(base, on=keys)
    out["encoder_divergence"] = out["encoder_mi_gen0"] - out[ENCODER_KEY]
    return out


def _partial_corr(x, y, z) -> dict:
    """corr(x, y | z): correlate the residuals of x and y after regressing each on z."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 4 or np.ptp(z) == 0:
        return {"r": float("nan"), "n": int(len(x))}
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"r": float("nan"), "n": int(len(x))}
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    dof = max(n - 3, 1)
    t = float(r * np.sqrt(dof / max(1 - r ** 2, 1e-12)))
    return {"r": r, "t": t, "n": int(n), "dof": dof}


def channel_analysis(df: pd.DataFrame) -> dict:
    """The C3 relationship, tested three ways. See the module docstring (decision D6)."""
    ed = _encoder_divergence(df)
    if ed.empty:
        return {"available": False,
                "reason": "encoder-divergence probes were disabled (e8.n_probes = 0), so "
                          "C3's second channel was not measured"}
    sub = ed[np.isfinite(ed["encoder_divergence"]) & np.isfinite(ed["kl_payload"])]

    partial = _partial_corr(sub["kl_payload"], sub["encoder_divergence"], sub["generation"])
    pooled_r = (float(np.corrcoef(sub["kl_payload"], sub["encoder_divergence"])[0, 1])
                if len(sub) > 2 else float("nan"))

    within = {}
    for g, s in sub.groupby("generation"):
        if len(s) > 2 and s["kl_payload"].std() > 0 and s["encoder_divergence"].std() > 0:
            within[int(g)] = float(np.corrcoef(s["kl_payload"], s["encoder_divergence"])[0, 1])

    # Lag-1, both directions. "Encoder divergence lags value divergence" predicts that
    # value(g) -> encoder(g+1) is the stronger of the two.
    keys = ["contamination", "signal", "seed_rep"]
    lag_ve, lag_ev = [], []
    for _, chain in sub.sort_values("generation").groupby(keys):
        v = chain["kl_payload"].to_numpy(float)
        e_ = chain["encoder_divergence"].to_numpy(float)
        if len(v) >= 4 and np.std(v[:-1]) > 0 and np.std(e_[1:]) > 0:
            lag_ve.append(float(np.corrcoef(v[:-1], e_[1:])[0, 1]))
        if len(v) >= 4 and np.std(e_[:-1]) > 0 and np.std(v[1:]) > 0:
            lag_ev.append(float(np.corrcoef(e_[:-1], v[1:])[0, 1]))

    return {
        "available": True,
        "decision": "D6 — partial correlation controlling for generation is THE test",
        "why_not_pooled": (
            "value and encoder divergence both trend with generation by construction, so a "
            "pooled regression across generations is significant even when the channels are "
            "independent; it cannot refute the shared-mechanism claim and is therefore "
            "reported but is not evidence"),
        "partial_corr_controlling_for_generation": partial,
        "pooled_corr_NOT_EVIDENCE": {"r": pooled_r, "n": int(len(sub))},
        "within_generation_corr": within,
        "within_generation_mean": (float(np.mean(list(within.values()))) if within
                                   else float("nan")),
        "lag1_value_leads_encoder": (float(np.mean(lag_ve)) if lag_ve else float("nan")),
        "lag1_encoder_leads_value": (float(np.mean(lag_ev)) if lag_ev else float("nan")),
        "lags_and_tracks_supported": bool(
            lag_ve and lag_ev and np.mean(lag_ve) > np.mean(lag_ev)),
        "interpretation": (
            "a strong partial correlation supports C3's claim that encoder divergence is "
            "downstream of value divergence through a shared representation; independence "
            "between the channels once generation is controlled for REFUTES it"),
    }


def trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition per-generation slopes with t statistics. This — not an equilibrium and
    not an extrapolation — is what E8 reports (V2 spec §2)."""
    rows = []
    keys = ["kl_payload", "mi_genuine", "regret", "mean_deep_genuine", "creator_col_kl"]
    if ENCODER_KEY in df.columns:
        keys += [ENCODER_KEY, "enc_free_encoder_mi"]
    for (f, mode), sub in df.groupby(["contamination", "signal"]):
        for key in keys:
            if key not in sub.columns:
                continue
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
            t = float(slope / se) if se > 0 else np.nan
            row = {"contamination": f, "signal": mode, "metric": key,
                   "slope": float(slope), "se": se, "t": t,
                   "quadratic_term": quad,
                   "significant": bool(se > 0 and abs(t) > 2.0)}
            # The repaired N11 verdict travels in the CSV next to the number it judges, under
            # the pre-registered conjunctive criterion (D1) rather than |t| alone.
            if key == "kl_payload":
                row["n11_passed"] = P3.n11_verdict(slope, t)["passed"]
            rows.append(row)
    return pd.DataFrame(rows)


def n11_report(trends: pd.DataFrame) -> dict:
    """The repaired N11, evaluated on this run at this run's scale (V3 §3).

    "A null must be evaluated at the scale of the experiment it gates." That rule is why this
    reads the actual E8 output rather than re-simulating a smaller chain.
    """
    zero = trends[(trends.contamination == 0.0) & (trends.metric == "kl_payload")]
    arms = {}
    for r in zero.itertuples():
        arms[r.signal] = P3.n11_verdict(r.slope, r.t)
    return {"criterion": f"|t| < {P3.N11_T_THRESHOLD} AND "
                         f"|slope| < {P3.N11_SLOPE_CEILING} nats/generation",
            "arms": arms,
            "passed": bool(arms) and all(v["passed"] for v in arms.values()),
            "consequence_if_failed": "E8 is not reported and is excluded from E11 (V3 §6)"}


def make_e8_figure(agg, trends, channels, path):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.3))
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
    if "encoder_mi" in agg.columns:
        for (f, mode), sub in agg.groupby(["contamination", "signal"]):
            sub = sub.sort_values("generation")
            ax.errorbar(sub.generation, sub.encoder_mi, yerr=sub.encoder_mi_sd, capsize=2,
                        ls=styles[mode], marker="^", ms=4, label=f"f={f}, {mode}")
        ax.set(xlabel="generation", ylabel="MI(inferred goal ; true goal) on clean probes [nats]",
               title="ENCODER divergence (C3 secondary)\ncan they still read clean human work?")
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()

    ax = axes[2]
    for (f, mode), sub in agg.groupby(["contamination", "signal"]):
        sub = sub.sort_values("generation")
        ax.plot(sub.generation, sub.regret, ls=styles[mode], marker="s", ms=4,
                label=f"f={f}, {mode}")
    ax.set(xlabel="generation", ylabel="behavioural regret",
           title="Harm, measured as regret\n(the headline harm claim)")
    ax.legend(fontsize=7)

    ax = axes[3]
    t = trends[trends.metric == "kl_payload"]
    labels = [f"f={r.contamination}\n{r.signal}" for r in t.itertuples()]
    colours = ["firebrick" if r.significant else "grey" for r in t.itertuples()]
    ax.bar(range(len(t)), t.slope, yerr=t.se, color=colours, capsize=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(t)), labels, fontsize=7)
    partial = (channels or {}).get("partial_corr_controlling_for_generation", {})
    r = partial.get("r", float("nan"))
    ax.set(ylabel="KL slope per generation",
           title=f"Per-generation trend (red = |t| > 2)\n"
                 f"C3 partial r (gen controlled) = {r:.2f}")

    fig.suptitle("E8 — Recursive degradation, two channels (trend, not equilibrium; "
                 "no extrapolation)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = C.standard_argparser("E8 — recursive degradation across generations")
    ap.add_argument("--only-contamination", type=float, default=None,
                    help="run only this f level. Used by run_all_v3 stage 3 to evaluate the "
                         "repaired N11 on E8's f=0 arm at full scale before the f>0 arms run.")
    ap.add_argument("--no-e12-gate", action="store_true",
                    help="run at the config's placeholder sample size instead of E12's. "
                         "Development only: V3 §1 C2 forbids hardcoding E8's sample size, "
                         "and a run made this way must not be reported.")
    args = ap.parse_args()
    cfg = C.resolve_config(args)
    workers = C.default_workers() if args.workers is None else args.workers
    require_e12 = not (args.no_e12_gate or args.quick)
    agg = run(cfg, out_dir=args.out, workers=workers, seed=args.seed,
              require_e12=require_e12, only_contamination=args.only_contamination)
    res_dir, _ = C.ensure_dirs(args.out)
    trends = pd.read_csv(res_dir / "e8_trends.csv")
    channels = json.loads((res_dir / "e8_channels.json").read_text(encoding="utf-8"))

    print("\nE8 — per generation:")
    print(agg.round(4).to_string(index=False))
    print("\nE8 — per-generation trends (the reported claim):")
    print(trends[trends.metric.isin(["kl_payload", "regret", ENCODER_KEY])]
          .round(4).to_string(index=False))

    rep = n11_report(trends)
    print(f"\n  N11 (zero-contamination recursion) — {rep['criterion']}")
    for signal, v in rep["arms"].items():
        print(f"    signal {signal:6s}: slope {v['slope']:+.5f}, t {v['t']:+.2f}  "
              f"-> {'passed' if v['passed'] else 'FAILED'}")
    print(f"  N11: {'PASSED — E8 is reportable' if rep['passed'] else 'FAILED — ' + rep['consequence_if_failed']}")

    if channels.get("available"):
        p = channels["partial_corr_controlling_for_generation"]
        print(f"\n  C3 two-channel relationship (decision D6):")
        print(f"    partial r (generation controlled) = {p['r']:+.3f} "
              f"(t {p.get('t', float('nan')):+.2f}, n {p['n']})")
        print(f"    within-generation mean r          = {channels['within_generation_mean']:+.3f}")
        print(f"    lag-1 value->encoder {channels['lag1_value_leads_encoder']:+.3f} vs "
              f"encoder->value {channels['lag1_encoder_leads_value']:+.3f}  "
              f"-> lags-and-tracks {'supported' if channels['lags_and_tracks_supported'] else 'NOT supported'}")
        print(f"    (pooled r = {channels['pooled_corr_NOT_EVIDENCE']['r']:+.3f}, "
              f"reported but NOT evidence — see D6)")


if __name__ == "__main__":
    main()
