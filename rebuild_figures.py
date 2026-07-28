#!/usr/bin/env python
"""Regenerate every figure from the CSVs already in ``results/``, without re-running anything.

WHY THIS EXISTS. The figures are the part of this repository most people will actually look at,
and their labels get rewritten far more often than the science does. Re-running the full
programme to change an axis label would cost hours and, worse, would produce figures drawn from
a *different* run than the numbers quoted in the RESULTS files. This script redraws from the
committed CSVs, so what you see is always the run that was reported.

It calls the same ``make_*_figure`` functions the experiments call, so there is exactly one
place where any given label is written. If a label looks wrong here, fix it in the experiment
module; this file only supplies the data.

    python rebuild_figures.py                 # results/ -> figures/
    python rebuild_figures.py --results DIR --figures DIR

Anything whose CSV is missing is skipped and reported, not faked.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _read(res: Path, name: str) -> pd.DataFrame:
    path = res / name
    if not path.exists():
        raise FileNotFoundError(name)
    return pd.read_csv(path)


def _verdict(res: Path, name: str) -> dict:
    path = res / name
    if not path.exists():
        raise FileNotFoundError(name)
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# One builder per figure. Each returns nothing and raises FileNotFoundError if
# its inputs are absent, which the driver turns into a SKIP line.
# --------------------------------------------------------------------------- #
def e1(res, figs):
    from ghostscale import figures
    figures.fig_e1(_read(res, "e1_timeseries.csv"), figs / "e1_crash.png")


def e2(res, figs):
    from ghostscale import figures
    points = _read(res, "e2_points.csv")
    stats = _read(res, "e2_cell_stats.csv")
    cell_stats = {(r.true_provenance, r.declared_signal):
                  {"within": float(r.within), "between": float(r.between),
                   "within_sd": float(r.within_sd), "between_sd": float(r.between_sd)}
                  for r in stats.itertuples()}
    figures.fig_e2(points[points.seed_rep == 0], cell_stats, figs / "e2_variance.png")


def e3(res, figs):
    from ghostscale.experiments import e3_titration as M
    M.make_e3_figures(_read(res, "e3_summary.csv"), figs)


def e4(res, figs):
    from ghostscale.experiments import e4_trust_exploit as M
    M.make_e4_figures(_read(res, "e4_summary.csv"), figs)


def e5(res, figs):
    from ghostscale.experiments import e5_precision_baseline as M
    M.make_e5_figure(_read(res, "e5_summary.csv"), figs / "e5_precision_baseline.png")


def e6(res, figs):
    from ghostscale.experiments import e6_corpus_corruption as M
    agg = _read(res, "e6_summary.csv")
    M.make_e6_figure(agg, sorted(agg.kappa.unique()), sorted(agg.signing_rate.unique()),
                     figs / "e6_corpus_corruption.png")


def e6b(res, figs):
    from ghostscale.experiments import e6b_corpus_biased as M
    df = _read(res, "e6b_raw.csv")
    agg = _read(res, "e6b_summary.csv")
    prereg = _verdict(res, "e6b_preregistration.json")
    M.make_e6b_figure(df, agg, prereg, figs / "e6b_corpus_biased.png")


def e7(res, figs):
    from ghostscale.experiments import e7_learn_ghost as M
    M.make_e7_figure(_read(res, "e7_raw.csv"), _read(res, "e7_summary.csv"),
                     figs / "e7_learn_ghost.png")


def e8(res, figs):
    from ghostscale.experiments import e8_recursive as M
    agg = _read(res, "e8_summary.csv")
    trends = _read(res, "e8_trends.csv")
    channels = _verdict(res, "e8_channels.json")
    M.make_e8_figure(agg, trends, channels, figs / "e8_recursive.png")


def e9(res, figs):
    from ghostscale.experiments import e9_poison_starve as M
    M.make_e9_figure(_read(res, "e9_raw.csv"), _read(res, "e9_summary.csv"), None,
                     figs / "e9_poison_starve.png")


def e10(res, figs):
    from ghostscale.experiments import e10_expertise as M
    M.make_e10_figure(_read(res, "e10_raw.csv"), _read(res, "e10_summary.csv"),
                      figs / "e10_expertise_gradient.png")


def e11(res, figs):
    from ghostscale.experiments import e11_regret_vs_kl as M
    df = _read(res, "e11_points.csv")
    M.make_e11_figure(df, _verdict(res, "e11_analysis.json"), figs / "e11_regret_vs_kl.png")


def e12(res, figs):
    from ghostscale.experiments import e12_leak_vs_samplesize as M
    M.make_e12_figure(_read(res, "e12_leak_vs_samplesize.csv"), _read(res, "e12_slopes.csv"),
                      _verdict(res, "e12_threshold.json"), figs / "e12_leak_convergence.png")


def e13(res, figs):
    from ghostscale.experiments import e13_freeze_leak_signature as M
    M.make_e13_figure(_read(res, "e13_freeze_leak_signature.csv"),
                      _verdict(res, "e13_verdict.json"), figs / "e13_shared_signature.png")


def e14(res, figs):
    from ghostscale.experiments import e14_engagement_floor as M
    M.make_figure(_read(res, "e14_engagement_floor.csv"), _verdict(res, "e14_verdict.json"),
                  figs / "e14_engagement_floor.png")


def e15(res, figs):
    from ghostscale.experiments import e15_competence_cliff as M
    M.make_figure(_read(res, "e15_competence_cliff.csv"), _verdict(res, "e15_verdict.json"),
                  figs / "e15_competence_cliff.png")


def e16(res, figs):
    from ghostscale.experiments import e16_label_coverage as M
    M.make_figure(_read(res, "e16_label_coverage.csv"), _verdict(res, "e16_verdict.json"),
                  figs / "e16_label_coverage.png")


def e17(res, figs):
    from ghostscale.experiments import e17_tier_dose_response as M
    M.make_figure(_read(res, "e17_tier_stats.csv"), _verdict(res, "e17_verdict.json"),
                  figs / "e17_tier_dose_response.png")


def e18(res, figs):
    from ghostscale.experiments import e18_deferred_estimator as M
    M.make_figure(_read(res, "e18_deferred_estimator.csv"), _verdict(res, "e18_verdict.json"),
                  figs / "e18_deferred_estimator.png")


BUILDERS = [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5), ("E6", e6),
            ("E6b", e6b), ("E7", e7), ("E8", e8), ("E9", e9), ("E10", e10), ("E11", e11),
            ("E12", e12), ("E13", e13), ("E14", e14), ("E15", e15), ("E16", e16),
            ("E17", e17), ("E18", e18)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--only", nargs="*", help="rebuild only these (e.g. --only E2 E17)")
    args = ap.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    wanted = {s.upper() for s in args.only} if args.only else None

    built, skipped, failed = [], [], []
    for name, fn in BUILDERS:
        if wanted and name.upper() not in wanted:
            continue
        try:
            fn(args.results, args.figures)
            built.append(name)
            print(f"  built   {name}")
        except FileNotFoundError as exc:
            skipped.append((name, str(exc)))
            print(f"  SKIP    {name}  (no {exc})")
        except Exception:
            failed.append(name)
            print(f"  FAILED  {name}")
            traceback.print_exc()

    print(f"\n{len(built)} built, {len(skipped)} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
