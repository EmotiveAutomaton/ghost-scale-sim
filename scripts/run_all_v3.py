#!/usr/bin/env python
"""Run the V3 experiment programme in the order V3 spec §2 requires, with the gates enforced.

    | stage | experiment    | purpose                          | gate                        |
    |-------|---------------|----------------------------------|-----------------------------|
    | 0     | prereg        | write the acceptance criteria    | must be written FIRST       |
    | 1     | E12           | leak-vs-sample-size; sets E8's N | must show 1/N (N13)         |
    | 2     | E13           | freeze/leak shared-axis          | classifies the E9 freeze    |
    | 3     | N11 re-run    | acceptance gate for the loop     | must pass at full E8 scale  |
    | 4     | E8            | the recursive result             | only if N11 passes          |
    | 5     | E11 re-pool   | fold E8 back into harm analysis  | only if E8 is reportable    |

THE ORDERING PROBLEM IN §2, AND HOW IT IS RESOLVED. The spec lists the N11 re-run as stage 3
and E8 as stage 4, but §4 also says N11 is "folded into E8's f=0 arm" — and N11 must be
evaluated at full E8 scale, which means running E8's f=0 arm. Those cannot both be literal.

Resolved by splitting E8 in two: stage 3 runs E8's f = 0 arm ALONE at full scale and gates on
it; stage 4 runs the f > 0 arms and merges. Nothing is evaluated at reduced scale, nothing is
run before its gate, and the f=0 numbers that the gate judges are the same numbers E8 reports
— which is what "measured on the same code path, in the same conditions" required in V2.

STALENESS. The V2 gate read ``results/e8_trends.csv`` and would happily have judged a
leftover file from a previous run. Stage 3 deletes the E8 outputs before it regenerates them,
so the gate cannot pass on stale evidence.

Usage:
    python scripts/run_all_v3.py --prereg-only     # write the criteria, run nothing
    python scripts/run_all_v3.py                   # full programme, gated
    python scripts/run_all_v3.py --quick           # smoke scale (NOT reportable)
    python scripts/run_all_v3.py --stage 1         # one stage
    python scripts/run_all_v3.py --from-stage 2    # resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PREREG_NAME = "v3_preregistration.json"

E8_OUTPUTS = ["e8_raw.csv", "e8_summary.csv", "e8_trends.csv", "e8_channels.json"]


def _results_dir(args) -> Path:
    """--quick writes to ``results_quick/`` (and its figures to ``results_quick/figures/``),
    NEVER into the committed ``results/`` and ``figures/``.

    A smoke run must not be able to destroy a reported run's evidence — during V3 development
    exactly that happened, twice, to V2's E8 outputs. Nor may it *gate* one: a quick
    ``e12_threshold.json`` left in ``results/`` would silently set the real E8's sample size.
    """
    return ROOT / ("results_quick" if args.quick else "results")


def _run(mod: str, args, extra: list[str] | None = None) -> float:
    cmd = [sys.executable, "-m", mod]
    if args.quick:
        cmd += ["--quick", "--out", str(_results_dir(args))]
    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]
    cmd += extra or []
    print(f"\n=== {mod} {' '.join(extra or [])} ===", flush=True)
    t = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    dt = (time.time() - t) / 60.0
    print(f"    ({dt:.1f} min, rc={rc})", flush=True)
    if rc != 0:
        print(f"FAILED: {mod}", file=sys.stderr)
        sys.exit(rc)
    return dt


# --------------------------------------------------------------------------- #
# Stage 0 — the pre-registration.
# --------------------------------------------------------------------------- #
def write_prereg(args) -> dict:
    from ghostscale.config import load_config
    from ghostscale.prereg_v3 import write_preregistration_v3
    cfg = load_config(quick=args.quick)
    prereg = _results_dir(args) / PREREG_NAME
    prereg.parent.mkdir(parents=True, exist_ok=True)
    payload = write_preregistration_v3(cfg, prereg, force=args.force_prereg)
    print(f"V3 acceptance criteria -> {prereg}")
    print(f"  N11: {payload['N11']['criterion']}")
    print(f"  N13: {payload['N13']['gate']} (predicted exponent "
          f"{payload['N13']['predicted_exponent']})")
    print(f"  E13: outcome {payload['E13']['prior_expectation']['expected_outcome']} expected "
          f"before the run")
    print(f"  hash: {payload['content_hash'][:16]}...")
    return payload


def check_prereg(args) -> dict:
    from ghostscale.prereg_v3 import assert_prereg_locked_v3
    return assert_prereg_locked_v3(_results_dir(args) / PREREG_NAME)


# --------------------------------------------------------------------------- #
# Stage 1 gate — E12 must show the leak shrinking with N (N13).
# --------------------------------------------------------------------------- #
def check_e12_gate(args) -> None:
    path = _results_dir(args) / "e12_threshold.json"
    if not path.exists():
        print("E12 threshold not found; run stage 1 first.", file=sys.stderr)
        sys.exit(2)
    v = json.loads(path.read_text(encoding="utf-8"))
    n13, dec = v["N13"], v["sample_size_decision"]
    print(f"\n--- E12 gate (N13) — b = {n13.get('b'):.3f}, t = {n13.get('t'):.2f} ---")
    if not n13["consistent_with_one_over_N"]:
        print(f"  NOTE: exponent outside the 1/N band {n13['consistent_band']}. The gate does "
              "not block on this, but it argues the error is not the 1/N kind and V3 §5 "
              "requires it stated in RESULTS_V3.md.", file=sys.stderr)
    if not v["e8_may_run"]:
        print("\n" + "!" * 76, file=sys.stderr)
        print("E12 DID NOT CLEAR E8 TO RUN.", file=sys.stderr)
        print(f"  N13 passed:       {n13['passed']}", file=sys.stderr)
        print(f"  sample size found: {dec['found']} {dec.get('reason', '')}", file=sys.stderr)
        print(v["if_e8_may_not_run"], file=sys.stderr)
        print("!" * 76, file=sys.stderr)
        if not args.force:
            sys.exit(3)
        print("--force given: continuing past a failed E12 gate.", file=sys.stderr)
    else:
        print(f"  E8 sample size := {dec['n_artifacts']} artifacts/generation")


# --------------------------------------------------------------------------- #
# Stage 3 — the repaired N11, at full E8 scale, on E8's own f=0 arm.
# --------------------------------------------------------------------------- #
def run_n11_gate(args) -> None:
    """Run E8's f = 0 arm alone at full scale and gate on it.

    This is the V2 lesson as a spec rule (V3 §3): a null must be evaluated at the scale of the
    experiment it gates. Nothing here is re-simulated smaller.
    """
    res = _results_dir(args)
    for name in E8_OUTPUTS:
        p = res / name
        if p.exists():
            p.unlink()      # the gate must not be able to pass on a previous run's evidence
    _run("ghostscale.experiments.e8_recursive", args,
         extra=["--only-contamination", "0.0"])

    from ghostscale.experiments.e8_recursive import n11_report
    trends = pd.read_csv(res / "e8_trends.csv")
    rep = n11_report(trends)
    print("\n--- N11 gate (repaired), at full E8 scale ---")
    print(f"  criterion: {rep['criterion']}")
    for signal, v in rep["arms"].items():
        print(f"    signal {signal:6s}: slope {v['slope']:+.6f}, t {v['t']:+.2f} -> "
              f"{'passed' if v['passed'] else 'FAILED'}")
    (res / "n11_verdict.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["passed"]:
        print("\nN11 FAILED — the recursion loop is still lossy at the E12-determined sample "
              "size. E8 will not run: its results would be implementation artifacts rather "
              "than findings. V3 §6: no exceptions, no 'close enough'.", file=sys.stderr)
        if not args.force:
            sys.exit(4)
        print("--force given: running E8 despite a failed N11.", file=sys.stderr)


def require_n11_passed(args) -> None:
    """Stages 4 and 5 are gated on N11 having PASSED, not merely on stage 3 having run.

    Checking only for the verdict file's existence would let ``--from-stage 4`` walk straight
    past a failed gate — which is the one thing V3 §6 says has no exceptions: E8 is not run and
    its results are not reported until the f=0 honest slope is insignificant at full E8 scale.
    """
    path = _results_dir(args) / "n11_verdict.json"
    if not path.exists():
        print("N11 has not been run; stage 3 gates this stage.", file=sys.stderr)
        if not args.force:
            sys.exit(5)
        return
    rep = json.loads(path.read_text(encoding="utf-8"))
    if not rep.get("passed"):
        print("\nN11 FAILED on the last stage-3 run — E8 is not reportable and E11 must not "
              "pool it. V3 §6: no exceptions, no 'close enough'.", file=sys.stderr)
        for signal, v in rep.get("arms", {}).items():
            print(f"    signal {signal}: slope {v['slope']:+.6f}, t {v['t']:+.2f}",
                  file=sys.stderr)
        if not args.force:
            sys.exit(5)
        print("--force given: proceeding despite a failed N11 (record as a deviation).",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Run the V3 programme, gated per spec §2")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stage", type=int, default=None, help="run only this stage")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--prereg-only", action="store_true",
                    help="write the acceptance criteria and stop")
    ap.add_argument("--force-prereg", action="store_true",
                    help="overwrite differing pre-registered criteria (a deviation; say so)")
    ap.add_argument("--force", action="store_true",
                    help="override a failed gate (records as a deviation)")
    args = ap.parse_args()

    write_prereg(args)
    if args.prereg_only:
        return
    check_prereg(args)
    if args.quick:
        print("\n*** --quick: smoke scale. Nothing produced by this run is reportable. ***")

    stages = ([args.stage] if args.stage is not None
              else [s for s in (1, 2, 3, 4, 5) if s >= args.from_stage])
    timings, t0 = {}, time.time()

    for s in stages:
        print("\n" + "=" * 76)
        if s == 1:
            print("STAGE 1 — E12: is the leak finite-sample? (sets E8's sample size)")
            print("=" * 76, flush=True)
            timings["E12"] = _run("ghostscale.experiments.e12_leak_vs_samplesize", args)
            check_e12_gate(args)
        elif s == 2:
            print("STAGE 2 — E13: the freeze and the leak on one axis")
            print("=" * 76, flush=True)
            timings["E13"] = _run("ghostscale.experiments.e13_freeze_leak_signature", args)
        elif s == 3:
            print("STAGE 3 — N11 re-run at full E8 scale (E8's f=0 arm alone)")
            print("=" * 76, flush=True)
            check_e12_gate(args)
            t = time.time()
            run_n11_gate(args)
            timings["N11"] = (time.time() - t) / 60.0
        elif s == 4:
            print("STAGE 4 — E8: the recursive result, finally reportable")
            print("=" * 76, flush=True)
            require_n11_passed(args)
            timings["E8"] = _run("ghostscale.experiments.e8_recursive", args)
        elif s == 5:
            print("STAGE 5 — E11 re-pool: fold E8 back into the harm analysis")
            print("=" * 76, flush=True)
            require_n11_passed(args)   # V3 §2: only if E8 is reportable
            timings["E11"] = _run("ghostscale.experiments.e11_regret_vs_kl", args)

    if not args.no_tests and args.stage is None:
        print("\n=== pytest (all nulls and invariants) ===", flush=True)
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode
        if rc != 0:
            sys.exit(rc)

    print("\n" + "=" * 76)
    print(f"V3 programme complete in {(time.time() - t0) / 60:.1f} min")
    print("=" * 76)
    print("Wall-clock per experiment (V3 spec §4 requires this in RESULTS_V3.md):")
    for name, dt in timings.items():
        print(f"  {name:8s} {dt:6.1f} min")
    print("\nCSVs in results/, figures in figures/. Now write RESULTS_V3.md from the CSVs.")


if __name__ == "__main__":
    main()
