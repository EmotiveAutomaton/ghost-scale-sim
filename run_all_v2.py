#!/usr/bin/env python
"""Run the V2 experiment programme in the order V2 spec §4 requires, with the gates enforced.

    "Required ordering. Each stage is conditional on the previous one."

    | stage | experiments      | gate                                        |
    |-------|------------------|---------------------------------------------|
    | 1     | E6b alone        | if falsified (naive KL < 0.10), STOP        |
    | 2     | E10, N8-N10      | cheap, no recursion; E10 is a headline      |
    | 2.5   | D2 calibration   | gates every learner experiment              |
    | 3     | E7               |                                             |
    | 4     | E9, N11          | N11 gates E8                                |
    | 5     | E8 (recursive)   | run last, overnight                         |
    | 6     | E11 analysis     | pure post-processing                        |

The gates are enforced here rather than left to discipline: stage 1 reads
``results/e6b_verdict.json`` and refuses to continue if E6b was falsified, and stage 5
refuses to start unless N11 has passed. ``--force`` overrides, and says so loudly.

Usage:
    python run_all_v2.py                 # full programme, gated
    python run_all_v2.py --quick         # smoke scale
    python run_all_v2.py --stage 1       # one stage
    python run_all_v2.py --from-stage 3  # resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

STAGES = {
    1: ("E6b — corpus corruption, biased synth (STAGE 1 GATE)",
        ["ghostscale.experiments.e6b_corpus_biased"]),
    2: ("E10 — the expertise gradient (headline standalone claim)",
        ["ghostscale.experiments.e10_expertise"]),
    3: ("D2 — prior_strength calibration (gates all learner experiments)",
        ["ghostscale.calibration"]),
    4: ("E7 — learning the GHOST column without labels",
        ["ghostscale.experiments.e7_learn_ghost"]),
    5: ("E9 — poisoning versus starvation",
        ["ghostscale.experiments.e9_poison_starve"]),
    6: ("E8 — recursive degradation (expensive; run last)",
        ["ghostscale.experiments.e8_recursive"]),
    7: ("E11 — regret versus KL (post-processing)",
        ["ghostscale.experiments.e11_regret_vs_kl"]),
}


def _run(mod: str, args) -> float:
    cmd = [sys.executable, "-m", mod]
    if args.quick:
        cmd.append("--quick")
    if args.workers is not None:
        cmd += ["--workers", str(args.workers)]
    print(f"\n=== {mod} ===", flush=True)
    t = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    dt = (time.time() - t) / 60.0
    print(f"    ({dt:.1f} min, rc={rc})", flush=True)
    if rc != 0:
        print(f"FAILED: {mod}", file=sys.stderr)
        sys.exit(rc)
    return dt


def check_stage1_gate(force: bool) -> None:
    """V2 spec §4: 'if falsified (KL < 0.10), stop and report'."""
    path = RESULTS / "e6b_verdict.json"
    if not path.exists():
        print("Stage-1 verdict not found; run stage 1 first.", file=sys.stderr)
        sys.exit(2)
    v = json.loads(path.read_text(encoding="utf-8"))
    if v.get("FALSIFIED"):
        msg = ("\n" + "!" * 76 +
               "\nE6b WAS FALSIFIED.\n"
               f"  naive KL at f={v['at_f']} = {v['observed_naive_kl_biased']:.4f} "
               f"< threshold {v['falsification_threshold']}\n"
               "  " + v["verdict"] + "\n" + "!" * 76)
        print(msg, file=sys.stderr)
        if not force:
            print("Stopping, as the spec requires. Use --force to continue anyway "
                  "(and record it as a deviation).", file=sys.stderr)
            sys.exit(3)
        print("--force given: continuing past a falsified stage-1 gate.", file=sys.stderr)


def check_n11_gate(force: bool) -> None:
    """N11 gates E8: 'if generational decay appears without any contamination, the recursion
    loop itself is lossy and every E8 result is an artifact'."""
    print("\n--- N11 gate: zero-contamination recursion must not degrade ---", flush=True)
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                         "tests/test_nulls_v2.py", "-k", "N11"], cwd=ROOT).returncode
    if rc != 0:
        print("\nN11 FAILED — the recursion loop is lossy. E8 will not run: its results "
              "would be implementation artifacts rather than findings.", file=sys.stderr)
        if not force:
            sys.exit(4)
        print("--force given: running E8 despite a failed N11.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Run the V2 programme, gated per spec §4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stage", type=int, default=None, help="run only this stage")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="override a failed gate (records as a deviation)")
    args = ap.parse_args()

    stages = ([args.stage] if args.stage is not None
              else [s for s in sorted(STAGES) if s >= args.from_stage])
    timings, t0 = {}, time.time()

    for s in stages:
        title, mods = STAGES[s]
        print("\n" + "=" * 76)
        print(f"STAGE {s} — {title}")
        print("=" * 76, flush=True)
        if s == 6:
            check_n11_gate(args.force)
        for mod in mods:
            timings[mod] = _run(mod, args)
        if s == 1 and args.stage is None:
            check_stage1_gate(args.force)

    if not args.no_tests and args.stage is None:
        print("\n=== pytest (all nulls and invariants) ===", flush=True)
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode
        if rc != 0:
            sys.exit(rc)

    print("\n" + "=" * 76)
    print(f"V2 programme complete in {(time.time() - t0) / 60:.1f} min")
    print("=" * 76)
    print("Wall-clock per experiment (V2 spec §4 requires this in RESULTS_V2.md):")
    for mod, dt in timings.items():
        print(f"  {mod:52s} {dt:6.1f} min")
    print("\nCSVs in results/, figures in figures/. Now (re)write RESULTS_V2.md from the CSVs.")


if __name__ == "__main__":
    main()
