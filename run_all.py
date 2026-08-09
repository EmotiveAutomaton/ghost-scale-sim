#!/usr/bin/env python
"""Run the original V1 experiment program (E1-E6), then the tests.

Later model generations (V2-V11) have their own runners under runners/; this script is
kept under its original name for compatibility and runs only the six V1 experiments.

Usage:
    python run_all.py                # full spec scale, parallel
    python run_all.py --quick        # fast smoke scale
    python run_all.py --workers 8    # cap parallelism
    python run_all.py --no-tests     # skip pytest

Each experiment writes a tidy CSV to results/ and a figure to figures/. This is the
`make all` equivalent (Definition of Done, Spec §12).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    "ghostscale.experiments.e1_crash",
    "ghostscale.experiments.e2_variance",
    "ghostscale.experiments.e3_titration",
    "ghostscale.experiments.e4_trust_exploit",
    "ghostscale.experiments.e5_precision_baseline",
    "ghostscale.experiments.e6_corpus_corruption",
]


def main():
    ap = argparse.ArgumentParser(description="Run all Ghost Scale experiments + tests")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    t0 = time.time()
    for mod in EXPERIMENTS:
        cmd = [sys.executable, "-m", mod]
        if args.quick:
            cmd.append("--quick")
        if args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        print(f"\n=== {mod} ===", flush=True)
        t = time.time()
        rc = subprocess.run(cmd, cwd=root).returncode
        print(f"    ({(time.time() - t) / 60:.1f} min, rc={rc})", flush=True)
        if rc != 0:
            print(f"FAILED: {mod}", file=sys.stderr)
            sys.exit(rc)

    if not args.no_tests:
        print("\n=== pytest ===", flush=True)
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root).returncode
        if rc != 0:
            sys.exit(rc)

    print(f"\nAll experiments + tests complete in {(time.time() - t0) / 60:.1f} min.")
    print("CSVs in results/, figures in figures/. Now (re)write RESULTS.md from the CSVs.")


if __name__ == "__main__":
    main()
