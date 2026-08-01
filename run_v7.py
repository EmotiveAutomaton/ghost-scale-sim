"""Run the V7 programme: the four closures, then E45 through E47.

    python run_v7.py
    python run_v7.py --only E45
    python run_v7.py --quick

Writes to results/v7/ and nowhere else. RESULTS_V7.md is generated afterwards by
scripts/write_results_v7.py and is never hand-written.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ghostscale.config import load_config
from ghostscale.prereg_v7 import assert_prereg_locked_v7, write_preregistration_v7
from ghostscale.v7 import PREREG_PATH, v7_dir

STAGES = [
    ("CLOSURES", "ghostscale.v7.closures", "the four results version 6 would not draw"),
    ("E45", "ghostscale.v7.e45_tom_efficiency", "what modelling a maker actually buys"),
    ("E46", "ghostscale.v7.e46_gate_leak", "the gate cannot fully close"),
    ("E47", "ghostscale.v7.e47_coverage_under_coupling", "does the coverage figure survive"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(quick=args.quick)
    cfg.set("inference.exact", True)

    write_preregistration_v7(cfg, PREREG_PATH)
    locked = assert_prereg_locked_v7(PREREG_PATH)
    print(f"pre-registration locked at {locked['content_hash'][:16]}")

    summary_path = v7_dir() / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    q = args.quick
    scale = {
        "CLOSURES": (dict(n_obs=8, n_timesteps=12, n_readers=4, n_encounters=6) if q
                     else dict(n_obs=40, n_timesteps=24, n_readers=24, n_encounters=30)),
        "E45": (dict(n_obs=20, n_timesteps=3, forced_k=3) if q
                else dict(n_obs=150, n_timesteps=3, forced_k=3)),
        "E46": (dict(n_readers=4, n_encounters=6, n_timesteps=12) if q
                else dict(n_readers=16, n_encounters=20, n_timesteps=24)),
        "E47": (dict(n_readers=4, n_artifacts=6, n_timesteps=12) if q
                else dict(n_readers=12, n_artifacts=16, n_timesteps=24)),
    }

    for name, module_path, blurb in STAGES:
        if args.only and name not in args.only:
            continue
        print(f"\n=== {name}: {blurb}")
        t0 = time.time()
        try:
            mod = __import__(module_path, fromlist=["run"])
            verdict = mod.run(cfg, workers=args.workers, **scale[name])
            summary[name] = {k: v for k, v in verdict.items() if k not in ("curve", "rows")}
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                      # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        # Written after every stage, not at the end. The repair pass learned this the hard way.
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
