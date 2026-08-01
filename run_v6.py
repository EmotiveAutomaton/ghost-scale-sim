"""Run the V6 programme: E35 through E43, plus the null suite.

    python run_v6.py                 # everything
    python run_v6.py --only E36 E41  # a subset
    python run_v6.py --quick         # smoke scale

Writes to results/v6/ and nowhere else, so a V6 run can never overwrite or be mistaken for the
committed V1-V5 record. RESULTS_V6.md is generated afterwards by scripts/write_results_v6.py and
is never hand-written.

THE PRE-REGISTRATION IS WRITTEN AND VERIFIED BEFORE ANY CELL RUNS. If the locked file has been
edited since it was written, the whole programme refuses to start.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ghostscale.config import load_config
from ghostscale.prereg_v6 import assert_prereg_locked_v6, write_preregistration_v6
from ghostscale.v6 import PREREG_PATH, v6_dir

STAGES = [
    ("E35", "ghostscale.v6.e35_depletion", "depletion, and whether it carries to unseen work"),
    ("E36", "ghostscale.v6.e36_process", "process recovery, the ordering claim, scale invariance"),
    ("E37", "ghostscale.v6.e37_wall", "the wall: legible and empty"),
    ("E38", "ghostscale.v6.e38_expertise", "does AI literacy stack or substitute"),
    ("E39", "ghostscale.v6.e39_tool", "the tool hypothesis"),
    ("E40", "ghostscale.v6.e40_cues", "aesthetics, endorsement, and the RLHF decoupling"),
    ("E41", "ghostscale.v6.e41_coupling", "two mechanisms for one phenomenon"),
    ("E42", "ghostscale.v6.e42_vulnerability", "engagement is not integration"),
    ("E43", "ghostscale.v6.e43_selfreport", "automaticity hides the work from its author"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(quick=args.quick)
    cfg.set("inference.exact", True)

    write_preregistration_v6(cfg, PREREG_PATH)
    locked = assert_prereg_locked_v6(PREREG_PATH)
    print(f"pre-registration locked at {locked['content_hash'][:16]}")

    scale = dict(n_obs=12, n_timesteps=12, forced_k=6) if args.quick else {}
    summary_path = v6_dir() / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}

    for name, module_path, blurb in STAGES:
        if args.only and name not in args.only:
            continue
        print(f"\n=== {name}: {blurb}")
        t0 = time.time()
        try:
            mod = __import__(module_path, fromlist=["run"])
            kwargs = dict(workers=args.workers)
            if args.quick:
                kwargs.update(n_obs=scale["n_obs"], n_timesteps=scale["n_timesteps"],
                              forced_k=scale["forced_k"])
                if name == "E35":
                    kwargs = dict(workers=args.workers, n_readers=6, n_encounters=6,
                                  n_timesteps=12, forced_k=6)
            elif name == "E35":
                kwargs = dict(workers=args.workers)
            verdict = mod.run(cfg, **kwargs)
            summary[name] = {k: v for k, v in verdict.items()
                             if k not in ("cells", "false_label_reference", "corners",
                                          "decoupling")}
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                      # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        # Written after EVERY stage, not at the end. The repair pass learned this the hard way:
        # a long run that stopped lost work it had already finished, which is exactly the
        # pressure that makes someone report a partial run as complete.
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
