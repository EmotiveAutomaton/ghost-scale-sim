"""Run the V8 programme.

    python run_v8.py
    python run_v8.py --only E48 E49
    python run_v8.py --quick

The severity pass runs LAST rather than first, deliberately: it sweeps every result the programme
has produced, so it needs them to exist. Its findings gate what may be CLAIMED, not what may be run.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ghostscale.config import load_config
from ghostscale.prereg_v8 import assert_prereg_locked_v8, write_preregistration_v8
from ghostscale.v8 import PREREG_PATH, v8_dir

STAGES = [
    ("E48", "ghostscale.v8.e48_reader_depth", "a reader can only see as far as it has built"),
    ("E49", "ghostscale.v8.e49_density", "the readymade: density rather than volume"),
    ("E50", "ghostscale.v8.e50_grab_vs_keep", "grabbing attention and keeping it"),
    ("E51", "ghostscale.v8.e51_creator", "a maker that can lie, and whether honesty pays"),
    ("E52", "ghostscale.v8.e52_avoidance", "an intent defined by what it will not do"),
    ("S1", "ghostscale.v8.s1_severity", "how often a random model of this shape does it too"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(quick=args.quick)
    cfg.set("inference.exact", True)

    write_preregistration_v8(cfg, PREREG_PATH)
    locked = assert_prereg_locked_v8(PREREG_PATH)
    print(f"pre-registration locked at {locked['content_hash'][:16]}")

    q = args.quick
    scale = {
        "E48": dict(n_obs=8, n_timesteps=12, n_exposures=4) if q
        else dict(n_obs=40, n_timesteps=24, n_exposures=12),
        "E49": dict(n_obs=16) if q else dict(n_obs=80),
        "E50": dict(n_obs=16, n_timesteps=12) if q else dict(n_obs=80, n_timesteps=16),
        "E51": dict(n_makers=6, n_encounters=4, n_timesteps=12) if q
        else dict(n_makers=24, n_encounters=12, n_timesteps=16),
        "E52": dict(n_obs=16, n_timesteps=12) if q else dict(n_obs=80, n_timesteps=20),
        "S1": dict(n_draws=8, n_obs=6, n_timesteps=8) if q
        else dict(n_draws=60, n_obs=12, n_timesteps=12),
    }

    summary_path = v8_dir() / "summary.json"
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
            verdict = mod.run(cfg, workers=args.workers, **scale[name])
            summary[name] = {k: v for k, v in verdict.items()
                             if k not in ("cells", "curve", "rows", "equilibrium")}
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                      # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
