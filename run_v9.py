"""Run the V9 programme — the last modelling version.

    python run_v9.py
    python run_v9.py --only MIN
    python run_v9.py --quick

The minimal-model programme runs first because it is the piece the version exists for. E53 and E54
ride along; both came out of the author's reading of places the published literature disagrees with
the simulation, and both are attempts to reconcile rather than to pick a side.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ghostscale.config import load_config
from ghostscale.prereg_v9 import verify, write_card
from ghostscale.v9 import v9_dir

STAGES = [
    ("MIN", "ghostscale.v9.minimal_models",
     "which structural commitment is each finding actually made of"),
    ("E53E54", "ghostscale.v9.e53_e54",
     "the surface detector, and the adversarial mode"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(quick=args.quick)
    cfg.set("inference.exact", True)

    card = write_card()
    assert verify(), "pre-registration hash does not verify"
    print(f"pre-registration locked at {card['sha256'][:16]}")

    q = args.quick
    scale = {
        "MIN": dict(n_obs=8, n_timesteps=10, forced_k=6) if q
        else dict(n_obs=20, n_timesteps=14, forced_k=8),
        "E53E54": dict(n_obs=12, n_timesteps=12, n_readers=8, n_encounters=8) if q
        else dict(n_obs=40, n_timesteps=16, n_readers=24, n_encounters=16),
    }

    summary_path = v9_dir() / "summary.json"
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
            summary[name] = mod.run(cfg, **scale[name])
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                      # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
