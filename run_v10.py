"""Run the V10 programme — the last simulation version.

    python run_v10.py
    python run_v10.py --only E56
    python run_v10.py --quick

E56 runs FIRST and that ordering is deliberate: it establishes whether the rider mechanism exists
in the reader this project has spent nine versions validating, before that mechanism is asserted of
a learner in E55.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ghostscale.config import load_config
from ghostscale.prereg_v10 import verify, write_card
from ghostscale.v10 import v10_dir

STAGES = [
    ("E56", "ghostscale.v10.e56_selective_gate",
     "is the gate selective -- the tennis players"),
    ("E57", "ghostscale.v10.e57_arms_race",
     "the arms race, and the reader who stopped updating"),
    ("E54R", "ghostscale.v10.e54r_rescore",
     "E54 rescored on error rather than movement"),
    ("E55", "ghostscale.v10.e55_intent_gate",
     "intent-gated learning -- can reading the maker defend you?"),
    ("S2", "ghostscale.v10.s2_severity",
     "how much of any of this was ever the theory"),
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
        "E56": dict(n_obs=12, n_timesteps=12) if q else dict(n_obs=40, n_timesteps=16),
        "E57": dict(n_obs=12, n_timesteps=12) if q else dict(n_obs=40, n_timesteps=16),
        "E54R": dict(n_readers=8, n_encounters=8, n_timesteps=12) if q
        else dict(n_readers=24, n_encounters=16, n_timesteps=16),
        "E55": dict(n_artifacts=16, n_learners=1, infer_steps=6) if q
        else dict(n_artifacts=40, n_learners=3, infer_steps=8),
        "S2": dict(n_draws=3, n_artifacts=12, n_obs=8) if q
        else dict(n_draws=12, n_artifacts=24, n_obs=16),
    }

    summary_path = v10_dir() / "summary.json"
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
            summary[name] = mod.run(cfg, workers=args.workers, **scale[name])
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                      # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
