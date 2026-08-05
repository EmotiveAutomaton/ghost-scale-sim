"""Run the tests another project asked this simulation to run.

    python runners/run_soundingline.py                 # everything
    python runners/run_soundingline.py --only S1        # one test

DELIBERATELY SEPARATE FROM ``run_validation.py``. The validation pass V-1 through V-9 is a closed,
reproducible artefact of this project and adding stages to it would change what re-running it
means. This runner shares none of its state, locks no pre-registration -- these are not Ghost Scale
hypotheses -- and writes only under ``results/validation/soundingline/``.

See ``ghostscale/validation/soundingline/__init__.py`` for why nothing here may call a versioned
``run()``.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from ghostscale.config import load_config
from ghostscale.validation.soundingline import sl_dir

STAGES = [
    ("S1", "ghostscale.validation.soundingline.s1_unlock_statistic",
     "is the unlock ratio measuring what process_error_reduction measures?"),
    ("S2", "ghostscale.validation.soundingline.s2_flattened_intent",
     "does flattened intent read as posterior concentration at matched density?"),
    ("S3", "ghostscale.validation.soundingline.s3_two_channels",
     "two affect layers, and whether a louder shield gives the concealer away"),
    ("S45", "ghostscale.validation.soundingline.s45_inference_order",
     "does inference order buy accuracy, or only cost?"),
    ("S6", "ghostscale.validation.soundingline.s6_surface_decay",
     "does surface decay across an artifact while content does not?"),
    # ---- second batch -------------------------------------------------------------------
    ("T1", "ghostscale.validation.soundingline.t1_triangle",
     "chain or triangle? six directed edges, all measured the same way"),
    ("T2", "ghostscale.validation.soundingline.t2_automaticity",
     "does drive diversity read as posterior breadth, or only as difficulty?"),
    ("T3", "ghostscale.validation.soundingline.t3_countability",
     "is 'a decision was recovered' ever a well-defined event?"),
    ("T4", "ghostscale.validation.soundingline.t4_uncertain_reader",
     "does channel divergence survive a reader that does not know the channel?"),
    ("T5", "ghostscale.validation.soundingline.t5_detection",
     "auxiliary: is the process posterior a better maker-detector than the goal posterior?"),
]

SCALE = {
    "S1": dict(n_obs=120, n_timesteps=24, forced_k=24),
    "S2": dict(n_obs=120, n_timesteps=24, forced_k=24),
    "S3": dict(n_obs=400, n_emissions=12),
    "S45": dict(n_obs=60, n_timesteps=24, forced_k=24),
    "S6": dict(n_obs=400),
    "T1": dict(n_obs=200),
    "T2": dict(n_obs=200),
    "T3": dict(n_obs=150),
    "T4": dict(n_obs=400, n_emissions=12),
    "T5": dict(n_obs=200),
}
QUICK = {
    "S1": dict(n_obs=12, n_timesteps=24, forced_k=24),
    "S2": dict(n_obs=12, n_timesteps=24, forced_k=24),
    "S3": dict(n_obs=40, n_emissions=12),
    "S45": dict(n_obs=6, n_timesteps=24, forced_k=24),
    "S6": dict(n_obs=40),
    "T1": dict(n_obs=20),
    "T2": dict(n_obs=20),
    "T3": dict(n_obs=20),
    "T4": dict(n_obs=60, n_emissions=12),
    "T5": dict(n_obs=30),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    scale = QUICK if args.quick else SCALE
    summary_path = sl_dir() / "summary.json"
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
            verdict = mod.run(cfg, **scale[name])
            summary[name] = verdict
            print(f"    ok in {time.time() - t0:.1f}s")
        except Exception as exc:                       # noqa: BLE001
            summary[name] = {"failed": True, "error": repr(exc),
                             "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}")
        # Written after every stage. The repair pass learned this the hard way.
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
