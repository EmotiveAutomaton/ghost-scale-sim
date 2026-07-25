#!/usr/bin/env python
"""Re-run E8 under V2's parameters and V2's seeding rule.

WHY THIS EXISTS. During V3 development the original ``results/e8_*.csv`` — the evidence
RESULTS_V2.md's "E8 — not reportable" section was written from — were overwritten by a
``--quick`` run that was launched without ``--out``. They are not recoverable. This script
regenerates that cell so the files hold a V2-parameter run rather than smoke-scale data.

It REPRODUCES THE REPORTED V2 RUN EXACTLY — f = 0 honest: slope +0.0119 nats/generation,
t = 3.75 — so the V2 evidence is intact and V3's premise is unaltered.

Two things this doubles as:

  * **A verification that the V3 no-averaging path IS the V2 path.** E12's control arm is only
    a control if it reproduces V2 rather than approximating it. It does, and bit-for-bit: this
    chain was also diffed against a verbatim reimplementation of V2's ``run_generation`` /
    ``run_chain`` (max |ΔKL| = 0.000e+00 over four generations).

  * **A regression test for a bug that nearly shipped.** The first two attempts returned
    +0.0067 (t = 1.31) and looked like environmental drift. They were not: ``resolve_sample_size``
    consulted ``results/e12_threshold.json`` whenever it existed, regardless of ``require_e12``,
    so a leftover threshold from a ``--quick`` E12 (120 artifacts) silently overrode the 300
    requested here. If this script stops reproducing +0.0119 / 3.75, suspect a stale artifact
    in ``results/`` before suspecting the physics.

Writes to ``results/``. V3's own experiments write to their own filenames and do not collide;
``--quick`` runs write to ``results_quick/`` and cannot reach these files at all.
"""
from __future__ import annotations

import sys

import pandas as pd

from ghostscale.config import load_config
from ghostscale.experiments import e8_recursive as E8, _common as C

# V2's E8 cell, exactly as config/default.yaml carried it before V3.
V2_E8 = {
    "experiments.e8.g_max": 4,
    "experiments.e8.min_g_max": 3,
    "experiments.e8.n_creators_next": 20,
    "experiments.e8.n_artifacts": 300,
    "experiments.e8.n_observers": 5,
    "experiments.e8.n_probes": 0,               # V2 had no encoder channel
    "experiments.e8.infer_steps": 6,
    "experiments.e8.n_replications": 3,
    "experiments.e8.d": 0.0,
    "experiments.e8.synth_draw_seed": 17,
    "experiments.e8.population_average_seed": False,   # the V2 seeding rule
}
V2_REFERENCE = {"slope": 0.0119, "t": 3.75}


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else C.default_workers()
    cfg = load_config(overrides=V2_E8)
    E8.run(cfg, workers=workers, require_e12=False)

    res_dir, _ = C.ensure_dirs(None)
    trends = pd.read_csv(res_dir / "e8_trends.csv")
    honest = trends[(trends.contamination == 0.0) & (trends.signal == "honest")
                    & (trends.metric == "kl_payload")]
    if not len(honest):
        print("no f=0 honest arm in the regenerated trends", file=sys.stderr)
        return 1
    slope, t = float(honest.slope.iloc[0]), float(honest.t.iloc[0])
    print(f"\nE8 f=0 honest arm, V2 parameters, this machine:")
    print(f"  slope {slope:+.4f}  (RESULTS_V2.md reports {V2_REFERENCE['slope']:+.4f})")
    print(f"  t     {t:+.2f}    (RESULTS_V2.md reports {V2_REFERENCE['t']:+.2f})")
    if abs(slope - V2_REFERENCE["slope"]) < 5e-4 and abs(t - V2_REFERENCE["t"]) < 0.05:
        print("  matches the reported V2 run")
        return 0
    print("  DOES NOT match the reported V2 run. Before concluding anything about the model, "
          "check for a stale artifact in results/ — a leftover e12_threshold.json silently "
          "overriding this script's sample size produced exactly this symptom once already "
          "(see the module docstring).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
