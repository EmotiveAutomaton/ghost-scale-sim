#!/usr/bin/env python
"""The diagnostics pass. Two probes from the spec, five checks that came out of reading the code.

    python run_diagnostics.py                  # everything, in order
    python run_diagnostics.py --stage 1        # one stage
    python run_diagnostics.py --from-stage 2   # resume
    python run_diagnostics.py --quick          # tiny scale, NOT reportable

-----------------------------------------------------------------------------------------
WHY THE ORDER IS THE ORDER.

    1  D-1, D-3, D-4, D-5, D-6   the cheap checks. Two of them change what the expensive sweeps
                                 should sweep, so they go first. D-1 needs no simulation at all and
                                 predicts several parameter-sweep outcomes from arithmetic.
    2  P-1                       parameter recovery. Needs `ghostscale/fitting.py`, which had to be
                                 written because three of the four parameters are not hidden states
                                 and the spec's estimator does not name an estimator for them.
    3  P-2                       the difficulty probe, with a fourth knob added because two of the
                                 spec's three were measured dead before the criteria were locked.
    4  D-2                       the uptake response curve. Runs last because it is only interesting
                                 once P-2 has established where on the difficulty axis anything sits.

Nothing here gates anything, and that is deliberate. These are diagnostics on the apparatus rather
than experiments, so an unwelcome answer changes what the repair should be rather than stopping the
pass. The one thing that would stop it is a criteria lock that does not verify, and that is checked
before the first check runs.

`DIAGNOSTICS.md` is not written here. It is generated from the verdict files afterwards by
`scripts/write_diagnostics_md.py`, for the reason the validation pass had: a document generated from
the same expectations that shaped the checks would agree with them by construction.
"""
from __future__ import annotations

import argparse
import json
import time

from ghostscale.config import load_config
from ghostscale.diagnostics import criteria as CR
from ghostscale.diagnostics import diagnostics_dir
from ghostscale.experiments import _common as C

STAGES = {
    1: ("the cheap checks: D-1 channels, D-3 disagreement, D-4 coverage, D-5 power, D-6 seeds",
        "two of these change what the expensive sweeps should sweep, so they run first"),
    2: ("P-1 parameter recovery",
        "an unidentifiable parameter cannot support a claim; a compressed one changes what may be "
        "said"),
    3: ("P-2 the goal-difficulty probe",
        "finds whether a regime exists where goal recovery is genuinely uncertain"),
    4: ("D-2 the uptake response curve",
        "only interesting once P-2 has placed things on the difficulty axis"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny scale for development; the outputs are NOT reportable")
    args = ap.parse_args()

    workers = args.workers or C.default_workers()
    cfg = load_config()
    scale = {"p1_seeds": 16, "p2_readers": 60, "p2_seeds": 6, "d2_readers": 80, "d2_seeds": 6}
    if args.quick:
        scale = {"p1_seeds": 3, "p2_readers": 10, "p2_seeds": 2, "d2_readers": 12, "d2_seeds": 2}

    lock = diagnostics_dir() / "criteria.json"
    payload = CR.ensure_criteria(cfg, lock)
    print(f"criteria locked: {payload['content_hash'][:16]}  ({lock})")
    print(f"scale: {json.dumps(scale)}"
          + ("   [QUICK — NOT REPORTABLE]" if args.quick else ""))

    stages = [args.stage] if args.stage else [s for s in sorted(STAGES) if s >= args.from_stage]
    summary_path = diagnostics_dir() / "summary.json"
    summary = {"quick": bool(args.quick), "criteria_hash": payload["content_hash"],
               "scale": scale, "stages": {}}
    if summary_path.exists() and (args.stage or args.from_stage > 1):
        # Merge rather than replace, so a partial run does not leave a file claiming the pass is one
        # stage long. The validation runner learned this the hard way.
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["stages"] = dict(prior.get("stages") or {})
            summary["partial_run"] = True
            summary["quick"] = bool(prior.get("quick") or args.quick)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    started = time.time()
    for stage in stages:
        title, why = STAGES[stage]
        print(f"\n{'=' * 78}\nstage {stage} — {title}\n  {why}\n{'=' * 78}")
        t0 = time.time()
        result = _run_stage(stage, cfg, workers, scale)
        elapsed = round(time.time() - t0, 1)
        verdicts = {k: (v.get("verdict") if isinstance(v, dict) else str(v))
                    for k, v in result.items()}
        summary["stages"][str(stage)] = {"title": title, "seconds": elapsed, "verdicts": verdicts}
        for name, v in verdicts.items():
            print(f"  {name}: {v}")
        print(f"  ({elapsed}s)")

    summary["total_seconds"] = round(time.time() - started, 1)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary_path}  ({summary['total_seconds']}s total)")
    print("next: python scripts/write_diagnostics_md.py")


def _run_stage(stage: int, cfg, workers: int, scale: dict):
    if stage == 1:
        from ghostscale.diagnostics import (d1_channels, d3_disagreement, d4_coverage,
                                            d5_d6_power_and_seeds)
        out = {"d1": d1_channels.run(cfg, workers=workers)}
        out["d3"] = d3_disagreement.run(cfg, workers=workers)
        out["d4"] = d4_coverage.run(cfg, workers=workers)
        out.update(d5_d6_power_and_seeds.run(cfg, workers=workers))
        return out
    if stage == 2:
        from ghostscale.diagnostics import p1_recovery
        return {"p1": p1_recovery.run(cfg, workers=workers, seeds=scale["p1_seeds"])}
    if stage == 3:
        from ghostscale.diagnostics import p2_difficulty
        return {"p2": p2_difficulty.run(cfg, workers=workers, n_readers=scale["p2_readers"],
                                        n_seeds=scale["p2_seeds"])}
    if stage == 4:
        from ghostscale.diagnostics import d2_uptake
        return {"d2": d2_uptake.run(cfg, workers=workers, n_readers=scale["d2_readers"],
                                    n_seeds=scale["d2_seeds"])}
    raise ValueError(f"no stage {stage}")


if __name__ == "__main__":
    main()
