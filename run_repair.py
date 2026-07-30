#!/usr/bin/env python
"""The repair pass. From demonstration to measurement.

    python run_repair.py                    # everything, in order
    python run_repair.py --stage 1          # one stage
    python run_repair.py --from-stage 3     # resume
    python run_repair.py --quick            # tiny scale, NOT reportable
    python run_repair.py --skip-sweep       # everything except the long matched-pair sweep

-----------------------------------------------------------------------------------------
THE ORDER, AND WHY IT IS THIS ORDER.

    1  R-1 to R-4   recomputation on committed data. No new simulation. Three headline criteria
                    that were undetermined become determined, and the interior peak gets an error
                    bar for the first time.
    2  R-5          the uptake decomposition. It changes what several experiments were measuring,
                    so it lands before anything is rerun.
    3  R-6, R-8a    estimators for every parameter and the identifiability map.
    4  R-8b         trust as something learned about a source rather than a knob.
    5  R-11, R-12   every reachable experiment, in matched pairs, exact against approximate. The
                    long one, and last of the code-driven stages so everything it exercises is
                    already repaired.
    6  R-13         the two reruns, which need the decomposition from stage 2 in place.

Nothing here gates anything. These are repairs rather than checks, and an unwelcome answer changes
what may be claimed rather than stopping the pass. The one thing that stops it is a criteria lock
that does not verify.

REPAIR.md is not written here. It is generated afterwards from the verdict files by
`scripts/write_repair_md.py`, for the reason both earlier passes had: a document generated from the
same expectations that shaped the work would agree with it by construction.
"""
from __future__ import annotations

import argparse
import json
import time

from ghostscale.config import load_config
from ghostscale.experiments import _common as C
from ghostscale.repair import criteria as CR
from ghostscale.repair import repair_dir

STAGES = {
    1: ("R-1 to R-4: recomputation, intervals, and the second disagreement statistic",
        "runs entirely on committed data; highest value per unit of work in the pass"),
    2: ("R-5: decompose uptake into movement, error reduction and the trust factor",
        "changes what several experiments were measuring, so it lands before any rerun"),
    3: ("R-6 and R-8a: an estimator for every parameter, and the identifiability map",
        "fitted to what the reader produces rather than to what it was shown"),
    4: ("R-8b: trust as something a reader learns about a named source",
        "converts a fixed disposition into an inference, and predicts a threshold"),
    5: ("R-11 and R-12: every reachable experiment, in matched pairs",
        "the long one; exact inference and collision-free seeding against the old code path"),
    6: ("R-13: the depth experiment and the generous fallback, rerun",
        "needs the decomposition from stage 2"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny scale for development; the outputs are NOT reportable")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip stage 5, which dominates the wall clock")
    args = ap.parse_args()

    workers = args.workers or C.default_workers()
    cfg = load_config()
    scale = {"r6_seeds": 6, "r8b_encounters": 30, "r8b_readers": 20,
             "r13_readers": 40, "r13_seeds": 8}
    if args.quick:
        scale = {"r6_seeds": 2, "r8b_encounters": 8, "r8b_readers": 5,
                 "r13_readers": 8, "r13_seeds": 2}

    lock = repair_dir() / "criteria.json"
    payload = CR.ensure_criteria(cfg, lock)
    print(f"criteria locked: {payload['content_hash'][:16]}  ({lock})")
    print(f"scale: {json.dumps(scale)}"
          + ("   [QUICK — NOT REPORTABLE]" if args.quick else ""))

    stages = [args.stage] if args.stage else [s for s in sorted(STAGES) if s >= args.from_stage]
    if args.skip_sweep:
        stages = [s for s in stages if s != 5]

    summary_path = repair_dir() / "summary.json"
    summary = {"quick": bool(args.quick), "criteria_hash": payload["content_hash"],
               "scale": scale, "stages": {}}
    if summary_path.exists() and (args.stage or args.from_stage > 1 or args.skip_sweep):
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
        result = _run_stage(stage, cfg, workers, scale, args.quick)
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
    print("next: python scripts/write_repair_md.py")


def _run_stage(stage: int, cfg, workers: int, scale: dict, quick: bool):
    if stage == 1:
        from ghostscale.repair import tier1_recompute
        return {"tier1": tier1_recompute.run(cfg, workers=workers)}
    if stage == 2:
        from ghostscale.repair import r5_uptake
        return {"r5": r5_uptake.run(cfg, workers=workers)}
    if stage == 3:
        from ghostscale.repair import r6_estimators
        return {"r6": r6_estimators.run(cfg, workers=workers, seeds=scale["r6_seeds"])}
    if stage == 4:
        from ghostscale.repair import r8b_learned_trust
        return {"r8b": r8b_learned_trust.run(cfg, workers=workers,
                                             n_encounters=scale["r8b_encounters"],
                                             n_readers=scale["r8b_readers"])}
    if stage == 5:
        from ghostscale.repair import r11_r12_sweep
        return {"sweep": r11_r12_sweep.run(cfg, workers=workers, quick=quick)}
    if stage == 6:
        from ghostscale.repair import r13_reruns
        return {"r13": r13_reruns.run(cfg, workers=workers,
                                      n_readers=scale["r13_readers"],
                                      n_seeds=scale["r13_seeds"])}
    raise ValueError(f"no stage {stage}")


if __name__ == "__main__":
    main()
