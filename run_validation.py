#!/usr/bin/env python
"""The validation pass — V-1 through V-9, in the order the spec gates them.

    python run_validation.py                    # the whole pass
    python run_validation.py --stage 1          # one stage
    python run_validation.py --from-stage 4     # resume
    python run_validation.py --quick            # tiny scale, NOT reportable

-----------------------------------------------------------------------------------------
WHAT GATES WHAT, and the gates are real rather than decorative.

    1  V-1, the solver check      if the interior peak moves, stop and re-anchor first
    2  V-2, nulls + random-model  any headline inside the random distribution is pulled
                                  from the figures
    3  V-3, V-4                   scoping; changes wording rather than blocking
    4  V-5 to V-7                 cheap, run alongside
    5  V-8, V-9                   V-8 does not block release; V-9 is written now, built later

Stage 1's gate is the one with teeth. The interior peak on the readability axis is the location a
great deal of this project is anchored to, including the prediction card written for a human study;
if exact inference moves it, everything downstream is describing a different place and there is no
point computing robustness matrices for it first. So a stage-1 failure STOPS THE PASS unless
``--ignore-gates`` is passed, and passing that flag is recorded in the summary so a reader knows.

-----------------------------------------------------------------------------------------
VALIDATION.md IS NOT WRITTEN HERE. It is written from the verdict files after the runs, by
``scripts/write_validation_md.py``, which is the spec's own constraint: a document generated from
the same expectations that shaped the checks would agree with them by construction.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ghostscale.config import load_config
from ghostscale.experiments import _common as C
from ghostscale.validation import criteria as CR
from ghostscale.validation import validation_dir


STAGES = {
    1: ("V-1  the solver check", "gates everything: if the interior peak moves, stop"),
    2: ("V-2  scrambled provenance and the random-model false-positive rate",
        "gates the figures: nothing inside the random distribution gets a chart"),
    3: ("V-3, V-4  robustness and construction audit", "scoping; changes wording, not blocking"),
    4: ("V-5, V-6, V-7  superseded criteria, cross-version consistency, seed and scale",
        "cheap, run alongside"),
    5: ("V-8, V-9  independent reimplementation and the forward prediction",
        "V-8 upgrades or downgrades the strongest claim; V-9 is written now and built later"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=None, help="run only this stage")
    ap.add_argument("--from-stage", type=int, default=1, help="resume from this stage")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny scale for development; the outputs are NOT reportable")
    ap.add_argument("--ignore-gates", action="store_true",
                    help="continue past a failed gate; recorded in the summary")
    args = ap.parse_args()

    workers = args.workers or C.default_workers()
    cfg = load_config()
    if args.quick:
        cfg.set("validation.n_observers", 8)
        cfg.set("validation.n_seeds", 3)
        cfg.set("validation.random_model_draws", 12)

    # The criteria are written and hash-locked before the first check runs.
    criteria_path = validation_dir() / "criteria.json"
    criteria = CR.ensure_criteria(cfg, criteria_path)
    print(f"criteria locked: {criteria['content_hash'][:16]}  ({criteria_path})")
    print(f"scale: {cfg.get('validation.n_observers')} readers x "
          f"{cfg.get('validation.n_seeds')} seeds"
          + ("   [QUICK — NOT REPORTABLE]" if args.quick else ""))

    stages = [args.stage] if args.stage else [s for s in sorted(STAGES) if s >= args.from_stage]

    # A PARTIAL RUN MERGES INTO THE EXISTING SUMMARY rather than replacing it. Writing a fresh
    # summary from `--stage 5` would leave a file claiming the pass consists of one stage, and
    # VALIDATION.md reads its header out of that file. Learned by doing it.
    summary_path = validation_dir() / "summary.json"
    summary = {"quick": bool(args.quick), "ignore_gates": bool(args.ignore_gates),
               "criteria_hash": criteria["content_hash"], "stages": {}}
    if summary_path.exists() and (args.stage or args.from_stage > 1):
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["stages"] = dict(prior.get("stages") or {})
            summary["partial_run"] = True
            summary["stages_rerun_this_invocation"] = [str(s) for s in stages]
            # A quick stage layered onto a full pass makes the whole file unreportable, and the
            # more cautious of the two flags has to win.
            summary["quick"] = bool(prior.get("quick") or args.quick)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    started = time.time()

    for stage in stages:
        title, gate = STAGES[stage]
        print(f"\n{'=' * 78}\nstage {stage} — {title}\n  gate: {gate}\n{'=' * 78}")
        t0 = time.time()
        result = _run_stage(stage, cfg, workers)
        elapsed = round(time.time() - t0, 1)
        summary["stages"][str(stage)] = {"title": title, "seconds": elapsed,
                                         "verdicts": _verdicts(result)}
        for name, v in _verdicts(result).items():
            print(f"  {name}: {v}")
        print(f"  ({elapsed}s)")

        noted = _gate_notes(stage, result)
        if noted:
            summary.setdefault("gate_notes", []).append({"stage": stage, "note": noted})
            print(f"\n  GATE NOTE: {noted}")

        blocked = _gate_blocks(stage, result)
        if blocked and not args.ignore_gates:
            summary["halted_at_stage"] = stage
            summary["halt_reason"] = blocked
            print(f"\nSTOPPED at stage {stage}: {blocked}\n"
                  f"Re-anchor the affected claims before running anything downstream, or pass "
                  f"--ignore-gates to continue with that recorded.")
            break
        if blocked:
            summary.setdefault("gates_ignored", []).append({"stage": stage, "reason": blocked})
            print(f"\n  GATE FAILED AND IGNORED: {blocked}")

    summary["total_seconds"] = round(time.time() - started, 1)
    (validation_dir() / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {validation_dir() / 'summary.json'}  ({summary['total_seconds']}s total)")
    print("next: python scripts/write_validation_md.py")


def _run_stage(stage: int, cfg, workers: int):
    if stage == 1:
        from ghostscale.validation import v1_solver
        return {"v1": v1_solver.run(cfg, workers=workers)}
    if stage == 2:
        from ghostscale.validation import v2_nulls
        return {"v2": v2_nulls.run(cfg, workers=workers)}
    if stage == 3:
        from ghostscale.validation import v3_robustness, v4_construction
        return {"v3": v3_robustness.run(cfg, workers=workers),
                "v4": v4_construction.run(cfg, workers=workers)}
    if stage == 4:
        from ghostscale.validation import v5_v7_secondary
        return v5_v7_secondary.run(cfg, workers=workers)
    if stage == 5:
        from ghostscale.validation import v8_v9
        return v8_v9.run(cfg, workers=workers)
    raise ValueError(f"no stage {stage}")


def _verdicts(result: dict) -> dict:
    return {k: (v.get("verdict") if isinstance(v, dict) else str(v))
            for k, v in result.items()}


def _gate_blocks(stage: int, result: dict) -> str | None:
    """The one HALTING gate. Only stage 1 has it, and the spec is why.

    Everything downstream of the readability axis is describing a location on it. Computing a
    robustness matrix for a peak that has moved is not just wasted time, it is misleading output:
    the matrix would be about the wrong place and would read as though it were about the right one.
    """
    if stage == 1:
        v1 = result.get("v1", {})
        if v1.get("verdict") == "PEAK_MOVED_RE_ANCHOR_EVERYTHING":
            return ("the interior peak on the readability axis moved under exact inference. Every "
                    "claim anchored to that location has to be re-anchored before anything "
                    "downstream means what it says.")
    return None


def _gate_notes(stage: int, result: dict) -> str | None:
    """Gates that change what the OUTPUT is allowed to say without stopping the pass.

    Surfaced here rather than left to be discovered in a verdict file later, because the thing they
    change is which claims are allowed a figure, and figures get built after this runs.
    """
    if stage == 2:
        b = result.get("v2", {}).get("random_parameter_false_positive_rate", {})
        if b.get("verdict") in ("EFFECT_IS_ARCHITECTURE_DEPENDENT",
                                "EFFECT_DOES_NOT_SEPARATE_FROM_RANDOM"):
            return ("a headline effect sits inside the distribution a randomly parameterised model "
                    "of the same architecture produces. It is reported as architecture-dependent "
                    "and it does not get a figure. The remaining checks are still informative, so "
                    "the pass continues.")
        if b.get("verdict") == "EFFECT_SEPARATES_BUT_APPARATUS_IS_PERMISSIVE":
            return ("the headline separates from the random distribution, but the apparatus's "
                    "false-positive rate is above the alpha committed before the run. Every "
                    "borderline finding has to be read against that rate, and the figures may not "
                    "imply a zero baseline.")
    if stage == 5:
        v8 = result.get("v8", {})
        if v8.get("verdict") == "MECHANISM_REPLICATES_MAGNITUDE_DOES_NOT":
            return ("the independent reimplementation reproduces the mechanism and not the "
                    "magnitude. The specific multiple may not be quoted as though it transfers; "
                    "the direction may.")
        if v8.get("verdict") == "DOES_NOT_REPLICATE":
            return ("the strongest result did not replicate in independent code. It comes out of "
                    "the public-facing material until that is understood.")
    return None


if __name__ == "__main__":
    main()
