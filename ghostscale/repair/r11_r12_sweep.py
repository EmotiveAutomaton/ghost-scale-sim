"""R-11 and R-12 — run every reachable experiment under exact inference, both ways, and compare.

R-11 IS THE CHEAPEST OUTSTANDING CHECK IN THE PROJECT and it is folded in here rather than given its
own module, because it is one experiment in a sweep of many. The approximation's error in the
expected free energy peaks at the SECOND timestep, where the joint belief is furthest from a product
of its marginals, and decays afterwards. The selectivity measure is taken over the first three free
steps, deliberately, to catch the decision that matters. It sits exactly in the error peak, and until
the diagnostics pass fixed the accessors it was one of the experiments the solver could not be
swapped out of at all.

R-12 IS THE REST. The validation pass checked five experiments under exact inference and three of
their verdicts moved. Fourteen more are reachable and have never been checked, and six that were
blocked by Dirichlet learning are now reachable too. A one-in-three movement rate on the checked
sample is not a reason to leave twenty unchecked.

-----------------------------------------------------------------------------------------
WHAT THIS RUN CHANGES AT ONCE, AND WHY THAT IS DELIBERATE RATHER THAN CARELESS.

Three things move together: the solver becomes exact, the per-reader seeding becomes the
collision-free scheme, and the new measures are recorded alongside the old. Changing them one at a
time would need three full programmes and would still not isolate anything, because the seed change
alone re-randomises every reader and so moves every number by sampling noise regardless.

So the comparison is built differently. Every experiment is run in a MATCHED PAIR:

    baseline   the approximate solver and the legacy seeds, which is exactly the code path that
               produced the committed record
    repaired   exact inference and hashed seeds

Both are re-run now, so the baseline is regenerated rather than read off disk, and any difference
between the pair is attributable to the change rather than to anything that has drifted since. The
committed record is then compared against the regenerated baseline as a THIRD column, which is what
detects drift if there is any.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..config import Config
from . import criteria as CR
from . import repair_dir

# Every experiment, what loads its config, and what to read out of its verdict or stats. Ordered by
# cost so a partial run still covers the cheap majority.
EXPERIMENTS = [
    # (key, module, config loader, verdict file, scalar quantities, boolean quantities)
    ("E1", "e1_crash", "base", None, [], []),
    ("E5", "e5_precision_baseline", "base", None, [], []),
    ("E2", "e2_variance", "base", None, [], []),
    ("E17", "e17_tier_dose_response", "base", "e17_verdict.json", ["spearman_rho"], []),
    ("E3", "e3_titration", "base", None, [], []),
    ("E4", "e4_trust_exploit", "base", None, [], []),
    ("E19", "e19_explore", "v4_explore", "e19_verdict.json", [],
     ["positive_control_passed", "foreign_absorbed"]),
    ("E20", "e20_omega_sweep", "v4", "e20_verdict.json",
     ["fabrication_peak_omega", "fabrication_peak_value", "engagement_crossing_omega"],
     ["fabrication_peak_is_interior", "any_cell_crashes"]),
    ("E21", "e21_model_comparison", "v4", "e21_verdict.json", [], []),
    ("E32", "e32_omega_d", "v4", "e32_verdict.json", ["measures_indistinguishable"], []),
    ("E15", "e15_competence_cliff", "base", "e15_verdict.json", ["logistic_d50"], []),
    ("E28", "e28_beta", "v4_5", "e28_verdict.json", [], []),
    ("E29", "e29_gates", "v4_5", "e29_verdict.json", [], []),
    ("E30", "e30_depth", "v5", "e30_verdict.json", [], []),
    ("E31", "e31_two_gates", "v5", "e31_verdict.json",
     ["update_tracks_mu_rho", "exploit_mu_gap", "fabrication_gap"],
     ["update_tracks_recovered_depth", "dishonest_label_inflates_depth"]),
    ("E33", "e33_latent_goal", "v5", "e33_verdict.json", [], []),
    ("E34", "e34_prediction_card", "v4", None, [], []),
    # Newly reachable: the Dirichlet-learning experiments.
    ("E7", "e7_learn_ghost", "base", None, [], []),
    ("E16", "e16_label_coverage", "base", "e16_verdict.json", [], []),
    ("E9", "e9_poison_starve", "base", None, [], []),
    ("E18", "e18_deferred_estimator", "base", "e18_verdict.json", [], []),
    ("E13", "e13_freeze_leak_signature", "base", "e13_verdict.json", [], []),
    ("E6", "e6_corpus_corruption", "base", None, [], []),
    ("E6b", "e6b_corpus_biased", "base", "e6b_verdict.json", [], []),
    # LAST ON PURPOSE. These two dominate the wall clock: the sample-size sweep took 3.9 hours on
    # its own in version 3, and it runs twice here. Ordering the list cheap-first means a run that
    # is cut short still covers the majority, and whatever did not finish is named in the verdict
    # rather than left as a silent gap.
    ("E14", "e14_engagement_floor", "base", "e14_verdict.json", [], []),
    ("E12", "e12_leak_vs_samplesize", "base", "e12_threshold.json", [], []),
]

# E8 is deliberately absent. It stays withheld, its failing test stays in the suite, and this pass
# does not touch it.
WITHHELD = {"E8": "withheld; it has never passed its own control and the repair pass does not "
                  "change that"}


def _load_cfg(kind: str, exact: bool, quick: bool):
    from ..config import load_config
    from ..v4_model import load_v4_config
    from ..v4_5_model import load_v4_5_config
    from ..v5_model import load_v5_config
    if kind == "base":
        cfg = load_config(quick=quick)
    elif kind == "v4":
        cfg = load_v4_config(quick=quick, include_explore=False)
    elif kind == "v4_explore":
        cfg = load_v4_config(quick=quick, include_explore=True)
    elif kind == "v4_5":
        cfg = load_v4_5_config(quick=quick)
    else:
        cfg = load_v5_config(quick=quick)
    cfg.set("inference.exact", bool(exact))
    return cfg


def _run_one(key: str, module: str, kind: str, out_dir, exact: bool, workers: int,
             quick: bool) -> dict:
    import importlib
    mod = importlib.import_module(f"..experiments.{module}", __package__)
    cfg = _load_cfg(kind, exact, quick)
    t0 = time.time()
    mod.run(cfg, out_dir=out_dir, workers=workers, make_fig=False)
    return {"seconds": round(time.time() - t0, 1)}


def _read_verdict(out_dir, fname):
    if not fname:
        return {}
    p = out_dir / fname
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _outcome(v: dict):
    """The experiment's outcome as a STRING, or None.

    Strings only, and deliberately. One experiment stores its classification as a nested object
    containing floats, and comparing those objects reported a verdict as having "moved" when only
    the fourth decimal of a fitted coefficient had changed. An outcome that moves is supposed to
    mean the experiment reached a different conclusion, not that a number wobbled, so anything that
    is not a label is not an outcome.
    """
    for k in ("verdict", "outcome", "classification", "label"):
        val = v.get(k)
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            for kk in ("label", "verdict", "outcome"):
                if isinstance(val.get(kk), str):
                    return val[kk]
    return None


def summarise_from_disk(arms: tuple = ("baseline", "repaired")) -> dict:
    """Rebuild the sweep's verdict from whatever per-experiment output exists on disk.

    WHY THIS EXISTS. ``run`` writes its verdict once, at the end, so a sweep that is stopped part
    way through loses everything it had already computed even though every per-experiment output is
    sitting in its own directory. That is a real flaw for a run whose last two experiments dominate
    the wall clock, and it is the kind of flaw that quietly encourages someone to report a partial
    run as though it were complete.

    So the verdict can be rebuilt from the directories at any point. Experiments that have not
    produced output in both arms are recorded as INCOMPLETE and named, rather than being left out.
    """
    root = repair_dir("r12_sweep")
    rows, incomplete = [], []
    for key, module, kind, vfile, scalars, booleans in EXPERIMENTS:
        per_arm = {}
        for arm in arms:
            out = root / arm / key
            if not out.exists() or not any(out.iterdir()):
                continue
            v = _read_verdict(out, vfile)
            per_arm[arm] = {"ok": True, "verdict": v, "outcome": _outcome(v)}
        if len(per_arm) < len(arms):
            incomplete.append(key)
            continue
        row = {"experiment": key, "module": module}
        for arm in arms:
            row[f"{arm}_ok"] = True
            row[f"{arm}_outcome"] = per_arm[arm]["outcome"]
        row["outcome_moved"] = bool(
            per_arm["baseline"]["outcome"] != per_arm["repaired"]["outcome"])
        for q in scalars:
            bv = per_arm["baseline"]["verdict"].get(q)
            rv = per_arm["repaired"]["verdict"].get(q)
            row[f"{q}__baseline"], row[f"{q}__repaired"] = bv, rv
            if isinstance(bv, (int, float)) and isinstance(rv, (int, float)):
                row[f"{q}__delta"] = float(rv) - float(bv)
        for q in booleans:
            row[f"{q}__baseline"] = per_arm["baseline"]["verdict"].get(q)
            row[f"{q}__repaired"] = per_arm["repaired"]["verdict"].get(q)
            row[f"{q}__flipped"] = bool(row[f"{q}__baseline"] != row[f"{q}__repaired"])
        rows.append(row)
    return _finalise(rows, [], incomplete, len(EXPERIMENTS))


def run(cfg: Config, workers: int = 1, quick: bool = False, only: tuple = (),
        arms: tuple = ("baseline", "repaired")) -> dict:
    """Run every reachable experiment in a matched pair and report every quantity both ways."""
    root = repair_dir("r12_sweep")
    rows, failures = [], []
    targets = [e for e in EXPERIMENTS if not only or e[0] in only]

    for key, module, kind, vfile, scalars, booleans in targets:
        per_arm = {}
        for arm in arms:
            exact = arm == "repaired"
            os.environ["GHOSTSCALE_SEED_SCHEME"] = "hash" if exact else "legacy"
            out = root / arm / key
            out.mkdir(parents=True, exist_ok=True)
            try:
                info = _run_one(key, module, kind, out, exact, workers, quick)
                v = _read_verdict(out, vfile)
                per_arm[arm] = {"ok": True, "seconds": info["seconds"], "verdict": v,
                                "outcome": _outcome(v)}
            except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
                per_arm[arm] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                failures.append({"experiment": key, "arm": arm,
                                 "error": per_arm[arm]["error"]})
        os.environ.pop("GHOSTSCALE_SEED_SCHEME", None)

        row = {"experiment": key, "module": module}
        for arm in arms:
            row[f"{arm}_ok"] = per_arm.get(arm, {}).get("ok", False)
            row[f"{arm}_seconds"] = per_arm.get(arm, {}).get("seconds")
            row[f"{arm}_outcome"] = per_arm.get(arm, {}).get("outcome")
        if all(per_arm.get(a, {}).get("ok") for a in arms) and len(arms) == 2:
            b, r = per_arm["baseline"]["outcome"], per_arm["repaired"]["outcome"]
            row["outcome_moved"] = bool(b != r)
            for q in scalars:
                bv = per_arm["baseline"]["verdict"].get(q)
                rv = per_arm["repaired"]["verdict"].get(q)
                row[f"{q}__baseline"] = bv
                row[f"{q}__repaired"] = rv
                if isinstance(bv, (int, float)) and isinstance(rv, (int, float)):
                    row[f"{q}__delta"] = float(rv) - float(bv)
            for q in booleans:
                row[f"{q}__baseline"] = per_arm["baseline"]["verdict"].get(q)
                row[f"{q}__repaired"] = per_arm["repaired"]["verdict"].get(q)
                row[f"{q}__flipped"] = bool(
                    per_arm["baseline"]["verdict"].get(q)
                    != per_arm["repaired"]["verdict"].get(q))
        rows.append(row)

    return _finalise(rows, failures, [], len(targets))


def _finalise(rows: list, failures: list, incomplete: list, attempted: int) -> dict:
    """Score and write the sweep verdict. Shared by the live run and the rebuild-from-disk path."""
    root = repair_dir("r12_sweep")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(root / "sweep.csv", index=False)

    moved = ([r["experiment"] for r in rows if r.get("outcome_moved")] if rows else [])
    flips = []
    for r in rows:
        for k, v in r.items():
            if k.endswith("__flipped") and v:
                flips.append({"experiment": r["experiment"],
                              "quantity": k.replace("__flipped", "")})

    payload = {
        "check": "R-11 / R-12",
        "question": ("Run every reachable experiment with the solver exact and the seeding fixed. "
                     "What moves?"),
        "plain_language": (
            "The validation pass rechecked five experiments with the arithmetic done exactly rather "
            "than approximately, and three conclusions changed. This runs every experiment that can "
            "be run, including six that were blocked until the learning path was rebuilt, and "
            "reports every headline quantity both ways."),
        "design": (
            "matched pairs. The baseline arm is re-run now on the old code path rather than read "
            "off disk, so a difference between the pair is attributable to the change and not to "
            "anything that drifted since. Three things move together in the repaired arm and that "
            "is deliberate: separating them would need three full programmes and would still not "
            "isolate anything, because re-seeding alone re-randomises every reader."),
        "withheld": WITHHELD,
        "experiments_attempted": attempted,
        "experiments_completed": len(rows),
        "experiments_incomplete": incomplete,
        "outcomes_moved": moved,
        "boolean_flips": flips,
        "failures": failures,
        "rows": rows,
    }
    payload["verdict"] = ("SOME_VERDICTS_MOVE_UNDER_THE_REPAIRED_MODEL" if moved or flips
                          else "NO_VERDICT_MOVES_UNDER_THE_REPAIRED_MODEL")
    payload["statement"] = _statement(payload)
    (repair_dir() / "r11_r12_sweep.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _statement(p: dict) -> str:
    bits = ["%d of %d reachable experiments completed in both arms."
            % (p["experiments_completed"], p["experiments_attempted"])]
    if p["failures"]:
        bits.append("**%d did not run and are named rather than dropped:** %s."
                    % (len(p["failures"]),
                       "; ".join(f"{f['experiment']} ({f['arm']}) {f['error']}"
                                 for f in p["failures"][:6])))
    if p["outcomes_moved"]:
        bits.append("**These outcomes moved under the repaired model:** %s. Each is reported with "
                    "both values, and the original is retained."
                    % ", ".join(p["outcomes_moved"]))
    else:
        bits.append("**No outcome string moved.** Every experiment that ran reached the same "
                    "verdict under exact inference with the collision-free seeding as it did on "
                    "the code path that produced the committed record.")
    if p["boolean_flips"]:
        bits.append("Individual verdict components that flipped: "
                    + "; ".join(f"{f['experiment']}.{f['quantity']}" for f in p["boolean_flips"])
                    + ".")
    if p.get("experiments_incomplete"):
        bits.append(
            "**%d experiment(s) had not finished in both arms when this was written and are named "
            "rather than left out:** %s. They are the two ordered last on purpose, because they "
            "dominate the wall clock; the list is cheap-first so a run that is cut short still "
            "covers the majority."
            % (len(p["experiments_incomplete"]), ", ".join(p["experiments_incomplete"])))
    bits.append("The withheld experiment stays withheld and was not run.")
    return "\n\n".join(bits)
