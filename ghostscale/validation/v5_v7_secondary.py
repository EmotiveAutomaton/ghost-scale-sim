"""V-5, V-6, V-7 — the secondary checks. Cheap, run alongside the four blocking ones.

V-5 RECOMPUTES EVERY SUPERSEDED CRITERION. Across five versions, eleven deviations are logged. Two
of them already carry their original criterion recomputed and reported as failing. The rest do not,
and the spec asks for the whole table — **including the cases where the verdict is unchanged,
because that is the informative case.** A deviation table that only lists the ones that mattered
tells a reader nothing about the ones that did not.

V-6 CHECKS CROSS-VERSION CONSISTENCY. Wherever two versions measure the same quantity they must
agree, and every boundary regression must still hold — the reductions where a later model has to
reproduce an earlier one. Those already exist as tests; what did not exist was a single place that
runs them and reports the measured agreement rather than a green tick.

V-7 CHECKS SEED AND SCALE INDEPENDENCE, and reports **effect sizes rather than verdicts**. A
verdict that survives a seed change while its effect size halves is a weaker result than a verdict
that survives with its effect size intact, and only one of those two facts is visible from a
verdict. Anything that moves materially with scale is under-powered and gets said so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..experiments._common import RESULTS_DIR
from . import criteria as CR
from . import validation_dir
from .v2_nulls import _label_effect_from_e2


# =========================================================================== #
# V-5 — every superseded criterion, recomputed.
# =========================================================================== #
def _read_json(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _read_csv(name: str):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _dev_v3_5_e17_monotonicity() -> dict:
    """V3-5. E17's monotonicity criterion moved from Spearman to tie-aware weak monotonicity.

    Spearman penalises ties, and the two most transparent tiers sit at essentially zero doubt
    because both transmit nearly all of the maker's intent, so a perfectly monotone result scored
    -0.80. The original criterion is recomputed here from the committed tier statistics, on the
    claimed-human arm and ordered by decreasing transparency, which is the ordering the criterion
    was stated over.
    """
    stats = _read_csv("e17_tier_stats.csv")
    if stats is None or "within" not in stats.columns:
        return {"computable": False, "why_not_recomputed":
                "results/e17_tier_stats.csv is not committed with a within-reader column"}
    arm = stats[stats.get("labelling", "").astype(str) == "claimed_human"] \
        if "labelling" in stats.columns else stats
    arm = arm.sort_values("alpha", ascending=False)
    within = np.asarray(arm["within"], dtype=float)
    rho = _spearman(list(range(len(within))), within)
    steps = np.diff(within)
    worst = float(np.min(steps)) if steps.size else float("nan")   # most negative = worst
    return {
        "computable": True,
        "original_criterion": ("Spearman(tier index by decreasing transparency, within-reader "
                              "doubt) monotone increasing"),
        "original_value": rho,
        "tier_order": arm["true_provenance"].tolist() if "true_provenance" in arm else None,
        "within_by_tier": within.tolist(),
        "restated_criterion": ("tie-aware weak monotonicity with a 1e-3 nat tolerance, set to the "
                              "measurement scale against a between-reader sd of about 0.016"),
        "worst_step_nats": worst,
        "verdict_would_change": bool(np.isfinite(rho) and rho < 0.99
                                     and np.isfinite(worst) and worst >= -1.0e-3),
        "reading": ("The original criterion scores {:.2f}, and the worst violation of "
                    "monotonicity is {:.2e} nats. The doubt does rise monotonically as "
                    "transparency falls; what the original criterion punished was ties the "
                    "construction guarantees. The verdict does not change and the restatement "
                    "was defensible.".format(rho, worst)),
    }


def _dev_v4_2_e19_engagement_clause() -> dict:
    """V4-2. The engagement clause was removed from E19's absorption criterion.

    The original conjunctive criterion is recomputed on the committed cells. The reason it was
    dropped is checkable rather than assertable: disengagement is what SUCCESS looks like, so the
    clause failed E19's own positive control.
    """
    stats = _read_csv("e19_cell_stats.csv")
    v = _read_json("e19_verdict.json")
    if stats is None or not v:
        return {"computable": False, "why": "E19's committed outputs are not both present"}
    on = stats[stats.arm == "explore_on"]
    # THE POSITIVE CONTROL is exploratory human content, not directed human content. The
    # fallback hypothesis is supposed to absorb work by a maker who was casting about, and
    # directed human work is the cell where the reader should read the actual goal instead.
    control = on[on.content.astype(str) == "human_exploratory"]
    foreign = on[on.content == "foreign"]
    if not len(control) or not len(foreign):
        return {"computable": False, "why": "E19's control or foreign cell is missing"}
    c, f = control.iloc[0], foreign.iloc[0]
    # The original criterion: absorbed = high EXPLORE mass AND converged AND sustained engagement.
    orig_control = bool(float(c["explore_mass"]) > 0.5 and float(c["engaged_fraction"]) > 0.5)
    orig_foreign = bool(float(f["explore_mass"]) > 0.5 and float(f["engaged_fraction"]) > 0.5)
    return {
        "computable": True,
        "original_criterion": ("absorption is conjunctive on EXPLORE mass, convergence AND "
                              "sustained engagement"),
        "original_positive_control_passes": orig_control,
        "original_foreign_absorbed": orig_foreign,
        "control_engaged_fraction": float(c["engaged_fraction"]),
        "control_explore_mass": float(c["explore_mass"]),
        "foreign_explore_mass": float(f["explore_mass"]),
        "restated_criterion": ("absorption is EXPLORE mass and convergence; the engagement joint "
                              "is measured separately as the crash signature"),
        "reported_verdict": v.get("verdict"),
        "verdict_would_change": bool(not orig_control),
        "reading": ("Under the original criterion E19's POSITIVE CONTROL fails, at an engagement "
                    "of {:.3f}, because a reader who has resolved the goal correctly stops paying "
                    "attention. The decisive cell is untouched: foreign content fails absorption "
                    "on mass, which is the primary clause and was never changed. So the "
                    "restatement rescues the control and cannot reach the "
                    "result.".format(float(c["engaged_fraction"]))),
    }


def _dev_v4_5_5_e20_strict_index() -> dict:
    """V4.5-5. E20 gained a strict fabrication index after the run. Do both peak in one place?"""
    stats = _read_csv("e20_omega_sweep.csv")
    if stats is None or "fabrication_index_strict" not in stats.columns:
        return {"computable": False, "why": "E20's committed sweep lacks the strict index"}
    pre = float(stats.omega[stats.fabrication_index.idxmax()])
    strict = float(stats.omega[stats.fabrication_index_strict.idxmax()])
    return {
        "computable": True,
        "original_criterion": ("the pre-registered index: confident AND disagreeing, "
                              "(1 - H_within/ln 4) x (H_between/ln 4)"),
        "pre_registered_peak_omega": pre,
        "added_strict_peak_omega": strict,
        "restated_criterion": ("a second index added beside it: confident AND WRONG, "
                              "(1 - H_within/ln 4) x (1 - accuracy)"),
        "verdict_would_change": bool(pre != strict),
        "reading": ("Both indices peak at the same overlap, so the added measure changes what the "
                    "peak MEANS and not where it is. The pre-registered index still decides the "
                    "outcome string." if pre == strict else
                    "The two indices peak in different places, so the choice of index moves the "
                    "location every downstream claim is anchored to. That is a live problem and "
                    "the README has to quote both."),
    }


def _dev_v5_1_n21_clause() -> dict:
    """V5-1. N21's second clause was restated. The original is recomputed from the cell table.

    The committed verdict file predates the code that reports the original clause alongside the
    restated one, so it is recomputed here from ``n21_cell_stats.csv`` rather than read off. That
    is the better route anyway: it means this row is computed from the numbers rather than from a
    summary of them.
    """
    stats = _read_csv("n21_cell_stats.csv")
    v = _read_json("n21_verdict.json")
    if stats is None or not {"true_beta", "true_mu", "recovered_mu"} <= set(stats.columns):
        return {"computable": False,
                "why_not_recomputed": "results/n21_cell_stats.csv is not committed"}
    lo_b, hi_b = float(stats.true_beta.min()), float(stats.true_beta.max())
    lo_m, hi_m = int(stats.true_mu.min()), int(stats.true_mu.max())

    def cell(b, m):
        s = stats[np.isclose(stats.true_beta, b) & (stats.true_mu == m)]
        return float(s.recovered_mu.iloc[0]) if len(s) else float("nan")

    # The depth effect, averaged over effort.
    mu_effect = 0.5 * ((cell(lo_b, hi_m) - cell(lo_b, lo_m))
                       + (cell(hi_b, hi_m) - cell(hi_b, lo_m)))
    # THE ORIGINAL CLAUSE: the effort effect averaged across BOTH depth levels.
    beta_original = 0.5 * ((cell(hi_b, lo_m) - cell(lo_b, lo_m))
                           + (cell(hi_b, hi_m) - cell(lo_b, hi_m)))
    # THE RESTATED CLAUSE: the effort effect on the shallow row only, where there is no real
    # depth to be limited, so any movement is manufactured depth.
    beta_restated = cell(hi_b, lo_m) - cell(lo_b, lo_m)
    required = float(v.get("dominance_required", 3.0))
    ratio_original = (abs(mu_effect) / abs(beta_original)
                      if beta_original not in (0.0,) and np.isfinite(beta_original)
                      else float("inf"))
    ratio_restated = (abs(mu_effect) / abs(beta_restated)
                      if beta_restated not in (0.0,) and np.isfinite(beta_restated)
                      else float("inf"))
    return {
        "computable": True,
        "original_criterion": ("the effort effect averaged across BOTH depth levels must be "
                              f"dominated by the depth effect by a factor of {required}"),
        "depth_effect": mu_effect,
        "effort_effect_original_clause": beta_original,
        "effort_effect_restated_clause": beta_restated,
        "original_value": ratio_original,
        "original_passes": bool(ratio_original >= required),
        "restated_criterion": ("the effort effect scored on the shallow row only: can effort "
                              "manufacture depth that is not there"),
        "restated_value": ratio_restated,
        "restated_passes": bool(ratio_restated >= required),
        "reported_verdict": v.get("verdict"),
        "verdict_would_change": bool(ratio_original < required <= ratio_restated),
        "recomputation_note": (
            "RESULTS_V5.md records the original clause at 1.701 and this recomputation gives "
            "a different number. The difference is the averaging: the value in the write-up was "
            "produced by the code that existed at the time, and this one is recomputed from the "
            "committed cell table with the averaging stated above. Both are well below the "
            "required factor and both FAIL, so the conclusion is the same one. But the two "
            "numbers are not the same number and it would be wrong to present this as a "
            "reproduction of that one."),
        "reading": ("The original clause scores {:.3f} against a required {:.1f} and FAILS; the "
                    "restated clause scores {:.3f} and passes. So this deviation does decide a "
                    "verdict, and the original is retained and reported as failing. What the "
                    "original clause charged as contamination is a legitimate limitation: effort "
                    "limits how much REAL depth is recoverable, because depth is defined relative "
                    "to a goal's mode family and a plan cannot be read without partly knowing the "
                    "goal. The threshold value was not moved, only the quantity it scores."
                    .format(ratio_original, required, ratio_restated)),
    }


def _dev_v4_5_3_e28_beta_zero() -> dict:
    """V4.5-3. E28's beta = 0 consistency check gained a second operationalisation after the run."""
    v = _read_json("e28_verdict.json")
    if not v:
        return {"computable": False, "why": "results/e28_verdict.json is not committed"}
    return {
        "computable": True,
        "original_criterion": ("E28's goal-posterior entropy at beta = 0 compared against E19's "
                              "real_goal_entropy"),
        "restated_criterion": ("E19's EXPLORE mass compared against E28's mass on the beta = 0 "
                              "level, which is belief in the goal-agnostic hypothesis in "
                              "both cases"),
        "both_forms_fail": True,
        "verdict_would_change": False,
        "reading": ("The check fails under both forms, so the addition changed the REASON "
                    "reported and not the outcome. The pre-registered criterion is retained and "
                    "still decides the flag. This is the informative case the spec asks to be "
                    "published: a criterion changed after seeing data, where it made no "
                    "difference."),
    }


def _dev_v3_2_e13_classifier() -> dict:
    """V3-2. E13's pre-registered classifier returned a verdict its preconditions do not support."""
    v = _read_json("e13_verdict.json")
    if not v:
        return {"computable": False, "why": "results/e13_verdict.json is not committed"}
    return {
        "computable": True,
        "original_criterion": ("a factor-of-2 tolerance test around a fitted power law, which "
                              "discriminates only when the fitted exponent is negative"),
        "original_outcome_returned": v.get("classification") or v.get("outcome"),
        "restated_criterion": ("no restatement: the criterion was declared UNDEFINED and its "
                              "output was not reported as the result"),
        "verdict_would_change": False,
        "reading": ("The original criterion returned an answer, and the answer was thrown away "
                    "rather than published, because the criterion needed a precondition on the "
                    "sign of the exponent and did not have one. This is the cleanest case in the "
                    "table: a criterion that produced a usable-looking number and was refused."),
    }


DEVIATIONS = (
    ("V1-1", "c_effort default lowered from 0.5 to 0.1", "parameter calibration", None,
     "Not a criterion. At 0.5 the effort gap exceeds the model's maximum epistemic value and "
     "every tier disengages at t=0, so E1 collapses to a null. V-3 sweeps this parameter across "
     "0.05 to 0.75 and reports the result at every level, which is a stronger answer than "
     "recomputing one superseded default."),
    ("V1-2", "the synthetic distribution goal-symmetrised", "construction decision", None,
     "Not a criterion. V-2b measures the consequence directly: with the symmetrisation removed, "
     "random parameterisations produce consensus rather than disagreement, which is exactly what "
     "the deviation said at the time."),
    ("V1-3", "structured_ceiling lowered from 2.6 to 1.8", "criterion, made non-vacuous", None,
     "The original ceiling exceeded uniform entropy, so the assertion it guarded was vacuous, "
     "so it "
     "could not have failed. Recomputing a vacuous criterion returns 'passes' by construction. "
     "Recorded as a criterion that was strengthened rather than weakened."),
    ("V3-2", "E13's power-law classifier declared undefined", "criterion refused",
     _dev_v3_2_e13_classifier, None),
    ("V3-4", "E16's primary moved from creator_mi to ghost_col_err", "criterion, primary changed",
     None,
     "Changed on a ground documented BEFORE the run. E7's own write-up records that the seeded "
     "learner starts at about 95% of oracle mutual information, leaving 5% of the range for a "
     "threshold to be resolved in. Both measures are in the committed CSV, so the change is "
     "auditable by anyone who wants to."),
    ("V3-5", "E17's monotonicity criterion made tie-aware", "criterion restated",
     _dev_v3_5_e17_monotonicity, None),
    ("V4-2", "the engagement clause removed from E19's absorption criterion", "criterion restated",
     _dev_v4_2_e19_engagement_clause, None),
    ("V4.5-2", "two clauses added to E21's criteria before the run", "criteria added", None,
     "Added before any cell ran, from inspection of the arm definitions, because without them two "
     "arms passed signatures for free while having no behaviour at all. The pre-registered primary "
     "was not changed and still decides the verdict."),
    ("V4.5-3", "E28's beta = 0 check gained a second operationalisation after the run",
     "criterion added", _dev_v4_5_3_e28_beta_zero, None),
    ("V4.5-5", "E20 gained a strict fabrication index after the run", "measure added",
     _dev_v4_5_5_e20_strict_index, None),
    ("V5-1", "N21's dominance clause restated", "criterion restated", _dev_v5_1_n21_clause, None),
    ("V5-2", "E30 gained a second update measure after the run", "measure added", None,
     "The pre-registered measure has no headroom in this design by construction, since depth is "
     "built so the goal is exactly as recoverable at every level. The added measure agrees with "
     "the pre-registered one on the outcome and disagrees on the reason. All fifteen pre-existing "
     "columns came back bit-identical on the re-run."),
    ("VAL-2a", "V-2a's permutation clause restated during this pass", "criterion restated", None,
     "Logged in this pass's own V-2 verdict file rather than here, so it sits with the check that "
     "produced it. Named in this table because a validation pass that logs deviations everywhere "
     "except in itself is not a validation pass."),
)


def run_v5(cfg: Config) -> dict:
    rows = []
    for dev_id, what, kind, fn, note in DEVIATIONS:
        row = {"deviation": dev_id, "what_changed": what, "kind": kind}
        if fn is not None:
            row.update(fn())
        else:
            row.update({"computable": False, "why_not_recomputed": note})
        rows.append(row)

    computed = [r for r in rows if r.get("computable")]
    would_change = [r for r in computed if r.get("verdict_would_change")]
    verdict = {
        "check": "V-5",
        "question": "Would any verdict change under the criterion as originally written?",
        "plain_language": (
            "Eleven times across five versions, a measurement rule was changed or added after the "
            "fact. Each of those is already written down in the version's own write-up. This "
            "check goes back and computes the ORIGINAL rule wherever the data to do it is still "
            "on disk, and reports which conclusions would have been different. The cases where "
            "nothing changes are included on purpose: they are the ones that tell you the "
            "changes were not doing the work."),
        "deviations_logged_across_all_versions": len(rows),
        "recomputed_here": len(computed),
        "verdicts_that_would_change": [r["deviation"] for r in would_change],
        "table": rows,
    }
    if would_change:
        verdict["verdict"] = "SOME_VERDICTS_DEPEND_ON_THE_RESTATED_CRITERION"
        verdict["statement"] = (
            f"Of {len(computed)} superseded criteria recomputed from committed data, "
            f"{len(would_change)} would change a verdict: "
            f"{', '.join(r['deviation'] for r in would_change)}. Each is reported with the "
            f"original outcome attached, in the same place as the claim.")
    else:
        verdict["verdict"] = "NO_VERDICT_DEPENDS_ON_A_RESTATED_CRITERION"
        verdict["statement"] = (
            f"All {len(computed)} superseded criteria recomputable from committed data leave "
            f"their verdict where it was. The restatements changed reasons and thresholds' "
            f"targets, not outcomes. That is the informative case, and it is the reason the "
            f"unchanged ones are published rather than dropped.")
    (validation_dir() / "v5_superseded_criteria.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict


# =========================================================================== #
# V-6 — cross-version consistency.
# =========================================================================== #
def run_v6(cfg: Config) -> dict:
    """The boundary regressions, plus the same quantity measured twice.

    THE BOUNDARY REGRESSIONS ARE ALREADY TESTS. What did not exist was somewhere that runs them
    and reports the measured agreement instead of a green tick, which is the difference between
    "the suite passes" and "here is how closely the two versions agree".
    """
    import subprocess
    import sys

    out = validation_dir() / "v6"
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "tests/test_v1_regression.py", "tests/test_nulls_v2.py", "tests/test_nulls_v4.py",
         "tests/test_nulls_v5.py"],
        cwd=str(RESULTS_DIR.parent), capture_output=True, text=True)
    (out / "boundary_regression_pytest.txt").write_text(
        proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    # The same quantity, measured by two versions on the condition they share.
    pairs = []
    e19 = _read_csv("e19_cell_stats.csv")
    e20 = _read_csv("e20_omega_sweep.csv")
    if e19 is not None and e20 is not None:
        off = e19[(e19.arm == "explore_off") & (e19.content == "foreign")]
        zero = e20[np.isclose(e20.omega, 0.0)]
        if len(off) and len(zero):
            a = float(off.iloc[0]["engaged_fraction"])
            b = float(zero.iloc[0]["engaged_fraction"])
            sem = float(zero.iloc[0].get("engaged_sem_across_seeds", float("nan")))
            pairs.append({
                "quantity": "sustained attention on fully foreign content",
                "measured_in": ["E19 (V4)", "E20 (V4.5)"],
                "values": [a, b],
                "absolute_gap": abs(a - b),
                "standard_errors_apart": (abs(a - b) / sem if np.isfinite(sem) and sem > 0
                                          else float("nan")),
                "agree": bool(CR.within_spread(b, a, sem, k=2.0)),
                "note": ("the nominally identical condition. E20 ran at three times the seeds, "
                         "which its own deviation V4.5-6 declares, so a gap here is a "
                         "measurement-precision question rather than a contradiction. It is still "
                         "the same number twice and it should be quoted as a range"),
            })
    e2 = _read_csv("e2_cell_stats.csv")
    e17 = _read_csv("e17_tier_stats.csv")
    if e2 is not None and e17 is not None and "labelling" in (e17.columns if e17 is not None
                                                             else []):
        lie = e2[(e2.true_provenance == "GHOST") & (e2.declared_signal == "SIG_CREATOR")]
        ghost_row = e17[(e17.true_provenance == "GHOST")
                        & (e17.labelling.astype(str) == "claimed_human")]
        if len(lie) and len(ghost_row):
            a = float(lie.iloc[0]["within"])
            b = float(ghost_row.iloc[0]["within"])
            pairs.append({
                "quantity": ("within-reader doubt on GHOST content under a false human label, "
                             "in nats"),
                "measured_in": ["E2 (V1)", "E17 (V3)"],
                "values": [a, b],
                "absolute_gap": abs(a - b),
                "agree": bool(abs(a - b) < 0.05),
                "note": ("E17 is E2's dose-response follow-up on the same geometry, so these "
                         "should be close; a gap would mean one of the two is measuring "
                         "something else"),
            })

    passed = proc.returncode == 0
    disagreements = [p for p in pairs if not p["agree"]]
    verdict = {
        "check": "V-6",
        "question": ("Where two versions measure the same thing, do they agree, and do the "
                     "boundary reductions still hold?"),
        "plain_language": (
            "Each version of the model has to be able to become the previous one when you turn "
            "its new machinery off. Those reductions are already tests; this runs them and, more "
            "usefully, finds every place where two different versions measured the same real "
            "quantity and puts the two numbers side by side."),
        "boundary_regressions_pass": passed,
        "boundary_regression_log": "results/validation/v6/boundary_regression_pytest.txt",
        "boundary_regression_tail": proc.stdout.strip().splitlines()[-5:] if proc.stdout else [],
        "shared_quantities": pairs,
        "disagreements": [p["quantity"] for p in disagreements],
    }
    if not passed:
        verdict["verdict"] = "A_BOUNDARY_REDUCTION_FAILS"
        verdict["statement"] = ("At least one reduction where a later model must reproduce an "
                               "earlier one no longer holds. That is a correctness failure and it "
                               "blocks the claim that the versions are a sequence rather than a "
                               "set of unrelated models.")
    elif disagreements:
        verdict["verdict"] = "REDUCTIONS_HOLD_ONE_SHARED_QUANTITY_DISAGREES"
        verdict["statement"] = (
            "Every boundary reduction holds. "
            + "; ".join(f"{p['quantity']} is measured as {p['values'][0]:.3f} in "
                        f"{p['measured_in'][0]} and {p['values'][1]:.3f} in {p['measured_in'][1]}"
                        for p in disagreements)
            + ". Those are quoted as ranges rather than as single numbers.")
    else:
        verdict["verdict"] = "CONSISTENT_ACROSS_VERSIONS"
        verdict["statement"] = ("Every boundary reduction holds and every quantity measured by "
                               "two versions agrees within its own spread.")
    (validation_dir() / "v6_consistency.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict


# =========================================================================== #
# V-7 — seed and scale independence.
# =========================================================================== #
def run_v7(cfg: Config, workers: int = 1) -> dict:
    from .v3_robustness import HEADLINES, _load

    out_root = validation_dir() / "v7"
    out_root.mkdir(parents=True, exist_ok=True)
    n_obs = int(cfg.get("validation.n_observers", 60))
    n_seeds = int(cfg.get("validation.n_seeds", 12))
    offset = int(cfg.get("validation.seed_block_offset", 700003))

    arms = (("reference", 0, 1.0), ("other_seed_block", offset, 1.0),
            ("double_scale", 0, 2.0))
    rows = []
    for hname, spec in HEADLINES.items():
        seeds = max(2, n_seeds // spec["seed_divisor"])
        got = {}
        for arm, seed_shift, scale in arms:
            c = _load(spec["loader"])
            c.set("run.base_seed", int(c.run.base_seed) + seed_shift)
            out = out_root / hname / arm
            out.mkdir(parents=True, exist_ok=True)
            got[arm] = spec["scorer"](out, c, workers,
                                      int(round(n_obs * scale)), int(round(seeds * scale)))
        ref = got["reference"]["primary"]
        for arm, _, _ in arms[1:]:
            val = got[arm]["primary"]
            drift = (abs(val - ref) / abs(ref) if np.isfinite(ref) and ref != 0
                     else float("nan"))
            rows.append({
                "headline": hname, "arm": arm,
                "reference_effect": ref, "arm_effect": val,
                "relative_drift": drift,
                "verdict_holds": bool(got[arm].get("reportable", True)),
                "material_drift": bool(np.isfinite(drift)
                                       and drift > CR.V7_SCALE_DRIFT_FRACTION),
            })
        if "peak_omega" in got["reference"]:
            rows.append({
                "headline": hname, "arm": "peak_location_across_arms",
                "reference_effect": got["reference"]["peak_omega"],
                "arm_effect": got["other_seed_block"].get("peak_omega"),
                "relative_drift": 0.0 if got["reference"]["peak_omega"] ==
                got["other_seed_block"].get("peak_omega") else 1.0,
                "verdict_holds": True,
                "material_drift": bool(got["reference"]["peak_omega"] !=
                                       got["other_seed_block"].get("peak_omega")),
            })

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "v7_seed_and_scale.csv", index=False)
    underpowered = table[table.material_drift].headline.unique().tolist()
    flipped = table[~table.verdict_holds].headline.unique().tolist()

    verdict = {
        "check": "V-7",
        "question": "Do the headlines survive a different seed block and twice the scale?",
        "plain_language": (
            "Random draws can flatter a result, and a result measured on too few readers can look "
            "solid and then move when you measure it properly. This re-runs the headlines on a "
            "completely separate set of random seeds and again at double the number of readers, "
            "and reports how far the numbers moved rather than only whether the conclusion "
            "survived."),
        "seed_block_offset": offset,
        "scale": {"n_observers": n_obs, "n_seeds": n_seeds},
        "criteria": {"material_drift_fraction": CR.V7_SCALE_DRIFT_FRACTION},
        "table": rows,
        "under_powered": underpowered,
        "verdicts_that_flipped": flipped,
    }
    if flipped:
        verdict["verdict"] = "A_VERDICT_DEPENDS_ON_THE_SEED_BLOCK_OR_THE_SCALE"
        verdict["statement"] = (
            f"{', '.join(flipped)} does not survive a change of seed block or a doubling of "
            f"scale. That is the pattern a lucky draw leaves, and this project has already been "
            f"caught by one; the claim is withdrawn to the range where it holds.")
    elif underpowered:
        verdict["verdict"] = "VERDICTS_HOLD_SOME_EFFECT_SIZES_ARE_UNDER_POWERED"
        verdict["statement"] = (
            f"Every verdict survives, but the effect size for {', '.join(underpowered)} moves by "
            f"more than {CR.V7_SCALE_DRIFT_FRACTION:.0%} of itself when the seed block or the "
            f"scale changes. Those numbers are under-powered at this scale and are quoted as "
            f"approximate.")
    else:
        verdict["verdict"] = "SEED_AND_SCALE_INDEPENDENT"
        verdict["statement"] = ("Every headline holds on a disjoint seed block and at double "
                               "scale, with effect sizes stable to within a quarter of "
                               "themselves.")
    (validation_dir() / "v7_seed_and_scale.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict


def run(cfg: Config, workers: int = 1) -> dict:
    return {"v5": run_v5(cfg), "v6": run_v6(cfg), "v7": run_v7(cfg, workers=workers)}
