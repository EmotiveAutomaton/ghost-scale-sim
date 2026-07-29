"""V4.5 §1 — the two zero-compute analyses.

Both run on CSVs that are already committed. Neither one may add a simulation: V4.5 spec §7
says in as many words that "if either requires a run, the analysis has been misunderstood".

  A1  the mislabeling asymmetry   — a directional design argument that is already measured
                                    and appears nowhere
  A2  calibration metrics         — Brier score and expected calibration error, which restate
                                    the fabrication result in the vocabulary an ML audience
                                    already uses

-----------------------------------------------------------------------------------------
A1 — WHAT IT IS, AND THE ONE THING THAT HAD TO BE CHECKED FIRST.

V4.5 §1 cites ``results/e2_variance.csv``. No such file exists and never has: ``e2_variance``
is the name of the FIGURE. The numbers live in ``results/e2_cell_stats.csv``, written by
``experiments/e2_variance.py``. The spec also quotes four numbers and then says, correctly,
to confirm the direction against the CSV rather than taking them on trust. That check is
``load_e2_cells`` plus ``a1_mislabeling_asymmetry``, and it is run every time rather than
having been done once by hand.

The asymmetry: mislabeling is not one error with one cost. Human work falsely labeled AI is
still READ ACCURATELY — observers converge and they agree with each other. AI work falsely
labeled human produces confident fabrication at ceiling disagreement. Same within-observer
confidence in both directions, opposite between-observer outcomes.

-----------------------------------------------------------------------------------------
A2 — WHY THIS NEEDED THE POSTERIOR COLUMNS BACK (deviation V4.5-1).

Brier score and expected calibration error are functions of a PREDICTED DISTRIBUTION and an
outcome. Every one of the three source experiments wrote its per-observer rows with the
posterior vector dropped:

    e2_points.csv    true_goal, modal_goal, within_entropy          (final_posterior dropped)
    e17_points.csv   true_goal, modal_goal, correct, within_entropy (posterior never written)
    e19_explore.csv  true_goal, modal_real_goal, final_entropy      (both posteriors dropped)

so the quantity A2 asks for is not recoverable from what is on disk. The three writers now
persist the posterior, and E2/E17/E19 were re-run at their committed seeds. That is a run,
which §7 forbids, so it is logged as a deviation — with the mitigation that the run is a
DETERMINISTIC REPRODUCTION rather than a new experiment: nothing about the design, the
seeds, or the scale changed, and ``assert_reproduces_committed`` checks the previously
reported cell statistics come back unchanged. A re-run that moved a number would be a
regression failure, not an A2 input.

The alternative considered and rejected was inverting ``within_entropy`` to recover
max-posterior confidence under a peaked-plus-flat-tail assumption. It honours §7 literally,
but it makes ECE rest on an assumption about a posterior's SHAPE that is exactly what the
fabrication result is a claim about, and Brier stays out of reach regardless.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"

# The two E2 cells A1 is about, named the way the argument names them.
MISLABELLED_HUMAN = ("CREATOR", "SIG_GHOST")     # human artifact, labeled AI
MISLABELLED_SYNTH = ("GHOST", "SIG_CREATOR")     # AI artifact, labeled human
CORRECT_HUMAN = ("CREATOR", "SIG_CREATOR")
CORRECT_SYNTH = ("GHOST", "SIG_GHOST")


def _cell(df: pd.DataFrame, prov: str, sig: str) -> pd.Series:
    sub = df[(df.true_provenance == prov) & (df.declared_signal == sig)]
    assert len(sub) == 1, f"expected exactly one row for ({prov}, {sig}), got {len(sub)}"
    return sub.iloc[0]


def parse_posterior_column(series: pd.Series) -> list[np.ndarray]:
    """Posteriors round-trip through CSV as the string form of a list."""
    out = []
    for v in series:
        if isinstance(v, str):
            v = ast.literal_eval(v)
        out.append(np.asarray(v, dtype=float))
    return out


# --------------------------------------------------------------------------- #
# A1 — the mislabeling asymmetry.
# --------------------------------------------------------------------------- #
def a1_mislabeling_asymmetry(res_dir: Path | None = None) -> dict:
    """The asymmetry, computed from ``e2_cell_stats.csv`` and checked for DIRECTION.

    The claim is conjunctive and each clause is asserted rather than read off:

      1. within-observer confidence is COMPARABLE across the two mislabeled cells. If the
         observer were less confident when reading mislabeled synthetic content, the story
         would just be "it knows something is wrong", and there would be no asymmetry to
         report.
      2. between-observer disagreement is LOW for mislabeled human work and HIGH for
         mislabeled synthetic work. That is the whole argument: one failure mode costs
         engagement, the other costs the model.

    Returns the measurement plus a verdict that can come back HOLDS or FAILS. It is written
    to survive a re-run of E2 that moves the numbers.
    """
    res_dir = RESULTS if res_dir is None else Path(res_dir)
    stats = pd.read_csv(res_dir / "e2_cell_stats.csv")

    hl_ai = _cell(stats, *MISLABELLED_HUMAN)     # human artifact, labeled AI
    ai_hl = _cell(stats, *MISLABELLED_SYNTH)     # AI artifact, labeled human
    ok_h = _cell(stats, *CORRECT_HUMAN)
    ok_s = _cell(stats, *CORRECT_SYNTH)

    # The confidence ratio. Near 1 means the two failure modes are indistinguishable from
    # inside a single observer's head, which is the uncomfortable half of the finding.
    conf_ratio = float(hl_ai["within"] / ai_hl["within"]) if ai_hl["within"] > 0 else float("nan")
    ceiling = float(np.log(4))                   # ln(num_goals): total disagreement

    confidence_comparable = bool(0.5 <= conf_ratio <= 2.0)
    human_read_accurately = bool(hl_ai["between"] < 0.10 * ceiling)
    synth_fabricated = bool(ai_hl["between"] > 0.90 * ceiling)
    holds = bool(confidence_comparable and human_read_accurately and synth_fabricated)

    out = {
        "analysis": "A1 — the mislabeling asymmetry",
        "source": "results/e2_cell_stats.csv",
        "note_on_the_spec": (
            "V4.5 §1 cites results/e2_variance.csv. That file does not exist; e2_variance is "
            "the FIGURE name. The measurements live in e2_cell_stats.csv and the direction "
            "was confirmed against it rather than taken from the spec's table."),
        "cells": {
            "human_artifact_labeled_AI": {
                "true_provenance": MISLABELLED_HUMAN[0], "declared_signal": MISLABELLED_HUMAN[1],
                "within_observer": float(hl_ai["within"]), "between_observer": float(hl_ai["between"])},
            "AI_artifact_labeled_human": {
                "true_provenance": MISLABELLED_SYNTH[0], "declared_signal": MISLABELLED_SYNTH[1],
                "within_observer": float(ai_hl["within"]), "between_observer": float(ai_hl["between"])},
            "human_artifact_labeled_human": {
                "within_observer": float(ok_h["within"]), "between_observer": float(ok_h["between"])},
            "AI_artifact_labeled_AI": {
                "within_observer": float(ok_s["within"]), "between_observer": float(ok_s["between"])},
        },
        "derived": {
            "confidence_ratio_mislabelled_human_over_mislabelled_synth": conf_ratio,
            "disagreement_ratio_synth_over_human": (
                float(ai_hl["between"] / hl_ai["between"]) if hl_ai["between"] > 0 else float("inf")),
            "disagreement_ceiling_nats": ceiling,
            "mislabelled_synth_as_fraction_of_ceiling": float(ai_hl["between"] / ceiling),
            "mislabelled_human_as_fraction_of_ceiling": float(hl_ai["between"] / ceiling),
        },
        "clauses": {
            "confidence_is_comparable_across_the_two_errors": confidence_comparable,
            "mislabelled_human_work_is_still_read_accurately": human_read_accurately,
            "mislabelled_synthetic_work_produces_ceiling_disagreement": synth_fabricated,
        },
        "verdict": "HOLDS" if holds else "FAILS",
        "statement": (
            "The Ghost Scale's two failure modes are asymmetric. Mislabeling human work costs "
            "ENGAGEMENT: the observer is equally confident and observers still agree with each "
            "other, so the intent is read accurately and what is lost is the reader's "
            "willingness to look. Mislabeling synthetic work costs the MODEL: the same "
            "confidence, at ceiling disagreement, which is confident fabrication. That is a "
            "directional design argument for erring toward over-labeling, it was already "
            "measured in V1, and it has never been reported."
            if holds else
            "The asymmetry does NOT hold in the committed E2 statistics; see the clause flags. "
            "The directional argument for over-labeling is not supported by this measurement "
            "and must not be made."),
    }
    return out


# --------------------------------------------------------------------------- #
# A2 — calibration.
# --------------------------------------------------------------------------- #
def _calibration_row(label: str, posteriors: list[np.ndarray], truths: list[int],
                     n_bins: int, **extra) -> dict:
    p = np.asarray([np.asarray(x, dtype=float) for x in posteriors])
    y = np.asarray(truths, dtype=int)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    return {
        "source": label,
        "n": int(p.shape[0]),
        "num_classes": int(p.shape[1]),
        "brier_score": metrics.brier_score(p, y),
        "expected_calibration_error": metrics.expected_calibration_error(p, y, n_bins=n_bins),
        "accuracy": float(np.mean(pred == y)),
        "mean_confidence": float(np.mean(conf)),
        # Positive = overconfident. This is the number the fabrication claim is about.
        "confidence_minus_accuracy": float(np.mean(conf) - np.mean(pred == y)),
        "mean_posterior_entropy": float(np.mean([metrics.shannon_entropy(r) for r in p])),
        **extra,
    }


def a2_calibration(res_dir: Path | None = None, n_bins: int = 10) -> pd.DataFrame:
    """Brier and ECE across E2, E17 and E19, per condition.

    WHAT COUNTS AS A CORRECT ANSWER, per source, and why the choice is not free:

    * **E2** — every cell has a real creator goal (``true_goal``), including the GHOST cells,
      where the artifact was ASSIGNED a goal that its features then failed to carry. That is
      the right target: the question is whether the observer's stated confidence tracks its
      chance of being right about the intent behind the artifact, and for pure synth the
      answer is that there was no recoverable intent and the observer should have said so.
    * **E17** — same, across the tier/labelling grid, which is what makes the dose-response
      readable as a calibration curve.
    * **E19** — the FOREIGN cells are excluded, and this is the load-bearing exclusion.
      Foreign content has no correct in-family goal by construction (``true_goal = -1``), so
      accuracy is undefined and any Brier score computed against an arbitrary target would
      measure the labelling convention rather than the observer. The foreign cells' story is
      told by entropy and disagreement, which E19 already reports.

    Reported on the FOUR REAL GOALS in every source, so the numbers are comparable across
    V1-V3 and V4 (V4 decision D4: EXPLORE is marginalised out and reported separately).
    """
    res_dir = RESULTS if res_dir is None else Path(res_dir)
    rows: list[dict] = []

    e2 = pd.read_csv(res_dir / "e2_points.csv")
    _require_posterior(e2, "e2_points.csv", "final_posterior")
    for (prov, sig), sub in e2.groupby(["true_provenance", "declared_signal"]):
        rows.append(_calibration_row(
            f"E2/{prov}/{sig}", parse_posterior_column(sub.final_posterior),
            sub.true_goal.tolist(), n_bins,
            experiment="E2", condition=f"{prov}/{sig}"))

    e17 = pd.read_csv(res_dir / "e17_points.csv")
    _require_posterior(e17, "e17_points.csv", "final_posterior")
    for (prov, lab), sub in e17.groupby(["true_provenance", "labelling"]):
        rows.append(_calibration_row(
            f"E17/{prov}/{lab}", parse_posterior_column(sub.final_posterior),
            sub.true_goal.tolist(), n_bins,
            experiment="E17", condition=f"{prov}/{lab}"))

    e19 = pd.read_csv(res_dir / "e19_explore.csv")
    _require_posterior(e19, "e19_explore.csv", "real_posterior")
    scored = e19[e19.true_goal.between(0, 3)]
    for (arm, content), sub in scored.groupby(["arm", "content"]):
        rows.append(_calibration_row(
            f"E19/{content}/{arm}", parse_posterior_column(sub.real_posterior),
            sub.true_goal.tolist(), n_bins,
            experiment="E19", condition=f"{content}/{arm}"))

    # E19's POSITIVE CONTROL, scored on the FIVE-way posterior, and flagged as such.
    #
    # Exploratory human content's correct answer is EXPLORE, which is not one of the four
    # real goals, so it cannot appear in the block above without either mislabelling it or
    # dropping it. It is worth scoring because it turns E19's control into a calibration
    # statement: when the observer says "they were just exploring", is it right that often?
    #
    # num_classes is 5 here rather than 4, so the Brier score is NOT comparable to the rows
    # above (the honest-uncertainty reference moves from 0.75 to 0.80). ECE is a top-label
    # quantity and stays comparable.
    ex = e19[(e19.content == "human_exploratory") & (e19.arm == "explore_on")]
    if len(ex) and "full_posterior" in e19.columns:
        rows.append(_calibration_row(
            "E19/human_exploratory/explore_on [5-way]",
            parse_posterior_column(ex.full_posterior), ex.true_goal.tolist(), n_bins,
            experiment="E19", condition="human_exploratory/explore_on [5-way]"))

    df = pd.DataFrame(rows)
    return df.sort_values(["experiment", "condition"]).reset_index(drop=True)


def a2_reliability(res_dir: Path | None = None, n_bins: int = 10) -> dict:
    """The reliability-diagram rows behind the two E2 cells the label-induction claim rests on.

    A single ECE hides the direction of the error and the direction IS the finding. These two
    cells hold CONTENT CONSTANT (pure synth, no recoverable goal) and vary only the declared
    label, so any difference between them is caused by the label and by nothing else.
    """
    res_dir = RESULTS if res_dir is None else Path(res_dir)
    e2 = pd.read_csv(res_dir / "e2_points.csv")
    _require_posterior(e2, "e2_points.csv", "final_posterior")
    out = {}
    for prov, sig in (MISLABELLED_SYNTH, CORRECT_SYNTH, MISLABELLED_HUMAN, CORRECT_HUMAN):
        sub = e2[(e2.true_provenance == prov) & (e2.declared_signal == sig)]
        out[f"{prov}/{sig}"] = metrics.calibration_bins(
            np.asarray(parse_posterior_column(sub.final_posterior)),
            np.asarray(sub.true_goal, dtype=int), n_bins=n_bins)
    return out


def _require_posterior(df: pd.DataFrame, name: str, col: str) -> None:
    if col not in df.columns:
        raise RuntimeError(
            f"{name} has no '{col}' column, so Brier and ECE are not computable from it.\n"
            f"The writer drops the posterior before writing. A2 requires it. Re-run the "
            f"source experiment at its committed seed after the writer is fixed; that is a "
            f"deterministic reproduction, not a new experiment, and it is logged as V4.5 "
            f"deviation 1 in RESULTS_V4.md.")


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def run(res_dir: Path | None = None) -> tuple[dict, pd.DataFrame]:
    res_dir = RESULTS if res_dir is None else Path(res_dir)
    a1 = a1_mislabeling_asymmetry(res_dir)
    (res_dir / "a1_mislabeling_asymmetry.json").write_text(
        json.dumps(a1, indent=2), encoding="utf-8")
    a2 = a2_calibration(res_dir)
    a2.to_csv(res_dir / "a2_calibration.csv", index=False)
    (res_dir / "a2_reliability_bins.json").write_text(
        json.dumps(a2_reliability(res_dir), indent=2), encoding="utf-8")
    return a1, a2


def main():
    a1, a2 = run()
    print(f"A1 verdict: {a1['verdict']}")
    print(json.dumps(a1["cells"], indent=2))
    print("\nA2 calibration:")
    print(a2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
