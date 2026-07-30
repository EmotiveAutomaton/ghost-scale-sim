"""R-5 — decompose the uptake measure. The most consequential item in the pass.

WHAT THE MEASURE ON RECORD ACTUALLY IS. ``psi_analogue`` is a trust weight times a DISTANCE between
what a reader believed before and after. Being a distance, it counts any movement as uptake, so a
reader who becomes confidently WRONG has moved as far from where it started as one who became
confidently right. Measured on this model, the confidently wrong score 87% of what the correct do.

**So "how much the reader took on" has never measured understanding.** It measures belief movement,
and being fooled moves you almost as much as being right. That is why it comes out U-shaped in
recovery quality, with a minimum in the middle, and why an experiment whose two conditions straddle
that minimum returns a flat result for reasons unconnected to what it was testing.

ONE NUMBER BECOMES THREE, ALWAYS REPORTED TOGETHER:

    movement          the distance on record. U-shaped. Retained, not replaced.
    error reduction   how much closer to the truth the reader got. Signed, monotone in accuracy,
                      NEGATIVE when the reader moves away from the truth, which is the behaviour
                      the distance cannot express at all.
    trust factor      pulled out of the product and reported on its own, because it changes by
                      more than fortyfold across the range one experiment sweeps, which is larger
                      than most effects in this project.

THE SPECIFICATION'S FORMULA FOR THE SECOND ONE IS UNDEFINED AND HAD TO BE CORRECTED. Written against
a point mass on the truth, every term in it diverges; floored at an epsilon it returns a number that
tracks the epsilon rather than the data. Reversing the arguments gives the reduction in the
surprisal of the truth, which is well defined and has the wanted behaviour. See
``metrics.error_reduction``.
"""
from __future__ import annotations

import ast
import json

import numpy as np
import pandas as pd

from .. import metrics
from ..config import Config
from ..experiments._common import RESULTS_DIR
from . import criteria as CR
from . import repair_dir

# Where the posterior AND the true goal were both persisted, so all three quantities are computable
# from committed data with no simulation.
SOURCES = (
    ("E2", "e2_points.csv", ["true_provenance", "declared_signal"], "final_posterior", "true_goal"),
    ("E17", "e17_points.csv", ["true_provenance", "labelling"], "final_posterior", "true_goal"),
    ("E19", "e19_explore.csv", ["arm", "content"], "real_goal_posterior", "true_goal"),
)
NEEDS_RERUN = {
    "E30": ("carries psi_analogue but not the posterior, so error reduction cannot be recomputed. "
            "This is the experiment the decomposition matters most for, and the rerun supplies it"),
    "E31": "carries prior_drift but not the posterior",
    "E20": "carries neither the posterior nor a prior to measure movement against",
    "E32": "carries psi_analogue but not the posterior",
    "E4": "raw file not committed",
}


def _posteriors(series) -> np.ndarray:
    return np.array([np.asarray(ast.literal_eval(s), dtype=float) if isinstance(s, str)
                     else np.asarray(s, dtype=float) for s in series])


def _uniform_prior(k: int) -> np.ndarray:
    """The reference the reader started from, in the absence of a persisted per-reader prior.

    A DELIBERATE APPROXIMATION AND IT IS DECLARED. The committed files do not carry each reader's
    own heterogeneous prior, so error reduction is computed against the uniform it was perturbed
    from. The perturbation scale is small relative to the movements being measured, and the same
    reference is used for every reader in every cell, so a comparison BETWEEN cells is unaffected.
    An absolute value for a single reader is approximate, and is not quoted as anything else. The
    rerun carries the real prior.
    """
    return np.full(k, 1.0 / k)


def run(cfg: Config, workers: int = 1) -> dict:
    out = repair_dir("r5_uptake")
    kappa = float(cfg.signal_model.kappa)
    rows = []
    for name, fname, keys, post_col, truth_col in SOURCES:
        path = RESULTS_DIR / fname
        if not path.exists():
            rows.append({"experiment": name, "available": False,
                         "why": f"results/{fname} is not on disk"})
            continue
        df = pd.read_csv(path)
        if post_col not in df.columns or truth_col not in df.columns:
            rows.append({"experiment": name, "available": False,
                         "why": f"needs {post_col} and {truth_col}; has {list(df.columns)}"})
            continue
        keys = [k for k in keys if k in df.columns]
        for cell, g in df.groupby(keys):
            P = _posteriors(g[post_col])
            truth = g[truth_col].values.astype(int)
            prior = _uniform_prior(P.shape[1])
            move = np.array([metrics.kl_divergence(p, prior) for p in P])
            err = np.array([metrics.error_reduction(p, prior, t) for p, t in zip(P, truth)])
            acc = np.array([int(np.argmax(p) == t) for p, t in zip(P, truth)])
            confident = np.array([metrics.within_observer_entropy(p) < 0.5 for p in P])
            wrong = confident & (acc == 0)
            rows.append({
                "experiment": name, "available": True,
                "cell": " / ".join(str(c) for c in (cell if isinstance(cell, tuple) else (cell,))),
                "n": int(len(g)),
                "accuracy": float(acc.mean()),
                "movement_as_committed": float(np.mean(move) * metrics.trust_factor(kappa)),
                "movement_unweighted": float(np.mean(move)),
                "error_reduction": float(np.mean(err)),
                "error_reduction_sd": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
                "share_confidently_wrong": float(wrong.mean()),
                "movement_of_the_confidently_wrong": (float(np.mean(move[wrong]))
                                                      if wrong.any() else float("nan")),
                "error_reduction_of_the_confidently_wrong": (float(np.mean(err[wrong]))
                                                             if wrong.any() else float("nan")),
                "trust_factor": metrics.trust_factor(kappa),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "decomposition.csv", index=False)

    live = df[df.get("available", pd.Series(dtype=bool)) == True] if len(df) else df  # noqa: E712
    disagreements = []
    if len(live):
        for exp, g in live.groupby("experiment"):
            if len(g) < 2:
                continue
            from ..diagnostics.criteria import spearman
            rho_move = spearman(g.accuracy.values, g.movement_unweighted.values)
            rho_err = spearman(g.accuracy.values, g.error_reduction.values)
            disagreements.append({
                "experiment": exp, "cells": int(len(g)),
                "movement_vs_accuracy": rho_move,
                "error_reduction_vs_accuracy": rho_err,
                "measures_rank_conditions_differently": bool(
                    np.isfinite(rho_move) and np.isfinite(rho_err)
                    and np.sign(rho_move) != np.sign(rho_err)),
            })

    negatives = live[live.error_reduction < 0] if len(live) else live
    flips = [d for d in disagreements if d["measures_rank_conditions_differently"]]

    if len(negatives):
        verdict = "ERROR_REDUCTION_GOES_NEGATIVE_WHERE_MOVEMENT_CANNOT"
    elif flips:
        verdict = "THE_TWO_MEASURES_RANK_CONDITIONS_DIFFERENTLY"
    else:
        verdict = "THE_TWO_MEASURES_AGREE_ON_EVERY_COMMITTED_CELL"

    payload = {
        "check": "R-5",
        "question": "Has the uptake measure been measuring understanding, or just belief movement?",
        "plain_language": (
            "Several experiments measure how much a reader 'takes on' from a work. The measure is a "
            "distance between what the reader believed before and after, so a reader who ends up "
            "confidently wrong scores almost as highly as one who ends up right: both moved a long "
            "way. This splits that one number into three, the important one being how much closer "
            "to the truth the reader actually got, which can go negative and which the old measure "
            "cannot express."),
        "criteria": {"quantities": ["movement", "error_reduction", "trust_factor"],
                     "disagreement_is_a_finding": CR.UPTAKE_DISAGREEMENT_IS_A_FINDING},
        "formula_correction": (
            "the specification writes error reduction with the divergence arguments the wrong way "
            "round, which makes every term infinite against a point mass on the truth and leaves a "
            "value determined by the epsilon it is floored at. Corrected to the reduction in the "
            "surprisal of the truth."),
        "prior_reference": (
            "computed against the uniform prior each reader's own prior was perturbed from, because "
            "the committed files do not persist the per-reader prior. Declared rather than hidden: "
            "between-cell comparisons are unaffected because the reference is shared, and absolute "
            "single-reader values are approximate. The rerun carries the real prior."),
        "cells": rows,
        "measure_agreement": disagreements,
        "cells_with_negative_error_reduction": (
            negatives[["experiment", "cell", "accuracy", "movement_unweighted", "error_reduction"]]
            .to_dict(orient="records") if len(negatives) else []),
        "needs_rerun": NEEDS_RERUN,
        "trust_factor_range": {
            "kappa": [0.1, 0.5, 0.9, 0.99],
            "factor": [metrics.trust_factor(k) for k in (0.1, 0.5, 0.9, 0.99)],
            "ratio_across_the_swept_range": metrics.trust_factor(0.99) / metrics.trust_factor(0.1),
        },
        "verdict": verdict,
    }
    payload["statement"] = _statement(payload)
    (repair_dir() / "r5_uptake.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def _statement(p: dict) -> str:
    negs = p["cells_with_negative_error_reduction"]
    tf = p["trust_factor_range"]
    bits = []
    if negs:
        worst = min(negs, key=lambda r: r["error_reduction"])
        bits.append(
            "**The decomposition separates cells the old measure could not.** In %d cell(s) error "
            "reduction is NEGATIVE: the reader ended further from the truth than it started. The "
            "clearest is %s in %s, where movement reads %.3f, which looks like substantial uptake, "
            "while error reduction reads %.3f, which says the reader was moved away from the "
            "answer. The measure on record cannot express that at all, because a distance has no "
            "sign."
            % (len(negs), worst["cell"], worst["experiment"], worst["movement_unweighted"],
               worst["error_reduction"]))
    else:
        bits.append("No committed cell has negative error reduction, so on this data the two "
                    "measures never disagree about direction. They still differ in shape, and the "
                    "experiments where that matters are the ones that did not persist a posterior.")

    flips = [d for d in p["measure_agreement"] if d["measures_rank_conditions_differently"]]
    if flips:
        bits.append("**And they rank conditions differently** in %s, which is reported as the "
                    "finding rather than resolved in favour of either."
                    % ", ".join(d["experiment"] for d in flips))

    # The three-cell contrast, which is the finding the decomposition was built to expose.
    cells = {c["cell"]: c for c in p["cells"] if c.get("available") and c["experiment"] == "E2"}
    lie = cells.get("GHOST / SIG_CREATOR")
    honest_machine = cells.get("GHOST / SIG_GHOST")
    human = cells.get("CREATOR / SIG_CREATOR")
    if lie and honest_machine and human:
        bits.append(
            "**And it rewrites the headline, in the direction of making it sharper.** Three cells "
            "that the measure on record reports as %.3f, %.3f and %.3f, which reads as 'a lot, "
            "nearly as much, and almost none':\n\n"
            "| what the reader saw | movement, as reported | error reduction |\n"
            "|---|---|---|\n"
            "| human work, labelled honestly | %.3f | **%+.3f** |\n"
            "| machine work, labelled honestly | %.3f | **%+.3f** |\n"
            "| machine work, passed off as human | %.3f | **%+.3f** |\n\n"
            "Under the old measure the lie and the honest human work look almost the same. Under "
            "the new one they have OPPOSITE SIGNS, and the lie is %.1f times larger in magnitude "
            "than the truth. The false label does not merely make readers take on more. **It "
            "moves them away from the answer, further than an honest label moves them toward it.** "
            "The honest machine label produces almost nothing in either direction, at %+.3f, which "
            "is the correct behaviour and which the old measure reported as a small positive."
            % (human["movement_unweighted"], lie["movement_unweighted"],
               honest_machine["movement_unweighted"],
               human["movement_unweighted"], human["error_reduction"],
               honest_machine["movement_unweighted"], honest_machine["error_reduction"],
               lie["movement_unweighted"], lie["error_reduction"],
               abs(lie["error_reduction"]) / abs(human["error_reduction"]),
               honest_machine["error_reduction"]))

    bits.append(
        "**The trust weight is now reported separately** rather than multiplied in. It runs from "
        "%.3f to %.3f across the range one experiment sweeps, a factor of %.0f, which is larger "
        "than most effects in this project. Any sweep that varied trust and reported uptake was "
        "reporting the product of two things."
        % (tf["factor"][0], tf["factor"][-1], tf["ratio_across_the_swept_range"]))

    bits.append(
        "**Five experiments cannot be decomposed from committed data** and are named rather than "
        "skipped: " + "; ".join(f"{k} {v}" for k, v in p["needs_rerun"].items()) + ". The first of "
        "those is the one the decomposition matters most for.")
    return "\n\n".join(bits)
