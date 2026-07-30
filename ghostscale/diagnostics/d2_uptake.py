"""D-2 — the shape of uptake as a function of goal recovery. Probably why the depth result was flat.

THE MEASURE. `metrics.psi_analogue` is what every experiment means by "how much the reader takes on":

    uptake = [engaged] * (-ln(1 - kappa)) * KL( posterior || prior )

It is a DISTANCE between what the reader ended up believing and what it started believing. Nothing in
that definition says the distance is larger when the reader is right.

WHY THAT MATTERS. A reader who recovers the goal correctly ends far from a near-uniform prior, so
uptake is high. A reader who recovers nothing ends AT its prior, so uptake is low. But a reader who
becomes confidently WRONG also ends far from its prior, so uptake is high again. Three regimes, and
the middle one is the only one with low uptake.

If that is the shape, then uptake is **not monotone in recovery quality**, and any experiment that
regresses uptake on a manipulation whose arms land on opposite sides of the trough will return a null
for reasons having nothing to do with the manipulation. The depth experiment did exactly that: it
regressed uptake on depth level, found it flat, and reported the construction at fault for having no
headroom. "No headroom" and "a non-monotone response with the arms straddling the minimum" look
identical from a flat regression and are different problems with different repairs.

So this maps the curve directly, before anything is rerun in a new regime. It is a prerequisite for
P-2's repair rather than a check on it: knowing where the trough is decides which difficulty setting
a rerun should use, and a rerun at the trough would fail for the third time.

TWO EXTRA THINGS ARE MEASURED because the same argument applies to them. Uptake is gated on
engagement, so a reader that disengages contributes exactly zero, which puts a hard discontinuity in
the response. And uptake is scaled by `-ln(1 - kappa)`, which is a monotone function of trust, so any
sweep that varies trust and reports uptake is reading a product of two things.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from . import criteria as CR
from . import diagnostics_dir


def _sweep(cfg: Config, n_readers: int = 80, n_seeds: int = 6) -> list:
    """Uptake against recovery quality, along the one axis that moves recovery.

    Reader inexpertise is the axis, for the reason P-2 establishes: it is the only knob that moves
    goal recovery continuously, because it is a systematic error rather than a sampling one. Human
    content only, so nothing about machine-made content enters and the curve is a property of the
    uptake measure rather than of the reframe.
    """
    from ..creators import build_creator_bank
    from ..environment import Artifact, Environment
    from ..generative_model import build_shared_model
    from ..observer import make_observer, rollout_observer

    c = cfg.copy()
    c.set("inference.exact", True)
    gm = build_shared_model(c)
    bank = build_creator_bank(c, gm)
    T = int(c.run.n_timesteps)
    k = min(T, 10)
    ng = int(c.cardinalities.num_goals)
    true_goal = 1
    kappa = float(c.signal_model.kappa)

    rows = []
    for d in np.round(np.arange(0.0, 1.0001, 0.05), 3):
        per = []
        for s in range(n_seeds):
            env = Environment(c, gm, np.random.default_rng(31_000 + s), creator_bank=bank)
            art = Artifact(provenance=K.CREATOR, goal=true_goal, declared_signal=K.SIG_CREATOR)
            for i in range(n_readers):
                r = np.random.default_rng(32_000 + 811 * s + i)
                agent = make_observer(gm, c, r, d_i=float(d))
                res = rollout_observer(agent, art, env, c, r, T, force_deep_k=k)
                q = np.asarray(res.final_goal_posterior, dtype=float)
                free = np.asarray(res.attention)[k:]
                engaged = bool(free.size and np.mean(free == K.DEEP) > 0)
                per.append({
                    "correct": int(np.argmax(q) == true_goal),
                    "confidence_in_the_truth": float(q[true_goal]),
                    "goal_entropy": metrics.within_observer_entropy(q),
                    # Ungated, so the curve is about the DISTANCE and not about the gate. The gated
                    # version is reported beside it, and the gap between them is the gate's effect.
                    "uptake_ungated": metrics.psi_analogue(q, res.goal_prior, kappa, True),
                    "uptake_as_reported": metrics.psi_analogue(q, res.goal_prior, kappa, engaged),
                    "engaged": int(engaged),
                    # Whether the reader is confident, and whether that confidence is misplaced.
                    "confident": int(metrics.within_observer_entropy(q) < 0.5),
                })
        df = pd.DataFrame(per)
        conf_wrong = df[(df.confident == 1) & (df.correct == 0)]
        rows.append({
            "inexpertise": float(d), "expertise": float(1.0 - d),
            "accuracy": float(df.correct.mean()),
            "goal_entropy": float(df.goal_entropy.mean()),
            "confidence_in_the_truth": float(df.confidence_in_the_truth.mean()),
            "uptake_ungated": float(df.uptake_ungated.mean()),
            "uptake_ungated_sd": float(df.uptake_ungated.std(ddof=1)),
            "uptake_as_reported": float(df.uptake_as_reported.mean()),
            "engaged_fraction": float(df.engaged.mean()),
            "confidently_wrong_fraction": float(len(conf_wrong) / len(df)),
            "uptake_of_the_confidently_wrong": (float(conf_wrong.uptake_ungated.mean())
                                                if len(conf_wrong) else float("nan")),
            "uptake_of_the_correct": (float(df[df.correct == 1].uptake_ungated.mean())
                                      if int(df.correct.sum()) else float("nan")),
            "n": len(df),
        })
    return rows


def _kappa_scaling(cfg: Config) -> dict:
    """The multiplicative trust term, stated as arithmetic rather than measured.

    Nothing needs simulating: the factor is `-ln(1 - kappa)` and it is in the definition. It is
    reported because any experiment that sweeps trust and reports uptake is reading a product, and
    the factor spans more than a decade over the range the project sweeps.
    """
    ks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    vals = [float(-np.log(1.0 - min(k, 1.0 - 1e-6))) for k in ks]
    return {
        "kappa": ks, "scaling_factor": vals,
        "ratio_across_the_swept_range": float(vals[-1] / vals[1]) if vals[1] > 0 else float("inf"),
        "why_it_matters": ("uptake is defined as this factor times a belief distance, so a sweep "
                           "that varies trust and reports uptake is reporting the product. Over the "
                           "range E4 sweeps, the factor alone changes by more than a factor of "
                           "forty, which is larger than most effects in the project."),
    }


def run(cfg: Config, workers: int = 1, n_readers: int = 80, n_seeds: int = 6) -> dict:
    out = diagnostics_dir("d2_uptake")
    rows = _sweep(cfg, n_readers=n_readers, n_seeds=n_seeds)
    df = pd.DataFrame(rows)
    df.to_csv(out / "uptake_curve.csv", index=False)

    # The shape. Uptake against accuracy, which is the axis the depth experiment implicitly assumed
    # was monotone.
    u = df.uptake_ungated.values
    acc = df.accuracy.values
    i_min = int(np.argmin(u))
    endpoints = float(min(u[0], u[-1]))
    span = float(np.max(u) - np.min(u))
    dip = float((endpoints - u[i_min]) / span) if span > 0 else 0.0
    interior = bool(0 < i_min < len(u) - 1)
    non_monotone = bool(interior and dip >= CR.D2_NONMONOTONE_DIP)

    rho_u_acc = CR.spearman(acc, u)
    scaling = _kappa_scaling(cfg)

    if non_monotone:
        verdict = "UPTAKE_IS_NON_MONOTONE_IN_RECOVERY"
        statement = (
            "**Uptake is not monotone in how well the reader recovers the goal.** It falls from "
            "%.3f at full expertise to a minimum of %.3f at inexpertise %.2f, where accuracy is "
            "%.3f, and then rises again to %.3f at zero expertise where accuracy is %.3f. The dip "
            "is %.0f%% of the measure's whole range and its minimum is in the interior.\n\n"
            "The mechanism is visible in the definition rather than mysterious. Uptake is the "
            "distance between what the reader ended up believing and what it started believing. A "
            "reader who gets it right ends far from its prior. A reader who gets nothing ends AT its "
            "prior. A reader who becomes confidently WRONG also ends far from its prior. Measured "
            "here, the confidently wrong readers score %.3f against %.3f for the correct ones, "
            "which is %.0f%% of the correct readers' uptake for a belief that is false.\n\n"
            "**This is a better explanation of the flat depth result than the one on record.** That "
            "experiment regressed uptake on depth level, found it flat, and reported its own "
            "construction at fault for leaving no headroom. A non-monotone response whose arms "
            "straddle the minimum produces the same flat regression, and the two have different "
            "repairs: one needs a construction change and the other needs the arms moved to the "
            "same side of the trough. The rank correlation between uptake and accuracy across this "
            "sweep is %.2f, which is what a flat regression on a U looks like.\n\n"
            "The practical consequence for the repair P-2 sets up: a rerun must not sit at the "
            "trough. That is inexpertise %.2f here, and it is close to the middle of the band P-2 "
            "identifies as the difficulty regime, so the two constraints pull against each other "
            "and the rerun needs both on the table at once."
            % (u[0], u[i_min], df.inexpertise.values[i_min], acc[i_min], u[-1], acc[-1],
               100 * dip,
               float(np.nanmean(df.uptake_of_the_confidently_wrong)),
               float(np.nanmean(df.uptake_of_the_correct)),
               100 * float(np.nanmean(df.uptake_of_the_confidently_wrong)
                           / np.nanmean(df.uptake_of_the_correct)),
               rho_u_acc if np.isfinite(rho_u_acc) else float("nan"),
               df.inexpertise.values[i_min]))
    else:
        verdict = "UPTAKE_IS_MONOTONE_IN_RECOVERY"
        statement = (
            "Uptake decreases monotonically as goal recovery degrades, with a rank correlation of "
            "%.2f against accuracy and no interior minimum. It can therefore be regressed on a "
            "difficulty manipulation without the arms' positions mattering, and the flat depth "
            "result needs the explanation already on record rather than this one."
            % (rho_u_acc if np.isfinite(rho_u_acc) else float("nan")))

    payload = {
        "check": "D-2",
        "question": ("Does the amount a reader takes on rise and fall with how well it read the "
                     "work, or is the relationship a different shape?"),
        "plain_language": (
            "Several experiments measure how much a reader 'takes on' from a work, and treat it as "
            "going up when the reader understands more. The measure is actually a distance between "
            "the reader's belief before and after, and a reader who becomes confidently WRONG has "
            "also moved a long way. So the relationship may be U-shaped rather than a slope, and if "
            "it is, an experiment whose two conditions sit either side of the bottom of the U finds "
            "nothing for reasons unconnected to what it was testing."),
        "criteria": {"nonmonotone_dip": CR.D2_NONMONOTONE_DIP},
        "curve": rows,
        "minimum_at_inexpertise": float(df.inexpertise.values[i_min]),
        "minimum_is_interior": interior,
        "dip_fraction_of_range": dip,
        "spearman_uptake_vs_accuracy": rho_u_acc,
        "kappa_scaling": scaling,
        "engagement_gate_note": (
            "uptake is gated on engagement, so a disengaged reader contributes exactly zero and the "
            "response has a hard discontinuity in it as well as a trough. The ungated series is "
            "reported beside the gated one; the gap between them is the gate."),
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "d2_uptake.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
