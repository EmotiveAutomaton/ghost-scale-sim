"""T-9 — S-2's actual question, asked on an emitter that works.

S-2 IS WITHDRAWN AND ITS QUESTION WAS NEVER TESTED. That distinction is the reason this module
exists. S-2's manipulation was drawn and discarded -- ``V5Environment.sample_feature`` ignores
``artifact.goal`` once a creator is bound -- so the feature streams were bit-identical with the
mixture switched off. What was withdrawn is the measurement, not the hypothesis.

THE HYPOTHESIS, in the curator's own correction of a stronger earlier version:

    It is unfair to say corporate goals are singular... the SHARE of the goal is
    disproportionately large. You'd expect that to be a larger piece, NOT the whole pie.

So the construct is **concentration**, not thinness, and the two are confounded in every real
corpus. That is the entire reason it belongs in a simulation.

WHAT IS DIFFERENT FROM T-2. T-2 swept diversity from a point mass to flat and asked whether a
reader sees variety as posterior breadth. It found breadth rises, goal accuracy collapses to below
chance, and -- decisively -- that at MATCHED goal accuracy diverse work carries no more breadth
than single-drive work made equally hard to read. `BREADTH_IS_LARGELY_A_DIFFICULTY_METER`.

T-9 asks the narrower question S-2 was aimed at and T-2 swept past: not *flat versus point mass*
but **dominant versus flat**, the regime where a real corporate artifact lives. A 70% share is not
a point mass and it is not a flat mixture, and T-2's sweep only ever visited it in passing at
automaticity 0.25 and 0.5.

THE SAME DIFFICULTY CONTROL IS ATTACHED, because without it this module would repeat S-2's
mistake in a subtler form: reporting a breadth difference that is really a legibility difference.

THE LIVE GATE HERE IS THE ONE S-2 FAILED, AND IT MUST PASS. Forcing the mixture to a constant has
to change what the reader sees. S-2 records the same gate at exactly 0.0 and is marked
expected_to_fail; this module is the control that shows the gate discriminates.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...baselines import ObservationTape, TapedEnvironment
from ...config import Config
from ...methods import gates as G
from ...methods import inference as INF
from ...methods import provenance as PROVENANCE
from ...v5_model import make_v5_observer
from ...v6 import SEED_OFFSET, harness as H
from . import sl_dir
from . import t_common as T
from .common import build
from .t2_automaticity import READ_GLANCES, READ_TIER, mixed_deep_features

MU = 3
BETA = 1.0
N_TIMESTEPS = 24
#: S-2's own dominant share, plus the neighbours that say whether the answer is a knife edge.
SHARES = (0.40, 0.55, 0.70, 0.85, 1.00)
#: Alphas for the difficulty curve: single-drive work made progressively harder to read.
ALPHAS = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20)


def _weights(share: float, ng: int) -> np.ndarray:
    """One dominant drive at ``share``, the rest splitting the remainder evenly."""
    w = np.full(ng, (1.0 - float(share)) / max(ng - 1, 1))
    w[0] = float(share)
    return w / w.sum()


def _one(world, cfg_r, n_mu, n_sub, ng, w, seed_base, i, alpha=None,
         force_constant=False) -> dict:
    art_rng = np.random.default_rng(seed_base * 31 + i)
    modal = int(np.argmax(w))
    creator, artifact, env = H.make_artifact_and_env(
        world, cfg_r, modal, MU, BETA, N_TIMESTEPS, art_rng, provenance=READ_TIER)
    tape = ObservationTape(env, artifact, np.random.default_rng(seed_base * 104729 + i),
                           N_TIMESTEPS)
    actives = art_rng.choice(ng, size=N_TIMESTEPS, p=w)
    if force_constant:
        actives = np.full(N_TIMESTEPS, modal, dtype=int)
    tape.deep = mixed_deep_features(world, env, creator, actives, MU, n_sub, ng, N_TIMESTEPS,
                                    art_rng, alpha_override=alpha)
    agent = make_v5_observer(world, np.random.default_rng(seed_base * 7907 + i))
    enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent, creator,
                          np.random.default_rng(seed_base * 7907 + i), N_TIMESTEPS,
                          READ_GLANCES, n_sub, n_mu, ng,
                          float(world.cfg.signal_model.kappa), true_goal=modal)
    hmax = float(np.log(max(ng, 2)))
    counts = np.bincount(actives, minlength=ng).astype(float)
    return {
        "purpose_breadth": float(metrics.within_observer_entropy(enc.goal_posterior)) / hmax,
        "goal_correct": int(enc.correct),
        "process": float(enc.process["process_error_reduction"]),
        "true_mixture_breadth": float(
            metrics.within_observer_entropy(counts / counts.sum())) / hmax,
        "distinct_goals_used": int(len(set(actives.tolist()))),
    }


def run(cfg: Config, n_obs: int = 250) -> dict:
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    rng = np.random.default_rng(SEED_OFFSET + 91_900)
    rows = []

    for share in SHARES:
        w = _weights(share, ng)
        for i in range(int(n_obs)):
            r = _one(world, cfg_r, n_mu, n_sub, ng, w, 73_000 + int(share * 100), i)
            r.update({"axis": "share", "share": float(share), "alpha": None, "i": i})
            rows.append(r)
    for alpha in ALPHAS:
        w = _weights(1.0, ng)                      # single drive, made harder to read
        for i in range(int(n_obs)):
            r = _one(world, cfg_r, n_mu, n_sub, ng, w, 74_000 + int(alpha * 100), i, alpha=alpha)
            r.update({"axis": "difficulty", "share": 1.0, "alpha": float(alpha), "i": i})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t9_concentration_points.csv", index=False)
    (df.groupby(["axis", "share", "alpha"], dropna=False)
       [["purpose_breadth", "goal_correct", "process", "true_mixture_breadth"]]
       .agg(["mean", "std", "count"]).to_csv(sl_dir() / "t9_concentration_summary.csv"))

    def sel(**kw):
        d = df
        for k, v in kw.items():
            d = d[d[k] == v]
        return d

    series = {f"{s:.2f}": {
        "purpose_breadth": float(sel(axis="share", share=s).purpose_breadth.mean()),
        "goal_accuracy": float(sel(axis="share", share=s).goal_correct.mean()),
        "process": float(sel(axis="share", share=s).process.mean()),
        "true_mixture_breadth": float(sel(axis="share", share=s).true_mixture_breadth.mean()),
    } for s in SHARES}

    dom, flat = sel(axis="share", share=0.70), sel(axis="share", share=0.40)
    pure = sel(axis="share", share=1.00)
    contrasts = {
        "dominant_70_minus_flat_40_breadth": T.boot_paired(
            dom.purpose_breadth.to_numpy(), flat.purpose_breadth.to_numpy(), rng),
        "dominant_70_minus_flat_40_goal_accuracy": T.boot_paired(
            dom.goal_correct.to_numpy(), flat.goal_correct.to_numpy(), rng),
        "dominant_70_minus_flat_40_process": T.boot_paired(
            dom.process.to_numpy(), flat.process.to_numpy(), rng),
        "dominant_70_minus_single_drive_breadth": T.boot_paired(
            dom.purpose_breadth.to_numpy(), pure.purpose_breadth.to_numpy(), rng),
    }

    # ---- the difficulty control, exactly T-2's ------------------------------------------------
    curve = []
    for a in ALPHAS:
        d = sel(axis="difficulty", alpha=float(a))
        curve.append({"alpha": float(a), "goal_accuracy": float(d.goal_correct.mean()),
                      "purpose_breadth": float(d.purpose_breadth.mean())})
    curve.sort(key=lambda r: r["goal_accuracy"])
    acc_axis, br_axis = {}, []
    for p in curve:
        acc_axis.setdefault(p["goal_accuracy"], []).append(p["purpose_breadth"])
    xs = sorted(acc_axis)
    br_axis = [float(np.mean(acc_axis[a])) for a in xs]

    matched = {}
    for s in SHARES:
        d = sel(axis="share", share=s)
        acc, br = float(d.goal_correct.mean()), float(d.purpose_breadth.mean())
        eq = float(np.interp(acc, xs, br_axis))
        matched[f"{s:.2f}"] = {
            "goal_accuracy": acc, "breadth": br,
            "breadth_of_equally_hard_single_drive_work": eq,
            "excess": float(br - eq),
            "in_range_of_difficulty_curve": bool(min(xs) <= acc <= max(xs)),
        }
    base = matched["1.00"]["excess"]        # the single-drive arm IS a point on the curve
    for v in matched.values():
        v["excess_above_single_drive_baseline"] = float(v["excess"] - base)
    contested = [v["excess_above_single_drive_baseline"] for k, v in matched.items()
                 if float(k) < 1.0 and v["in_range_of_difficulty_curve"]]
    separates = bool(contested and min(contested) > 0.02)

    # ---- gates ---------------------------------------------------------------------------------
    gr = G.GateReport()
    off = []
    for i in range(min(int(n_obs), 40)):
        w = _weights(0.40, ng)
        a = _one(world, cfg_r, n_mu, n_sub, ng, w, 73_040, i, force_constant=False)
        b = _one(world, cfg_r, n_mu, n_sub, ng, w, 73_040, i, force_constant=True)
        off.append(abs(a["purpose_breadth"] - b["purpose_breadth"]))
    gr.live("mixture_reaches_the_reader", float(np.mean(off)), 0.01,
            detail=("forcing the mixture to a constant must change what the reader sees. THIS IS "
                    "THE GATE S-2 FAILED AT EXACTLY 0.0, and it is here so the gate is shown to "
                    "discriminate rather than merely to exist: same check, same shape, working "
                    "emitter, and it passes."))
    gr.positive("single_drive_has_zero_true_mixture_breadth",
                float(pure.true_mixture_breadth.mean()), 0.0, 1e-9,
                detail="a mixture concentrated on one drive has zero entropy by construction.")
    gr.identity("true_mixture_breadth_is_monotone_in_share",
                float(max(0.0, max(np.diff([series[f'{s:.2f}']['true_mixture_breadth']
                                            for s in SHARES])))), 0.0, 1e-9,
                detail=("the generative mixture's own entropy must fall as the dominant share "
                        "rises. A check on the construction, not on the reader."))

    bound, src = INF.smallest_effect_of_interest(
        contrasts["dominant_70_minus_single_drive_breadth"]["difference"], 0.10,
        "the 70%-versus-single-drive breadth difference, the largest live effect here")
    equiv = INF.equivalence_from_interval(
        contrasts["dominant_70_minus_flat_40_goal_accuracy"]["difference"],
        contrasts["dominant_70_minus_flat_40_goal_accuracy"]["interval"], 0.05,
        "5 percentage points of goal accuracy: below this a breadth difference cannot be "
        "attributed to the work having become harder to read")

    verdict = {
        "test": "T-9 — is concentrated intent visible as posterior concentration?",
        "for": "Sounding Line, C-22 and purpose_breadth. S-2's question, on a working emitter",
        "WHY_THIS_EXISTS": (
            "S-2 is withdrawn because its manipulation never reached the reader, not because its "
            "question was answered. This asks the question. The emitter is "
            "t2_automaticity.mixed_deep_features, which builds each position from "
            "world.subsig[mu, active_goal, mode] directly, so the drive mixture is actually in "
            "the observations."),
        "construction": {"read_tier": "CURATOR", "read_glances": READ_GLANCES, "mu": MU,
                         "beta": BETA, "n_goals": int(ng), "dominant_shares": list(SHARES),
                         "n_per_arm": int(n_obs),
                         "note": "decision density identical across share arms by construction"},
        "series_by_dominant_share": series,
        "contrasts": contrasts,
        "difficulty_control": {
            "curve": curve, "matched": matched,
            "excess_above_baseline_by_share": {
                k: v["excess_above_single_drive_baseline"] for k, v in matched.items()},
            "verdict": ("BREADTH_SEPARATES_CONCENTRATION_FROM_DIFFICULTY" if separates
                        else "BREADTH_IS_LARGELY_A_DIFFICULTY_METER"),
            "how_to_read": (
                "the curve is single-drive work made progressively harder to read by lowering the "
                "channel's alpha. If a mixture arm sits ABOVE it at the same goal accuracy, the "
                "extra breadth is concentration rather than difficulty. If it sits on the curve, "
                "purpose_breadth cannot tell 'a dominant purpose' from 'a hard read'."),
        },
        "goal_accuracy_should_not_move": equiv,
        "equivalence_bound": {"bound": bound, "source": src},
        "what_would_have_falsified_it": (
            "breadth flat across the dominant-share sweep, which would make posterior entropy "
            "blind to concentration entirely; or the excess vanishing against the difficulty "
            "curve, which would make it a legibility meter with a suggestive name."),
        "what_this_cannot_show": (
            "anything about corporate text. 'Concentration' here is the entropy of a drive "
            "mixture this module sets by hand. Whether real institutional writing has that shape "
            "is exactly what a simulation cannot say."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "t9_concentration.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
