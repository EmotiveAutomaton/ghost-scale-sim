"""P-2 — the goal-difficulty probe. Is there a regime where reading the goal is genuinely uncertain?

THE HYPOTHESIS, and it is the spec's: goal recovery in this model is close to perfect almost
everywhere, and that ceiling is what makes two experiments unreadable. The generous-fallback control
fails because the reader resolves the goal and then correctly stops paying attention. The depth
experiment has no headroom because the goal is recovered perfectly at every depth. Both are downstream
of the same thing, so one change might address both, and repairing them separately would waste both
attempts.

The probe characterises the model's own difficulty axis. It tests nothing.

-----------------------------------------------------------------------------------------
THE SPEC NAMES THREE KNOBS AND TWO OF THEM DO NOT WORK. Measured before the criteria were locked, and
declared there:

    peak mass 0.90 (default) -> accuracy 1.000
    peak mass 0.22 (nearly flat signatures) -> accuracy 1.000
    goals 4 -> 6 -> 8 -> accuracy 1.000 throughout

The reason is structural rather than a matter of range. Every DEEP observation is an independent draw
from a goal-dependent distribution, so ANY non-zero per-observation evidence accumulates to certainty
over the run. Separation and goal count change the RATE at which the reader gets there, not the
destination. Both are therefore run as single confirmatory cells at their extremes and reported as
measured-dead rather than swept, which is cheaper and more honest than a sweep of a flat axis.

Observation count is the one named knob that bites, and it bites over the integer range one to three,
which also strips out the free steps engagement is measured over.

-----------------------------------------------------------------------------------------
SO A FOURTH KNOB IS ADDED AND MADE PRIMARY: READER INEXPERTISE.

It works for the reason the others do not. Inexpertise perturbs the reader's own goal signatures, which
is a SYSTEMATIC error rather than a sampling one, and systematic error does not average out however
long the run. It is also the model's own account of why the two-dimensions result works: a mis-aimed
template fails silently and keeps failing.

That it is not in the spec's list is not a criticism of the spec. It is the thing the measurement
found, and it was declared in the criteria lock before this sweep ran.

-----------------------------------------------------------------------------------------
WHAT THE PROBE IS ACTUALLY LOOKING FOR, per section 2.4: not that accuracy lands in a band, which is
necessary and not sufficient, but that **uptake has room to move** there. The variance of uptake
across readers is reported at every cell and it is the deciding quantity.
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


def _cell(cfg: Config, *, d_i: float = 0.0, n_timesteps: int | None = None,
          peak: float | None = None, num_goals: int | None = None,
          n_readers: int = 60, n_seeds: int = 6, forced_k: int | None = None) -> dict:
    """One cell of the difficulty grid. Human content only, exact inference, nothing else varying."""
    from ..creators import build_creator_bank
    from ..environment import Artifact, Environment
    from ..generative_model import build_shared_model
    from ..observer import make_observer, rollout_observer

    c = cfg.copy()
    c.set("inference.exact", True)
    if peak is not None:
        c.set("artifact_model.sig_peak_mass", float(peak))
        c.set("artifact_model.js_threshold", 0.0)   # the floor is what is being swept past
    if num_goals is not None:
        g = int(num_goals)
        c.set("cardinalities.num_goals", g)
        c.set("cardinalities.num_features", 2 * g)
        c.set("artifact_model.goal_feature_pairs", [[2 * i, 2 * i + 1] for i in range(g)])
        c.set("artifact_model.structured_ceiling", float(0.9 * np.log(2 * g)))
    T = int(c.run.n_timesteps if n_timesteps is None else n_timesteps)
    k = int(min(T, 10) if forced_k is None else min(T, forced_k))
    ng = int(c.cardinalities.num_goals)
    true_goal = min(1, ng - 1)

    gm = build_shared_model(c)
    bank = build_creator_bank(c, gm)
    acc, ent, upt, eng, correct_conf = [], [], [], [], []
    per_seed_between, per_seed_js = [], []
    for s in range(n_seeds):
        env = Environment(c, gm, np.random.default_rng(21_000 + s), creator_bank=bank)
        art = Artifact(provenance=K.CREATOR, goal=true_goal, declared_signal=K.SIG_CREATOR)
        posts = []
        for i in range(n_readers):
            r = np.random.default_rng(22_000 + 977 * s + i)
            agent = make_observer(gm, c, r, d_i=float(d_i))
            res = rollout_observer(agent, art, env, c, r, T, force_deep_k=k)
            q = np.asarray(res.final_goal_posterior, dtype=float)
            posts.append(q)
            acc.append(int(np.argmax(q) == true_goal))
            ent.append(metrics.within_observer_entropy(q))
            upt.append(metrics.psi_analogue(q, res.goal_prior, float(c.signal_model.kappa), True))
            free = np.asarray(res.attention)[k:]
            eng.append(float(np.mean(free == K.DEEP)) if free.size else float("nan"))
            correct_conf.append(float(q[true_goal]))
        per_seed_between.append(metrics.between_observer_entropy(posts))
        from .d3_disagreement import mean_pairwise_js
        per_seed_js.append(mean_pairwise_js(posts, rng=np.random.default_rng(23_000 + s)))
    return {
        "accuracy": float(np.mean(acc)),
        "goal_entropy": float(np.mean(ent)),
        "goal_entropy_ceiling": float(np.log(ng)),
        "confidence_in_the_truth": float(np.mean(correct_conf)),
        "uptake": float(np.mean(upt)),
        "uptake_sd": float(np.std(upt, ddof=1)) if len(upt) > 1 else 0.0,
        "uptake_iqr": float(np.subtract(*np.percentile(upt, [75, 25]))),
        "engaged_fraction": float(np.nanmean(eng)) if len(eng) else float("nan"),
        "between_reader_entropy": float(np.mean(per_seed_between)),
        "mean_pairwise_js": float(np.mean(per_seed_js)),
        "n_readers": n_readers, "n_seeds": n_seeds, "n_timesteps": T, "forced_deep_k": k,
        "num_goals": ng,
    }


D_GRID = (0.0, 0.30, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.90, 0.95, 1.00)
T_GRID = (1, 2, 3, 4, 6, 12, 24, 40)
CONFIRMATORY = (
    ("signature separation at its extreme", {"peak": 0.22},
     "the spec's first knob, pushed until the signatures are nearly flat"),
    ("goal count doubled", {"num_goals": 8},
     "the spec's third knob, at twice the default"),
)


def run(cfg: Config, workers: int = 1, n_readers: int = 60, n_seeds: int = 6) -> dict:
    out = diagnostics_dir("p2_difficulty")
    rows = []

    default = _cell(cfg, n_readers=n_readers, n_seeds=n_seeds)
    rows.append({"knob": "default", "level": "default", "value": None, **default})

    for d in D_GRID:
        rows.append({"knob": "reader inexpertise", "level": f"d={d}", "value": float(d),
                     **_cell(cfg, d_i=d, n_readers=n_readers, n_seeds=n_seeds)})
    for T in T_GRID:
        rows.append({"knob": "observations before the decision", "level": f"T={T}",
                     "value": float(T),
                     **_cell(cfg, n_timesteps=T, n_readers=n_readers, n_seeds=n_seeds)})
    for name, kw, why in CONFIRMATORY:
        r = _cell(cfg, n_readers=n_readers, n_seeds=n_seeds, **kw)
        rows.append({"knob": "confirmatory (measured dead before the lock)", "level": name,
                     "value": list(kw.values())[0], "why_confirmatory_only": why, **r})

    # The two most promising knobs together, as section 2.2 asks. Inexpertise is one; the other is
    # the observation count, because it is the only other one that moved anything.
    for d in (0.70, 0.80, 0.85):
        for T in (3, 6, 12):
            rows.append({"knob": "joint: inexpertise x observations", "level": f"d={d}, T={T}",
                         "value": float(d),
                         **_cell(cfg, d_i=d, n_timesteps=T, n_readers=n_readers,
                                 n_seeds=n_seeds, forced_k=T)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "difficulty.csv", index=False)

    lo, hi = CR.P2_TARGET_BAND
    base_var = float(default["uptake_sd"])
    df["in_target_band"] = (df.accuracy >= lo) & (df.accuracy <= hi)
    df["uptake_variance_ratio"] = df.uptake_sd / base_var if base_var > 0 else np.nan
    df["uptake_has_room"] = df.uptake_variance_ratio >= CR.P2_UPTAKE_VARIANCE_FACTOR
    df.to_csv(out / "difficulty.csv", index=False)

    band = df[df.in_target_band]
    usable = band[band.uptake_has_room]
    dead = df[df.knob.str.startswith("confirmatory")]
    all_above = bool((df[df.knob != "confirmatory (measured dead before the lock)"].accuracy
                      > CR.P2_CEILING).all())

    if len(usable):
        best = usable.sort_values("uptake_variance_ratio", ascending=False).iloc[0]
        verdict = "REGIME_FOUND"
        statement = (
            "A regime exists. **%s** puts goal accuracy at %.3f, inside the %.2f to %.2f band "
            "committed before the run, and uptake varies %.2f times as much across readers as it "
            "does at the default. %d of the %d cells in the band clear the variance requirement.\n\n"
            "The knob that gets there is reader inexpertise, which the spec does not list. The two "
            "knobs it does list and this pass ran as confirmatory cells are dead as predicted: %s. "
            "Separation and goal count change how fast the reader arrives at certainty, not whether "
            "it arrives, because every deep look is an independent draw and any non-zero evidence "
            "accumulates.\n\n"
            "What makes inexpertise different is that it is a SYSTEMATIC error in the reader's own "
            "templates rather than a sampling one, so it does not average out however long the run. "
            "That is also the model's own account of why a mis-aimed template fails silently, which "
            "means the difficulty axis and the two-dimensions result are the same mechanism seen "
            "from two directions."
            % (str(best["level"]), float(best["accuracy"]), lo, hi,
               float(best["uptake_variance_ratio"]), len(usable), len(band),
               "; ".join("%s leaves accuracy at %.3f" % (r["level"], r["accuracy"])
                         for _, r in dead.iterrows())))
    elif len(band):
        verdict = "ACCURACY_MOVES_UPTAKE_DOES_NOT"
        statement = (
            "Accuracy can be brought into the %.2f to %.2f band, at %d cells, and uptake stays "
            "pinned in all of them: none reaches %.2f times its variance at the default. Per the "
            "spec this is reported as a finding about uptake rather than a failure of the probe. "
            "**Uptake is not sensitive to goal recovery**, and the depth experiment's flat result "
            "was therefore never about depth."
            % (lo, hi, len(band), CR.P2_UPTAKE_VARIANCE_FACTOR))
    elif all_above:
        verdict = "NO_REGIME"
        statement = (
            "Goal accuracy stays above %.2f across every cell of every knob. The difficulty axis is "
            "not continuous in this model, and that is structural rather than a matter of range: "
            "evidence accumulates to certainty whatever the per-observation rate."
            % CR.P2_CEILING)
    else:
        verdict = "NO_REGIME"
        statement = ("No cell lands in the band with anything usable between the ceiling and "
                     "chance.")

    payload = {
        "check": "P-2",
        "question": ("Is there a setting where reading the goal is genuinely uncertain, and does "
                     "anything downstream come alive there?"),
        "plain_language": (
            "In almost every experiment the simulated reader works out the maker's purpose "
            "perfectly. That sounds good and it is actually a problem: if the answer is always "
            "right there is no room for anything downstream to vary, so two experiments came back "
            "uninterpretable. This looks for a setting where the reader is genuinely unsure, and "
            "then checks whether the thing those experiments were trying to measure can move there."),
        "criteria": {"target_band": list(CR.P2_TARGET_BAND), "ceiling": CR.P2_CEILING,
                     "uptake_variance_factor": CR.P2_UPTAKE_VARIANCE_FACTOR},
        "default_cell": default,
        "knob_note": ("the spec names three knobs; two were measured dead before the criteria were "
                      "locked and are run here as single confirmatory cells, and a fourth, reader "
                      "inexpertise, was added and made primary. Both decisions are in the lock."),
        "cells": df.to_dict(orient="records"),
        "cells_in_band": band.level.tolist(),
        "cells_in_band_with_uptake_room": usable.level.tolist(),
        "confirmatory_dead_knobs": dead[["level", "accuracy", "goal_entropy"]].to_dict(
            orient="records"),
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "p2_difficulty.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload
