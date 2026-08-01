"""E52 — a maker defined by what it will not do, and the reader that has no word for that.

THE SEALED PREDICTION, AND AN HONEST DOWNGRADE OF ITS STATUS.

During the validation pass one prediction was written down and content-hash locked for an
experiment that did not exist. It has been described ever since as the project's only forward test,
because everything else was predicted by people who already knew the theory and the literature check
happened afterwards.

**The author does not recognise authoring it.** It appears to have come out of an exchange with a
different model during an earlier phase of the work. A sealed prediction is worth exactly what the
commitment behind it was worth, and a commitment nobody remembers making is not a forward test.

So the experiment is built and run -- it is a good experiment, and avoidance-defined intent is a
real gap -- but **it is not counted as this project's forward test, and the scoreboard is corrected
to say the project still has none.** That correction costs the project its only such claim. It is
made anyway, because the alternative is banking a commitment that was not really made.

-----------------------------------------------------------------------------------------
WHY IT IS STILL WORTH RUNNING.

Every hypothesis space in this project is a space of things a maker might be TRYING TO DO. Avoidance
is a different shape of intention: the purpose is specified by what is absent rather than by what is
present, so the evidence for it is a HOLE in the distribution rather than a peak in it.

The sealed primary: a reader equipped with avoidance hypotheses recovers the constraint and a reader
without does not.

The sealed secondary, which is the more interesting one: the unequipped reader is predicted to be
CONFIDENTLY WRONG rather than uncertain. An avoidance leaves a hole; every positive hypothesis has
support inside that hole; and the one whose peak sits furthest from it wins by default.

If that holds it is the interior-peak mechanism arriving by a completely different route, which
would make it two independent instances of one thing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v8_dir

# The sealed thresholds, carried over verbatim from the validation pass's locked file.
SEALED_ACCURACY_GAP = 0.30
SEALED_CONFIDENCE = 0.5
SEALED_ATTENTION_RATIO = 1.5


def _avoidant_signature(sig_true: np.ndarray, avoided: int) -> np.ndarray:
    """What a maker emits when its purpose is a region it will not enter.

    The maker is otherwise ordinary. What defines it is the ABSENCE: it never emits the avoided
    goal's features. So the surface is the average of everything else, with a hole where the
    constraint is -- and a reader looking for a peak has nothing to find, while a reader looking
    for a hole has everything.
    """
    sig = np.asarray(sig_true, dtype=float)
    others = [g for g in range(sig.shape[0]) if g != int(avoided)]
    out = sig[others].mean(axis=0)
    # The hole: mass is removed from the avoided goal's own support and redistributed.
    hole = sig[int(avoided)] > sig[int(avoided)].mean()
    out = out.copy()
    out[hole] *= 0.05
    return out / out.sum()


def run(cfg: Config, n_obs: int = 60, n_timesteps: int = 20, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    sig_true = np.asarray(world.gm.sig_true, dtype=float)

    # READER TWO holds avoidance hypotheses: one per region, in the same likelihood family. Built
    # as an alternate world whose goals ARE the avoidance signatures, so nothing about the reader
    # differs except what it can entertain.
    avoid_sigs = np.stack([_avoidant_signature(sig_true, g) for g in range(ng)])
    try:
        w2, b2, r2, _, _, ng2 = H.build_alt_world(cfg, avoid_sigs)
        buildable = True
    except AssertionError as exc:
        return {"experiment": "E52", "buildable": False, "error": repr(exc),
                "outcome": "NOT_BUILDABLE"}

    rows = []
    for reader in ("pursuit_only", "holds_avoidance"):
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 91_000 + i)
            avoided = int(rng.integers(ng))
            # The artifact comes from the avoidant maker either way; only the reader changes.
            creator, art, env = H.make_artifact_and_env(
                w2, r2, avoided, 2, 1.0, n_timesteps, rng, provenance=K.CREATOR)
            if reader == "holds_avoidance":
                agent, use_world, use_cfg = H.make_alt_observer(w2, rng, ng), w2, r2
            else:
                agent, use_world, use_cfg = make_v5_observer(world, rng), world, cfg_r
            enc = H.run_encounter(use_world, use_cfg, art, env, agent, creator, rng,
                                  n_timesteps, 8, n_sub, n_mu, ng, kappa, true_goal=avoided)
            rows.append({
                "reader": reader, "observer": i, "avoided": avoided,
                "correct": int(enc.correct),
                "final_entropy": float(enc.final_entropy),
                "engaged_fraction": float(enc.engaged_fraction),
                "cum_deep": int(np.sum(np.asarray(enc.attention) == K.DEEP)),
                "posterior": enc.goal_posterior.tolist(),
            })

    df = pd.DataFrame(rows)
    out = v8_dir("e52_avoidance")
    df.to_csv(out / "e52_avoidance.csv", index=False)

    cells = {}
    for r in ("pursuit_only", "holds_avoidance"):
        s = df[df.reader == r]
        posts = [np.asarray(p, dtype=float) for p in s.posterior]
        cells[r] = {
            "accuracy": float(s.correct.mean()),
            "within_observer": float(np.mean([metrics.within_observer_entropy(p) for p in posts])),
            "engaged_fraction": float(s.engaged_fraction.mean()),
            "deep_looks": float(s.cum_deep.mean()),
        }

    chance = 1.0 / ng
    gap = cells["holds_avoidance"]["accuracy"] - cells["pursuit_only"]["accuracy"]
    primary = bool(gap >= SEALED_ACCURACY_GAP)
    confidently_wrong = bool(cells["pursuit_only"]["within_observer"] < SEALED_CONFIDENCE
                             and cells["pursuit_only"]["accuracy"] <= chance + 0.05)
    ratio = (cells["holds_avoidance"]["deep_looks"]
             / max(cells["pursuit_only"]["deep_looks"], 1e-9))
    cost = bool(ratio >= SEALED_ATTENTION_RATIO)

    if primary and confidently_wrong:
        branch = "SEALED_PREDICTION_HOLDS_INCLUDING_THE_SECONDARY"
    elif primary:
        branch = "GAP_BUT_READER_ONE_IS_UNCERTAIN"
    elif gap < -0.05:
        branch = "GAP_REVERSES"
    else:
        branch = "NO_GAP"

    verdict = {
        "experiment": "E52",
        "question": ("Can a reader recover an intention defined by what its maker WOULD NOT do, "
                     "and what happens to a reader that has no hypothesis of that shape?"),
        "plain_language": (
            "Every hypothesis in this project is a thing a maker might be trying to do. Avoidance "
            "is a different shape: the purpose is what is absent, so the evidence is a hole rather "
            "than a peak. A reader with no word for that has to explain the hole with something, "
            "and the prediction is that it picks confidently and wrongly."),
        "epistemic_status": {
            "was_described_as": "the project's only forward test, sealed with a content hash",
            "is_now": "a good experiment whose sealed status is withdrawn",
            "why": ("the author does not recognise authoring the sealed prediction; it appears to "
                    "have come out of an exchange with a different model. A sealed prediction is "
                    "worth what the commitment behind it was worth, and a commitment nobody "
                    "remembers making is not a forward test."),
            "consequence": ("the project has NO forward test, and the scoreboard is corrected to "
                            "say so. That costs it its only such claim and the correction is made "
                            "anyway."),
        },
        "cells": cells,
        "sealed_criteria": {
            "accuracy_gap": SEALED_ACCURACY_GAP, "confidence": SEALED_CONFIDENCE,
            "attention_ratio": SEALED_ATTENTION_RATIO,
        },
        "measured": {"accuracy_gap": gap, "attention_ratio": ratio, "chance": chance,
                     "primary_holds": primary, "confidently_wrong": confidently_wrong,
                     "cost_prediction_holds": cost},
        "branch": branch,
        "why_it_is_still_worth_running": (
            "the secondary prediction -- that the unequipped reader is confidently WRONG rather "
            "than uncertain -- would be the interior-peak mechanism arriving by a different route, "
            "and that would make it two independent instances of one thing."),
        "n_obs": int(n_obs),
    }
    (v8_dir() / "e52_avoidance.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                 encoding="utf-8")
    return verdict
