"""E50 — grabbing attention and keeping it are two decisions, and slop is not shock art.

The essay is precise about this and the model has never been able to represent it:

    aesthetics is "something attention-grabbing, but not necessarily attention-keeping"
    shock art is "raw aesthetic attentional capture without the follow-up in observable
    decision-making required to maintain that attention"

With a single engagement decision repeated at every timestep, the model scores SLOP and SHOCK ART
identically: both end with nothing recovered, so both look like a reader that did not engage. The
theory says they are different objects. One was never looked at. The other was looked at hard, and
then abandoned -- and the abandonment is the finding, because it is a promise that was not kept.

Splitting the decision gives each a signature in the ATTENTION TRACE rather than in the outcome:

    shock art   captured, not sustained   -> a spike and a collapse
    slop        neither                   -> a flat line at the floor
    real work   both                      -> capture that holds

    fails if the two are indistinguishable, in which case capture buys nothing the model can see
    and the essay's distinction has no mechanical content
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import v8_model as V8
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v8_dir

# (name, surface appeal, social endorsement, how much is actually behind it)
ARMS = (
    ("real_work", 0.7, 0.5, 3),
    ("shock_art", 1.0, 0.2, 1),
    ("slop", 0.15, 0.0, 1),
    ("quiet_masterpiece", 0.2, 0.0, 3),
)


def run(cfg: Config, n_obs: int = 60, n_timesteps: int = 16, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    two = V8.TwoStageAttention()

    rows = []
    for name, salience, social, depth in ARMS:
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 89_000 + i)
            g = int(rng.integers(ng))
            creator, art, env = H.make_artifact_and_env(
                world, cfg_r, g, int(depth), 1.0 if depth > 1 else 0.0, n_timesteps, rng)
            enc = H.run_encounter(world, cfg_r, art, env,
                                  make_v5_observer(world, rng), creator, rng,
                                  n_timesteps, 4, n_sub, n_mu, ng, kappa)

            # What the model itself found worth staying for, which is what has to do the sustaining.
            gain = float(max(enc.process["process_error_reduction"], 0.0))
            tr = two.trace(salience, social, min(gain * 4.0, 1.0), n_steps=n_timesteps)
            rows.append({
                "arm": name, "observer": i, "salience": salience, "social": social,
                "depth": int(depth),
                "peak": tr["peak"], "tail": tr["tail"], "collapse": tr["collapse"],
                "captured": int(tr["captured"]), "sustained": int(tr["sustained"]),
                "recovered": gain,
                # The single-decision measure, so the two can be compared directly.
                "single_stage_engagement": float(enc.engaged_fraction),
            })

    df = pd.DataFrame(rows)
    out = v8_dir("e50_grab_vs_keep")
    df.to_csv(out / "e50_traces.csv", index=False)

    cells = df.groupby("arm").agg(
        peak=("peak", "mean"), tail=("tail", "mean"), collapse=("collapse", "mean"),
        captured=("captured", "mean"), sustained=("sustained", "mean"),
        recovered=("recovered", "mean"),
        single_stage=("single_stage_engagement", "mean")).reset_index()

    def _c(name, col):
        return float(cells[cells.arm == name][col].iloc[0])

    # Do the two failures separate on the trace, where the single measure cannot tell them apart?
    two_stage_gap = abs(_c("shock_art", "peak") - _c("slop", "peak"))
    single_gap = abs(_c("shock_art", "single_stage") - _c("slop", "single_stage"))
    separates = bool(two_stage_gap >= 0.30 and two_stage_gap > single_gap * 2.0)

    # And the case the essay cares about in the other direction: work with nothing on the surface
    # and everything behind it. Nobody looks, so nobody finds out.
    quiet = {"peak": _c("quiet_masterpiece", "peak"),
             "recovered_if_looked_at": _c("quiet_masterpiece", "recovered"),
             "captured": _c("quiet_masterpiece", "captured")}

    verdict = {
        "experiment": "E50",
        "hypothesis": "H8.5",
        "question": "Are grabbing attention and keeping it two different decisions?",
        "plain_language": (
            "With one engagement decision repeated, this model scores slop and shock art the same: "
            "both end with nothing recovered. The theory says they are different things. One was "
            "never looked at; the other was looked at hard and then abandoned, and the "
            "abandonment is the point, because it is a promise that was not kept."),
        "cells": cells.to_dict(orient="records"),
        "separation": {
            "on_the_two_stage_trace": two_stage_gap,
            "on_the_single_engagement_measure": single_gap,
            "outcome": ("SHOCK_ART_AND_SLOP_SEPARATE_ON_THE_TRACE" if separates
                        else "CAPTURE_BUYS_NOTHING_THE_MODEL_CAN_SEE"),
        },
        "the_quiet_masterpiece": {
            **quiet,
            "note": ("the other direction the essay cares about: everything behind it and nothing "
                     "on the surface. Nobody looks, so nobody finds out, and a single-decision "
                     "model scores it the same as slop."),
        },
        "n_obs": int(n_obs),
    }
    (v8_dir() / "e50_grab_vs_keep.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                    encoding="utf-8")
    return verdict
