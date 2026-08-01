"""E38 — does AI literacy stack with art literacy, or replace it?

THE PREDICTION, AND IT IS THE AUTHOR'S OWN. Someone expert in diffusion could ratchet through
generated work using AI skill instead of art skill, because that is the skill the artifact
actually demands now. If so, the people most familiar with these systems are the least affected
by the crash -- which is an odd and interesting prediction, since it says the phenomenon spares
exactly the population that produced it.

THE SECOND-ORDER PREDICTION IS THE ONE WORTH MEASURING, and it is darker. If expertise
SUBSTITUTES rather than stacks, the machine-matched reader recovers machine content AND LOSES
HUMAN CONTENT by a comparable margin. A crossover, not a dominance. The adaptation that protects
you from the crash is the same adaptation that costs you the human channel -- expertise being
eaten and replaced rather than extended.

THE DESIGN IS A 2x2 and the interesting cell is the off-diagonal one nobody would think to run:
the machine-matched reader on human work.

    reader \\ content    human            machine
    human               the baseline     the crash
    machine             THE TEST         the recovery

Both readers are built by the same code over different hypothesis families, so nothing differs
between them except what they expect a maker to be like.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h66_verdict
from . import harness as H
from . import SEED_OFFSET, v6_dir

READERS = ("human", "machine")
CONTENTS = ("human", "machine")


def run(cfg: Config, n_obs: int = 50, n_timesteps: int = 24, forced_k: int = 12,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    hw, hcfg_b, hcfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(hw.cfg.signal_model.kappa)

    # THE MACHINE FAMILY MUST BE WELL-FORMED, AND THE FIRST VERSION OF THIS EXPERIMENT USED A
    # DEGENERATE ONE. Built from the non-invertible surfaces, the machine reader's hypotheses
    # were collapsed two-to-one, so it could not recover its own content in principle -- ceiling
    # about 0.5 -- and the experiment was measuring the WALL (E37) rather than expertise.
    #
    # The question here is whether a DIFFERENT skill reads generated work, not whether a
    # degraded one does. So the machine family is the foreign signature family: four distinct,
    # well-formed signatures on a feature block the human reader's hypotheses do not cover.
    # Fully invertible for a reader that holds it; opaque to one that does not.
    sigs = FN.build_v4_signatures(hcfg_b, omega=0.0, include_explore=False,
                                  foreign_seed=int(cfg.get("v4.foreign_seed", 20250401)))
    machine_sigs = np.asarray(sigs.sig_foreign, dtype=float)
    mw, mcfg_b, mcfg_r, _, _, mng = H.build_alt_world(cfg, machine_sigs)

    worlds = {"human": (hw, hcfg_r), "machine": (mw, mcfg_r)}

    recs = []
    for reader in READERS:
        r_world, r_cfg = worlds[reader]
        for content in CONTENTS:
            c_world, c_cfg = worlds[content]
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(SEED_OFFSET + 50_000 + i)
                g = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    c_world, c_cfg, g, 2, 1.0, n_timesteps, art_rng)
                agent = H.make_alt_observer(r_world, np.random.default_rng(SEED_OFFSET + 60_000 + i), ng)
                enc = H.run_encounter(r_world, r_cfg, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 60_000 + i),
                                      n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
                recs.append({
                    "reader": reader, "content": content, "observer": i,
                    "goal_correct": int(enc.correct),
                    "final_entropy": float(enc.final_entropy),
                    "engaged_fraction": float(enc.engaged_fraction),
                    "error_reduction": float(enc.error_reduction),
                    "process_accuracy": float(enc.process["process_accuracy"]),
                })

    df = pd.DataFrame(recs)
    out = v6_dir("e38_expertise")
    df.to_csv(out / "e38_expertise.csv", index=False)

    cells = df.groupby(["reader", "content"]).agg(
        goal_accuracy=("goal_correct", "mean"),
        final_entropy=("final_entropy", "mean"),
        engaged_fraction=("engaged_fraction", "mean"),
        error_reduction=("error_reduction", "mean"),
        process_accuracy=("process_accuracy", "mean")).reset_index()
    cells.to_csv(out / "e38_cells.csv", index=False)

    acc = {(r, c): float(cells[(cells.reader == r) & (cells.content == c)].goal_accuracy.iloc[0])
           for r in READERS for c in CONTENTS}
    h66 = h66_verdict(acc)

    verdict = {
        "experiment": "E38",
        "hypothesis": "H6.6",
        "question": "Does AI literacy stack with art literacy, or replace it?",
        "plain_language": (
            "If reading generated work needs a different skill rather than more of the same "
            "skill, then people who know how these systems work should be able to read them. "
            "The question worth asking is what that costs: does the new expertise sit alongside "
            "the old one, or eat it?"),
        "cells": cells.to_dict(orient="records"),
        "H6.6": h66,
        "design_note": ("both readers are built by the same code over different hypothesis "
                        "families, so nothing differs between them except what they expect a "
                        "maker to be like. The interesting cell is the off-diagonal one: the "
                        "machine-matched reader on human work."),
        "n_obs": int(n_obs),
    }
    (v6_dir() / "e38_expertise.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                 encoding="utf-8")
    return verdict
