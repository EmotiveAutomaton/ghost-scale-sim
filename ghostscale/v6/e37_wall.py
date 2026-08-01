"""E37 — the wall: content that is legible and empty.

THE CRITIQUE THIS ANSWERS, WHICH LANDS ON THE CONSTRUCTION RATHER THAN ON A RESULT.

Readability and reader inexpertise are both built in this model as scalars on a shared-vocabulary
axis, differing only in whether the content moved or the reader moved. E32 found they behave
OPPOSITELY in consequence -- the unskilled reader quits and feels settled, the expert facing
foreign content keeps working and stays lost -- which is real evidence that they are not the same
thing. But the objection was about the build: **the wall in front of generated content is not a
degree of vocabulary overlap.**

Humans cheat when reading other humans, because they assume the maker's decisions bottom out in
human-shaped sensory experience. He felt sad, so the colours went that way. That assumption is
what makes the inversion tractable at all. With a generative model there is no mapping
translation -- not less overlap, but no invertible route from the surface back to a state you
could occupy.

SO THE THIRD CONDITION IS A MANY-TO-ONE GENERATOR ON FAMILIAR FEATURES. Several maker states emit
the same surface distribution, on features the reader knows perfectly well. Full vocabulary,
familiar structure, and the inversion still does not exist.

WHAT IT PREDICTS THAT NEITHER EXISTING CONDITION DOES: **legible and empty.** Low uncertainty
about what is on the surface, no recovery of what is behind it. That is the complaint people
actually make about generated text, and it is not "I cannot parse this" -- which is what foreign
content produces and what the model has been offering as its account of the phenomenon.

NULL N30. The non-invertible family must stay on the HUMAN feature block. Without that this is
foreign content wearing new vocabulary and the hypothesis cannot fail.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h65_verdict
from ..v5_model import make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

ARMS = ("human", "foreign", "noninvertible")


def run(cfg: Config, n_obs: int = 60, n_timesteps: int = 24, forced_k: int = 10,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    # The READER is always the ordinary human-family reader. That is the point: this is about
    # what the content does to a normal reader, not about a mismatched one.
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)

    sig_true = np.asarray(world.gm.sig_true, dtype=float)
    family = V6.build_noninvertible_family(sig_true, n_states=ng, collapse_to=2)

    # The non-invertible CONTENT world: a generator whose maker states collapse onto shared
    # surfaces. The reader is not given this world; it is what the artifacts come from.
    content_sigs = V6.build_machine_matched_signatures(family, ng)
    nw, ncfg_b, ncfg_r, _, _, nng = H.build_alt_world(cfg, content_sigs)

    # Foreign content: the existing account, at zero overlap.
    fworld, fcfg_b, fcfg_r, _, _, fng = H.build_world_and_config(cfg)

    recs = []
    for arm in ARMS:
        for i in range(int(n_obs)):
            art_rng = np.random.default_rng(SEED_OFFSET + 40_000 + i)
            g = int(art_rng.integers(ng))
            if arm == "noninvertible":
                creator, artifact, env = H.make_artifact_and_env(
                    nw, ncfg_r, g, 2, 1.0, n_timesteps, art_rng, provenance=K.CREATOR)
            elif arm == "foreign":
                # Foreign content is the V4 account: real structure on a block the reader's
                # hypotheses do not cover. Produced here by driving alpha to the GHOST tier so
                # the reader's own family cannot explain the surface.
                creator, artifact, env = H.make_artifact_and_env(
                    fworld, fcfg_r, g, 2, 0.0, n_timesteps, art_rng, provenance=K.GHOST)
            else:
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0, n_timesteps, art_rng, provenance=K.CREATOR)

            agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 88_000 + i))
            enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                  np.random.default_rng(SEED_OFFSET + 88_000 + i),
                                  n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
            recs.append({
                "arm": arm, "observer": i,
                "goal_correct": int(enc.correct),
                "final_entropy": float(enc.final_entropy),
                "engaged_fraction": float(enc.engaged_fraction),
                "error_reduction": float(enc.error_reduction),
                "movement": float(enc.movement),
                "process_accuracy": float(enc.process["process_accuracy"]),
                "posterior": enc.goal_posterior.tolist(),
            })

    df = pd.DataFrame(recs)
    out = v6_dir("e37_wall")
    df.to_csv(out / "e37_wall.csv", index=False)

    cells = {}
    for arm in ARMS:
        a = df[df.arm == arm]
        posts = [np.asarray(p, dtype=float) for p in a.posterior]
        cells[arm] = {
            "goal_accuracy": float(a.goal_correct.mean()),
            "final_entropy": float(a.final_entropy.mean()),
            "engaged_fraction": float(a.engaged_fraction.mean()),
            "error_reduction": float(a.error_reduction.mean()),
            "movement": float(a.movement.mean()),
            "process_accuracy": float(a.process_accuracy.mean()),
            "between_observer": float(metrics.between_observer_entropy(posts)),
            "pairwise_divergence": float(metrics.mean_pairwise_js(posts)),
        }

    h65 = h65_verdict(cells["noninvertible"], cells["foreign"])

    verdict = {
        "experiment": "E37",
        "hypothesis": "H6.5",
        "question": ("Is the wall in front of generated content a vocabulary deficit, or the "
                     "absence of an invertible route back to a maker state?"),
        "plain_language": (
            "The model has been describing unreadable machine content as content written in a "
            "vocabulary the reader lacks. The objection is that this is not what it feels like: "
            "you can read every word and there is still nothing behind it. This builds content "
            "on FAMILIAR features whose maker cannot be inverted from the surface, and asks "
            "whether that is a different failure from unfamiliar content."),
        "cells": cells,
        "H6.5": h65,
        "null_n30": {
            "statement": "the non-invertible family stays on the human feature block",
            "max_foreign_mass": float(family["max_foreign_mass"]),
            "passed": bool(family["max_foreign_mass"] < 0.10),
            "why": ("without it this is foreign content in new vocabulary and the hypothesis "
                    "cannot fail"),
        },
        "family": {"n_states": family["n_states"], "collapse_to": family["collapse_to"],
                   "invertible": family["invertible"]},
        "n_obs": int(n_obs),
    }
    (v6_dir() / "e37_wall.json").write_text(json.dumps(verdict, indent=2, default=str),
                                            encoding="utf-8")
    return verdict
