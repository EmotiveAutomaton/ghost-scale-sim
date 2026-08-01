"""E41 — two mechanisms for one phenomenon, and the values layer.

THE MOST CONSEQUENTIAL ITEM IN V6, because it is not an addition. It is a disagreement between
the theory and its own implementation that nobody had noticed.

THE PREPRINT'S TRUST EXPLOIT: trust SUPPRESSES the disgust threshold. "When kappa -> 1, the DMN
activates oxytocinergic pathways that directly inhibit the threat and error-detection responses
of the anterior insula and amygdala. This inhibition drives theta_EC -> 0, holding the sensory
gate open despite the bottom-up signal of omega -> 0." The gate is held open while the signal
collapses, and the brain must then resolve a flat partition function, so it over-fits.

THE CODE'S TRUST EXPLOIT: the label channel out-argues the content channel. Both arrive every
glance, they disagree on a lie, and which wins is an inequality with an analytic crossover at
kappa = 0.538. Above it the reader ends up believing the label; below it, the content.

BOTH PRODUCE THE PHENOMENON AND THEY ARE NOT THE SAME CLAIM. In the code's version the exploit
works THROUGH the reader's belief about provenance -- it is fooled about who made the thing. In
the preprint's version the exploit works on the GATE, so it should still fire on a reader that
CORRECTLY BELIEVES the content is machine-made, as long as it trusts the source. The code cannot
produce that at all: with an honest label the reader's provenance belief is right and there is
nothing left to exploit.

SO THE DISCRIMINATING CELL IS THE HONEST LABEL AT HIGH TRUST, and it is the whole experiment.

    coupled     an honest label, high trust  ->  integrates anyway   (the gate is suppressed)
    uncoupled   an honest label, high trust  ->  does not integrate  (nothing to be fooled about)

Three outcomes are all informative and all are named in advance, which matters because a
comparison with no pre-registered branches can be reframed after the answer arrives:

    COUPLING_PREDICTS_AN_EXPLOIT_THE_RACE_CANNOT   the preprint's mechanism does work the code's
                                                   cannot, and the code is missing something
    MECHANISMS_BEHAVIOURALLY_IDENTICAL             the distinction is formal only; keep the
                                                   simpler one
    RACE_ALONE_ACCOUNTS_FOR_THE_EXPLOIT            the coupling buys nothing and the preprint
                                                   should adopt the code's account

-----------------------------------------------------------------------------------------
THE VALUES LAYER RIDES ALONG, because it changes what the gate's divergence term is computed on
and this is the only experiment that reads the gate directly.

The gate in V1-V5 compares the recovered GOAL to the reader's value prior. The theory says
something one step removed: you infer the goal, the goal implies VALUES, and the values decide
whether the process is allowed to integrate. Non-injective by construction (null N26), because
what the layer buys is that two different goals can imply the same values -- "I disagree with
what you were doing but we want the same things", which the code could not previously represent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h62_verdict
from ..v5_model import make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

KAPPA_GRID = (0.20, 0.40, 0.538, 0.70, 0.90)
LABELS = ("honest", "false")


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, forced_k: int = 10,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    lam = float(cfg.get("v6.gate.lam", 1.0))
    k_gain = float(cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(cfg.get("v6.gate.theta_0", 0.35))

    values_map = V6.build_values_map(ng, n_values=2)
    value_prior = np.array([0.9, 0.1])          # the reader's own values, deliberately lopsided

    # N26: the map must be non-injective, or the layer is the goal renamed.
    n26_passed = bool(values_map.shape[0] < ng)

    recs = []
    for label in LABELS:
        for kappa in KAPPA_GRID:
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(SEED_OFFSET + 15_000 + i)
                g = int(art_rng.integers(ng))
                # Machine-made content throughout: this is about what a reader does with work
                # that has no human goal behind it, told the truth or told a lie.
                sig = K.SIG_GHOST if label == "honest" else K.SIG_CREATOR
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 0.0, n_timesteps, art_rng,
                    provenance=K.GHOST, declared_signal=sig, signing_rate=1.0)
                agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 16_000 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 16_000 + i),
                                      n_timesteps, forced_k, n_sub, n_mu, ng, float(kappa))

                for coupling in (0.0, 1.0):
                    for use_values in (False, True):
                        out = H.integration(
                            enc, float(kappa), None, lam, coupling, k_gain,
                            values_map=values_map if use_values else None,
                            value_prior=value_prior if use_values else None,
                            theta_0=theta_0)
                        recs.append({
                            "label": label, "kappa": float(kappa), "coupling": coupling,
                            "values_layer": bool(use_values), "observer": i,
                            "integration": float(out["integration"]),
                            "theta_EC": float(out["theta_EC"]),
                            "value_divergence": float(out["value_divergence"]),
                            "psi": float(out["psi"]),
                            "goal_correct": int(enc.correct),
                            "final_entropy": float(enc.final_entropy),
                            "error_reduction": float(enc.error_reduction),
                        })

    df = pd.DataFrame(recs)
    out_dir = v6_dir("e41_coupling")
    df.to_csv(out_dir / "e41_coupling.csv", index=False)

    cells = df.groupby(["label", "kappa", "coupling", "values_layer"]).agg(
        integration=("integration", "mean"), theta_EC=("theta_EC", "mean"),
        value_divergence=("value_divergence", "mean"), psi=("psi", "mean"),
        goal_accuracy=("goal_correct", "mean"),
        error_reduction=("error_reduction", "mean")).reset_index()
    cells.to_csv(out_dir / "e41_cells.csv", index=False)

    # THE DISCRIMINATING CELL: honest label, goal-based gate, across the trust grid.
    honest = cells[(cells.label == "honest") & (~cells.values_layer)].sort_values("kappa")
    coupled = honest[honest.coupling == 1.0].integration.to_numpy()
    uncoupled = honest[honest.coupling == 0.0].integration.to_numpy()
    h62 = h62_verdict(coupled, uncoupled, KAPPA_GRID)

    false_lab = cells[(cells.label == "false") & (~cells.values_layer)].sort_values("kappa")

    verdict = {
        "experiment": "E41",
        "hypothesis": "H6.2",
        "question": ("The preprint and the code explain the trust exploit by two different "
                     "mechanisms. Do they predict the same thing?"),
        "plain_language": (
            "The paper says a trusting reader is exploited because trust switches off the alarm "
            "that would normally stop it absorbing something. The code says a trusting reader is "
            "exploited because the label out-argues the work. Those sound alike and they are "
            "not: the first should still happen when the reader is told the TRUTH, and the "
            "second cannot. That is the whole experiment."),
        "H6.2": h62,
        "discriminating_cell": {
            "what": "an honest label on machine content, at high trust",
            "why": ("under the channel race the reader's provenance belief is correct and there "
                    "is nothing to exploit; under the coupled gate the threshold is suppressed "
                    "and the reader integrates anyway"),
            "coupled": [float(x) for x in coupled],
            "uncoupled": [float(x) for x in uncoupled],
            "kappa_grid": [float(k) for k in KAPPA_GRID],
        },
        "false_label_reference": false_lab.to_dict(orient="records"),
        "values_layer": {
            "n_values": int(values_map.shape[0]), "n_goals": int(ng),
            "null_n26_non_injective": n26_passed,
            "why": ("two different goals can imply the same values, so the gate opens for both. "
                    "That is 'I disagree with what you were doing but we want the same things', "
                    "which the code could not previously represent."),
            "cells": cells[cells.values_layer].to_dict(orient="records"),
        },
        "channel_crossover_from_diagnostics": 0.538,
        "cells": cells.to_dict(orient="records"),
        "n_obs": int(n_obs),
    }
    (v6_dir() / "e41_coupling.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                encoding="utf-8")
    return verdict
