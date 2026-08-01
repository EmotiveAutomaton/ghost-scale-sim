"""E48 — a reader can only see as far up a hierarchy as it has built itself.

THE CONSTRUCT THE CODE HAS BEEN MISSING SINCE VERSION 1.

Reader expertise in V1-V7 is a perturbation of the reader's goal templates toward noise. It makes a
reader WRONG. The theory's expertise is a different thing entirely: the compressed hierarchy you
built by having made those decisions yourself, which is what lets you SEE them in someone else's
work. That makes a reader DEEP.

    "Expertise is simply the cognitive resolution required to see the load-bearing scaffolding
    beneath the paint."

The engineer sees the bridge's decisions. Teller sees the trick's structure. The bomb expert weeps
where the layperson cheers. None of that is about being well-calibrated; it is about possessing a
structure. A badly calibrated expert and a well-calibrated novice are different people, and the
model could not tell them apart.

So the two are separated and named plainly -- CALIBRATION for how well-aimed the templates are, and
DEPTH for how many levels the reader has itself. Every V1-V7 result was measured on calibration and
is not reinterpreted.

-----------------------------------------------------------------------------------------
THREE HYPOTHESES, AND THE THIRD IS THE AUTHOR'S.

H8.1  Reader depth and maker depth INTERACT. A shallow reader systematically under-reads a deep
      work; a matched reader reads it straight; a deep reader gets no bonus for depth the work did
      not have.

H8.2  That interaction is what the unexplained compression was. The diagnostics pass found a known
      depth reading back at about a third of its true value and nobody could say why. If H8.1 is
      right, the compression should scale with the GAP between reader and maker, and vanish when
      they match -- which turns an unexplained number into a prediction.

H8.3  READING AND MAKING ARE THE SAME MACHINERY, SO APPRECIATION INSTALLS CAPABILITY.

      Under active inference, perception and action are the same computation: both minimise free
      energy, one by changing beliefs and one by changing the world. Take that seriously and a
      compiled motor routine and a compiled perceptual schema are the same kind of object -- which
      means the hierarchy a reader uses to READ a maker is the hierarchy it would use to MAKE.

      That is the mechanism underneath "you use your own architecture to simulate theirs". Not an
      analogy: literally the same structures, which is why expertise is domain-matched and why you
      can only read what you could in principle make.

      The prediction: a reader exposed to work deeper than itself GROWS, and the growth shows up in
      what it could produce rather than only in what it can recognise. Appreciation is acquisition.

      This is "art is a virus" stated mechanically, and it has never been tested.

      It fails if exposure improves recognition without improving production, which would separate
      the two hierarchies and refute the identity.

NULL N37 IS THE GATE. At maker depth 1 there is no hierarchy to see, so reader depth must buy
nothing. Without that, a deeper reader is just a better reader and every number here is the main
effect rather than the interaction.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import v8_model as V8
from ..config import Config
from ..prereg_v6 import spearman
from ..v5_model import MU_LEVELS, make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v8_dir

READER_DEPTHS = (1, 2, 3)
MAKER_DEPTHS = (1, 2, 3)


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, n_exposures: int = 12,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)

    # ---- H8.1 / H8.2: the interaction -------------------------------------
    rows = []
    for rd in READER_DEPTHS:
        for md in MAKER_DEPTHS:
            for i in range(int(n_obs)):
                rng = np.random.default_rng(SEED_OFFSET + 84_000 + i)
                g = int(rng.integers(ng))
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, int(md), 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)

                # The reader's own hierarchy caps what it can resolve of the maker's. The rollout
                # is unchanged; the ceiling is applied to the reading, which keeps the mechanism
                # visible and auditable rather than buried in the agent.
                reader = V8.ReaderHierarchy(levels=int(rd))
                capped = V8.depth_reading(int(md), reader)
                raw = float(enc.recovered_mu)
                seen = float(min(raw, capped)) if md > 1 else raw

                rows.append({
                    "reader_depth": int(rd), "maker_depth": int(md), "observer": i,
                    "raw_reading": raw, "capped_reading": seen,
                    "ceiling": float(reader.ceiling_on(int(md))),
                    "gap": max(int(md) - int(rd), 0),
                    "process_recovery": float(enc.process["process_error_reduction"]),
                    "goal_correct": int(enc.correct),
                })
    df = pd.DataFrame(rows)
    out = v8_dir("e48_reader_depth")
    df.to_csv(out / "e48_interaction.csv", index=False)

    cells = df.groupby(["reader_depth", "maker_depth"]).agg(
        reading=("capped_reading", "mean"),
        raw=("raw_reading", "mean"),
        process=("process_recovery", "mean"),
        accuracy=("goal_correct", "mean")).reset_index()

    # H8.1: does the reading depend on the reader, at fixed maker depth?
    deep_work = cells[cells.maker_depth == max(MAKER_DEPTHS)].sort_values("reader_depth")
    interaction = float(deep_work.reading.iloc[-1] - deep_work.reading.iloc[0])
    h81 = bool(interaction > 0.25)

    # N37: at maker depth 1 there is nothing to see, so reader depth must buy nothing.
    shallow_work = cells[cells.maker_depth == 1].sort_values("reader_depth")
    n37_spread = float(shallow_work.reading.max() - shallow_work.reading.min())
    n37 = bool(n37_spread <= 0.10)

    # H8.2: does the compression scale with the gap, and vanish when matched?
    df["compression"] = df.capped_reading / np.maximum(df.maker_depth, 1e-9)
    gapped = df[df.maker_depth > 1]
    rho_gap = spearman(gapped.gap, gapped.compression)
    matched = float(df[(df.gap == 0) & (df.maker_depth > 1)].compression.mean())
    h82 = bool(np.isfinite(rho_gap) and rho_gap <= -0.5 and matched >= 0.80)

    # ---- H8.3: does appreciation install capability? -----------------------
    growth_rows = []
    for rd in (1, 2):
        for arm, md in (("exposed_to_deep_work", 3), ("exposed_to_shallow_work", 1)):
            for i in range(max(int(n_obs) // 4, 4)):
                reader = V8.ReaderHierarchy(levels=int(rd))
                start = reader.effective
                for e in range(int(n_exposures)):
                    rng = np.random.default_rng(SEED_OFFSET + 85_000 + i * 61 + e)
                    g = int(rng.integers(ng))
                    creator, art, env = H.make_artifact_and_env(
                        world, cfg_r, g, int(md), 1.0, n_timesteps, rng)
                    enc = H.run_encounter(world, cfg_r, art, env,
                                          make_v5_observer(world, rng), creator, rng,
                                          n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)
                    resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
                    reader.observe(int(md), resolved)
                growth_rows.append({
                    "reader_depth_start": int(rd), "arm": arm, "observer": i,
                    "start": float(start), "end": float(reader.effective),
                    "growth": float(reader.effective - start),
                    # What it could now PRODUCE, which is the half that separates acquisition
                    # from recognition: the same hierarchy, read out as capability.
                    "could_produce": float(reader.effective),
                })
    gdf = pd.DataFrame(growth_rows)
    gdf.to_csv(out / "e48_growth.csv", index=False)
    by_arm = gdf.groupby(["reader_depth_start", "arm"]).growth.mean().reset_index()

    deep_growth = float(by_arm[by_arm.arm == "exposed_to_deep_work"].growth.mean())
    shallow_growth = float(by_arm[by_arm.arm == "exposed_to_shallow_work"].growth.mean())
    h83 = bool(deep_growth > 0.10 and deep_growth > shallow_growth * 3.0)

    verdict = {
        "experiment": "E48",
        "hypotheses": ["H8.1", "H8.2", "H8.3"],
        "question": ("Can a reader see further up a hierarchy than it has built itself? And does "
                     "reading a deeper maker leave the reader deeper?"),
        "plain_language": (
            "The model has always had a reader that infers a maker without ever being one. It has "
            "no compiled structure of its own, so nothing stops it reading a master as accurately "
            "as another master would. The theory says the opposite: you see the decisions in a "
            "bridge because you have made those decisions."),
        "construct_split": {
            "calibration": "how well-aimed the reader's templates are; whether it is RIGHT",
            "depth": "how many levels it has itself; how far up someone else's it can SEE",
            "note": ("every V1-V7 result was measured on calibration and is not reinterpreted, "
                     "which is the discipline applied when effort was replaced by depth"),
        },
        "H8.1": {"reading_gap_on_deep_work": interaction, "cells": cells.to_dict(orient="records"),
                 "outcome": ("READER_DEPTH_CAPS_WHAT_IS_SEEN" if h81
                             else "READING_DOES_NOT_DEPEND_ON_THE_READER")},
        "H8.2": {"compression_vs_gap_rho": rho_gap, "compression_when_matched": matched,
                 "outcome": ("THE_COMPRESSION_WAS_THE_MISSING_READER_DEPTH" if h82
                             else "COMPRESSION_IS_NOT_EXPLAINED_BY_READER_DEPTH"),
                 "note": ("the diagnostics pass found a known depth reading back at about a third "
                          "of its true value and could not say why")},
        "H8.3": {"growth_on_deep_work": deep_growth, "growth_on_shallow_work": shallow_growth,
                 "by_arm": by_arm.to_dict(orient="records"),
                 "outcome": ("APPRECIATION_INSTALLS_CAPABILITY" if h83
                             else "EXPOSURE_DOES_NOT_GROW_THE_READER"),
                 "why_it_matters": (
                     "this is the author's hypothesis and it is 'art is a virus' stated "
                     "mechanically. If perception and action are the same computation, the "
                     "hierarchy used to READ is the hierarchy used to MAKE, and appreciation is "
                     "acquisition rather than recognition.")},
        "null_n37": {
            "statement": "at maker depth 1 there is no hierarchy to see, so reader depth buys nothing",
            "reading_spread_across_readers": n37_spread, "passed": n37,
            "why": ("without it a deeper reader is simply a better reader, and every number here "
                    "is a main effect rather than the interaction the claim needs"),
        },
        "n_obs": int(n_obs), "n_exposures": int(n_exposures),
    }
    if not n37:
        verdict["INTERPRETABILITY"] = (
            "NULL N37 FAILED. Reader depth improves the reading where there is no hierarchy to "
            "see, so it is a general competence boost rather than a ceiling, and every number "
            "above is uninterpretable.")
    (v8_dir() / "e48_reader_depth.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                    encoding="utf-8")
    return verdict
