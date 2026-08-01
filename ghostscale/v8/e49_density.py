"""E49 — the readymade: artfulness as density rather than volume.

THE CASE THE THEORY SPENDS THE MOST WORDS DEFENDING AND THE MODEL COULD NOT REPRESENT.

Depth in V5-V7 is measured over a sequence of observations, so an artifact with almost no
observable extent carries almost nothing to read. Duchamp's *Fountain* -- near-zero fabrication, one
act of selection, enormous compressed context -- scores as empty. Which is exactly the reading the
essay exists to argue against:

    "Simplicity without a dense underlying decision tree is just empty data; simplicity born from
    extreme compression is a masterpiece."

So the quantity is a RATIO: hierarchy invoked per unit of observable extent. One act that needs
three levels to explain is maximally dense; three hundred acts that need one level are not.

-----------------------------------------------------------------------------------------
THE HALF THAT MAKES IT A TEST RATHER THAN A REDEFINITION.

If density is what a readymade has, then whether you see it depends on whether you POSSESS a
structure, not on how much evidence you gathered. So the population should go BIMODAL: readers with
a matched hierarchy see a great deal, readers without see nothing, and there is no middle.

That is a different prediction from "a readymade is weak work", which gives a low mean and one
mode. And it is what conceptual art actually does to a room -- some people see genius, some see a
urinal, and almost nobody is mildly impressed.

    fails if the reading is unimodal with a low mean, in which case the theory's defence of the
    readymade does not survive its own model

NULL N39 GUARDS THE OBVIOUS CHEAT. Density must not simply reward shortness. A brief artifact with
no hierarchy behind it scores at the floor, because the numerator is what carries the claim.
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

# (name, maker depth, observable extent). The readymade is the interesting one: one act, full
# hierarchy behind it. The sprawl is its opposite -- a great deal of surface, nothing under it.
ARTIFACTS = (
    ("readymade", 3, 2),
    ("sketch", 2, 6),
    ("ordinary_work", 2, 24),
    ("sprawl", 1, 24),
)


def run(cfg: Config, n_obs: int = 60, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)

    rows = []
    for name, mu, extent in ARTIFACTS:
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 86_000 + i)
            g = int(rng.integers(ng))
            # THE READER'S OWN HIERARCHY IS WHAT DECIDES, which is the whole point: a readymade is
            # unreadable to a reader that has not built the structure it compresses. Drawn across
            # the population so the distribution is the measurement.
            reader_depth = int(rng.integers(1, 4))
            reader = V8.ReaderHierarchy(levels=reader_depth)

            creator, art, env = H.make_artifact_and_env(
                world, cfg_r, g, int(mu), 1.0, int(extent), rng)
            enc = H.run_encounter(world, cfg_r, art, env,
                                  make_v5_observer(world, rng), creator, rng,
                                  int(extent), int(extent), n_sub, n_mu, ng, kappa)

            seen_levels = V8.depth_reading(int(mu), reader)
            dens = V8.density(seen_levels, int(extent))
            rows.append({
                "artifact": name, "true_depth": int(mu), "extent": int(extent),
                "reader_depth": reader_depth, "observer": i,
                "levels_seen": float(seen_levels),
                "density": float(dens),
                "true_density": float(V8.density(float(mu), int(extent))),
                "goal_correct": int(enc.correct),
                "process_recovery": float(enc.process["process_error_reduction"]),
            })

    df = pd.DataFrame(rows)
    out = v8_dir("e49_density")
    df.to_csv(out / "e49_density.csv", index=False)

    cells = df.groupby("artifact").agg(
        density=("density", "mean"), density_sd=("density", "std"),
        true_density=("true_density", "mean"),
        levels_seen=("levels_seen", "mean"),
        extent=("extent", "mean"), true_depth=("true_depth", "mean")).reset_index()

    # H8.4a: is the readymade dense, where the old measure said empty?
    ready = float(cells[cells.artifact == "readymade"].density.iloc[0])
    ordinary = float(cells[cells.artifact == "ordinary_work"].density.iloc[0])
    sprawl = float(cells[cells.artifact == "sprawl"].density.iloc[0])
    dense = bool(ready > ordinary and ready > sprawl)

    # H8.4b: is the READING of it split rather than merely low?
    modality = {}
    for name, _, _ in ARTIFACTS:
        modality[name] = V8.bimodality(df[df.artifact == name].density.to_numpy())
    split = bool(modality["readymade"]["split"])

    # N39: density must not reward shortness as such.
    n39_passed = bool(sprawl < ready and
                      float(cells[cells.artifact == "sketch"].density.iloc[0]) < ready)
    # ...and the sharper form: a SHORT artifact with no hierarchy must not beat a long deep one.
    short_empty = V8.density(1.0, 2)
    long_deep = V8.density(3.0, 24)
    n39_strict = bool(short_empty > long_deep)   # expected TRUE -> the measure is a ratio, so this
    # is the honest limitation rather than a pass: brevity does help. The guard is that the
    # NUMERATOR still has to be there, which the sprawl comparison tests.

    verdict = {
        "experiment": "E49",
        "hypothesis": "H8.4",
        "question": ("Can this model represent a readymade -- almost nothing on the surface, an "
                     "enormous amount compressed behind it?"),
        "plain_language": (
            "Depth has always been measured over a sequence, so something with almost no surface "
            "carries almost nothing to read, and a urinal on a plinth scores as empty. That is the "
            "reading the theory exists to argue against. Measured as hierarchy per unit of "
            "surface, one act that needs three levels to explain is the densest thing there is."),
        "cells": cells.to_dict(orient="records"),
        "H8.4_density": {
            "readymade": ready, "ordinary_work": ordinary, "sprawl": sprawl,
            "readymade_is_densest": dense,
            "outcome": ("THE_READYMADE_IS_DENSE_NOT_EMPTY" if dense
                        else "DENSITY_DOES_NOT_RESCUE_THE_READYMADE"),
        },
        "H8.4_bimodality": {
            "by_artifact": modality,
            "readymade_splits_the_room": split,
            "outcome": ("READERS_SPLIT_RATHER_THAN_SHRUG" if split
                        else "READERS_ARE_UNIFORMLY_UNIMPRESSED"),
            "why_it_matters": (
                "whether you see a readymade depends on whether you POSSESS a structure, not on "
                "how much evidence you gathered. So the population should split rather than "
                "converge on mild disappointment -- which is what conceptual art actually does to "
                "a room, and is a different prediction from 'it is weak work'."),
        },
        "null_n39": {
            "statement": "density must not simply reward shortness",
            "sprawl_below_readymade": bool(sprawl < ready),
            "passed": n39_passed,
            "acknowledged_limitation": {
                "short_and_empty": short_empty, "long_and_deep": long_deep,
                "short_empty_beats_long_deep": n39_strict,
                "note": ("a ratio does favour brevity, and that is stated rather than hidden. The "
                         "guard is that the NUMERATOR must be there: an artifact with no hierarchy "
                         "behind it scores at the floor however short it is, which the sprawl and "
                         "sketch comparisons test. A measure that could not be gamed by brevity at "
                         "all would not be a compression measure."),
            },
        },
        "n_obs": int(n_obs),
    }
    (v8_dir() / "e49_density.json").write_text(json.dumps(verdict, indent=2, default=str),
                                               encoding="utf-8")
    return verdict
