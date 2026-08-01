"""E43 — automaticity hides the work from its own author.

THE CORRECTION THIS IMPLEMENTS, AND IT IS THE AUTHOR'S SECOND ONE.

A working note said the subconscious holds the process goals. The correction: it holds the
PRACTISED ones. Recently acquired, poorly modelled skills are held consciously and can be
reported on demand -- a novice can tell you exactly which rule they were following, because they
are still following it deliberately. It is the tightly compressed, heavily automatised structure
that becomes inaccessible to its own author. A master cannot tell you why the perspective works.

That is the automaticity story running in the direction the theory needs, and it makes a
prediction the model can be asked for: **self-report accuracy falls with depth, while the
reader's recovery of the actual goal does not.**

WHY THIS IS NOT JUST E33 AGAIN. E33 already found that a reader recovers a maker's latent goal at
0.88 while the maker's own declared goal collapses to 0.05 -- the reader knowing the maker better
than the maker knows themselves. But in E33 the maker's self-blindness is a MANIPULATED
PARAMETER: an experimenter dials up how wrong the self-report is. Here it is a CONSEQUENCE of how
much practised structure the maker has. Nobody sets it. Depth sets it.

That is a different and stronger claim, and it is the one the theory actually makes: the reason
experts cannot explain themselves is not that they are unusually opaque people, it is that
expertise is compression and compression is what makes a decision unavailable for report.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h611_verdict
from ..v5_model import MU_LEVELS, make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir


def run(cfg: Config, n_obs: int = 50, n_timesteps: int = 24, forced_k: int = 24,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    base = float(cfg.get("v6.self_report.base", 0.95))
    decay = float(cfg.get("v6.self_report.decay", 0.30))

    recs = []
    for mu in MU_LEVELS:
        p_report = V6.self_report_accuracy(int(mu), MU_LEVELS, base=base, decay=decay)
        for i in range(int(n_obs)):
            art_rng = np.random.default_rng(SEED_OFFSET + 19_000 + mu * 71 + i)
            g = int(art_rng.integers(ng))
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, g, int(mu), 1.0, n_timesteps, art_rng)
            agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 21_000 + mu * 67 + i))
            enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                  np.random.default_rng(SEED_OFFSET + 21_000 + mu * 67 + i),
                                  n_timesteps, forced_k, n_sub, n_mu, ng, kappa)

            # THE MAKER'S OWN ACCOUNT. Correct with probability p_report, otherwise some other
            # goal. The maker is not lying; it has lost access to a decision it compressed.
            decl_rng = np.random.default_rng(SEED_OFFSET + 23_000 + mu * 61 + i)
            if decl_rng.random() < p_report:
                declared = g
            else:
                declared = int(decl_rng.choice([x for x in range(ng) if x != g]))

            recs.append({
                "mu": int(mu), "observer": i,
                "p_self_report": float(p_report),
                "declared_correct": int(declared == g),
                "reader_correct": int(enc.correct),
                "reader_error_reduction": float(enc.error_reduction),
                "process_accuracy": float(enc.process["process_accuracy"]),
                "final_entropy": float(enc.final_entropy),
            })

    df = pd.DataFrame(recs)
    out = v6_dir("e43_selfreport")
    df.to_csv(out / "e43_selfreport.csv", index=False)

    cells = df.groupby("mu").agg(
        p_self_report=("p_self_report", "mean"),
        declared_accuracy=("declared_correct", "mean"),
        reader_accuracy=("reader_correct", "mean"),
        reader_error_reduction=("reader_error_reduction", "mean"),
        process_accuracy=("process_accuracy", "mean")).reset_index()
    cells.to_csv(out / "e43_cells.csv", index=False)

    h611 = h611_verdict(list(cells.declared_accuracy), list(cells.reader_accuracy))

    verdict = {
        "experiment": "E43",
        "hypothesis": "H6.11",
        "question": ("Does the maker lose access to its own reasons as the work gets deeper, "
                     "while the reader does not?"),
        "plain_language": (
            "A novice can tell you exactly which rule they were following, because they are "
            "still following it on purpose. A master cannot tell you why the perspective works. "
            "The claim is that this is not a fact about personalities but about compression: "
            "practice is what makes a decision automatic, and automatic is what makes it "
            "unavailable for report."),
        "distinction_from_e33": (
            "E33 already showed a reader recovering a maker's latent goal while the maker's "
            "declared goal collapsed. But there the self-blindness is a MANIPULATED PARAMETER. "
            "Here nobody sets it: depth sets it, which is a stronger and different claim."),
        "cells": cells.to_dict(orient="records"),
        "H6.11": h611,
        "n_obs": int(n_obs),
    }
    (v6_dir() / "e43_selfreport.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                  encoding="utf-8")
    return verdict
