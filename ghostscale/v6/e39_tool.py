"""E39 — the tool hypothesis: what the Ghost Scale is actually supposed to do.

THE GAP. In V1-V5 the reader has a machine-made hypothesis only in the sense that it can FAIL to
settle on any human goal. There has never been a hypothesis that SAYS there is no maker here, so
the reader could not conclude one. It could only keep failing, which is why the honest-label
condition still leaves it inside the read-a-maker frame.

That is not what the affordance is for. Its job is not to make you distrust a label; it is to let
your brain RELAX. The preprint says so directly: the signal lets the viewer "save energy
calculating theta_EC by quickly feeding the user the answer of a low-value kappa", and climb
"into higher-order reasoning, such as guessing a prompt's motivation". A reader that has concluded
there is no maker should stop CLEANLY -- resolved, disengaged, and not fabricating.

So this adds one hypothesis value: NO_MAKER, with a likelihood matched to intent-empty content.
Structurally it is the move V4 made with EXPLORE, pointed at a different target, and it inherits
EXPLORE's failure mode along with its null.

NULL N27 IS NOT OPTIONAL AND IS THE SAME CHECK V4 RAN. A hypothesis that sits close to every
human goal absorbs human work and trivially destroys every result in the project. V4 caught
exactly that: at V1-V3 cardinality ``sig_EXPLORE`` was flat over the whole feature space and sat
four times closer to synthetic content than to any goal the observer held, so E19 could only ever
have returned one answer. The check is run at construction, before any cell.

THE THREE SIGNATURES BEING TOLD APART. This is the whole design and it is why a bare
before/after comparison would not do:

    the crash          unresolved AND still looking      (foreign content, no tool hypothesis)
    the exploit        resolved, confident, and wrong    (a false label)
    the tool reading   resolved, stopped, not inventing  (the prediction)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h67_verdict
from ..v4_model import build_v4_synth
from ..v5_model import make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

ARMS = ("no_tool_hypothesis", "with_tool_hypothesis", "human_control")


def run(cfg: Config, n_obs: int = 60, n_timesteps: int = 24, forced_k: int = 10,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    base, bcfg_b, bcfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(base.cfg.signal_model.kappa)
    sig_true = np.asarray(base.gm.sig_true, dtype=float)

    no_maker = V6.build_no_maker_signature(build_v4_synth(bcfg_b))
    n27 = V6.assert_no_maker_does_not_absorb(no_maker, sig_true)

    # The reader WITH the tool hypothesis holds one extra value on the goal factor.
    sig_plus = np.vstack([sig_true, no_maker[None, :]])
    tw, tcfg_b, tcfg_r, t_mu, t_sub, t_ng = H.build_alt_world(cfg, sig_plus)

    recs = []
    for arm in ARMS:
        for i in range(int(n_obs)):
            art_rng = np.random.default_rng(SEED_OFFSET + 70_000 + i)
            g = int(art_rng.integers(ng))
            if arm == "human_control":
                # The control that decides whether NO_MAKER has eaten everything: genuine human
                # work, read by the reader that holds the extra hypothesis.
                creator, artifact, env = H.make_artifact_and_env(
                    tw, tcfg_r, g, 2, 1.0, n_timesteps, art_rng, provenance=K.CREATOR)
                world, cfg_use, use_ng = tw, tcfg_r, t_ng
                agent = H.make_alt_observer(tw, np.random.default_rng(SEED_OFFSET + 90_000 + i), t_ng)
            elif arm == "with_tool_hypothesis":
                creator, artifact, env = H.make_artifact_and_env(
                    tw, tcfg_r, g, 2, 0.0, n_timesteps, art_rng, provenance=K.GHOST)
                world, cfg_use, use_ng = tw, tcfg_r, t_ng
                agent = H.make_alt_observer(tw, np.random.default_rng(SEED_OFFSET + 90_000 + i), t_ng)
            else:
                creator, artifact, env = H.make_artifact_and_env(
                    base, bcfg_r, g, 2, 0.0, n_timesteps, art_rng, provenance=K.GHOST)
                world, cfg_use, use_ng = base, bcfg_r, ng
                agent = make_v5_observer(base, np.random.default_rng(SEED_OFFSET + 90_000 + i))

            enc = H.run_encounter(world, cfg_use, artifact, env, agent, creator,
                                  np.random.default_rng(SEED_OFFSET + 90_000 + i),
                                  n_timesteps, forced_k, t_sub, t_mu, use_ng, kappa)
            # The fabrication index is confidence multiplied by disagreement, and it is the
            # project's own measure of INVENTION as distinct from honest confusion. A tool
            # reading must be quiet on it: concluding there is no maker is not the same as
            # confidently inventing one.
            recs.append({
                "arm": arm, "observer": i,
                "goal_correct": int(enc.correct),
                "final_entropy": float(enc.final_entropy),
                "engaged_fraction": float(enc.engaged_fraction),
                "error_reduction": float(enc.error_reduction),
                "posterior": enc.goal_posterior.tolist(),
            })

    df = pd.DataFrame(recs)
    out = v6_dir("e39_tool")
    df.to_csv(out / "e39_tool.csv", index=False)

    cells = {}
    for arm in ARMS:
        a = df[df.arm == arm]
        posts = [np.asarray(p, dtype=float) for p in a.posterior]
        within = float(np.mean([metrics.within_observer_entropy(p) for p in posts]))
        between = float(metrics.between_observer_entropy(posts))
        ceiling = float(np.log(len(posts[0])))
        cells[arm] = {
            "goal_accuracy": float(a.goal_correct.mean()),
            "final_entropy": float(a.final_entropy.mean()),
            "engaged_fraction": float(a.engaged_fraction.mean()),
            "error_reduction": float(a.error_reduction.mean()),
            "within_observer": within,
            "between_observer": between,
            "pairwise_divergence": float(metrics.mean_pairwise_js(posts)),
            # confidence x disagreement, the project's invention measure
            "fabrication_index": float((1.0 - within / ceiling) * (between / ceiling)),
        }

    h67 = h67_verdict(cells["with_tool_hypothesis"], cells["no_tool_hypothesis"])

    # N27 behaviourally, not only at construction: the reader holding the extra hypothesis must
    # still read genuine human work correctly.
    n27_behavioural = float(cells["human_control"]["goal_accuracy"])
    n27_passed = bool(n27_behavioural >= 0.80)

    verdict = {
        "experiment": "E39",
        "hypothesis": "H6.7",
        "question": ("Does a reader that can conclude 'there is no maker here' stop cleanly, "
                     "rather than either crashing or inventing?"),
        "plain_language": (
            "Until now the reader could only FAIL to find a maker. It had no way to decide that "
            "there was not one. That is what the Ghost Scale is meant to give it -- not distrust "
            "of a label, but permission to stop. This gives the reader that hypothesis and asks "
            "whether stopping looks different from failing."),
        "cells": cells,
        "H6.7": h67,
        "null_n27": {
            "statement": "NO_MAKER must not absorb human work (the failure V4 caught with EXPLORE)",
            "construction": n27,
            "behavioural_human_accuracy": n27_behavioural,
            "passed": bool(n27_passed),
        },
        "three_signatures": {
            "crash": "unresolved and still looking",
            "exploit": "resolved, confident and wrong",
            "tool_reading": "resolved, stopped, not inventing",
        },
        "n_obs": int(n_obs),
    }
    if not n27_passed:
        verdict["INTERPRETABILITY"] = (
            "NULL N27 FAILED. The tool hypothesis absorbs human work, which is the EXPLORE "
            "failure V4 caught, and every number above is uninterpretable.")
    (v6_dir() / "e39_tool.json").write_text(json.dumps(verdict, indent=2, default=str),
                                            encoding="utf-8")
    return verdict
