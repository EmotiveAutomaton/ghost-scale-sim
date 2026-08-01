"""E47 — does the coverage threshold survive the other trust mechanism?

THE QUESTION E41 OPENED AND COULD NOT ANSWER.

E16 produced the project's one policy number: about a third of machine content has to be labelled
before a reader keeps a clean picture of what people are like, and three-quarters if the reader does
not know the convention exists. That number is quoted.

E41 then found that the published theory and the code explain the trust exploit by different
mechanisms. The code's version says a trusted label MISINFORMS you: the fix is honesty, and honest
labels are fully protective. The paper's version says trust LOWERS THE GUARD: an honest label still
tells the truth, but a reader that trusts the source integrates what it is given anyway.

**If the second mechanism is the right one, honest labels stop being fully protective, and the
coverage a disclosure regime needs should go up.** E16's number was produced by a model that did not
contain that mechanism.

    predicted:  the threshold rises with the coupling
    fails if:   it does not move, in which case the coverage figure is robust to which mechanism is
                correct -- which would be good news, and worth knowing either way

THIS IS A CONDITIONAL RESULT AND IS REPORTED AS ONE. Nothing here establishes that the coupled
mechanism is the true one; E41 established only that the two are distinguishable. What this says is
what the policy number would be IF the paper's mechanism is right, which is the question a
policymaker would actually ask before relying on a threshold.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v7_dir

COVERAGE_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COUPLINGS = (0.0, 1.0)
CLEAN_PICTURE = 0.25          # error in the reader's carried picture, below which it is "clean"


def _corpus_pass(world, cfg_r, n_mu, n_sub, ng, coverage: float, coupling: float,
                 n_readers: int, n_artifacts: int, n_timesteps: int, base_seed: int) -> float:
    """How wrong a reader's picture of human intent ends up, after a mixed corpus.

    The reader carries a belief about what makes work human-made. Machine content that gets past
    the gate corrupts it; content the gate refuses does not. Coverage decides how much of the
    machine content is labelled, and the coupling decides whether being told the truth is enough
    to keep it out.
    """
    lam = float(world.cfg.get("v6.gate.lam", 1.0))
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))
    kappa = float(world.cfg.signal_model.kappa)
    values_map = V6.build_values_map(ng, n_values=2)

    errors = []
    for r in range(int(n_readers)):
        carried = np.full(ng, 1.0 / ng)
        truth = np.full(ng, 1.0 / ng)
        for a in range(int(n_artifacts)):
            rng = np.random.default_rng(SEED_OFFSET + base_seed + r * 733 + a)
            g = int(rng.integers(ng))
            is_machine = rng.random() < 0.5
            labelled = is_machine and (rng.random() < coverage)

            if is_machine:
                artifact, env = H.make_foreign_artifact_and_env(
                    world, cfg_r, g, n_timesteps, rng, omega=0.10,
                    declared_signal=K.SIG_GHOST if labelled else K.UNSIGNED)
                enc = H.run_encounter(world, cfg_r, artifact, env,
                                      make_v5_observer(world, rng), None, rng,
                                      n_timesteps, 6, n_sub, n_mu, ng, kappa, true_goal=g)
            else:
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, artifact, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 6, n_sub, n_mu, ng, kappa)

            # A labelled machine artifact SHOULD be refused. Whether it is depends on which
            # mechanism is running: the gate's threshold is suppressed by trust under the coupling.
            recovered = enc.goal_posterior
            if is_machine and labelled:
                implied = V6.implied_values(recovered, values_map)
                vp = np.clip(implied[::-1], 1e-6, None)
                vp = vp / vp.sum()
                divergence = V6.value_divergence_via_values(recovered, vp, values_map)
            else:
                divergence = 0.0

            resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
            theta = V6.disgust_threshold(divergence, kappa, None, lam, theta_0, coupling)
            g_open = V6.gate(float(np.clip(resolved, 0.0, 1.0)), theta, k_gain)

            w = float(g_open) * float(np.mean(np.asarray(enc.attention) == K.DEEP))
            carried = (1.0 - w) * carried + w * np.asarray(recovered, dtype=float)
            carried = carried / carried.sum()
            if not is_machine:
                truth = (1.0 - w) * truth + w * np.asarray(recovered, dtype=float)
                truth = truth / truth.sum()
        errors.append(float(metrics.kl_divergence(carried, truth)))
    return float(np.mean(errors))


def _threshold(cov, err, bar):
    for c, e in zip(cov, err):
        if e <= bar:
            return float(c)
    return None


def run(cfg: Config, n_readers: int = 10, n_artifacts: int = 14, n_timesteps: int = 24,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)

    rows = []
    for coupling in COUPLINGS:
        for cov in COVERAGE_GRID:
            e = _corpus_pass(world, cfg_r, n_mu, n_sub, ng, cov, coupling,
                             n_readers, n_artifacts, n_timesteps, 55_000)
            rows.append({"coupling": float(coupling), "coverage": float(cov),
                         "picture_error": e})

    df = pd.DataFrame(rows)
    out = v7_dir("e47_coverage")
    df.to_csv(out / "e47_coverage.csv", index=False)

    thresholds = {}
    for coupling in COUPLINGS:
        s = df[df.coupling == coupling].sort_values("coverage")
        thresholds[float(coupling)] = _threshold(s.coverage, s.picture_error, CLEAN_PICTURE)

    uncoupled, coupled = thresholds[0.0], thresholds[1.0]
    moved = (uncoupled is not None and coupled is not None and coupled > uncoupled) or \
            (uncoupled is not None and coupled is None)

    verdict = {
        "experiment": "E47",
        "hypothesis": "H7.4",
        "question": ("If trust lowers the guard rather than merely misinforming, does a disclosure "
                     "regime need more coverage than the current figure?"),
        "plain_language": (
            "The project's one policy number says about a third of machine content has to be "
            "labelled. That number was produced by a model in which telling the truth is fully "
            "protective. If the published theory is right that trust lowers the guard, then an "
            "honest label no longer keeps the material out, and the number should move."),
        "threshold_under_the_channel_race": uncoupled,
        "threshold_under_the_coupled_gate": coupled,
        "threshold_moved": bool(moved),
        "clean_picture_bar": CLEAN_PICTURE,
        "curve": df.to_dict(orient="records"),
        "outcome": ("COVERAGE_MUST_RISE_UNDER_THE_COUPLED_MECHANISM" if moved
                    else "THE_COVERAGE_FIGURE_IS_ROBUST_TO_THE_MECHANISM"),
        "this_is_conditional_and_is_reported_as_such": (
            "nothing here establishes that the coupled mechanism is the true one. E41 established "
            "only that the two are distinguishable. This says what the policy number would be IF "
            "the paper's mechanism is right, which is the question that has to be asked before "
            "relying on a threshold."),
        "relationship_to_e16": (
            "E16's figure is not overturned. It is the answer under one of two mechanisms, and "
            "until the mechanism is settled the honest statement is a range rather than a number."),
        "n_readers": int(n_readers), "n_artifacts": int(n_artifacts),
    }
    (v7_dir() / "e47_coverage.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                encoding="utf-8")
    return verdict
