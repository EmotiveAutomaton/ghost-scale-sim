"""C-1 to C-4 — closing the four results version 6 deliberately did not draw.

Version 6 produced a visual walkthrough and held four findings out of it, because a picture makes a
claim hard to qualify and each of those four carried an open question. Holding them was right.
Leaving them held is not, so each is settled here or explicitly retired.

    C-1  the two-gates result, scored on the quantity the theory names rather than the one the
         construction holds constant
    C-2  the depth-versus-effort null, same treatment, and the last unresolved solver disagreement
    C-3  the crash and the invention peak: establish the co-location under exact inference, or
         retire the claim from the README and the prediction card
    C-4  depletion, at a length the mechanism needs and on a criterion that can actually work
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from ..config import Config
from ..prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval, spearman
from ..v5_model import MU_LEVELS, make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v7_dir

# E31's own design: content type x label x whether the reader's gate is open.
E31_CONTENT = (("human", 1.0), ("foreign", 0.0))
E31_LABELS = (("sig_creator", K.SIG_CREATOR), ("sig_ghost", K.SIG_GHOST),
              ("unsigned", K.UNSIGNED))
E31_RHO_BAR = 0.70          # the bar the criterion was originally written against

N21_CELLS = (("committed_shallow", 1.00, 1), ("committed_deep", 1.00, 3),
             ("offhand_shallow", 0.25, 1), ("offhand_deep", 0.25, 3))
N21_DOMINANCE = 3.0


def _cells(world, cfg_r, n_mu, n_sub, ng, spec, n_obs, n_timesteps, base_seed):
    """Run a list of (name, beta, mu, label) cells and score goal AND process uptake."""
    kappa = float(world.cfg.signal_model.kappa)
    rows = []
    for name, beta, mu, label, sig in spec:
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + base_seed + i)
            g = int(rng.integers(ng))
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, g, int(mu), float(beta), n_timesteps, rng,
                provenance=K.CREATOR, declared_signal=sig,
                signing_rate=0.0 if sig == K.UNSIGNED else 1.0)
            enc = H.run_encounter(world, cfg_r, artifact, env,
                                  make_v5_observer(world, rng), creator, rng,
                                  n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)
            rows.append({
                "cell": name, "label": label, "beta": float(beta), "true_mu": int(mu),
                "observer": i,
                "recovered_mu": float(enc.recovered_mu),
                "goal_uptake": float(enc.error_reduction),
                "process_uptake": float(enc.process["process_error_reduction"]),
                "goal_correct": int(enc.correct),
            })
    return rows


# =========================================================================== #
# C-1 — the two gates, on the quantity the theory names.
# =========================================================================== #
def c1_two_gates(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    """E31's own design, with process uptake scored as a primary.

    THE HISTORY, BECAUSE IT IS THE POINT. E31's criterion is that how much a reader takes on tracks
    its own estimate of how much thinking went in -- one mechanism, whichever channel produced the
    estimate. It scored 0.886 under the approximate solver and 0.600 under exact arithmetic, twice,
    by two independent routes, and the diagnostics pass then found it was computed over six cells
    while 2,400 per-reader pairs sat unused. It has been the project's longest-running open
    question.

    The retrofit suggested the whole thing was a measurement error: the criterion was scored on
    GOAL uptake, and depth is constructed so the goal is equally recoverable at every depth. But
    that came from a reconstruction of E31's cells rather than from E31, which is why version 6
    would not draw it. This is E31's own design.
    """
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    spec = []
    for cname, beta in E31_CONTENT:
        for lname, sig in E31_LABELS:
            for mu in MU_LEVELS:
                spec.append((f"{cname}/{lname}/mu{mu}", beta, mu, lname, sig))
    rows = _cells(world, cfg_r, n_mu, n_sub, ng, spec, n_obs, n_timesteps, 51_000)
    df = pd.DataFrame(rows)
    cells = df.groupby("cell").agg(
        recovered_mu=("recovered_mu", "mean"),
        goal_uptake=("goal_uptake", "mean"),
        process_uptake=("process_uptake", "mean"),
        goal_accuracy=("goal_correct", "mean")).reset_index()

    rho_goal = spearman(cells.recovered_mu, cells.goal_uptake)
    rho_process = spearman(cells.recovered_mu, cells.process_uptake)

    rng = np.random.default_rng(20260803)
    idx = np.arange(len(cells))
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        take = rng.choice(idx, idx.size, replace=True)
        draws.append(spearman(cells.recovered_mu.to_numpy()[take],
                              cells.process_uptake.to_numpy()[take]))
    lo, hi = percentile_interval(draws)

    holds = bool(np.isfinite(rho_process) and rho_process >= E31_RHO_BAR)
    return {
        "closure": "C-1",
        "experiment": "E31",
        "question": ("Does what a reader takes on track its own estimate of how much thinking went "
                     "into the work, whichever channel produced that estimate?"),
        "scored_on_the_method": {"rho": rho_process, "interval": [lo, hi],
                                 "bar": E31_RHO_BAR, "holds": holds},
        "scored_on_the_purpose": {"rho": rho_goal,
                                  "note": ("the measure the criterion originally used, retained. "
                                           "Depth is constructed so the purpose is equally "
                                           "recoverable at every depth, so this quantity cannot "
                                           "track depth whatever is true of the reader.")},
        "history": {"approximate_solver": 0.886, "exact_solver": 0.600,
                    "retrofit_reconstruction": 0.833},
        "cells": cells.to_dict(orient="records"),
        "n_cells": int(len(cells)), "n_obs": int(n_obs),
        "outcome": ("ONE_MECHANISM_CONFIRMED_ON_THE_METHOD" if holds
                    else "STILL_DOES_NOT_TRACK"),
    }


# =========================================================================== #
# C-2 — depth is not effort, on the same measure.
# =========================================================================== #
def c2_depth_not_effort(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    """N21's four cells, scored on the method as well as on the recovered depth.

    N21 reverses under exact inference -- the null returns "effort can manufacture depth" -- and it
    has never been scored on the method measure. Same fix as C-1 and the last unresolved solver
    disagreement in the project.
    """
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    spec = [(name, beta, mu, "unsigned", K.UNSIGNED) for name, beta, mu in N21_CELLS]
    rows = _cells(world, cfg_r, n_mu, n_sub, ng, spec, n_obs, n_timesteps, 52_000)
    df = pd.DataFrame(rows)
    cells = df.groupby("cell").agg(
        recovered_mu=("recovered_mu", "mean"),
        process_uptake=("process_uptake", "mean"),
        goal_uptake=("goal_uptake", "mean")).reset_index()
    c = {r["cell"]: r for r in cells.to_dict(orient="records")}

    def eff(a, b, col):
        return float(c[a][col] - c[b][col])

    # The depth effect at each effort level, and effort's FALSE-DEPTH effect at shallow depth.
    mu_effect_committed = eff("committed_deep", "committed_shallow", "recovered_mu")
    mu_effect_offhand = eff("offhand_deep", "offhand_shallow", "recovered_mu")
    beta_false_depth = abs(eff("committed_shallow", "offhand_shallow", "recovered_mu"))
    mu_effect = 0.5 * (mu_effect_committed + mu_effect_offhand)
    ratio = mu_effect / beta_false_depth if beta_false_depth > 1e-9 else float("inf")

    # And the same contrast on the method measure, which is what C-1 established is the right one.
    mu_effect_process = 0.5 * (eff("committed_deep", "committed_shallow", "process_uptake")
                               + eff("offhand_deep", "offhand_shallow", "process_uptake"))
    beta_false_process = abs(eff("committed_shallow", "offhand_shallow", "process_uptake"))
    ratio_process = (mu_effect_process / beta_false_process
                     if beta_false_process > 1e-9 else float("inf"))

    holds = bool(mu_effect_committed > 0 and mu_effect_offhand > 0 and ratio >= N21_DOMINANCE)
    return {
        "closure": "C-2",
        "experiment": "N21",
        "question": "Is depth just effort wearing a hat?",
        "on_recovered_depth": {
            "mu_effect_at_full_effort": mu_effect_committed,
            "mu_effect_at_low_effort": mu_effect_offhand,
            "effort_false_depth_effect": beta_false_depth,
            "dominance_ratio": ratio, "bar": N21_DOMINANCE, "holds": holds,
        },
        "on_the_method": {
            "mu_effect": mu_effect_process,
            "effort_false_effect": beta_false_process,
            "dominance_ratio": ratio_process,
            "note": ("the measure C-1 established is the right one for depth. Reported beside the "
                     "pre-registered quantity, deciding nothing."),
        },
        "cells": cells.to_dict(orient="records"),
        "outcome": ("DEPTH_IS_NOT_EFFORT" if holds else "EFFORT_CAN_MANUFACTURE_DEPTH"),
    }


# =========================================================================== #
# C-3 — do the crash and the invention peak sit in the same band?
# =========================================================================== #
def c3_crash_and_peak(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    """The claim the README and the prediction card both assert, under exact inference.

    Version 4.5 reported that the collapse and the invention peak occupy one narrow band, and the
    framework had always treated them as two separate phenomena, so the coincidence was striking
    and it got quoted. Under the repaired model the peak is exactly where it was and the crash
    signature fires NOWHERE -- because the reader at partial overlap ends up LESS uncertain and
    drops below a threshold in a conjunctive criterion.

    Either it is re-established here or it comes out of both documents. A striking coincidence that
    only holds under a superseded solver is not a finding.
    """
    from ..experiments import e20_omega_sweep as E20  # noqa: F401  (documents the source design)

    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    grid = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0)

    rows = []
    for omega in grid:
        # EVERY OBSERVER IN A CELL READS THE SAME ARTIFACT, and the first version of this did not.
        # With a fresh artifact per observer, "between-observer disagreement" measures the sampler
        # rather than the readers: they disagree because they are looking at different things, and
        # the column stays near its ceiling at every overlap including full overlap, which is what
        # it did. E19 found this the hard way and the codebase warns about it in as many words.
        cell_rng = np.random.default_rng(SEED_OFFSET + 53_000 + int(omega * 1000))
        g = int(cell_rng.integers(ng))

        posts, ents, engs = [], [], []
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 54_000 + i)
            artifact, env = H.make_foreign_artifact_and_env(
                world, cfg_r, g, n_timesteps, rng, omega=float(omega))
            enc = H.run_encounter(world, cfg_r, artifact, env,
                                  make_v5_observer(world, rng), None, rng,
                                  n_timesteps, 10, n_sub, n_mu, ng, kappa, true_goal=g)
            posts.append(enc.goal_posterior)
            ents.append(float(enc.final_entropy))
            engs.append(float(enc.engaged_fraction))
        within = float(np.mean(ents))
        between = float(metrics.between_observer_entropy(posts))
        ceiling = float(np.log(ng))
        rows.append({
            "omega": float(omega),
            "within_observer": within,
            "between_observer": between,
            "engaged_fraction": float(np.mean(engs)),
            "fabrication_index": float((1.0 - within / ceiling) * (between / ceiling)),
            # The crash signature as version 4 defined it: unresolved AND no longer looking.
            "unresolved": bool(within > 0.5),
            "disengaged": bool(np.mean(engs) < 0.5),
            "crashed": bool(within > 0.5 and np.mean(engs) < 0.5),
        })

    df = pd.DataFrame(rows)
    peak_row = df.loc[df.fabrication_index.idxmax()]
    crashed = df[df.crashed]
    co_located = bool(len(crashed) and
                      float(crashed.omega.min()) <= float(peak_row.omega) <= float(crashed.omega.max()))

    return {
        "closure": "C-3",
        "experiment": "E20",
        "question": "Do the collapse and the invention peak occupy the same narrow band?",
        "peak_omega": float(peak_row.omega),
        "peak_fabrication": float(peak_row.fabrication_index),
        "omegas_where_the_crash_signature_fires": [float(x) for x in crashed.omega],
        "co_located": co_located,
        "rows": df.to_dict(orient="records"),
        "outcome": ("CO_LOCATION_HOLDS" if co_located else "CO_LOCATION_RETIRED"),
        "consequence": (
            "the claim stays in the README and the prediction card" if co_located else
            "the claim comes OUT of the README and the prediction card. The peak is unchanged and "
            "is not in question; what does not survive is the coincidence, which held under the "
            "approximate solver because the reader was left more uncertain at partial overlap "
            "than exact arithmetic leaves it."),
    }


# =========================================================================== #
# C-4 — depletion, longer, and on a criterion that can work.
# =========================================================================== #
def c4_depletion(cfg: Config, n_readers: int, n_encounters: int, n_timesteps: int) -> dict:
    """E35 at a length the mechanism needs, scored on the relative drop.

    E35's pre-registered clause is an ABSOLUTE drop of 0.10 in engagement with a fixed human probe.
    Across three seed blocks the mechanism reproduced every time and that clause passed once,
    because baseline probe engagement itself differs about two-fold between blocks and an absolute
    threshold cannot be stable on a quantity whose baseline moves that much.

    So: longer, and scored on the relative drop with the original clause retained and reported.
    """
    from ..v6 import e35_depletion as E35

    out = E35.run(cfg, n_readers=n_readers, n_encounters=n_encounters,
                  n_timesteps=n_timesteps, forced_k=6)
    rel = float(out.get("relative_drop", float("nan")))
    rho = float(out.get("probe_monotone_rho", float("nan")))
    holds = bool(np.isfinite(rel) and rel >= 0.40 and np.isfinite(rho) and rho <= -0.70)
    return {
        "closure": "C-4",
        "experiment": "E35",
        "question": ("Does a reader worn down by content that gives it nothing disengage from work "
                     "it has never seen?"),
        "relative_drop": rel,
        "absolute_drop": float(out.get("probe_drop", float("nan"))),
        "monotone_rho": rho,
        "fold_reduction": float(out.get("fold_reduction", float("nan"))),
        "n_encounters": int(n_encounters),
        "pre_registered_clause": {
            "statement": "an ABSOLUTE drop of at least 0.10",
            "value": float(out.get("probe_drop", float("nan"))),
            "verdict_under_it": out.get("outcome"),
            "why_it_is_retained_and_not_used": (
                "an absolute bar on a quantity whose baseline varies two-fold between seed blocks "
                "cannot be stable, and it was not: the mechanism reproduced on three blocks and "
                "the clause passed on one. Retained and reported, as every superseded criterion in "
                "this project is."),
        },
        "outcome": ("DEPLETION_CARRIES_TO_UNSEEN_WORK" if holds else "DEPLETION_DOES_NOT_CARRY"),
    }


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, n_readers: int = 24,
        n_encounters: int = 30, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    payload = {
        "check": "V7 closures",
        "question": "The four results version 6 would not draw. Settle them or retire them.",
        "C-1": c1_two_gates(cfg, n_obs, n_timesteps),
        "C-2": c2_depth_not_effort(cfg, n_obs, n_timesteps),
        "C-3": c3_crash_and_peak(cfg, n_obs, n_timesteps),
        "C-4": c4_depletion(cfg, n_readers, n_encounters, n_timesteps),
    }
    (v7_dir() / "closures.json").write_text(json.dumps(payload, indent=2, default=str),
                                            encoding="utf-8")
    return payload
