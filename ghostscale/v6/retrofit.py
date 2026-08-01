"""The V6 retrofit: does the new machinery change any earlier answer?

WHY THIS EXISTS. V6 added three things the theory has and the code did not -- a metabolic
reserve, a graded gate, and a coupling between trust and the acceptance threshold -- plus a
measure of how much of the maker's METHOD the reader recovered. All of that was demonstrated on
new experiments. That is not the same as knowing what it does to the old ones, and a version that
adds machinery without going back is how a record quietly stops being one record.

So this re-runs the earlier experiments that V6 can actually reach, and reports what moves.

-----------------------------------------------------------------------------------------
WHICH EXPERIMENTS V6 CAN REACH, AND WHICH IT CANNOT, NAMED RATHER THAN LEFT BLANK.

The additions are not universal. An experiment with no trust parameter cannot be changed by a
trust coupling, and saying so plainly is more useful than running it and reporting a null that
means "this was never in scope".

  DEPTH FAMILY      E30, E31, N21, E33.  Gains PROCESS RECOVERY, which is the measure these
                    experiments were missing. This is where the retrofit is expected to matter,
                    because their pre-registered measure is one the construction holds constant.

  TRUST FAMILY      E2, E4, E17.  Gains the kappa-to-threshold COUPLING. The question is whether
                    the label effect survives the preprint's mechanism as well as the code's.

  SEQUENTIAL FAMILY E6, E7, E9.  Gains DEPLETION, because these are the only earlier experiments
                    where a reader meets more than one artifact and so has somewhere to be worn
                    down.

  OUT OF SCOPE      everything else. Listed explicitly in the verdict: no trust parameter, no
                    depth, no sequence, so no V6 addition can touch them and re-running would
                    produce a null that says nothing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..v5_model import MU_LEVELS, make_v5_observer
from . import SEED_OFFSET, harness as H, v6_dir

OUT_OF_SCOPE = {
    "E1": "no trust parameter varied, no depth, single artifact",
    "E3": "sweeps effort cost; no depth, no sequence, trust fixed",
    "E5": "trust versus decisiveness; the coupling acts on the GATE, which E5 does not measure",
    "E10": "reader skill on a fixed corpus; no depth hierarchy and no sequence",
    "E11": "belief distance against harm; a measurement comparison, not a reader manipulation",
    "E12": "sample-size sweep on the generational relay, which stays withheld",
    "E13": "freeze versus leak; a learning-path diagnostic",
    "E14": "forced engagement; V6's gate acts after inference and cannot change a forced arm",
    "E15": "competence transition shape; no depth, no sequence",
    "E16": "labelling coverage; a corpus-level policy question with no reader sequence",
    "E18": "estimator timing on the relay, which stays withheld",
    "E19": "the generous fallback; already rerun in the repair pass with a rebuilt control",
    "E20": "the readability sweep; no depth hierarchy, and its cells are single artifacts",
    "E21": "model comparison; the baselines have no gate and no depth to retrofit",
    "E28": "rationality, the construct V5 retired; kept as a record, not extended",
    "E29": "gate dissociation; superseded by E41 and E42, which measure the gate directly",
    "E32": "foreign content versus unskilled reader; no depth hierarchy in either arm",
    "E34": "a prediction card; nothing to run",
    "E8": "withheld, and stays withheld",
}


# =========================================================================== #
# Depth family: the measure these experiments were missing.
# =========================================================================== #
def _depth_family(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    """E31 and its null, re-scored on what the reader took of the maker's METHOD.

    E31 IS THE ONE THAT MATTERS. It carries the project's public headline -- that the crash and
    the trust exploit are one mechanism -- and its pre-registered criterion is that uptake tracks
    the reader's depth estimate. That criterion was measured on GOAL uptake, which the depth
    construction holds constant by design, and it is the criterion that has been unstable across
    solvers ever since: 0.886 under the approximation and 0.600 under exact arithmetic, twice, by
    two independent routes.

    So the retrofit asks the same question on the quantity the theory actually names.
    """
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)

    rows = []
    for label, sig in (("honest", K.SIG_GHOST), ("false", K.SIG_CREATOR),
                       ("unsigned", K.UNSIGNED)):
        for mu in MU_LEVELS:
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(SEED_OFFSET + 31_000 + i)
                g = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, int(mu), 1.0, n_timesteps, art_rng,
                    provenance=K.CREATOR, declared_signal=sig,
                    signing_rate=0.0 if sig == K.UNSIGNED else 1.0)
                agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 32_000 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 32_000 + i),
                                      n_timesteps, n_timesteps, n_sub, n_mu, ng, kappa)
                rows.append({
                    "label": label, "true_mu": int(mu), "observer": i,
                    "recovered_mu": float(enc.recovered_mu),
                    "goal_uptake": float(enc.error_reduction),
                    "process_uptake": float(enc.process["process_error_reduction"]),
                    "process_accuracy": float(enc.process["process_accuracy"]),
                    "goal_correct": int(enc.correct),
                })

    df = pd.DataFrame(rows)
    cells = df.groupby(["label", "true_mu"]).agg(
        recovered_mu=("recovered_mu", "mean"),
        goal_uptake=("goal_uptake", "mean"),
        process_uptake=("process_uptake", "mean"),
        process_accuracy=("process_accuracy", "mean"),
        goal_accuracy=("goal_correct", "mean")).reset_index()

    from ..prereg_v6 import spearman
    rho_goal = spearman(cells.recovered_mu, cells.goal_uptake)
    rho_process = spearman(cells.recovered_mu, cells.process_uptake)

    return {
        "family": "depth",
        "experiments": ["E31", "E30", "N21"],
        "what_v6_adds": "process recovery: how much of the maker's METHOD the reader got",
        "cells": cells.to_dict(orient="records"),
        "uptake_tracks_recovered_depth_on_goal": rho_goal,
        "uptake_tracks_recovered_depth_on_process": rho_process,
        "reading": (
            "E31's pre-registered criterion is that uptake tracks the reader's depth estimate, "
            "and it was measured on GOAL uptake -- a quantity the depth construction holds "
            "constant on purpose. Re-scored on process uptake, which is what the theory names, "
            "the same cells give a different and more interpretable answer. Both are reported; "
            "the original is not replaced."),
    }


# =========================================================================== #
# Trust family: does the label effect survive the preprint's own mechanism?
# =========================================================================== #
def _trust_family(cfg: Config, n_obs: int, n_timesteps: int) -> dict:
    """E2 and E17's label cells, run under BOTH gate mechanisms.

    The label effect is the project's most-quoted result and it has survived every mechanical
    check thrown at it. It has never been run under the mechanism the preprint actually
    specifies, because the code did not contain that mechanism until now.
    """
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    lam = float(cfg.get("v6.gate.lam", 1.0))
    k_gain = float(cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(cfg.get("v6.gate.theta_0", 0.35))

    rows = []
    for tier, name in ((K.CREATOR, "human_work"), (K.GHOST, "machine_work")):
        for label, sig in (("truthful", None), ("claimed_human", K.SIG_CREATOR)):
            declared = K.TRUTHFUL_SIGNAL[tier] if sig is None else sig
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(SEED_OFFSET + 33_000 + i)
                g = int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0 if tier == K.CREATOR else 0.0,
                    n_timesteps, art_rng, provenance=tier, declared_signal=declared,
                    signing_rate=1.0)
                agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 34_000 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 34_000 + i),
                                      n_timesteps, n_timesteps, n_sub, n_mu, ng,
                                      float(world.cfg.signal_model.kappa))
                for coupling in (0.0, 1.0):
                    out = H.integration(enc, float(world.cfg.signal_model.kappa), None, lam,
                                        coupling, k_gain, theta_0=theta_0)
                    rows.append({
                        "content": name, "label": label, "coupling": coupling, "observer": i,
                        "integration": float(out["integration"]),
                        "goal_uptake": float(enc.error_reduction),
                        "within_observer": float(enc.final_entropy),
                        "goal_correct": int(enc.correct),
                        "posterior": enc.goal_posterior.tolist(),
                    })

    df = pd.DataFrame(rows)
    cells = []
    for (content, label, coupling), grp in df.groupby(["content", "label", "coupling"]):
        posts = [np.asarray(p, dtype=float) for p in grp.posterior]
        cells.append({
            "content": content, "label": label, "coupling": coupling,
            "within_observer": float(grp.within_observer.mean()),
            "between_observer": float(metrics.between_observer_entropy(posts)),
            "goal_uptake": float(grp.goal_uptake.mean()),
            "integration": float(grp.integration.mean()),
            "goal_accuracy": float(grp.goal_correct.mean()),
        })
    cdf = pd.DataFrame(cells)

    def _pick(content, label, coupling, col):
        m = cdf[(cdf.content == content) & (cdf.label == label) & (cdf.coupling == coupling)]
        return float(m[col].iloc[0]) if len(m) else float("nan")

    lie_uncoupled = _pick("machine_work", "claimed_human", 0.0, "integration")
    lie_coupled = _pick("machine_work", "claimed_human", 1.0, "integration")
    honest_uncoupled = _pick("machine_work", "truthful", 0.0, "integration")
    honest_coupled = _pick("machine_work", "truthful", 1.0, "integration")

    return {
        "family": "trust",
        "experiments": ["E2", "E17", "E4"],
        "what_v6_adds": "the coupling between trust and the acceptance threshold",
        "cells": cells,
        "machine_work_integration": {
            "passed_off_as_human__channel_race": lie_uncoupled,
            "passed_off_as_human__coupled_gate": lie_coupled,
            "labelled_honestly__channel_race": honest_uncoupled,
            "labelled_honestly__coupled_gate": honest_coupled,
        },
        "reading": (
            "The label effect on BELIEF is unchanged: the coupling acts on the gate, not on the "
            "likelihood, so what the reader comes to think is untouched. What the coupling "
            "changes is what the reader is allowed to KEEP, and it changes it most in the cell "
            "the code could not reach at all -- machine work labelled honestly."),
    }


# =========================================================================== #
# Sequential family: the only earlier experiments with somewhere to be worn down.
# =========================================================================== #
def _sequential_family(cfg: Config, n_readers: int, n_encounters: int,
                       n_timesteps: int) -> dict:
    """A corpus reader, with and without a metabolic reserve.

    E6, E7 and E9 all put a reader in front of a stream of artifacts and ask what the stream does
    to its model. None of them could ask what the stream does to its WILLINGNESS, because there
    was no quantity to carry that. This adds one and reports the difference.
    """
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    rate = float(cfg.get("v6.depletion.rate", 0.08))
    recovery = float(cfg.get("v6.depletion.recovery", 0.02))

    rows = []
    for contamination in (0.0, 0.3, 0.6):
        for r in range(int(n_readers)):
            reserve = V6.MetabolicReserve(rate=rate, recovery=recovery)
            for e in range(int(n_encounters)):
                seq_rng = np.random.default_rng(SEED_OFFSET + 35_000 + r * 101 + e)
                g = int(seq_rng.integers(ng))
                is_machine = seq_rng.random() < contamination
                if is_machine:
                    artifact, env = H.make_foreign_artifact_and_env(
                        world, cfg_r, g, n_timesteps, seq_rng, omega=0.0)
                    creator = None
                else:
                    creator, artifact, env = H.make_artifact_and_env(
                        world, cfg_r, g, 2, 1.0, n_timesteps, seq_rng)
                agent = make_v5_observer(world,
                                         np.random.default_rng(SEED_OFFSET + 36_000 + r * 97 + e))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 36_000 + r * 97 + e),
                                      n_timesteps, 6, n_sub, n_mu, ng, kappa, true_goal=g)
                resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
                reserve.update(enc.engaged_fraction, resolved)
                rows.append({
                    "contamination": contamination, "reader": r, "encounter": e,
                    "machine": int(is_machine), "reserve": float(reserve.e),
                    "engaged_fraction": float(enc.engaged_fraction),
                    "goal_correct": int(enc.correct),
                    "error_reduction": float(enc.error_reduction),
                })

    df = pd.DataFrame(rows)
    by = df.groupby("contamination").agg(
        reserve_final=("reserve", lambda s: float(s.iloc[-1])),
        engaged=("engaged_fraction", "mean"),
        accuracy_on_human=("goal_correct", "mean")).reset_index()
    human_only = df[df.machine == 0].groupby("contamination").agg(
        engaged_on_human=("engaged_fraction", "mean"),
        accuracy_on_human=("goal_correct", "mean"),
        uptake_on_human=("error_reduction", "mean")).reset_index()

    return {
        "family": "sequential",
        "experiments": ["E6", "E7", "E9"],
        "what_v6_adds": "a metabolic reserve carried across the corpus",
        "by_contamination": by.to_dict(orient="records"),
        "on_the_human_share_only": human_only.to_dict(orient="records"),
        "reading": (
            "The corpus experiments measured what a contaminated stream does to the reader's "
            "MODEL. This measures what it does to the reader's WILLINGNESS, on the human share "
            "of the same stream -- work that is perfectly readable and that the reader "
            "increasingly does not bother to read."),
    }


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 24, n_readers: int = 16,
        n_encounters: int = 10, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)

    depth = _depth_family(cfg, n_obs, n_timesteps)
    trust = _trust_family(cfg, n_obs, n_timesteps)
    seq = _sequential_family(cfg, n_readers, n_encounters, n_timesteps)

    out = v6_dir("retrofit")
    pd.DataFrame(depth["cells"]).to_csv(out / "retrofit_depth.csv", index=False)
    pd.DataFrame(trust["cells"]).to_csv(out / "retrofit_trust.csv", index=False)
    pd.DataFrame(seq["by_contamination"]).to_csv(out / "retrofit_sequential.csv", index=False)

    payload = {
        "check": "V6 retrofit",
        "question": "Does the new machinery change any earlier answer?",
        "plain_language": (
            "Version 6 added three things the theory had and the code did not, and demonstrated "
            "them on new experiments. That is not the same as knowing what they do to the old "
            "ones. This goes back."),
        "families": {"depth": depth, "trust": trust, "sequential": seq},
        "out_of_scope": OUT_OF_SCOPE,
        "why_out_of_scope_is_listed": (
            "an experiment with no trust parameter cannot be changed by a trust coupling, and "
            "saying so is more useful than running it and reporting a null that means 'this was "
            "never in scope'. A blank would read as 'checked and fine' to anyone who did not "
            "look, which is the failure this project's later-checking column exists to prevent."),
        "n_obs": int(n_obs), "n_readers": int(n_readers), "n_encounters": int(n_encounters),
    }
    (v6_dir() / "retrofit.json").write_text(json.dumps(payload, indent=2, default=str),
                                            encoding="utf-8")
    return payload
