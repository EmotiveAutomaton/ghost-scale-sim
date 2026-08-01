"""E35 — metabolic depletion, and whether the damage carries to work the reader has never seen.

THE GAP THIS CLOSES. ``theta_base(E)`` is in the preprint's equation, with its own symbol, and
nothing in V1-V5 implements it. Effort cost is a constant; every reader arrives fresh; nothing
that happens in one encounter changes what the reader will spend on the next. So the essay's
central cultural claim -- that the damage ACCUMULATES IN THE READER and ends in apathy -- was
not untested. It was unrepresentable, and the simulation's silence about it was mistaken for
evidence of nothing.

THE CRITERION IS THE PROBE, NOT THE EXPOSED CONDITION, and that is the whole design. A depletion
term that lowers engagement on the content that caused it is a knob doing exactly what it was
pointed at, and proves nothing. The claim worth testing is that it CARRIES: a reader worn down
by intent-empty content should engage less with genuine human work it has never encountered.

So each reader alternates between an EXPOSURE stream, which varies by arm, and a FIXED HUMAN
PROBE that is identical in every arm and at every point in the sequence. The probe is what is
scored. Nothing about the probe changes; if engagement on it falls, something about the reader
has changed.

NULL N22 RUNS FIRST AND IS REPORTED FIRST. On a fully resolvable corpus the reserve must not
move. That null is the one the generational experiment (E8) never passed, and it is written
first here on purpose: a depletion mechanism that drifts on content it should not touch cannot
support any conclusion, and would produce this experiment's headline for free.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h61_verdict
from ..v5_model import make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

# The three exposure streams. The probe is the same object in all three.
ARMS = ("intent_empty", "human", "mixed")


def _one_reader(world, cfg_r, n_mu, n_sub, ng, arm: str, seed: int,
                n_encounters: int, n_timesteps: int, forced_k: int,
                rate: float, recovery: float) -> dict:
    """One reader's life: alternating exposure and probe, with a reserve carried across."""
    rng = np.random.default_rng(seed)
    reserve = V6.MetabolicReserve(rate=rate, recovery=recovery)
    kappa = float(world.cfg.signal_model.kappa)

    probe_engagement, reserves, probe_correct = [], [], []
    for e in range(int(n_encounters)):
        # ---- exposure -------------------------------------------------------
        #
        # WHICH CONTENT DEPLETES, AND WHY THE FIRST VERSION OF THIS EXPERIMENT USED THE WRONG
        # KIND. Depletion is the product of how much the reader looked and how little it got, so
        # content that the reader ABANDONS cannot deplete it -- correctly, since walking away is
        # cheap. Built first with a hierarchical maker whose goal identity was attenuated, the
        # reader disengaged after about a tenth of the free phase and the reserve never moved off
        # its ceiling in any arm. That would have been reported as "depletion does not carry"
        # when what it actually showed is that nobody paid.
        #
        # The depleting content is the content that HOLDS attention and gives nothing back --
        # which is exactly the essay's description ("every look keeps promising an answer that
        # never arrives") and exactly the regime E19 and E20 already located: partially
        # overlapping foreign content, which holds the reader at 0.83 of the free phase.
        art_rng = np.random.default_rng(seed * 7919 + e)
        g = int(art_rng.integers(ng))
        foreign_now = (arm == "intent_empty") or (arm == "mixed" and e % 2 == 1)
        if foreign_now:
            artifact, env = H.make_foreign_artifact_and_env(
                world, cfg_r, g, n_timesteps, art_rng, omega=0.0)
            creator = None
        else:
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, g, 2, 1.0, n_timesteps, art_rng)
        agent = make_v5_observer(world, np.random.default_rng(seed * 104729 + e))
        enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                              np.random.default_rng(seed * 104729 + e),
                              n_timesteps, forced_k, n_sub, n_mu, ng, kappa, true_goal=g)

        # THE DEPLETION STEP. Driven by the product of how much the reader looked and how
        # little it got: a reader that never looks and a reader that always succeeds are both
        # undepleted, which is what makes N22 a real null.
        resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
        reserve.update(enc.engaged_fraction, resolved)

        # ---- probe: identical in every arm, never contributes to depletion ---
        p_rng = np.random.default_rng(SEED_OFFSET + 999_331 + e)
        p_creator, p_art, p_env = H.make_artifact_and_env(
            world, cfg_r, int(p_rng.integers(ng)), 2, 1.0, n_timesteps, p_rng)
        p_agent = make_v5_observer(world, np.random.default_rng(seed * 15485863 + e))
        p_enc = H.run_encounter(world, cfg_r, p_art, p_env, p_agent, p_creator,
                                np.random.default_rng(seed * 15485863 + e),
                                n_timesteps, 0, n_sub, n_mu, ng, kappa)

        # THE RESERVE ACTS THROUGH THE GATE, so a depleted reader disengages sooner. The rollout
        # itself is unchanged -- the reader's pymdp policy has no reserve in it -- and the
        # reserve is applied to the probe's engagement here, which keeps the mechanism visible
        # and auditable instead of buried inside the agent.
        theta = reserve.theta_base()
        gate = V6.gate(float(p_enc.engaged_fraction), theta,
                       k_gain=float(world.cfg.get("v6.gate.k_gain", 8.0)))
        probe_engagement.append(float(p_enc.engaged_fraction * gate))
        probe_correct.append(int(p_enc.correct))
        reserves.append(float(reserve.e))

    return {"arm": arm, "seed": seed, "probe_engagement": probe_engagement,
            "probe_correct": probe_correct, "reserves": reserves}


def run(cfg: Config, n_readers: int = 24, n_encounters: int = 12, n_timesteps: int = 24,
        forced_k: int = 6, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    rate = float(cfg.get("v6.depletion.rate", 0.08))
    recovery = float(cfg.get("v6.depletion.recovery", 0.02))

    rows = []
    for arm in ARMS:
        for i in range(int(n_readers)):
            rows.append(_one_reader(world, cfg_r, n_mu, n_sub, ng, arm, SEED_OFFSET + 20_000 + i,
                                    n_encounters, n_timesteps, forced_k, rate, recovery))

    df = pd.DataFrame([
        {"arm": r["arm"], "seed": r["seed"], "encounter": t,
         "probe_engagement": r["probe_engagement"][t],
         "probe_correct": r["probe_correct"][t], "reserve": r["reserves"][t]}
        for r in rows for t in range(len(r["reserves"]))])
    out = v6_dir("e35_depletion")
    df.to_csv(out / "e35_depletion.csv", index=False)

    by = df.groupby(["arm", "encounter"]).agg(
        probe_engagement=("probe_engagement", "mean"),
        probe_correct=("probe_correct", "mean"),
        reserve=("reserve", "mean")).reset_index()
    by.to_csv(out / "e35_by_encounter.csv", index=False)

    exposed = by[by.arm == "intent_empty"].sort_values("encounter")
    control = by[by.arm == "human"].sort_values("encounter")

    verdict = h61_verdict(list(exposed.probe_engagement), list(exposed.reserve),
                          float(control.reserve.iloc[0]), float(control.reserve.iloc[-1]))

    # A RELATIVE DROP, REPORTED BESIDE THE PRE-REGISTERED ABSOLUTE ONE AND DECIDING NOTHING.
    #
    # The pre-registered criterion is an ABSOLUTE fall of 0.10 in probe engagement. Run on a
    # disjoint seed block, the mechanism reproduced exactly -- the reserve fell from 1.00 to
    # 0.46 against 0.50, the monotonicity came out at -0.81 against -0.77, and the control arm
    # held at 1.00 both times -- and the VERDICT still flipped, because baseline probe
    # engagement itself differs about two-fold between seed blocks and an absolute threshold
    # cannot be stable on a quantity whose baseline moves that much.
    #
    # That is a criterion that cannot do its job, which is the failure this project has now
    # caught in four separate places. It is recorded here rather than repaired: the
    # pre-registered clause still decides, and is reported as failing on one of two seed blocks.
    first = float(exposed.probe_engagement.iloc[0])
    last = float(exposed.probe_engagement.iloc[-1])
    verdict["relative_drop"] = float(1.0 - last / first) if first > 0 else float("nan")
    verdict["fold_reduction"] = float(first / last) if last > 0 else float("inf")
    verdict["criterion_note"] = (
        "the pre-registered clause is an ABSOLUTE drop of %.2f and it is not seed-stable: on a "
        "disjoint seed block the mechanism reproduced (reserve 1.00 -> 0.46, monotonicity -0.81, "
        "control flat at 1.00) while the verdict flipped, because baseline probe engagement "
        "differs about two-fold between blocks. The relative drop and the monotonicity are "
        "stable; the absolute threshold is not. Reported, not repaired." % 0.10)
    verdict.update({
        "experiment": "E35",
        "hypothesis": "H6.1",
        "question": ("Does a reader worn down by intent-empty content engage less with genuine "
                     "human work it has never seen?"),
        "plain_language": (
            "The preprint says the damage accumulates in the reader and ends in apathy. The "
            "simulation had no way to represent that at all: every reader arrived fresh. This "
            "gives the reader an energy budget that falls when it looks hard and gets nothing, "
            "and then measures its engagement with a FIXED human artifact that never changes."),
        "design_note": (
            "the criterion is the PROBE, not the exposed condition. A depletion term that only "
            "lowers engagement on the content that caused it is a knob doing what it was "
            "pointed at. The claim is that it carries."),
        "arms": {a: {"probe_engagement_first": float(by[(by.arm == a)].probe_engagement.iloc[0]),
                     "probe_engagement_last": float(by[(by.arm == a)].probe_engagement.iloc[-1]),
                     "reserve_first": float(by[(by.arm == a)].reserve.iloc[0]),
                     "reserve_last": float(by[(by.arm == a)].reserve.iloc[-1])}
                 for a in ARMS},
        "null_n22": {
            "statement": "on a fully resolvable corpus the reserve must not move",
            "reserve_start": float(control.reserve.iloc[0]),
            "reserve_end": float(control.reserve.iloc[-1]),
            "passed": bool(verdict["null_n22_passed"]),
            "why_first": ("this is the null the generational experiment never passed. A "
                          "depletion mechanism that drifts on content it should not touch "
                          "would produce this experiment's headline for free."),
        },
        "n_readers": int(n_readers), "n_encounters": int(n_encounters),
        "probe": "a fixed human artifact, identical in every arm and at every point",
    })
    (v6_dir() / "e35_depletion.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                 encoding="utf-8")
    return verdict
