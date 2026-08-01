"""The shared V6 rollout harness.

One place where a reader meets an artifact, so that nine experiments cannot drift apart on what
"engagement" or "recovery" means. V5 learned this the hard way in the other direction: E19's
between-observer column silently stopped meaning what its name said once each observer got a
fresh content draw, and that was only found because someone re-read the sampler.

WHAT THIS ADDS OVER ``observer.rollout_observer``, which it calls rather than replaces:

  * the PER-STEP SUB-GOAL POSTERIOR, marginalised out of factor 1, which is what process
    recovery is scored against and which no previous version ever read;
  * the alignment between the reader's timestep and the creator's position in its own plan,
    which is off by one and is handled explicitly here rather than in nine call sites;
  * the metabolic reserve, carried ACROSS encounters, which is the thing V1-V5 has no place
    to put.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as K
from .. import foreign as FN
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..environment import Artifact
from ..n21_depth_not_effort import merged_config
from ..observer import rollout_observer
from ..v5_model import (MU_LEVELS, HierarchicalCreator, V5Environment, as_mgs, build_v5_world,
                        make_v5_observer, marginal_goal, marginal_subgoal, recovered_mu)

_EPS = 1e-12


@dataclass
class Encounter:
    """Everything one reader-meets-one-artifact produces, scored the same way every time."""

    goal_posterior: np.ndarray
    goal_prior: np.ndarray
    goal_posteriors_by_step: list
    subgoal_posteriors: list
    true_modes: list
    attention: np.ndarray
    engaged_fraction: float
    correct: int
    final_entropy: float
    recovered_mu: float
    error_reduction: float
    movement: float
    process: dict
    true_goal: int


def creator_positions(creator: HierarchicalCreator, n_timesteps: int,
                      initial_glance: bool = True) -> list:
    """The creator's true execution mode at each of the reader's timesteps.

    THE OFF-BY-ONE IS REAL AND IS HANDLED HERE ONCE. The free initial glance consumes position
    zero of the creator's plan (``V5Environment.sample_feature`` calls ``creator.advance()`` on
    a SKIM), so the reader's timestep ``t`` sees the creator's position ``t + 1``. Getting this
    wrong would depress process recovery by exactly one step's worth of chain mixing and would
    look like a small, plausible, entirely uninformative result — which is the failure mode this
    project has been bitten by seven times.
    """
    traj = np.asarray(creator.trajectory, dtype=int)
    off = 1 if initial_glance else 0
    return [int(traj[min(t + off, traj.size - 1)]) for t in range(int(n_timesteps))]


def run_encounter(world, cfg_rollout: Config, artifact: Artifact, env, agent,
                  creator: HierarchicalCreator | None, rng, n_timesteps: int,
                  force_deep_k: int, n_sub: int, n_mu: int, ng: int,
                  kappa: float, initial_glance: bool = True,
                  true_goal: int | None = None) -> Encounter:
    """One reader, one artifact, scored on both goal and process.

    ``creator`` may be None for content that is not produced by a hierarchical maker -- foreign
    content has no execution chain to recover, so process scoring is skipped rather than faked.
    ``true_goal`` then supplies what accuracy is scored against.
    """
    res = rollout_observer(agent, artifact, env, cfg_rollout, rng,
                           n_timesteps=n_timesteps, force_deep_k=force_deep_k,
                           initial_glance=initial_glance, early_stop=False)

    q = np.asarray(res.final_goal_posterior, dtype=float)
    g_post = marginal_goal(q, n_mu, ng, n_sub)
    g_prior = marginal_goal(np.asarray(res.goal_prior, dtype=float), n_mu, ng, n_sub)

    rows = np.asarray(res.goal_posterior, dtype=float)
    sub_posts = [marginal_subgoal(r, n_mu, ng, n_sub) for r in rows]
    goal_posts = [marginal_goal(r, n_mu, ng, n_sub) for r in rows]
    modes = (creator_positions(creator, n_timesteps, initial_glance)
             if creator is not None else [])

    free = np.asarray(res.attention)[force_deep_k:]
    engaged = float(np.mean(free == K.DEEP)) if free.size else float("nan")

    g_true = int(creator.goal) if creator is not None else int(true_goal)
    process = (V6.process_recovery(sub_posts, modes, n_sub) if creator is not None else
               {"process_accuracy": float("nan"), "process_logprob": float("nan"),
                "process_error_reduction": float("nan"), "n_steps": 0})

    return Encounter(
        goal_posterior=g_post, goal_prior=g_prior,
        goal_posteriors_by_step=goal_posts,
        subgoal_posteriors=sub_posts, true_modes=modes,
        attention=np.asarray(res.attention), engaged_fraction=engaged,
        correct=int(int(np.argmax(g_post)) == g_true),
        final_entropy=float(metrics.within_observer_entropy(g_post)),
        recovered_mu=recovered_mu(q, n_mu, ng, n_sub),
        error_reduction=float(metrics.error_reduction(g_post, g_prior, g_true)),
        movement=float(metrics.kl_divergence(g_post, g_prior)),
        process=process,
        true_goal=g_true)


def make_foreign_artifact_and_env(world, cfg_rollout: Config, foreign_goal: int,
                                  n_timesteps: int, rng, omega: float = 0.10,
                                  declared_signal: int = K.UNSIGNED):
    """Content produced by a real policy over a goal the reader has no hypothesis for.

    THE REGIME THIS PROJECT ALREADY KNOWS SUSTAINS ATTENTION, which is why it is here rather
    than a hierarchical maker with its legibility turned down. E19 and E20 both found that a
    reader facing partially-overlapping foreign content keeps looking indefinitely -- 0.63 to
    0.79 of the free phase -- because every glance keeps promising an answer that never arrives.
    A hierarchical maker with a low rationality setting does NOT do that: its execution modes are
    still resolvable, so the reader settles and stops, at around 0.01.

    That distinction matters for any experiment whose claim is about ENGAGEMENT rather than about
    recovery, because the two regimes differ by a factor of fifty on exactly that quantity.
    """
    from ..v5_model import V5Environment
    from .. import foreign as FN

    sigs = FN.build_v4_signatures(world.cfg, omega=float(omega), include_explore=False,
                                  foreign_seed=int(world.cfg.get("v4.foreign_seed", 20250401)))
    artifact = Artifact(provenance=K.GHOST, goal=0, declared_signal=int(declared_signal),
                        foreign_goal=int(foreign_goal))
    env = V5Environment(cfg_rollout, world.gm, rng, honesty=1.0, signing_rate=0.0,
                        creator=None, foreign_sig=sigs.sig_foreign)
    return artifact, env


def build_world_and_config(cfg: Config, delta: float | None = None,
                           enforce_separation: bool = True):
    """A V5 world at V4 cardinality, plus the merged config the rollout needs.

    Factored out because nine V6 experiments need the identical five lines and because getting
    the cardinality dance wrong is silent: ``build_v5_world`` needs the UNMERGED goal count to
    build from and ``rollout_observer`` needs the MERGED one to size its arrays.
    """
    cfg = cfg.copy()
    cfg.set("cardinalities.num_goals", FN.NUM_REAL_GOALS)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))
    if delta is not None:
        cfg.set("v5.depth.delta", float(delta))
    world = build_v5_world(cfg, enforce_mode_separation=enforce_separation)
    n_mu, n_sub, ng = len(MU_LEVELS), world.n_subgoals, FN.NUM_REAL_GOALS
    return world, cfg, merged_config(cfg, n_mu, n_sub), n_mu, n_sub, ng


def build_alt_world(cfg: Config, sig_true: np.ndarray, delta: float | None = None,
                    enforce_separation: bool = False):
    """A V5 world built over an ARBITRARY signature family rather than the human one.

    Three V6 experiments need a world whose hypotheses are not the human goal family, and each
    of them needs it for a different reason:

      * the WALL (§4.3) needs content whose maker-state cannot be inverted from a surface the
        reader knows perfectly well;
      * the MACHINE-MATCHED READER (§4.4) needs a reader whose likelihood family is built from
        the machine generator, to ask whether AI literacy stacks with art literacy or replaces
        it;
      * the TOOL HYPOTHESIS (§4.5) needs one extra hypothesis -- "there is no maker here" --
        which changes the goal cardinality and so cannot be bolted onto a finished world.

    ``build_v5_world`` hard-codes the human family and the real-goal count, so this rebuilds the
    same object with both supplied. Everything else -- the depth construction, the chains, the
    channel, the assertions that matter -- is the shipped code, called directly, so this cannot
    drift from V5's world.

    The mode-separation floor is OFF by default here. On an arbitrary family it is not
    meaningful: the floor exists to catch a HUMAN mode family too degenerate to carry depth, and
    a deliberately collapsed family would trip it by design. Which is the point of that cell.
    """
    from pymdp.legacy import utils

    from ..generative_model import (GenerativeModel, alpha_by_provenance, assert_preferences_zero,
                                    build_C)
    from ..v4_model import build_v4_synth
    from ..v5_model import (build_subgoal_chains, build_subgoal_signatures, build_v5_A0,
                            build_v5_A1, build_v5_A2, build_v5_B, V5World)

    cfg = cfg.copy()
    sig = np.asarray(sig_true, dtype=float)
    ng = int(sig.shape[0])
    cfg.set("cardinalities.num_goals", ng)
    cfg.set("cardinalities.num_features",
            int(cfg.get("v4.cardinalities.num_features", FN.NUM_FEATURES_V4)))

    n_sub = int(cfg.get("v5.subgoals.n", 4))
    d = float(cfg.get("v5.depth.delta", 1.0) if delta is None else delta)
    dwell = float(cfg.get("v5.subgoals.dwell", 9.2))
    next_share = float(cfg.get("v5.subgoals.next_share", 0.7))
    kappa = float(cfg.signal_model.kappa)

    synth = build_v4_synth(cfg)
    alpha = alpha_by_provenance(cfg)
    subsig = build_subgoal_signatures(cfg, sig, n_sub, d)
    chains = build_subgoal_chains(ng, n_sub, dwell, next_share)
    n_mu = len(MU_LEVELS)

    A = utils.obj_array(3)
    A[0] = build_v5_A0(cfg, subsig, synth, alpha)
    A[1] = build_v5_A1(cfg, kappa, n_mu * ng * n_sub)
    A[2] = build_v5_A2(cfg, n_mu, ng, n_sub)

    gm = GenerativeModel(A=A, B=build_v5_B(cfg, chains, n_mu), C=build_C(cfg),
                         sig=sig, noise_free_synth=synth, alpha=alpha, kappa=kappa,
                         cfg=cfg, d=0.0, sig_true=sig, synth_k=-1, synth_lean=0.0,
                         goal_symmetric=True)
    assert_preferences_zero(gm.C)

    world = V5World(gm=gm, sigs=None, cfg=cfg, mu_levels=MU_LEVELS, n_subgoals=n_sub,
                    subsig=subsig, chains=chains,
                    diagnostics={"alt_family": True, "n_goals": ng,
                                 "separation_enforced": bool(enforce_separation)})
    return world, cfg, merged_config_for(cfg, n_mu, n_sub, ng), n_mu, n_sub, ng


def merged_config_for(cfg: Config, n_mu: int, n_sub: int, ng: int) -> Config:
    """``merged_config`` with an explicit goal count, for worlds that are not the human family."""
    merged = cfg.copy()
    merged.set("cardinalities.num_goals", int(n_mu) * int(ng) * int(n_sub))
    return merged


def make_alt_observer(world, rng, ng: int, mu_prior=None):
    """An observer over an alternate world. ``build_v5_D`` needs the real goal count passed in."""
    from ..v5_model import F_ATTENTION, build_v5_D

    D = build_v5_D(world.cfg, rng, len(world.mu_levels), int(ng), world.n_subgoals,
                   mu_prior=mu_prior)
    a = world.cfg.agent
    if bool(world.cfg.get("inference.exact", False)):
        from ..exact import ExactAgent
        return ExactAgent(A=world.gm.A, B=world.gm.B, C=world.gm.C, D=D,
                          control_fac_idx=[F_ATTENTION], policy_len=int(a.policy_len),
                          gamma=float(a.gamma), action_selection=str(a.action_selection),
                          use_utility=bool(a.use_utility),
                          use_states_info_gain=bool(a.use_states_info_gain), rng=rng)
    from pymdp.legacy.agent import Agent
    return Agent(A=world.gm.A, B=world.gm.B, C=world.gm.C, D=D,
                 control_fac_idx=[F_ATTENTION], policy_len=int(a.policy_len),
                 inference_horizon=int(a.inference_horizon),
                 use_utility=bool(a.use_utility),
                 use_states_info_gain=bool(a.use_states_info_gain),
                 action_selection=str(a.action_selection), gamma=float(a.gamma))


def make_artifact_and_env(world, cfg_rollout: Config, true_goal: int, true_mu: int,
                          beta: float, n_timesteps: int, rng, provenance: int = K.CREATOR,
                          declared_signal: int = K.UNSIGNED, honesty: float = 1.0,
                          signing_rate: float = 0.0):
    creator = HierarchicalCreator(world, int(true_goal), int(true_mu), float(beta),
                                  n_positions=int(n_timesteps) + 2, rng=rng)
    artifact = Artifact(provenance=int(provenance), goal=int(true_goal),
                        declared_signal=int(declared_signal))
    env = V5Environment(cfg_rollout, world.gm, rng, honesty=honesty,
                        signing_rate=signing_rate, creator=creator)
    return creator, artifact, env


def integration(enc: Encounter, kappa: float, reserve: V6.MetabolicReserve | None,
                lam: float, coupling: float, k_gain: float,
                values_map: np.ndarray | None = None,
                value_prior: np.ndarray | None = None,
                theta_0: float = 0.0) -> dict:
    """How much of what the reader recovered is allowed to change it.

    THIS IS THE QUANTITY THE AUTHOR'S CORRECTION IS ABOUT. Deep looking and permitting
    integration are separate things in this model and V1-V5 never reported them apart: a reader
    can look intently, recover the maker accurately, and integrate nothing. That is a closed
    gate with high engagement, and it is what the objection to calling engagement "willingness
    to be vulnerable" correctly identified.

    ``omega`` here is the reader's own PRECISION WEIGHTING, which is the preprint's omega and
    is NOT the code's overlap parameter -- the two share a letter and are different objects,
    one an output and one an input. Estimated from how much the reader actually resolved,
    which is what "the brain drops the precision weighting of incoming data" describes.
    """
    if values_map is not None and value_prior is not None:
        divergence = V6.value_divergence_via_values(enc.goal_posterior, value_prior, values_map)
    else:
        prior = np.asarray(enc.goal_prior, dtype=float)
        divergence = float(metrics.kl_divergence(enc.goal_posterior, prior))

    resolved = 1.0 - float(enc.final_entropy) / float(np.log(enc.goal_posterior.size))
    omega_precision = float(np.clip(resolved, 0.0, 1.0))

    psi = V6.psi_intent_extraction_limit(
        enc.goal_posterior, enc.goal_prior, kappa, omega_precision, divergence,
        reserve=reserve, lam=lam, theta_0=theta_0, coupling=coupling, k_gain=k_gain)
    psi["integration"] = float(psi["gate"])
    psi["value_divergence_source"] = "values" if values_map is not None else "goal"
    return psi
