"""The observer: a per-observer agent wrapper and the rollout loop.

The observer is a control-of-perception agent: at each timestep it first DECIDES how
deeply to look (DEEP vs SKIM) from its current beliefs, then receives an artifact_features
observation whose goal-content depends on that choice. Effort observations deterministically
reveal the attention state, so the attention posterior is always pinned to the chosen action.

Note on C (pymdp legacy): the utility term softmaxes C per modality, so C[0]=C[1]=0 become
*uniform* preferences — a constant offset identical across policies that cancels in policy
comparison. Only the effort preference C[2] differentiates DEEP from SKIM. This is precisely
the "no preference over provenance or signal" requirement (Spec §14): disengagement can only
come from the epistemic term collapsing, never from a distaste.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from .config import Config
from .environment import Artifact, Environment
from .generative_model import GenerativeModel, build_D, make_agent
from . import metrics


def make_observer(gm: GenerativeModel, cfg: Config, rng: np.random.Generator,
                  kappa: float | None = None):
    """Build one heterogeneous observer agent (its own D drawn from ``rng``)."""
    D = build_D(cfg, rng)
    return make_agent(gm, D, cfg, kappa=kappa)


def find_named_policies(agent):
    """Return (all_DEEP_policy, all_SKIM_policy) from the agent's policy set."""
    deep = skim = None
    for p in agent.policies:
        att = np.asarray(p)[:, K.F_ATTENTION]
        if np.all(att == K.DEEP):
            deep = p
        if np.all(att == K.SKIM):
            skim = p
    return deep, skim


def _deep_action(agent) -> np.ndarray:
    """Action array that keeps attention DEEP (non-control factors stay 0)."""
    action = np.zeros(len(agent.num_controls))
    action[K.F_ATTENTION] = K.DEEP
    return action


@dataclass
class RolloutResult:
    # Per-timestep series (length T).
    attention: np.ndarray                 # attention actually used each step
    goal_posterior: np.ndarray            # (T, num_goals) post-observation belief
    prov_posterior: np.ndarray            # (T, num_provenance)
    within_entropy: np.ndarray            # (T,) Shannon entropy of goal posterior
    efe: dict = field(default_factory=dict)   # optional EFE-term series (pre-obs decision values)

    # Summary scalars.
    first_skim_t: int = -1                # first free (non-forced) SKIM; -1 if never
    cum_deep: int = 0                     # cumulative DEEP steps
    cum_high_cost: int = 0                # cumulative HIGH_COST observations (== DEEP steps)
    final_goal_posterior: np.ndarray = None
    goal_prior: np.ndarray = None         # observer's D over goal (for KL/psi)
    modal_goal: int = -1
    kappa: float = None


def rollout_observer(agent, artifact: Artifact, env: Environment, cfg: Config,
                     rng: np.random.Generator, n_timesteps: int,
                     force_deep_k: int = 0, record_efe: bool = False,
                     kappa: float | None = None, initial_glance: bool = True) -> RolloutResult:
    """Roll one observer forward over ``n_timesteps`` looking at a single artifact.

    ``force_deep_k``: force DEEP for the first k steps (E2), so every condition gets a
    comparable inference attempt before the agent is free to disengage.

    ``initial_glance``: before the decision loop, the observer takes one free cheap glance
    (a SKIM observation) that delivers the provenance signal and updates its provenance
    belief WITHOUT resolving any goal (synth features). This is the Ghost Scale working as
    intended (Spec §0): the observer reads the provenance tag cheaply and early, and can
    then disengage from GHOST content *before* burning DEEP budget. The glance is not
    counted as a timestep or a DEEP step.
    """
    agent.reset()
    deep_pol, skim_pol = find_named_policies(agent)
    goal_prior = np.asarray(agent.D[K.F_GOAL], dtype=float).copy()

    if initial_glance:
        glance_feat = env.sample_feature(artifact, K.SKIM, rng)  # synth: no goal info
        agent.infer_states([glance_feat, artifact.declared_signal, K.LOW_COST])

    attn = np.zeros(n_timesteps, dtype=int)
    goal_post = np.zeros((n_timesteps, cfg.cardinalities.num_goals))
    prov_post = np.zeros((n_timesteps, cfg.cardinalities.num_provenance))
    ent = np.zeros(n_timesteps)
    efe = {k: np.zeros(n_timesteps) for k in
           ("deep_prag", "deep_epi_total", "deep_epi_goal",
            "skim_prag", "skim_epi_total", "skim_epi_goal")} if record_efe else {}

    first_skim = -1
    for t in range(n_timesteps):
        agent.infer_policies()  # decision from current (pre-observation) beliefs

        if record_efe:
            pd, ed_tot = metrics.policy_efe_terms(agent, deep_pol)
            ed_goal = metrics.epistemic_value(agent, deep_pol)
            ps, es_tot = metrics.policy_efe_terms(agent, skim_pol)
            es_goal = metrics.epistemic_value(agent, skim_pol)
            efe["deep_prag"][t], efe["deep_epi_total"][t], efe["deep_epi_goal"][t] = pd, ed_tot, ed_goal
            efe["skim_prag"][t], efe["skim_epi_total"][t], efe["skim_epi_goal"][t] = ps, es_tot, es_goal

        forced = t < force_deep_k
        if forced:
            attention = K.DEEP
            agent.action = _deep_action(agent)
        else:
            action = agent.sample_action()
            attention = int(action[K.F_ATTENTION])
            if attention == K.SKIM and first_skim < 0:
                first_skim = t

        obs = env.observation(artifact, attention, rng)
        qs = agent.infer_states(obs)

        attn[t] = attention
        goal_post[t] = np.asarray(qs[K.F_GOAL])
        prov_post[t] = np.asarray(qs[K.F_PROVENANCE])
        ent[t] = metrics.within_observer_entropy(goal_post[t])

    final_goal = goal_post[-1]
    return RolloutResult(
        attention=attn, goal_posterior=goal_post, prov_posterior=prov_post,
        within_entropy=ent, efe=efe,
        first_skim_t=first_skim,
        cum_deep=int(np.sum(attn == K.DEEP)),
        cum_high_cost=int(np.sum(attn == K.DEEP)),
        final_goal_posterior=final_goal,
        goal_prior=goal_prior,
        modal_goal=int(np.argmax(final_goal)),
        kappa=float(kappa if kappa is not None else cfg.signal_model.kappa),
    )
