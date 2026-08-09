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
from .generative_model import (GenerativeModel, build_D, make_agent,
                               build_observer_model)
from . import metrics
from . import learning as _learning

# Fallback salt, used only if a Generator exposes no seed sequence to spawn from.
_SIG_STREAM_SALT = 0x5164A17
# Inexpertise at or below this is treated as exactly zero (fast path / N8).
_D_ZERO_TOL = 0.0


def observer_sig_rng(rng: np.random.Generator) -> np.random.Generator:
    """A DEDICATED RNG stream for the C1 ``sig_i`` perturbation.

    N8 HAZARD — the reason this exists. The observer's D prior is drawn from the caller's
    ``rng``. If ``sig_i`` were drawn from the same stream, then turning expertise on would
    consume variates and shift every subsequent D draw, so V2-at-d=0 would no longer
    reproduce V1 and N8 would fail for a reason having nothing to do with the refactor.

    ``seed_seq.spawn`` is the correct primitive here: it derives an independent child stream
    WITHOUT drawing a variate from the parent, so the D sequence is bit-identical whatever
    ``d`` is. (``spawn`` advances the parent's child counter only, not its bit stream.)
    """
    seq = getattr(rng.bit_generator, "seed_seq", None)
    if seq is not None:
        return np.random.default_rng(seq.spawn(1)[0])
    state = rng.bit_generator.state["state"]
    base = state["state"] if isinstance(state, dict) else int(state)
    return np.random.default_rng(int(base) % (2**63 - 1) ^ _SIG_STREAM_SALT)


def make_observer(gm: GenerativeModel, cfg: Config, rng: np.random.Generator,
                  kappa: float | None = None, d_i: float = 0.0):
    """Build one heterogeneous observer agent.

    Heterogeneity now has TWO sources (C1): the observer's own D prior (V1, drawn from
    ``rng``) and its own goal-signature likelihood ``sig_i`` at inexpertise ``d_i`` (V2,
    drawn from an independent stream). At ``d_i=0`` this is exactly the V1 observer.
    """
    rng_sig = observer_sig_rng(rng) if d_i > _D_ZERO_TOL else None
    om = build_observer_model(gm, cfg, d_i, rng_sig=rng_sig, kappa=kappa)
    D = build_D(cfg, rng)
    return make_agent(om, D, cfg, kappa=kappa)


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

    # V4.5: posteriors over any ADDITIONAL hidden state factors the caller asked to record,
    # keyed by factor index. Empty for every V1-V4 rollout, since none of them has a fourth
    # factor. This exists so beta can be read out of the same loop the goal is read out of,
    # rather than duplicating the rollout for E28.
    extra_posterior: dict = field(default_factory=dict)       # factor -> (T, cardinality)
    final_extra_posterior: dict = field(default_factory=dict) # factor -> (cardinality,)


def rollout_observer(agent, artifact: Artifact, env: Environment, cfg: Config,
                     rng: np.random.Generator, n_timesteps: int,
                     force_deep_k: int = 0, record_efe: bool = False,
                     kappa: float | None = None, initial_glance: bool = True,
                     early_stop: bool = True, stop_patience: int = 3,
                     stop_tol: float = 1e-3, learn: bool = False,
                     learn_mode: str = "online",
                     extra_factors: tuple[int, ...] = ()) -> RolloutResult:
    """Roll one observer forward over ``n_timesteps`` looking at a single artifact.

    ``force_deep_k``: force DEEP for the first k steps (E2), so every condition gets a
    comparable inference attempt before the agent is free to disengage.

    ``initial_glance``: before the decision loop, the observer takes one free cheap glance
    (a SKIM observation) that delivers the provenance signal and updates its provenance
    belief WITHOUT resolving any goal (synth features). This is the Ghost Scale working as
    intended (Spec §0): the observer reads the provenance tag cheaply and early, and can
    then disengage from GHOST content *before* burning DEEP budget. The glance is not
    counted as a timestep or a DEEP step.

    ``early_stop``: once the observer has disengaged (SKIM) and its goal posterior has been
    stable for ``stop_patience`` steps, the rollout stops and the remaining timesteps are
    padded with the final value. This is exact — after disengagement the agent skims
    forever and its beliefs are static (identity B on provenance/goal, no goal-resolving
    observations) — and it is the dominant runtime win. Never triggers before the forced
    phase ends.

    ``learn`` (V2 C3): after each timestep's state inference, run a Dirichlet concentration
    update on A[0]. Requires a Learner agent (one built with a pA). Two deliberate choices:

      * The FREE INITIAL GLANCE never contributes a learning update. The glance is a
        modelling device for delivering the provenance tag cheaply; letting it also update
        the model would hand out free model improvements that no metabolic cost gates, and
        that would blunt E9's starvation channel — a fully disengaged observer would keep
        learning from glances alone and never starve.
      * EARLY STOP is disabled whenever ``learn`` is set. Its correctness argument is that a
        disengaged observer's beliefs are static, which is false once the model itself is
        changing: a learner that skims is still accumulating counts on its SKIM columns.

    ``learn_mode`` (V3 follow-up) selects WHEN evidence is committed:

      * ``"online"``   — V1/V2/V3 behaviour: update at every timestep with the posterior held
        at that moment. Biased flat, because the first DEEP observation is committed before
        the goal has resolved; see ``learning.learn_deferred`` for the measurement.
      * ``"deferred"`` — buffer the artifact's observations and commit them once inference has
        finished, attributed under the resolved posterior. No new parameter, and the bias is
        absent at any ``n_timesteps``.

    ``extra_factors`` (V4.5) records the posterior over additional hidden state factors, by
    index, into ``RolloutResult.extra_posterior``. Empty by default and therefore inert for
    every V1-V4 call site. E28 gives the observer a fourth factor (beta, the creator's
    rationality) and needs to read its posterior out of the same loop that reads the goal's;
    duplicating the rollout to do that would have created two copies of the early-stop
    argument and the learning hooks, which is how they drift.
    """
    if learn:
        early_stop = False   # the early-stop correctness argument does not survive learning
    if learn_mode not in ("online", "deferred"):
        raise ValueError(f"learn_mode must be 'online' or 'deferred', got {learn_mode!r}")
    buffered: list[tuple[list[int], np.ndarray]] = []
    agent.reset()
    # PYMDP GOTCHA, the same one regret._reset_to_prior documents: Agent.reset() sets qs to
    # uniform but does NOT clear self.action, so on a REUSED agent the first infer_states of the
    # next artifact takes empirical_prior = B-propagated(uniform) instead of D. Every observer's
    # heterogeneous prior — which null N3 calls load-bearing — was silently replaced by a uniform
    # one from the second artifact onward in every corpus loop (E6, E6b, E7, E8, E9, E12, E13,
    # and the calibration criterion). A fresh agent has action None already, and ExactAgent.reset
    # clears it itself, so this line changes reused pymdp agents and nothing else.
    agent.action = None
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
    extra = {f: np.zeros((n_timesteps, np.asarray(agent.D[f]).size))
             for f in extra_factors}

    first_skim = -1
    for t in range(n_timesteps):
        forced = t < force_deep_k

        # COST: infer_policies is ~85% of runtime in the forced-DEEP aggregator path
        # (measured: 2.25s of 2.63s), and on a forced step its entire output is discarded —
        # the action is set directly below and ``sample_action`` is never called, so q_pi and
        # G are never read. Skipping it there is exact, not an approximation. It is still run
        # when EFE terms are being recorded, since those ARE the measurement (E1/E5).
        # This is the dominant cost lever for E6/E6b/E7, not the corpus caching V2 spec §4
        # assumes: an Artifact is metadata only, and draw_corpus(200) costs 0.9 ms against
        # 5091 ms of rollouts for the same 200 artifacts.
        if record_efe or not forced:
            agent.infer_policies()  # decision from current (pre-observation) beliefs

        if record_efe:
            pd, ed_tot = metrics.policy_efe_terms(agent, deep_pol)
            ed_goal = metrics.epistemic_value(agent, deep_pol)
            ps, es_tot = metrics.policy_efe_terms(agent, skim_pol)
            es_goal = metrics.epistemic_value(agent, skim_pol)
            efe["deep_prag"][t], efe["deep_epi_total"][t], efe["deep_epi_goal"][t] = pd, ed_tot, ed_goal
            efe["skim_prag"][t], efe["skim_epi_total"][t], efe["skim_epi_goal"][t] = ps, es_tot, es_goal

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
        if learn:
            if learn_mode == "online":
                _learning.learn_step(agent, obs, cfg)
            else:
                # Attention is kept per-step: effort observations reveal it exactly, so it is
                # not one of the uncertain factors the resolved posterior is meant to correct.
                buffered.append((obs, np.asarray(qs[K.F_ATTENTION], dtype=float).copy()))

        attn[t] = attention
        goal_post[t] = np.asarray(qs[K.F_GOAL])
        prov_post[t] = np.asarray(qs[K.F_PROVENANCE])
        ent[t] = metrics.within_observer_entropy(goal_post[t])
        for f in extra_factors:
            extra[f][t] = np.asarray(qs[f], dtype=float)

        # Early stop: after the forced phase, if disengaged (SKIM) and the goal posterior
        # has been stable for `stop_patience` steps, the future is static -> pad and break.
        if early_stop and not forced and attention == K.SKIM and t >= force_deep_k + stop_patience:
            recent = goal_post[t - stop_patience:t + 1]
            if np.max(np.abs(recent - goal_post[t])) < stop_tol:
                attn[t + 1:] = K.SKIM
                goal_post[t + 1:] = goal_post[t]
                prov_post[t + 1:] = prov_post[t]
                ent[t + 1:] = ent[t]
                if record_efe:
                    for key in efe:
                        efe[key][t + 1:] = efe[key][t]
                for f in extra_factors:
                    extra[f][t + 1:] = extra[f][t]
                break

    if learn and learn_mode == "deferred":
        # Inference over this artifact is finished; commit its evidence now, under the
        # posterior that finishing produced.
        _learning.learn_deferred(agent, buffered, cfg)

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
        extra_posterior=extra,
        final_extra_posterior={f: extra[f][-1].copy() for f in extra_factors},
    )
