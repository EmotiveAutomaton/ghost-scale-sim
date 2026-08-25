"""The bounded active reader: the repository's pinned legacy PyMDP agent, given a belief over
maker hypotheses and a controllable probe (spec section 4.3).

Factors:   [maker hypothesis (K)]  [probe context (P)]
Modalities:[feature (F)]           [probe echo (P)]
The probe factor is fully controllable; the feature likelihood depends on both the hypothesis
and the probe, so choosing a probe is choosing which hypotheses become distinguishable. The reader
has zero preference over feature outcomes and over which hypothesis is true (the N7 assertion
is applied at construction); its policy is driven by epistemic value and, where a cost modality
is attached, by probe cost.
"""
from __future__ import annotations

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

_EPS = 1e-12


def build_reader(emissions: np.ndarray, prior: np.ndarray, probe_costs: np.ndarray | None = None,
                 gamma: float = 16.0, use_info_gain: bool = True, use_utility: bool = True,
                 use_param_info_gain: bool = False, policy_len: int = 1) -> Agent:
    """``emissions[p, h]`` is the feature distribution under probe p and hypothesis h.
    ``prior`` is over hypotheses. Costs, if given, enter through a second modality whose
    preference is -cost (the only preference the reader holds)."""
    P, K, F = emissions.shape
    A = utils.obj_array(2 if probe_costs is not None else 1)
    A0 = np.zeros((F, K, P))
    for p in range(P):
        for h in range(K):
            A0[:, h, p] = emissions[p, h]
    A[0] = A0
    if probe_costs is not None:
        # cost outcome is a deterministic function of the probe: one outcome per probe level
        A1 = np.zeros((P, K, P))
        for p in range(P):
            A1[p, :, p] = 1.0
        A[1] = A1
    B = utils.obj_array(2)
    B[0] = np.eye(K)[:, :, None]
    B1 = np.zeros((P, P, P))
    for a in range(P):
        B1[a, :, a] = 1.0
    B[1] = B1
    C = utils.obj_array(len(A))
    C[0] = np.zeros(F)
    if probe_costs is not None:
        C[1] = -np.asarray(probe_costs, dtype=float)
    D = utils.obj_array(2)
    D[0] = np.asarray(prior, dtype=float) / np.sum(prior)
    D[1] = np.full(P, 1.0 / P)
    assert np.all(C[0] == 0.0), "the reader must hold no preference over features (N7)"
    agent = Agent(A=A, B=B, C=C, D=D, control_fac_idx=[1], policy_len=int(policy_len),
                  gamma=float(gamma), use_utility=bool(use_utility),
                  use_states_info_gain=bool(use_info_gain),
                  use_param_info_gain=bool(use_param_info_gain),
                  action_selection="deterministic")
    return agent


def choose_probe(agent: Agent, current_probe: int = 0) -> tuple:
    """One decision step: infer states from the current probe echo (no feature yet), evaluate
    policies, and return (chosen probe, negative expected free energy per policy)."""
    obs = [0] if len(agent.A) == 1 else [0, int(current_probe)]
    # A feature observation is required by the API; use the first feature as a dummy only when
    # the caller has no observation yet. Callers with real evidence use ``observe`` instead.
    agent.infer_states(obs)
    q_pi, G = agent.infer_policies()
    action = agent.sample_action()
    return int(action[1]), np.asarray(G, dtype=float)


def observe_sequence(agent: Agent, features: np.ndarray, probe: int) -> np.ndarray:
    """Feed a sequence of feature observations under a fixed probe; return the final posterior
    over hypotheses."""
    qs = None
    for f in features:
        obs = [int(f)] if len(agent.A) == 1 else [int(f), int(probe)]
        qs = agent.infer_states(obs)
        # THE LEGACY AGENT ONLY ADVANCES TIME WHEN AN ACTION IS TAKEN. Without this, every
        # infer_states restarts from D and the posterior reflects only the last observation
        # (the same reset gotcha V11's observer fix documents). Hold the probe fixed and step.
        agent.action = np.array([0.0, float(probe)]) if len(agent.B) == 2 else np.array([0.0])
        agent.step_time()
    return np.asarray(qs[0], dtype=float)


def exact_sequence_posterior(emissions: np.ndarray, prior: np.ndarray, features: np.ndarray,
                             probe: int) -> np.ndarray:
    """The closed-form posterior the agent should converge to under a fixed probe."""
    ll = np.log(np.maximum(emissions[probe][:, features], _EPS)).sum(axis=1)
    v = ll + np.log(np.maximum(prior, _EPS))
    v = np.exp(v - v.max())
    return v / v.sum()


def policy_disagreement(exact_ranking: np.ndarray, agent_choice: int) -> dict:
    """Whether the agent's chosen probe is the exact-optimal one, and its rank."""
    order = list(np.argsort(-np.asarray(exact_ranking)))
    return {"agent_probe": int(agent_choice), "exact_best": int(order[0]),
            "agent_rank": int(order.index(int(agent_choice))) + 1,
            "agrees": bool(order[0] == int(agent_choice))}
