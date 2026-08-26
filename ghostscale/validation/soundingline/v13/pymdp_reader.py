"""The bounded active reader: the repository's pinned legacy PyMDP agent over a V13 hypothesis
grid with a controllable probe (spec §4). Used only where the reader acts (trunk Q, G14).

Factors:   [hypothesis (K)]   [probe (P)]
Modalities:[feature (F)]      [probe echo (P)]   [+ cost (P) when probe costs are given]
The probe factor is fully controllable; the feature likelihood depends on hypothesis and probe,
so choosing a probe chooses which hypotheses become distinguishable. The reader holds no
preference over features or hypotheses (asserted at construction); its policy is driven by
epistemic value and, with a cost modality, by probe cost.
"""
from __future__ import annotations

import numpy as np

from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

_EPS = 1e-12


def build_reader(emissions: np.ndarray, prior: np.ndarray, probe_costs=None, gamma: float = 16.0,
                 use_info_gain: bool = True, use_utility: bool = True, policy_len: int = 1) -> Agent:
    """``emissions[p, h]`` is the feature distribution under probe p and hypothesis h."""
    P, K, F = emissions.shape
    n_mod = 2 + (1 if probe_costs is not None else 0)
    A = utils.obj_array(n_mod)
    A0 = np.zeros((F, K, P))
    for p in range(P):
        for h in range(K):
            A0[:, h, p] = emissions[p, h]
    A[0] = A0
    A1 = np.zeros((P, K, P))
    for p in range(P):
        A1[p, :, p] = 1.0
    A[1] = A1
    if probe_costs is not None:
        A[2] = A1.copy()
    B = utils.obj_array(2)
    B[0] = np.eye(K)[:, :, None]
    B1 = np.zeros((P, P, P))
    for a in range(P):
        B1[a, :, a] = 1.0
    B[1] = B1
    Cm = utils.obj_array(n_mod)
    Cm[0] = np.zeros(F)
    Cm[1] = np.zeros(P)
    if probe_costs is not None:
        Cm[2] = -np.asarray(probe_costs, dtype=float)
    D = utils.obj_array(2)
    D[0] = np.asarray(prior, dtype=float) / np.sum(prior)
    D[1] = np.full(P, 1.0 / P)
    assert np.all(Cm[0] == 0.0), "the reader must hold no preference over features"
    return Agent(A=A, B=B, C=Cm, D=D, control_fac_idx=[1], policy_len=int(policy_len), gamma=float(gamma),
                 use_utility=bool(use_utility), use_states_info_gain=bool(use_info_gain),
                 use_param_info_gain=False, action_selection="deterministic")


def choose_probe(agent: Agent, current_probe: int = 0) -> tuple:
    obs = [0, int(current_probe)] + ([int(current_probe)] if len(agent.A) == 3 else [])
    agent.infer_states(obs)
    q_pi, G = agent.infer_policies()
    action = agent.sample_action()
    return int(action[1]), np.asarray(G, dtype=float)


def observe_sequence(agent: Agent, features: np.ndarray, probe: int) -> np.ndarray:
    qs = None
    for f in features:
        obs = [int(f), int(probe)] + ([int(probe)] if len(agent.A) == 3 else [])
        qs = agent.infer_states(obs)
        agent.action = np.array([0.0, float(probe)])
        agent.step_time()
    return np.asarray(qs[0], dtype=float)


def exact_sequence_posterior(emissions: np.ndarray, prior: np.ndarray, features: np.ndarray, probe: int) -> np.ndarray:
    ll = np.log(np.maximum(emissions[probe][:, features], _EPS)).sum(axis=1)
    v = ll + np.log(np.maximum(prior, _EPS))
    v = np.exp(v - v.max())
    return v / v.sum()


def exact_eig_per_probe(emissions: np.ndarray, prior: np.ndarray, n_steps: int, rng, draws: int = 200,
                        groups: np.ndarray | None = None) -> np.ndarray:
    """Monte-Carlo EIG of each probe about the hypothesis (or its group marginal)."""
    P, K, F = emissions.shape
    p = np.asarray(prior, float)
    H0 = -(p[p > 0] * np.log(p[p > 0])).sum() if groups is None else _ent(np.bincount(groups, weights=p))
    out = np.zeros(P)
    for pr in range(P):
        posts = []
        for _ in range(int(draws)):
            h = int(rng.choice(K, p=p))
            feats = rng.choice(F, size=int(n_steps), p=emissions[pr, h])
            q = exact_sequence_posterior(emissions, p, feats, pr)
            posts.append(_ent(q) if groups is None else _ent(np.bincount(groups, weights=q)))
        out[pr] = H0 - np.mean(posts)
    return out


def _ent(q):
    q = np.asarray(q, float)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def pairwise_divergence(emissions: np.ndarray) -> dict:
    """Probe non-equivalence audit (Q08): for every pair of probes, the mean JS between the
    feature distributions they induce under the same hypothesis, and per probe the mean JS
    between hypotheses (how discriminative the probe is)."""
    from .common import js
    P, K, F = emissions.shape
    between_probes = np.zeros((P, P))
    for a in range(P):
        for b in range(P):
            between_probes[a, b] = float(np.mean([js(emissions[a, h], emissions[b, h]) for h in range(K)]))
    within = np.array([float(np.mean([js(emissions[p, h], emissions[p, k]) for h in range(K) for k in range(h + 1, K)])) if K > 1 else 0.0
                       for p in range(P)])
    off = between_probes[~np.eye(P, dtype=bool)]
    return {"min_pairwise_probe_js": float(off.min()) if off.size else 0.0, "mean_pairwise_probe_js": float(off.mean()) if off.size else 0.0,
            "discriminativeness_per_probe": within.tolist(), "min_discriminativeness": float(within.min())}


def policy_disagreement(exact_ranking: np.ndarray, agent_choice: int) -> dict:
    order = list(np.argsort(-np.asarray(exact_ranking)))
    return {"agent_probe": int(agent_choice), "exact_best": int(order[0]),
            "agent_rank": int(order.index(int(agent_choice))) + 1, "agrees": bool(order[0] == int(agent_choice))}
