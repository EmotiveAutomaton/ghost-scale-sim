"""Creators and the environment's content sources (Spec §4).

The asymmetry between the two creator classes IS the entire theoretical content of the
model (Spec §4.2), and is the reason the simulation does not assume its own conclusion:

    * A HUMAN CREATOR is a real pymdp Agent — a small POMDP with a hidden preferred
      outcome (its C) that SELECTS ACTIONS to realize that preference. Its policy, not a
      direct sample of sig[g], produces the artifact's features. Human artifacts carry
      recoverable goals *because a reward-optimizing policy actually produced them*.

    * A SYNTHETIC GENERATOR has NO C and NO policy. It samples features from
      noise_free_synth directly. No goal caused its structure.

We verify post hoc (Spec §4.1) that a human creator's realized feature distribution
approximates sig[g], and record the approximation error. Because the creator's emission
distribution is exactly its policy's action distribution, drawing an artifact's features
from that distribution is equivalent to letting the creator act to produce each feature —
we characterize it once (fast) rather than re-running inference per feature.
"""
from __future__ import annotations

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from . import constants as K
from .config import Config
from .generative_model import GenerativeModel, cards

_LOG_FLOOR = 1e-8


class HumanCreator:
    """A real POMDP agent that produces goal-laden artifact features.

    The creator picks, at each step, which feature to emit (a controllable "feature"
    factor with an identity likelihood). Its preference C is log(sig[g]); with unit policy
    precision the resulting action distribution equals sig[g], so a reward-optimizing
    policy reproduces the goal signature — exactly the claim under test.
    """

    def __init__(self, cfg: Config, sig: np.ndarray, goal: int):
        cd = cards(cfg)
        nf = cd.features
        self.goal = int(goal)
        self.cfg = cfg

        A = utils.obj_array(1)
        A[0] = np.eye(nf)                       # observe the feature you emit
        B = utils.obj_array(1)
        B0 = np.zeros((nf, nf, nf))             # action a -> emit feature a
        for a in range(nf):
            B0[a, :, a] = 1.0
        B[0] = B0
        C = utils.obj_array(1)
        C[0] = np.log(sig[self.goal] + _LOG_FLOOR)   # preference = log goal-signature
        D = utils.obj_array(1)
        D[0] = np.full(nf, 1.0 / nf)

        # policy_len=1, utility only (no epistemic term needed: A is identity), gamma=1 so
        # the policy posterior equals sig[g]; stochastic selection so emission is a sample.
        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            control_fac_idx=[0],
            policy_len=1,
            inference_horizon=1,
            use_utility=True,
            use_states_info_gain=False,
            action_selection="stochastic",
            gamma=1.0,
        )
        self._p_emit: np.ndarray | None = None

    def emission_distribution(self) -> np.ndarray:
        """The policy's feature-emission distribution P(feature), read from the agent's
        policy posterior (one inference call). This IS the reward-optimizing policy's
        action distribution."""
        if self._p_emit is None:
            self.agent.reset()
            q_pi, _ = self.agent.infer_policies()
            nf = np.asarray(self.agent.A[0]).shape[0]
            p = np.zeros(nf)
            for policy, prob in zip(self.agent.policies, q_pi):
                feat = int(np.asarray(policy).ravel()[0])  # policy_len=1, single control
                p[feat] += float(prob)
            self._p_emit = p / p.sum()
        return self._p_emit

    def sample_features(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n feature emissions from the policy distribution (equivalent to the
        creator acting n times)."""
        p = self.emission_distribution()
        return rng.choice(len(p), size=n, p=p)


def build_creator_bank(cfg: Config, gm: GenerativeModel) -> dict[int, HumanCreator]:
    """One human creator per goal (they share the goal-signature machinery)."""
    cd = cards(cfg)
    return {g: HumanCreator(cfg, gm.sig, g) for g in range(cd.goals)}


def verify_empirical_matches_A0(gm: GenerativeModel, cfg: Config,
                                rng: np.random.Generator, n_samples: int = 20000) -> dict:
    """Spec §4 / §10: verify that the environment's realized DEEP feature distribution
    matches the corresponding A[0] column within tolerance, per tier.

    Uses the same emission machinery the environment uses (creator policy for the human
    fraction α[p], noise_free_synth for the rest; pure synth for GHOST). Returns a dict
    tier_name -> max absolute deviation from A[0][:, p, g, DEEP].
    """
    from .environment import Environment  # local import to avoid a cycle

    env = Environment(cfg, gm, rng_world=rng,
                      honesty=1.0, signing_rate=1.0)
    cd = cards(cfg)
    goal = 0  # fixed reference goal for the comparison
    report: dict[str, float] = {}
    for p in range(cd.provenance):
        artifact = env.make_artifact(provenance=p, goal=goal, rng=rng)
        feats = np.array([env.sample_feature(artifact, K.DEEP, rng) for _ in range(n_samples)])
        emp = np.bincount(feats, minlength=cd.features).astype(float)
        emp /= emp.sum()
        target = np.asarray(gm.A[0])[:, p, goal, K.DEEP]
        report[K.PROVENANCE_NAMES[p]] = float(np.max(np.abs(emp - target)))
    return report
