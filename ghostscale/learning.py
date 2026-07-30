"""Learned likelihoods — the Learner observer and its diagnostics (V2 C3).

This module is what converts E6 from "does a provenance label help someone who already
knows?" into "can you learn to tell without labels?". V1's aggregator arrived holding a
correct ``A[0]`` GHOST column: it already knew what goalless output looks like, so the
experiment had been answered before it was run (V2 spec §0, Error 3). The Learner does not
have that column and must accumulate it from a corpus that is already contaminated and
unlabelled.

Two agent classes, both required:

  * **Oracle**  — V1 behaviour, correct ``A[0]`` supplied. The ceiling condition.
    Built by ``observer.make_observer``; nothing in this module touches it.
  * **Learner** — a weak Dirichlet prior over ``A[0]`` plus ``update_A`` from experience.

-----------------------------------------------------------------------------------------
DECISION D1 — how the Learner's prior is seeded. This is a THEORETICAL COMMITMENT, not a
convenience, and it is the single most consequential implementation choice in V2.

C3 says the learner starts "uninformative over the provenance dimension". Read literally as
"uniform everywhere", the learner cannot work, and not merely because it disengages:

    measured, with DEEP FORCED (disengagement impossible by construction), 400 artifacts:
        MI(features; goal) = -0.0000 nats at n = 100, 200, 300, 400
        max spread between the four learned goal columns = 0.0000  (bit-identical)

Learning ``A[0]`` attributes each observed feature to the *believed* (provenance, goal).
Under a uniform ``A[0]`` the goal posterior never leaves its prior, so every observation
deposits ~1/G of a count into all G goal columns and they converge to a common marginal.
That is an IDENTIFIABILITY deadlock, not a disengagement one, so no amount of forced
engagement rescues it.

So the Learner is seeded uninformative over PROVENANCE only:

    pA[0][:, p, g, DEEP] = strength * sig_i[g]     for EVERY provenance p
    pA[0][:, p, g, SKIM] = strength * synth

Every tier starts out looking CREATOR-like. The claim this encodes is C1's own: observers
share a likelihood *family* because they share a body plan; what they do not share, and must
learn, is which sources actually carry intent. The naive reader assumes everything they read
was meant, and has to learn otherwise. E7's question is correspondingly scoped — not "can you
learn intent-reading from scratch" but "can you learn which sources are hollow, from content
alone, without labels". Measured: this reaches MI 1.025 against the oracle ceiling of 1.089.
-----------------------------------------------------------------------------------------

PYMDP MECHANICS — two traps, both verified against inferactively-pymdp 1.0.2.

1. ``Agent.reset()`` executes ``self.A = utils.norm_dist_obj_arr(self.pA)`` whenever pA is
   set. This is GOOD for us: learning accumulates in pA and survives the per-artifact reset
   in ``rollout_observer`` with no surgery. But it also means A[1] (the kappa-precision
   signal likelihood) and A[2] (deterministic effort) would be OVERWRITTEN from pA on every
   reset. We pin them by setting pA[m] = FIXED_SCALE * A[m] for m in {1,2}: normalizing that
   returns A[m] exactly (verified drift after learning: 0.0), and the large concentration
   drives their parameter info-gain contribution to ~0 so they do not pollute the EFE.

2. ``calc_pA_info_gain`` sums over ALL modalities in pA, not just ``modalities_to_learn``.
   Same fix; that is the second reason FIXED_SCALE is large.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from . import constants as K
from .config import Config
from .generative_model import (GenerativeModel, cards, make_agent,
                               assert_A_column_stochastic)
from .metrics import (shannon_entropy, kl_divergence,
                      mutual_information_features_goal)

# Concentration placed on the non-learned modalities (1: signal, 2: effort). Large enough
# that normalizing returns them exactly and their parameter info gain is negligible.
FIXED_SCALE = 1.0e6


# --------------------------------------------------------------------------- #
# Prior construction.
# --------------------------------------------------------------------------- #
def build_pA(om: GenerativeModel, cfg: Config,
             prior_strength: float | None = None,
             oracle_A0: bool = False) -> object:
    """Dirichlet parameters over the observation model.

    ``prior_strength`` is the total concentration placed on each ``A[0]`` column: low = a
    weak prior that experience overrides quickly, high = a strong one. Under D2 it is
    calibrated for the EFE scale of the parameter info-gain term, and learning SPEED is set
    separately by ``lr_pA``, so that one knob does not silently do two jobs.

    ``oracle_A0=True`` seeds pA[0] from the TRUE A[0] at ``FIXED_SCALE`` — used by null N10,
    where a learner with lr_pA=0 must behave identically to a fixed-A observer.
    """
    cd = cards(cfg)
    strength = (float(cfg.get("learning.prior_strength", 1.0))
                if prior_strength is None else float(prior_strength))

    pA = utils.obj_array(3)
    if oracle_A0:
        pA[0] = FIXED_SCALE * np.asarray(om.A[0], dtype=float)
    else:
        # D1: uninformative over PROVENANCE, informed about the goal->feature family.
        pA0 = np.zeros((cd.features, cd.provenance, cd.goals, cd.attention))
        for p in range(cd.provenance):
            for g in range(cd.goals):
                pA0[:, p, g, K.DEEP] = strength * np.asarray(om.sig)[g]
                pA0[:, p, g, K.SKIM] = strength * np.asarray(om.noise_free_synth)
        pA[0] = pA0
    pA[1] = FIXED_SCALE * np.asarray(om.A[1], dtype=float)
    pA[2] = FIXED_SCALE * np.asarray(om.A[2], dtype=float)
    return pA


def make_learner_agent(om: GenerativeModel, D: object, cfg: Config,
                       kappa: float | None = None,
                       prior_strength: float | None = None,
                       lr_pA: float | None = None,
                       use_param_info_gain: bool | None = None,
                       oracle_A0: bool = False) -> Agent:
    """Construct a Learner observer: a pymdp Agent with Dirichlet learning on modality 0.

    ``use_param_info_gain`` gives the learner an epistemic drive toward things it does not
    yet have a model of. Under D2 it is calibrated, and if calibration fails in one pass it
    is switched off wholesale — under D1's seeding the learner already has a reason to engage
    GHOST content (it believes GHOST is CREATOR-like), so the term is not load-bearing for
    that motivation.
    """
    a = cfg.agent
    lr = float(cfg.get("learning.lr_pA", 1.0)) if lr_pA is None else float(lr_pA)
    pig = (bool(cfg.get("learning.use_param_info_gain", True))
           if use_param_info_gain is None else bool(use_param_info_gain))
    pA = build_pA(om, cfg, prior_strength=prior_strength, oracle_A0=oracle_A0)

    # THE SOLVER SWITCH REACHES THE LEARNER TOO (repair R-10). Six experiments were unreachable
    # under exact inference because ExactAgent refused Dirichlet learning rather than approximating
    # it, and two of those carry public claims. It now attributes counts under the exact joint's
    # marginals instead of the factorised ones: an exact E-step with the same conjugate M-step. The
    # parameter uncertainty is still handled the way pymdp handles it, and ``ExactAgent.update_A``
    # says so in as many words rather than letting the name overstate it.
    if bool(cfg.get("inference.exact", False)):
        from .exact import make_exact_agent
        agent = make_exact_agent(om, D, cfg).attach_pA(pA)
        agent.lr_pA = lr
        agent.modalities_to_learn = [K.M_FEATURES]
        agent.reset()
        return agent

    agent = Agent(
        A=om.A, pA=pA, B=om.B, C=om.C, D=D,
        control_fac_idx=[K.F_ATTENTION],
        policy_len=int(a.policy_len),
        inference_horizon=int(a.inference_horizon),
        use_utility=bool(a.use_utility),
        use_states_info_gain=bool(a.use_states_info_gain),
        use_param_info_gain=pig,
        lr_pA=lr,
        modalities_to_learn=[K.M_FEATURES],   # learn A[0] only; A[1]/A[2] are pinned
        action_selection=str(a.action_selection),
        gamma=float(a.gamma),
    )
    agent.reset()   # materialise A from pA before first use
    return agent


def is_learner(agent) -> bool:
    return getattr(agent, "pA", None) is not None


# --------------------------------------------------------------------------- #
# The learning step, with the mandatory invariant.
# --------------------------------------------------------------------------- #
def learn_step(agent, obs: list[int], cfg: Config) -> None:
    """One Dirichlet concentration update, with the V2 §3 invariant asserted.

        "every pA update leaves A column-stochastic. Assert after each update."

    Asserted unconditionally rather than behind a debug flag: a silently un-normalized A
    would corrupt every learned-model diagnostic downstream, and the check is cheap.
    """
    agent.update_A(obs)
    assert_A_column_stochastic(agent.A, cfg, where="pA update")


# --------------------------------------------------------------------------- #
# DEFERRED COMMITMENT — the estimator fix (V3 follow-up).
# --------------------------------------------------------------------------- #
def learn_deferred(agent, buffered: list[tuple[list[int], np.ndarray]], cfg: Config) -> None:
    """Commit a whole artifact's observations at once, using the RESOLVED posterior.

    -------------------------------------------------------------------------------------
    THE BIAS THIS REMOVES, AND WHY RAISING ``infer_steps`` IS NOT THE FIX.

    ``learn_step`` fires at every timestep with the posterior the observer holds AT THAT
    MOMENT. The observer's first DEEP observation arrives before the goal has been resolved —
    the free initial glance is synthetic and carries no goal information — so that
    observation's evidence is filed across goal columns in proportion to a nearly-uniform
    posterior. A fixed fraction of all evidence, about ``1/infer_steps``, is therefore
    misattributed, and the learned column comes out systematically FLATTER than the truth.

    Measured, one generation, d=0, forced DEEP, honest signal, f=0, as the contraction ratio
    r = KL(learned || uniform) / KL(true || uniform):

        infer_steps      2       4       6      12      24
        1 - r        0.147   0.140   0.078   0.035   0.014

    Because the fraction is per-ROLLOUT it does not shrink with corpus size, is common-mode
    across observers (so averaging cannot cancel it), and is only weakly reduced by
    engagement — which is exactly the set of invariances E12 and E14 measured, and which made
    the effect look structural.

    Raising ``infer_steps`` divides the bias; it does not remove it. Choosing the value at
    which the residue happens to clear a gate is choosing an operating point to pass one's own
    threshold — the failure decision D1 exists to prevent. So the estimator is fixed instead.

    THE FIX, and it introduces no new parameter. Inference over one artifact is an E-step and
    the Dirichlet update is an M-step; committing counts mid-E-step is the error. Deferred
    commitment buffers the artifact's observations, runs inference to the end, and only then
    deposits each observed feature — attributed under the FINAL posterior over the uncertain
    factors. There is no threshold to tune and no schedule to pick: the bias is absent at any
    ``infer_steps``, which is the property that makes a result reportable rather than tuned.

    ATTENTION IS KEPT PER-STEP, DELIBERATELY. Effort observations reveal the attention state
    exactly, so attention is not an uncertain factor and its posterior is already correct at
    the time each observation arrives. Re-attributing a SKIM observation under a final
    DEEP-dominated posterior would introduce a NEW misattribution in place of the one being
    removed. Only the genuinely uncertain factors — provenance and goal — take the resolved
    posterior.
    -------------------------------------------------------------------------------------

    ``buffered`` is a list of ``(obs, attention_marginal)`` in the order observed.
    """
    if not buffered:
        return

    def _as_obj_array(marginals) -> object:
        """pymdp indexes ``qs`` with a LIST of factor indices (``qs[A_factor_list[m]]``), which
        a plain Python list cannot do. The posterior must stay an object array."""
        out = utils.obj_array(len(marginals))
        for i, q in enumerate(marginals):
            out[i] = np.asarray(q, dtype=float).copy()
        return out

    resolved = [np.asarray(q, dtype=float).copy() for q in agent.qs]
    for obs, attention_marginal in buffered:
        marginals = [q.copy() for q in resolved]
        marginals[K.F_ATTENTION] = np.asarray(attention_marginal, dtype=float).copy()
        agent.qs = _as_obj_array(marginals)
        agent.update_A(obs)
    agent.qs = _as_obj_array(resolved)
    assert_A_column_stochastic(agent.A, cfg, where="deferred pA update")


# --------------------------------------------------------------------------- #
# Diagnostics on a learned A. E9's whole design rests on the first two.
# --------------------------------------------------------------------------- #
def column_kl(A_learned: object, A_true: object, provenance: int, goal: int,
              attention: int = K.DEEP) -> float:
    """KL(learned column || true column) — **SHAPE distortion**.

    E9's poisoning signature: hallucinated goal posteriors written into pA pull a column
    toward the wrong shape. Requires full support in the reference, which is what the D3b
    synth floor guarantees.
    """
    p = np.asarray(A_learned)[:, provenance, goal, attention]
    q = np.asarray(A_true)[:, provenance, goal, attention]
    return kl_divergence(p, q)


def column_entropy(A_learned: object, provenance: int, goal: int,
                   attention: int = K.DEEP) -> float:
    """H(learned column) — **FLATNESS**.

    E9's starvation signature: disengagement means no updates on genuine content, so the
    column never sharpens away from its prior and stays flat. Poisoning distorts shape;
    starvation flattens. These are distinguishable and the diagnostic must separate them.
    """
    return shannon_entropy(np.asarray(A_learned)[:, provenance, goal, attention])


def ghost_column_error(A_learned: object, synth: np.ndarray,
                       goal: int | None = None) -> float:
    """How far the learned GHOST/DEEP column is from the TRUE synthetic distribution (E7).

    This is E7's convergence measure: acquiring a clean GHOST column means learning that
    GHOST content is drawn from ``noise_free_synth`` irrespective of goal. Averaged over
    goals unless one is named.
    """
    A0 = np.asarray(A_learned)
    goals = range(A0.shape[2]) if goal is None else [int(goal)]
    return float(np.mean([kl_divergence(A0[:, K.GHOST, g, K.DEEP], np.asarray(synth))
                          for g in goals]))


def human_column_mi(A_learned: object, provenance: int = K.CREATOR) -> float:
    """MI(features; goal) of the learned HUMAN columns (E7).

    E7's prediction: without labels the learner folds synthetic features into its model of
    human intent, so the human columns acquire synthetic mass and LOSE goal-discriminability.
    This is the number that falls when that happens.
    """
    return mutual_information_features_goal(np.asarray(A_learned), provenance, K.DEEP)


@dataclass
class LearnedModelReport:
    """The full per-column panel, recorded by every learning experiment so E9's
    shape-vs-flatness diagnostic is a report rather than a separate measurement."""
    mi_by_tier: list
    kl_by_tier: list          # mean over goals of column_kl, per provenance (SHAPE)
    entropy_by_tier: list     # mean over goals of column_entropy, per provenance (FLATNESS)
    ghost_column_error: float
    creator_column_mi: float

    def flat(self, prefix: str = "") -> dict:
        out = {f"{prefix}ghost_col_err": self.ghost_column_error,
               f"{prefix}creator_mi": self.creator_column_mi}
        for p, name in enumerate(K.PROVENANCE_NAMES):
            out[f"{prefix}mi_{name}"] = self.mi_by_tier[p]
            out[f"{prefix}kl_{name}"] = self.kl_by_tier[p]
            out[f"{prefix}ent_{name}"] = self.entropy_by_tier[p]
        return out


def report_learned_model(A_learned: object, A_true: object, synth: np.ndarray,
                         cfg: Config) -> LearnedModelReport:
    """Compute the whole diagnostic panel for one learned A."""
    cd = cards(cfg)
    mi, kls, ents = [], [], []
    for p in range(cd.provenance):
        mi.append(mutual_information_features_goal(np.asarray(A_learned), p, K.DEEP))
        kls.append(float(np.mean([column_kl(A_learned, A_true, p, g) for g in range(cd.goals)])))
        ents.append(float(np.mean([column_entropy(A_learned, p, g) for g in range(cd.goals)])))
    return LearnedModelReport(
        mi_by_tier=mi, kl_by_tier=kls, entropy_by_tier=ents,
        ghost_column_error=ghost_column_error(A_learned, synth),
        creator_column_mi=mi[K.CREATOR],
    )
