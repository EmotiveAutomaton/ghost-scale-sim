"""V5 C4 — the latent goal: what the creator does not know about itself.

THE OMISSION, AND IT IS THE MOST CONSEQUENTIAL ITEM IN V5. The model assumes the creator
represents its own goal. The author's position is that a large part of what makes something art
is a goal that shapes the creator's behaviour WHILE THE CREATOR HAS NO ACCESS TO IT, and that
an observer can sometimes extract it anyway — reading someone better than they read themselves.

Three objects where the model had one:

    goal_latent    drives the creator's policy; NOT represented in the creator's own model
    goal_declared  what the creator believes and would report
    goal_inferred  what the observer recovers

-----------------------------------------------------------------------------------------
THE DECLARATION IS AN OBSERVATION, WITH ITS OWN PRECISION, AND THAT IS NOT A NEW PATTERN.

``goal_declared`` arrives as a fourth observation modality, and the observer's belief about how
reliable a self-report is has a precision ``rho`` exactly parallel to ``kappa`` on the
provenance channel. The world-side declaration process is separate and parameterised by the
true divergence rate, exactly as ``honesty`` is separate from ``kappa``. That split — what the
observer BELIEVES about a channel versus what the channel actually DOES — is the machinery the
trust exploit is built from, it is already validated, and C4 needs precisely it.

rho is deliberately below 1. An observer that fully trusted the declaration could never beat
the creator's self-model, which is the question C4 exists to ask.

-----------------------------------------------------------------------------------------
PRODUCTIVE CONFLICT, AND WHY WITHOUT IT C4's THIRD QUESTION IS MECHANICALLY DEAD.

C4 asks whether latent/declared divergence RAISES extractable depth. In a naive build it
cannot: emission depends only on ``goal_latent``, so a wrong declaration adds nothing to the
artifact and can only mislead a reader who trusts it. The question would be settled by the
construction rather than measured.

So the conflict is made productive. When latent != declared the creator pursues the latent goal
WHILE SUPPRESSING the declared goal's features — the artifact carries a signature that is
neither, which is more structure than either alone, and that extra structure is the TRACE.

THE TRAP THIS HAD TO AVOID, and it is the same one V5 §5 warns about for this exact layer: if
the trace's strength scaled with the divergence RATE, then "divergence increases depth" would
be true by construction and E33 would prove nothing. So the trace is BINARY IN FORM — present
when this artifact's latent and declared goals differ, absent when they agree — and its
strength is a fixed constant that does not depend on the rate. Whether more divergence yields
more recoverable depth is then an empirical question about how much of the trace survives
transmission and inference, which is what E33 measures and which can come out no.

-----------------------------------------------------------------------------------------
THREE CREATOR ARMS, NOT TWO, AND THE DECEPTIVE ARM IS WHY THE GENERATIVE ONE MEANS ANYTHING.

The spec asks for a generative control that "cannot produce the divergence by construction".
An arm that cannot produce the effect BECAUSE IT WAS DEFINED NOT TO is a definition presented
as a result. And "a generative model has no unconscious" is the wrong description: what a
prompted generative system lacks is not an unconscious but a CONSCIOUS — there is no
introspective self-model to report from, so it emits a plausible self-report produced by the
same process as its output.

    human_unaware  latent drives emission; declared may diverge; the conflict LEAVES A TRACE
    deceptive      declared is chosen to mislead; latent drives emission; NO trace
    generative     declared is CONFABULATED — sampled from the output distribution rather
                   than from what drove emission; NO trace

So all three arms can produce declared != latent, and the categorical claim is restated in a
form that is falsifiable rather than definitional:

    WHAT A GENERATIVE SYSTEM CANNOT PRODUCE IS NOT A DIVERGENCE.
    IT IS A DIVERGENCE THAT LEAVES A TRACE IN THE ARTIFACT.

The deceptive arm is the control that makes that testable: deception and unconsciousness both
yield declared != latent, and only unconsciousness is supposed to mark the work. If all three
arms separate on content alone, the measure is reading something other than the trace, and
E33's verdict says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as K
from .config import Config

# The fourth observation modality: what the creator says it was doing. One state per goal,
# plus NO_DECLARATION for artifacts that carry no self-report at all.
M_DECLARED = 3


@dataclass
class Declaration:
    latent: int
    declared: int          # -1 means no declaration was made
    diverged: bool
    arm: str

    @property
    def obs(self) -> int:
        return int(self.declared) if self.declared >= 0 else -1


def no_declaration_index(n_goals: int) -> int:
    return int(n_goals)


# --------------------------------------------------------------------------- #
# The world side: how each kind of creator comes by its self-report.
# --------------------------------------------------------------------------- #
def draw_declaration(arm: str, latent: int, n_goals: int, p_divergence: float,
                     rng: np.random.Generator,
                     emission_marginal: np.ndarray | None = None,
                     sig_true: np.ndarray | None = None) -> Declaration:
    """Produce ``goal_declared`` for one artifact under one creator arm.

    ``human_unaware`` — with probability ``p_divergence`` the creator's self-model names a
    DIFFERENT goal, drawn UNIFORMLY over the others. Uniform is the assumption-free choice and
    it is pre-registered as the primary; a systematic or flattering divergence is a separate
    arm and is deliberately not run here, because a systematic one is a second free parameter
    on a layer V5 §5 already calls the most tunable object in the spec.

    ``deceptive`` — the declaration is chosen to mislead at the same rate, so the DIVERGENCE
    STATISTICS ARE IDENTICAL to the unaware arm. That is the point of the arm: everything
    about the declaration channel matches, and only the artifact differs.

    ``generative`` — the declaration is CONFABULATED. It is not drawn from the latent goal at
    all, nor chosen against it; it is read off the artifact's own emission marginal, which is
    what "a self-report produced by the same process as the output" means mechanically. It is
    uncorrelated with what drove emission except through the artifact itself.
    """
    others = [g for g in range(n_goals) if g != int(latent)]
    if arm == "generative":
        if emission_marginal is None or sig_true is None:
            declared = int(rng.choice(n_goals))
        else:
            # Score the artifact's own output against each goal signature and sample from the
            # result: the model says what its output most looks like, which is exactly the
            # move a prompted system makes when asked what it was trying to do.
            scores = sig_true @ np.asarray(emission_marginal, dtype=float)
            p = scores / scores.sum()
            declared = int(rng.choice(n_goals, p=p))
        return Declaration(int(latent), declared, declared != int(latent), arm)

    if rng.random() < float(p_divergence):
        declared = int(rng.choice(others))
        return Declaration(int(latent), declared, True, arm)
    return Declaration(int(latent), int(latent), False, arm)


def conflict_chain(chain: np.ndarray, declared: int, n_subgoals: int,
                   suppression: float) -> np.ndarray:
    """The PLAN of a creator pursuing ``latent`` while avoiding looking like ``declared``.

    THE TRACE IS IN THE PLAN, NOT IN THE FEATURE HISTOGRAM, AND THAT WAS DECIDED BY
    MEASUREMENT. The obvious construction — multiply the latent signature by one minus the
    declared signature, pushing mass away from the declared goal's features — does nothing
    here, and it cannot. Each goal's signature carries 87% of its mass on its own feature pair
    and sits at floor (0.0094) on every other goal's pair, so the latent creator already has
    almost no mass where the declared goal peaks. Suppression cannot remove what is not there:
    the total mass moved is under 2% and vanishes on renormalisation.

    So the conflict acts where a creator's self-deception would actually show — in WHAT IT
    DOES, not in what it happens to emit. The creator AVOIDS ONE EXECUTION MODE, the one whose
    tail features would most read as the declared goal, and its plan routes around it. The mode
    is deleted from the chain and the remaining transitions are renormalised.

    That has three properties the feature version lacked:

      * it is VISIBLE. The mode marginal stops being uniform, and C1's whole construction is
        built so that a non-uniform mode marginal is exactly what a reader tracking order can
        detect. The trace lives in the depth channel, which is where C4's third question —
        does divergence raise extractable depth — has to live to be answerable at all.
      * it is IN FAMILY. The observer's templates still cover every mode; the creator simply
        never visits one. So this is a legible artifact with an odd shape, not out-of-family
        content, and the observer is not being asked to read something its hypothesis space
        cannot represent.
      * it is BINARY. The mode is avoided or it is not. ``suppression`` sets how completely,
        and it is a FIXED CONSTANT that does not depend on the divergence RATE — if it did,
        "divergence increases depth" would be true by construction and E33 would be measuring
        its own parameterisation. See the module docstring and V5 §5.
    """
    ch = np.array(chain, dtype=float, copy=True)
    avoid = int(declared) % int(n_subgoals)
    keep = 1.0 - float(suppression)
    # ONLY THE ROW IS SCALED. Scaling the ``avoid`` COLUMN as well looks symmetric and is a
    # no-op: the columns are renormalised immediately afterwards, so whatever that column was
    # multiplied by is divided straight back out. What actually suppresses the mode is
    # lowering the probability of ENTERING it from anywhere, which is the row.
    ch[avoid, :] *= keep
    ch = np.clip(ch, 1e-12, None)
    return ch / ch.sum(axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# The observer side: the declaration likelihood.
# --------------------------------------------------------------------------- #
def build_declaration_likelihood(n_goals: int, n_states: int, n_attention: int,
                                 rho: float, goal_of_state) -> np.ndarray:
    """A[3] — P(declared | hidden state), shape (n_goals + 1, provenance, states, attention).

    ``rho`` is the observer's precision on the self-report channel: how much it trusts a maker
    to know its own goal. At rho = 1 the declaration is taken as proof and the observer can
    never beat the self-model; at rho = 1/n_goals it is ignored entirely. It is the exact
    analogue of kappa on the provenance channel and it is a BELIEF, not the truth — the world's
    actual divergence rate is set separately, and the gap between them is where C4's
    interesting cases live.

    NO_DECLARATION is given its own outcome with likelihood independent of the goal, so an
    artifact that carries no self-report contributes nothing through this channel rather than
    contributing a diffuse hint about every goal.
    """
    none_idx = no_declaration_index(n_goals)
    A = np.zeros((n_goals + 1, K.CREATOR + 4, int(n_states), int(n_attention)))
    off = (1.0 - float(rho)) / max(n_goals - 1, 1)
    for s in range(int(n_states)):
        g = int(goal_of_state(s))
        row = np.full(n_goals + 1, 0.0)
        for gg in range(n_goals):
            row[gg] = float(rho) if gg == g else off
        row[none_idx] = 0.0
        A[:, :, s, :] = row[:, None, None]
    # NO_DECLARATION is a separate, goal-independent outcome. Modelled by giving every state a
    # small equal probability of saying nothing and renormalising, so the channel stays
    # column-stochastic without making silence informative.
    A[none_idx, :, :, :] = 1.0e-6
    A /= A.sum(axis=0, keepdims=True)
    return A


def observer_beats_self_model(latent_correct: np.ndarray,
                              declaration_correct: np.ndarray) -> dict:
    """Does the reader recover the latent goal more often than the creator's own report names
    it? The baseline MOVES with the divergence rate — at rate p the declaration is right
    (1 - p) of the time — so this is a comparison against the creator, not against a constant.
    """
    lat = float(np.mean(latent_correct))
    dec = float(np.mean(declaration_correct))
    return {"observer_recovers_latent": lat, "self_report_names_latent": dec,
            "margin": lat - dec, "observer_wins": bool(lat > dec)}
