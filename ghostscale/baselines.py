"""E21's comparison arms — the observers that are NOT the full active-inference model.

THE QUESTION. A referee will ask whether the active-inference apparatus is doing any work, or
whether a heuristic reproduces the same phenomena for a fraction of the machinery. Nobody has
answered it. V4.5 §2 says E21 "must be able to return 'the machinery is unnecessary'", and
that outcome goes in the first line of its section.

-----------------------------------------------------------------------------------------
THE FAIRNESS PROBLEM, WHICH IS THE WHOLE DESIGN.

Two of E21's measures are between-observer quantities. A deterministic heuristic run on a
fixed stimulus produces IDENTICAL output for every observer, so its between-observer
disagreement is exactly zero — and it "fails" the dissociation for a reason that has nothing
to do with theory of mind. Run that way, E21 is rigged, and the rigging runs in the direction
that flatters the framework.

So every arm here receives:

  * THE SAME OBSERVATION STREAM, literally. ``ObservationTape`` pre-draws the DEEP and SKIM
    feature each timestep would deliver, once per (artifact, observer), and every arm indexes
    into it according to the attention it chooses. Without the tape the arms would diverge
    after their first differing attention choice and would be reading different content.
  * THE SAME D-PRIOR DRAW as the full observer, used as its prior or its smoothing prior.
    This is the source of heterogeneity the V1 model has (null N3), and withholding it from
    the baselines would be withholding the only thing that lets them disagree.

Where an arm has a free parameter with no principled value (arm D's disengagement threshold,
arm E's stopping confidence), it is run across a GRID and credited if ANY setting reproduces
a signature. Maximally generous, for the same reason: the unwelcome outcome is that the
apparatus is unnecessary, so the benefit of the doubt belongs to the baselines.

-----------------------------------------------------------------------------------------
WHAT EACH ARM DELIBERATELY LACKS.

  B  Bayesian inverse planning, always DEEP.  Identical generative model and identical
     inference to arm A. The ONLY thing removed is the engagement policy. Whatever B
     reproduces is not attributable to expected free energy over attention.

  C  Label-truster.  Believes the declared tier; performs NO content inference at all. Its
     goal belief is its prior, forever. This is a weak arm and it is meant to be: it is the
     null hypothesis "the Ghost Scale label is the whole story". It should reproduce
     disengagement — which is cost-benefit arithmetic and needs no theory of mind — and
     nothing else.

  D  Effort-allocation heuristic.  Disengages when the raw observation stream stops
     surprising it, measured as the total-variation change in its own running feature
     histogram. Model-free: no goal, no creator, no policy. It has no EXPECTATION to be
     wrong about, which is why V4.5 §2 predicts it should fail E19's sustained-attention
     result specifically.

  E  No-ToM classifier.  Naive Bayes over features, with the feature-to-goal likelihood
     LEARNED BY COUNTING on a finite labelled training sample rather than derived from "a
     goal-directed agent emitted this". It never represents a creator, a policy, or an
     intention. It is the strongest baseline here and it is the one most likely to reproduce
     something: finite training data plus a product over many features gives confident,
     observer-specific answers on content that contains no goal, which is a mechanism for
     confident fabrication that requires no theory of mind at all.

     Per V4.5's design decision it carries a goal head as well as a provenance head. The
     dissociation is measured on GOALS, so a provenance-only classifier could not be scored
     on the decisive axis and arm E would have been excluded from the question it exists to
     answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as K
from .config import Config
from .environment import Artifact, Environment

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# The common stimulus.
# --------------------------------------------------------------------------- #
class ObservationTape:
    """Every feature observation an artifact could deliver, pre-drawn.

    Both attention levels are drawn for every timestep, so an arm that chooses DEEP at t and
    an arm that chooses SKIM at t are reading the same artifact rather than two samples of
    it. The tape is drawn once per (artifact, observer) from a dedicated stream, which also
    means an arm's own internal randomness cannot shift the content it is shown.
    """

    def __init__(self, env: Environment, artifact: Artifact, rng: np.random.Generator,
                 n_timesteps: int):
        self.artifact = artifact
        self.n_timesteps = int(n_timesteps)
        self.glance = int(env.sample_feature(artifact, K.SKIM, rng))
        self.deep = np.array([env.sample_feature(artifact, K.DEEP, rng)
                              for _ in range(self.n_timesteps)], dtype=int)
        self.skim = np.array([env.sample_feature(artifact, K.SKIM, rng)
                              for _ in range(self.n_timesteps)], dtype=int)

    def feature(self, t: int, attention: int) -> int:
        return int(self.deep[t] if attention == K.DEEP else self.skim[t])


class TapedEnvironment:
    """An ``Environment``-shaped facade that serves an ``ObservationTape``.

    Exists so arm A and arm B can run through the unmodified ``rollout_observer`` while
    reading the same tape as the heuristics. It deliberately implements only the three
    methods the rollout calls, so that any other use fails loudly rather than silently
    sampling fresh content.
    """

    def __init__(self, tape: ObservationTape):
        self.tape = tape
        self._t = 0
        self._glance_taken = False

    def sample_feature(self, artifact: Artifact, attention: int, rng) -> int:
        # The rollout's free initial glance is the only caller that reaches this directly.
        if not self._glance_taken:
            self._glance_taken = True
            return self.tape.glance
        return self.tape.feature(min(self._t, self.tape.n_timesteps - 1), attention)

    def effort_obs(self, attention: int) -> int:
        return K.HIGH_COST if attention == K.DEEP else K.LOW_COST

    def observation(self, artifact: Artifact, attention: int, rng) -> list[int]:
        t = min(self._t, self.tape.n_timesteps - 1)
        self._t += 1
        return [self.tape.feature(t, attention), artifact.declared_signal,
                self.effort_obs(attention)]


@dataclass
class BaselineResult:
    """The measurement plane every arm reports on, so the arms are commensurable."""
    final_goal_posterior: np.ndarray      # over the four REAL goals
    attention: np.ndarray                 # (T,) chosen attention per timestep
    engaged_fraction: float               # over the FREE steps only
    goal_posterior_trace: np.ndarray = None


def _engaged_fraction(attention: np.ndarray, forced_k: int) -> float:
    free = np.asarray(attention)[forced_k:]
    return float(np.mean(free == K.DEEP)) if free.size else float("nan")


# --------------------------------------------------------------------------- #
# Arm C — the label-truster.
# --------------------------------------------------------------------------- #
def run_label_truster(cfg: Config, tape: ObservationTape, artifact: Artifact,
                      d_prior: np.ndarray, alpha: np.ndarray, n_timesteps: int,
                      forced_k: int, engage_alpha_threshold: float = 0.5) -> BaselineResult:
    """Believes the declared tier. Performs no content inference whatsoever.

    Engagement rule: read the label, look up the intent transmission that tier claims, and
    engage if it is worth engaging with. Unsigned artifacts get the prior-weighted average
    over tiers, which is the only thing a label-truster can do without looking at content.

    The goal posterior NEVER MOVES. That is not an oversight; it is the arm. "No content
    inference" means exactly that, and an arm that reads content to form a goal belief would
    be arm E. The consequence — arm C cannot be confident and therefore cannot reproduce the
    dissociation — is a property of the hypothesis being tested, not a handicap imposed on it.
    """
    if artifact.declared_signal == K.UNSIGNED:
        believed_alpha = float(np.dot(np.asarray(d_prior[K.F_PROVENANCE], dtype=float),
                                      np.asarray(alpha, dtype=float)))
    else:
        # The signal indices coincide with the provenance tiers for the four signed values.
        believed_alpha = float(alpha[int(artifact.declared_signal)])

    engage = believed_alpha >= float(engage_alpha_threshold)
    attention = np.array([K.DEEP if (t < forced_k or engage) else K.SKIM
                          for t in range(n_timesteps)], dtype=int)
    post = np.asarray(d_prior[K.F_GOAL], dtype=float).copy()
    return BaselineResult(final_goal_posterior=post / post.sum(), attention=attention,
                          engaged_fraction=_engaged_fraction(attention, forced_k))


# --------------------------------------------------------------------------- #
# Arm D — the effort-allocation heuristic.
# --------------------------------------------------------------------------- #
def run_effort_heuristic(cfg: Config, tape: ObservationTape, artifact: Artifact,
                         d_prior: np.ndarray, n_features: int, n_timesteps: int,
                         forced_k: int, threshold: float,
                         patience: int = 3) -> BaselineResult:
    """Disengages when the observation stream stops surprising it. No goal model.

    The information proxy is deliberately model-free: the arm keeps a running histogram of
    the features it has seen and measures how much that histogram MOVES per observation, in
    total variation. Early on it moves a lot; once the stream is well characterised it stops
    moving, there is nothing further to learn from another look, and the arm stops paying.

    THIS IS THE ARM THE SPEC SINGLES OUT. Sustained expensive attention that never resolves
    is a prediction about an agent that keeps EXPECTING to learn about something specific and
    keeps being wrong. Arm D expects nothing in particular, so once the raw stream is
    characterised it has no reason to keep looking however foreign the content is. If it
    reproduces E19's sustained-attention result anyway, that says the phenomenon needs less
    machinery than claimed, and V4.5 §2 requires that be reported rather than explained.

    Its goal posterior is its prior, unmoved: "no goal model" is the definition of the arm.
    """
    counts = np.full(int(n_features), _EPS)
    prev = counts / counts.sum()
    attention = np.zeros(int(n_timesteps), dtype=int)
    stable = 0
    disengaged = False
    for t in range(int(n_timesteps)):
        if t < forced_k or not disengaged:
            attention[t] = K.DEEP
        else:
            attention[t] = K.SKIM
        counts[tape.feature(t, int(attention[t]))] += 1.0
        cur = counts / counts.sum()
        movement = 0.5 * float(np.sum(np.abs(cur - prev)))
        prev = cur
        if movement < float(threshold):
            stable += 1
            if stable >= int(patience) and t >= forced_k:
                disengaged = True
        else:
            stable = 0
    post = np.asarray(d_prior[K.F_GOAL], dtype=float).copy()
    return BaselineResult(final_goal_posterior=post / post.sum(), attention=attention,
                          engaged_fraction=_engaged_fraction(attention, forced_k))


# --------------------------------------------------------------------------- #
# Arm E — the no-ToM classifier.
# --------------------------------------------------------------------------- #
class NoToMClassifier:
    """Naive Bayes over features, with P(feature | goal) LEARNED BY COUNTING.

    What makes this "no theory of mind" is not that it lacks a likelihood — every classifier
    has one — but that it never represents a creator, a policy, or an intention. It has seen
    labelled examples and it has counted co-occurrences. It cannot ask what an agent was
    trying to do, and it has no notion that the artifact was PRODUCED by anything.

    Each observer trains on its OWN finite sample, which is where its heterogeneity comes
    from and which matches the full observer's per-observer variation in kind: two readers
    who have seen different work read the same artifact differently. The D prior supplies the
    class prior, so the two arms enter the comparison with the same prior beliefs.

    THE MECHANISM TO WATCH. Finite counts plus a product over many observed features gives
    confident, observer-specific answers even on content that carries no goal at all: each
    observer's sampling noise picks a different spurious winner and the product amplifies it.
    That is a route to confident fabrication that involves no theory of mind whatsoever, and
    if it reproduces E2's dissociation then the dissociation does not evidence a generative
    model of another agent.
    """

    def __init__(self, cfg: Config, env: Environment, n_goals: int, n_features: int,
                 rng: np.random.Generator, n_train: int, smoothing: float = 1.0):
        self.n_goals = int(n_goals)
        self.n_features = int(n_features)
        counts = np.full((self.n_goals, self.n_features), float(smoothing))
        per_goal = max(int(n_train) // self.n_goals, 1)
        for g in range(self.n_goals):
            art = Artifact(provenance=K.CREATOR, goal=g, declared_signal=K.SIG_CREATOR)
            for _ in range(per_goal):
                counts[g, env.sample_feature(art, K.DEEP, rng)] += 1.0
        self.log_lik = np.log(counts / counts.sum(axis=1, keepdims=True))

    def posterior(self, log_prior: np.ndarray, features: list[int]) -> np.ndarray:
        lp = np.asarray(log_prior, dtype=float).copy()
        for f in features:
            lp = lp + self.log_lik[:, int(f)]
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()


def run_no_tom_classifier(cfg: Config, clf: NoToMClassifier, tape: ObservationTape,
                          artifact: Artifact, d_prior: np.ndarray, n_timesteps: int,
                          forced_k: int, stop_confidence: float) -> BaselineResult:
    """Classify from features; stop looking once confident enough.

    The stopping rule is the classifier analogue of an engagement policy and it is NOT an
    expected-free-energy calculation: it is "I am sure enough, stop paying", which is what a
    deployed classifier with a cost per query would do. It gives arm E a defined engagement
    measure so it can be scored on E19's discriminator; without one, arm E could only be
    scored on half the experiment.
    """
    log_prior = np.log(np.clip(np.asarray(d_prior[K.F_GOAL], dtype=float), _EPS, None))
    attention = np.zeros(int(n_timesteps), dtype=int)
    seen: list[int] = []
    post = np.exp(log_prior - log_prior.max())
    post = post / post.sum()
    trace = np.zeros((int(n_timesteps), clf.n_goals))
    stopped = False
    for t in range(int(n_timesteps)):
        attention[t] = K.DEEP if (t < forced_k or not stopped) else K.SKIM
        if attention[t] == K.DEEP:
            seen.append(tape.feature(t, K.DEEP))
            post = clf.posterior(log_prior, seen)
        trace[t] = post
        if t >= forced_k - 1 and float(post.max()) >= float(stop_confidence):
            stopped = True
    return BaselineResult(final_goal_posterior=post, attention=attention,
                          engaged_fraction=_engaged_fraction(attention, forced_k),
                          goal_posterior_trace=trace)
