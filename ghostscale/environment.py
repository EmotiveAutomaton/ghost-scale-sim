"""The environment — the true world the observer is embedded in (Spec §3.3, §4).

Kept strictly separate from the observer's beliefs:
    * signal emission here is A1_generative, parameterized by ``honesty`` and
      ``signing_rate`` — NOT the observer's A1_observer (whose precision is κ).
    * feature emission uses the TRUE intent-transmission coefficients α[p] and real
      creator policies; the observer's A[0] is its *model* of this process.

The world and the observer's model are allowed to differ, but only deliberately, and
every deliberate mismatch is behind a named flag (Spec §4.3). The one such flag here is
``ghost_pure_synth``: GHOST artifacts are pure noise_free_synth (Spec §4.2, "the
synthetic generator samples features from noise_free_synth directly"), whereas the
observer's A[0] GHOST column grants a residual α[GHOST]=0.05 goal-dependence. Because
α[GHOST] is tiny, the realized GHOST distribution still matches A[0] within tolerance
(verified in ``creators.verify_empirical_matches_A0``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as K
from .config import Config
from .creators import HumanCreator, build_creator_bank
from .generative_model import GenerativeModel, cards, alpha_by_provenance


@dataclass
class Artifact:
    """One artifact the observer inspects. ``declared_signal`` is emitted once (a fixed
    provenance tag) and seen by the observer at every timestep."""
    provenance: int
    goal: int
    declared_signal: int


def emit_signal(rng: np.random.Generator, true_provenance: int,
                honesty: float, signing_rate: float) -> int:
    """A1_generative — the true signal-emission process (Spec §3.3).

        with prob (1 - signing_rate):  UNSIGNED
        otherwise:
            with prob honesty:        signal matching true provenance
            with prob (1 - honesty):  SIG_CREATOR   (the bad actor's move)

    ``signing_rate`` may be PER-PROVENANCE (see ``Environment.signing_rate_for``), which is
    what lets E16 model a disclosure regime that labels synthetic content only.
    """
    if rng.random() >= signing_rate:
        return K.UNSIGNED
    if rng.random() < honesty:
        return K.TRUTHFUL_SIGNAL[true_provenance]
    return K.SIG_CREATOR


class Environment:
    """Generates the observations an observer receives, given an artifact and the
    observer's chosen attention."""

    def __init__(self, cfg: Config, gm: GenerativeModel,
                 rng_world: np.random.Generator,
                 honesty: float | None = None,
                 signing_rate: float | None = None,
                 alpha_overrides: dict | None = None,
                 alpha_permutation: list[int] | None = None,
                 ghost_pure_synth: bool = True,
                 creator_bank: dict[int, HumanCreator] | None = None,
                 signing_rate_by_provenance: dict[int, float] | None = None):
        self.cfg = cfg
        self.gm = gm
        self.cd = cards(cfg)
        self.honesty = float(cfg.signal_model.honesty) if honesty is None else float(honesty)
        self.signing_rate = (float(cfg.signal_model.signing_rate)
                             if signing_rate is None else float(signing_rate))
        # E16: a labelling regime may cover tiers unequally — e.g. "disclose synthetic content"
        # labels GHOST at rate p and human work not at all. None = the uniform V1/V2 behaviour.
        self.signing_rate_by_provenance = (
            None if signing_rate_by_provenance is None
            else {int(k): float(v) for k, v in signing_rate_by_provenance.items()})
        # TRUE alpha (may be overridden/permuted for nulls N1/N4, matching the model build).
        self.alpha = alpha_by_provenance(cfg, overrides=alpha_overrides,
                                         permutation=alpha_permutation)
        self.ghost_pure_synth = bool(ghost_pure_synth)
        self.synth = np.asarray(gm.noise_free_synth, dtype=float)
        self.creators = creator_bank if creator_bank is not None else build_creator_bank(cfg, gm)

    # -- artifact construction ------------------------------------------------
    def signing_rate_for(self, provenance: int) -> float:
        """The labelling coverage this tier actually receives."""
        if self.signing_rate_by_provenance is None:
            return self.signing_rate
        return self.signing_rate_by_provenance.get(int(provenance), self.signing_rate)

    def make_artifact(self, provenance: int, goal: int,
                      rng: np.random.Generator) -> Artifact:
        signal = emit_signal(rng, provenance, self.honesty,
                             self.signing_rate_for(provenance))
        return Artifact(provenance=int(provenance), goal=int(goal), declared_signal=signal)

    # -- observation emission -------------------------------------------------
    def sample_feature(self, artifact: Artifact, attention: int,
                       rng: np.random.Generator) -> int:
        """Emit one artifact_features observation under the observer's ``attention``.

        SKIM never resolves goals -> pure synth. DEEP: with prob α[p] a feature from the
        creator's policy (goal-laden), else from synth. GHOST is pure synth (§4.2)."""
        nf = self.cd.features
        if attention == K.SKIM:
            return int(rng.choice(nf, p=self.synth))
        # DEEP
        if artifact.provenance == K.GHOST and self.ghost_pure_synth:
            return int(rng.choice(nf, p=self.synth))
        a = self.alpha[artifact.provenance]
        if rng.random() < a:
            return int(self.creators[artifact.goal].sample_features(1, rng)[0])
        return int(rng.choice(nf, p=self.synth))

    def effort_obs(self, attention: int) -> int:
        return K.HIGH_COST if attention == K.DEEP else K.LOW_COST

    def observation(self, artifact: Artifact, attention: int,
                    rng: np.random.Generator) -> list[int]:
        """Full observation vector [features, signal, effort] for one timestep."""
        return [self.sample_feature(artifact, attention, rng),
                artifact.declared_signal,
                self.effort_obs(attention)]

    # -- corpus draw (E6) -----------------------------------------------------
    def draw_corpus(self, n_artifacts: int, contamination: float,
                    creator_goals: np.ndarray, rng: np.random.Generator) -> list[Artifact]:
        """A corpus of N artifacts: a fraction ``contamination`` are GHOST synthetics, the
        rest CREATOR artifacts whose goals are drawn from the population distribution (via
        sampled ``creator_goals``). BOTH classes carry their honest provenance signal at the
        environment's ``signing_rate`` (SIG_GHOST / SIG_CREATOR when signed, else UNSIGNED).

        This is what makes ``signing_rate`` the lever in E6: at a high signing rate a trusting
        (high-kappa) aggregator can exclude synthetics via the signal; at a low signing rate it
        is left to content alone and is the most corruptible."""
        artifacts = []
        for _ in range(n_artifacts):
            if rng.random() < contamination:
                art = self.make_artifact(provenance=K.GHOST,
                                         goal=int(rng.integers(self.cd.goals)), rng=rng)
            else:
                goal = int(rng.choice(creator_goals))
                art = self.make_artifact(provenance=K.CREATOR, goal=goal, rng=rng)
            artifacts.append(art)
        return artifacts
