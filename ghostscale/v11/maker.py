"""The V11 world: a persistent maker with a value profile, and the exact reader over it.

Everything here is closed-form categorical probability — no pymdp agent. That is a decision, not
a shortcut (SPEC §1): with attention forced deep and every latent static, the goal posterior the
V1 agent computes IS exact categorical Bayes over the same A-matrix columns, and V-1 established
exact inference as the reference solver. What this omits — the attention policy, the metabolic
budget — makes this reader the CEILING, so every error floor V11 reports is a lower bound on what
a costlier reader would leave. An identifiability question wants the ceiling.

Two emitters, because the theory holds two accounts of what a value profile does and they cannot
both be right (SPEC §1):

    construction A   amplification.  g ~ Cat(w) per artifact; emission = tier(sig[g]).
                     The profile is visible only ACROSS artifacts.
    construction B   conjunctive satisfaction.  emission = tier(poe(w)), the weighted geometric
                     mean of the goal signatures. The profile is in every marginal.

The drive mask (S-14) rides on the theory's own mechanism: instruction amplifies
MULTIPLICATIVELY (w' ∝ w · exp(A·e_k)), so an absent channel (w[k] = 0) cannot be amplified and
the maker routes around it, while a present-but-unused trace is amplified to dominance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as K
from ..config import Config
from ..generative_model import (build_goal_signatures, build_noise_free_synth,
                                build_observer_signature, alpha_by_provenance)

_EPS = 1e-300


# --------------------------------------------------------------------------- #
# Profiles.
# --------------------------------------------------------------------------- #
def profile_family(ng: int = 4) -> dict[str, np.ndarray]:
    """The named finite family (SPEC §1). The finiteness is load-bearing: it is the
    convergent-midbrains bound made literal, and removing it is an experimental arm."""
    assert ng == 4, "the family is written for the four-goal world"
    fam = {"uniform": np.full(4, 0.25)}
    for k in range(4):
        w = np.full(4, 0.10)
        w[k] = 0.70
        fam[f"peaked_{k}"] = w
    fam["bimodal"] = np.array([0.40, 0.40, 0.10, 0.10])
    for name, w in fam.items():
        assert abs(w.sum() - 1.0) < 1e-12, f"profile {name} must be a distribution"
    return fam


def random_family(size: int, ng: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The unbounded-family arm: Dirichlet(1) draws, truth NOT included. Removing the bounded
    family is removing the convergent-midbrains assumption; the L1 asymptote gap against the
    bounded arm is that assumption's measured value (criterion C2)."""
    return {f"rand_{i}": rng.dirichlet(np.ones(ng)) for i in range(int(size))}


def amplify(w: np.ndarray, k: int, a_amp: float) -> np.ndarray:
    """Instructed amplification, multiplicative in the component (SPEC §1). w[k] = 0 stays 0:
    attention can only amplify a drive that exists. That single line is S-14's mechanism."""
    w2 = np.asarray(w, dtype=float) * np.exp(a_amp * np.eye(len(w))[k])
    s = w2.sum()
    assert s > 0, "a maker with no drives at all cannot produce"
    return w2 / s


# --------------------------------------------------------------------------- #
# The world.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MakerWorld:
    sig: np.ndarray            # (goals, features) — the shared latent signatures
    synth: np.ndarray          # (features,) — the frozen structured synthetic distribution
    alpha: np.ndarray          # (provenance,) — tier transmission


def build_maker_world(cfg: Config) -> MakerWorld:
    sig = build_goal_signatures(cfg)
    synth = build_noise_free_synth(cfg)
    alpha = alpha_by_provenance(cfg)
    return MakerWorld(sig=sig, synth=synth, alpha=alpha)


def tier_mix(world: MakerWorld, dist: np.ndarray, tier: int) -> np.ndarray:
    """The channel: alpha·dist + (1−alpha)·synth — V1's A0 composition, unchanged."""
    a = float(world.alpha[tier])
    out = a * np.asarray(dist, dtype=float) + (1.0 - a) * world.synth
    return out / out.sum()


def poe(sig: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted geometric mean of the goal signatures — construction B's emission.
    'Every drive partially satisfied at once, rather than served one at a time.'"""
    w = np.asarray(w, dtype=float)
    logp = w @ np.log(np.maximum(sig, _EPS))
    p = np.exp(logp - logp.max())
    return p / p.sum()


def commissioned_emission(world: MakerWorld, w: np.ndarray, k: int, tier: int,
                          a_amp: float, lam: float) -> np.ndarray:
    """S-14's commissioned artifact: a compliance channel carrying the instructed surface, and a
    pursuit channel carrying HOW — the amplified standing drives. Both maker types deliver the
    surface; they differ only in the pursuit."""
    how = poe(world.sig, amplify(w, k, a_amp))
    dist = lam * world.sig[k] + (1.0 - lam) * how
    return tier_mix(world, dist, tier)


# --------------------------------------------------------------------------- #
# Artifacts.
# --------------------------------------------------------------------------- #
def draw_artifact(dist: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(dist), size=int(n_steps), p=np.asarray(dist, dtype=float))


def maker_artifacts_A(world: MakerWorld, w: np.ndarray, tier: int, n_artifacts: int,
                      n_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Construction A: one goal drawn per artifact from the standing profile.
    Returns (artifacts (n, steps), goals (n,))."""
    goals = rng.choice(len(w), size=int(n_artifacts), p=np.asarray(w, dtype=float))
    arts = np.stack([draw_artifact(tier_mix(world, world.sig[g], tier), n_steps, rng)
                     for g in goals])
    return arts, goals


def maker_artifacts_B(world: MakerWorld, w: np.ndarray, tier: int, n_artifacts: int,
                      n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Construction B: every artifact emits the conjunction. Returns (n, steps)."""
    dist = tier_mix(world, poe(world.sig, w), tier)
    return np.stack([draw_artifact(dist, n_steps, rng) for _ in range(int(n_artifacts))])


# --------------------------------------------------------------------------- #
# The exact reader.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reader:
    """The reader's own generative objects. d = 0 is the expert (sig_r = sig_true); d > 0
    perturbs the signatures through the C1 machinery — the wrong-expertise arm."""
    sig_r: np.ndarray
    synth: np.ndarray
    alpha: np.ndarray


def build_reader(world: MakerWorld, cfg: Config, d: float = 0.0,
                 rng: np.random.Generator | None = None) -> Reader:
    sig_r = (world.sig.copy() if d <= 0.0
             else build_observer_signature(world.sig, float(d),
                                           rng if rng is not None
                                           else np.random.default_rng(0)))
    return Reader(sig_r=sig_r, synth=world.synth, alpha=world.alpha)


def goal_loglik(reader: Reader, artifact: np.ndarray, tier: int) -> np.ndarray:
    """log P(artifact | goal g), the forced-deep exact likelihood over the reader's own
    signatures. Shape (goals,)."""
    ng = reader.sig_r.shape[0]
    a = float(reader.alpha[tier])
    out = np.empty(ng)
    for g in range(ng):
        p = a * reader.sig_r[g] + (1.0 - a) * reader.synth
        p = p / p.sum()
        out[g] = float(np.log(np.maximum(p[artifact], _EPS)).sum())
    return out


def profile_loglik_A(reader: Reader, artifacts: np.ndarray, tier: int,
                     family: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-profile CUMULATIVE log-likelihood after each artifact, under construction A:
    log P(a_1..a_n | w) = sum_a log sum_g w[g] P(a | g). Returns name -> (n_artifacts,) of
    cumulative log-liks, so a convergence curve is one pass over prefixes."""
    per_artifact = np.stack([goal_loglik(reader, a, tier) for a in artifacts])   # (n, goals)
    out = {}
    for name, w in family.items():
        m = per_artifact.max(axis=1, keepdims=True)
        lik = np.log(np.maximum((np.exp(per_artifact - m) * w[None, :]).sum(axis=1),
                                _EPS)) + m[:, 0]
        out[name] = np.cumsum(lik)
    return out


def profile_loglik_B(reader: Reader, artifacts: np.ndarray, tier: int,
                     family: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cumulative log-likelihood under construction B: the profile is in the emission itself."""
    out = {}
    for name, w in family.items():
        p = tier_mix_reader(reader, poe(reader.sig_r, w), tier)
        logp = np.log(np.maximum(p, _EPS))
        out[name] = np.cumsum(np.array([float(logp[a].sum()) for a in artifacts]))
    return out


def tier_mix_reader(reader: Reader, dist: np.ndarray, tier: int) -> np.ndarray:
    a = float(reader.alpha[tier])
    out = a * np.asarray(dist, dtype=float) + (1.0 - a) * reader.synth
    return out / out.sum()


def posterior_from_logliks(cumlik: dict[str, np.ndarray], at: int) -> dict[str, float]:
    """Normalised posterior over the family after ``at`` artifacts (uniform prior over the
    family — the family IS the prior)."""
    names = list(cumlik)
    v = np.array([cumlik[n][at - 1] for n in names])
    v = np.exp(v - v.max())
    v = v / v.sum()
    return dict(zip(names, v))


def readout(post: dict[str, float], family: dict[str, np.ndarray],
            true_w: np.ndarray) -> tuple[str, float]:
    """(argmax profile name, L1 between the posterior-mean profile and the truth)."""
    best = max(post, key=post.get)
    mean_w = np.sum([p * family[n] for n, p in post.items()], axis=0)
    return best, float(np.abs(mean_w - np.asarray(true_w)).sum())
