"""Sequential Monte Carlo over evolving maker-state hypotheses (spec §3.4.7).

Why a particle filter belongs in this program at all: a staged reader commits and cannot revise,
an exact reader revises perfectly but only where the grid can be enumerated, and the interesting
middle is a reader that carries a *population* of hypotheses and can lose the right one. Particle
impoverishment is not a nuisance here -- it is the failure mode attack X22 targets, and card M06
measures how fast a filter recovers from an early wrong commitment.

The filter is deliberately plain: multinomial resampling on a low effective-sample-size trigger,
an optional jitter move over the drifting component, and no adaptive tuning. Every likelihood
evaluation debits the caller's budget, because the whole point of the tournament is that a method
which searches harder must pay for it (spec §3.5).
"""
from __future__ import annotations

import numpy as np

from . import common as C
from .ontology import COMPONENTS

DEFAULT_N = 240
ESS_FRACTION = 0.5


class ParticleFilter:
    """Particles are latent triples; weights are normalized importance weights."""

    def __init__(self, F, world, n_particles: int = DEFAULT_N, rng=None, jitter: float = 0.0,
                 model=None, budget=None):
        self.F, self.world = F, world
        self.model = model if model is not None else world
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n = int(n_particles)
        self.jitter = float(jitter)
        self.budget = budget
        flat = np.asarray(self.model.prior, float).ravel()
        flat = flat / flat.sum()
        idx = self.rng.choice(flat.size, size=self.n, p=flat)
        self.particles = np.array(np.unravel_index(idx, self.model.prior.shape)).T
        self.logw = np.zeros(self.n)
        self.history = []
        if budget is not None:
            budget.prop(self.n)

    # -- diagnostics ---------------------------------------------------------------------- #
    def weights(self) -> np.ndarray:
        return C.softmax(self.logw)

    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / np.maximum((w ** 2).sum(), 1e-300))

    def unique(self) -> int:
        return int(len({tuple(p) for p in self.particles}))

    def posterior(self) -> np.ndarray:
        """Particle approximation projected back onto the grid, for comparison with exact."""
        out = np.zeros(self.model.prior.shape)
        w = self.weights()
        for p, wi in zip(self.particles, w):
            out[tuple(p)] += wi
        return C.normalize(out.ravel()).reshape(out.shape)

    # -- the filter ----------------------------------------------------------------------- #
    def step(self, ep, t: int) -> None:
        """Absorb the observations of step ``t`` (a one-step slice, not a prefix)."""
        F, w = self.F, self.model
        for i, part in enumerate(self.particles):
            self.logw[i] += F.step_loglik(w, tuple(part), ep, t, self.budget)
        if self.jitter > 0:
            self._jitter()
        e = self.ess()
        self.history.append({"t": int(t), "ess": e, "unique": self.unique()})
        if e < ESS_FRACTION * self.n:
            self.resample()

    def resample(self) -> None:
        w = self.weights()
        idx = self.rng.choice(self.n, size=self.n, p=w)
        self.particles = self.particles[idx]
        self.logw = np.zeros(self.n)
        if self.budget is not None:
            self.budget.prop(self.n)

    def _jitter(self) -> None:
        """Let the foreground goal move: the drifting component of the maker state.

        Without this the filter cannot represent a goal switch at all, and M06's recovery question
        would be answered by construction rather than by the algorithm.
        """
        n_g = self.model.prior.shape[1]
        move = self.rng.random(self.n) < self.jitter
        if move.any():
            self.particles[move, 1] = self.rng.integers(0, n_g, size=int(move.sum()))

    def run(self, ep, upto: int) -> np.ndarray:
        for t in range(upto):
            self.step(ep, t)
        return self.posterior()


def divergence_from_exact(pf_post: np.ndarray, exact_post: np.ndarray) -> dict:
    """What M01 reports: how far the approximation sits from the answer that is not approximate."""
    return {"kl_exact_to_approx": C.kl(exact_post.ravel(), pf_post.ravel()),
            "total_variation": C.tv(exact_post.ravel(), pf_post.ravel()),
            "js": C.js(exact_post.ravel(), pf_post.ravel()),
            "max_abs": float(np.abs(exact_post - pf_post).max())}


def impoverishment(pf: ParticleFilter) -> dict:
    return {"unique_particles": pf.unique(), "n_particles": pf.n,
            "unique_fraction": float(pf.unique() / max(pf.n, 1)),
            "final_ess": pf.ess(), "ess_trace": [h["ess"] for h in pf.history]}


def recovery_curve(F, world, ep, upto: int, wrong: tuple, n_particles: int = DEFAULT_N,
                   rng=None, jitter: float = 0.05, budget=None) -> dict:
    """Seed every particle on a *wrong* committed hypothesis and watch whether the filter comes
    back. Reported as the number of steps until the true triple's mass first exceeds 0.5, and the
    mass it reaches by the end. ``None`` means it never recovered."""
    rng = rng if rng is not None else np.random.default_rng(0)
    pf = ParticleFilter(F, world, n_particles, rng, jitter, budget=budget)
    pf.particles = np.tile(np.array(wrong, int), (pf.n, 1))
    pf._jitter()
    truth = ep.latent.triple()
    trace, half_life = [], None
    for t in range(upto):
        pf.step(ep, t)
        m = float(pf.posterior()[truth])
        trace.append(m)
        if half_life is None and m > 0.5:
            half_life = t + 1
    return {"recovery_step": half_life, "final_true_mass": trace[-1] if trace else float("nan"),
            "trace": trace, "seeded_wrong": list(wrong), "truth": list(truth)}
