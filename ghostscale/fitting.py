"""Parameter estimation by exact likelihood. The capability P-1 needs and the project did not have.

WHY THIS FILE HAD TO BE WRITTEN BEFORE P-1 COULD RUN.

Parameter recovery is the routine first check in computational cognitive modelling: generate data from
the model at known parameter values, then fit the model to that data and ask whether the fit returns
what you put in. The diagnostics spec asks for it on four parameters, and phrases the estimation step
as "run the model's own inference on those observations".

That phrasing works for exactly one of the four. **mu** is a hidden state; the observer carries a
posterior over it and the posterior mean is an estimate. **kappa**, **omega** and **theta** are not
hidden states. The observer has no posterior over any of them, nothing in the codebase produced an
estimate of any of them, and "the model's own inference" does not name an estimator for them.

So they are estimated the standard way instead: **maximise the likelihood of a fixed dataset over a
grid of candidate values**, using a model that does contain the parameter. Three different datasets
and three different likelihoods, because the three parameters enter the model at three different
points:

  kappa  enters the OBSERVER'S likelihood over the provenance signal. Estimated from the observation
         tape by replaying it through observers built at each candidate kappa and reading off the
         exact sequence log-evidence. This is the cleanest of the three: the parameter is inside the
         model whose likelihood is being evaluated.

  omega  enters the WORLD's foreign signature family and appears nowhere in the observer's model at
         all. Estimated by an ANALYST who holds the correct generative model: the DEEP feature draws
         are i.i.d. from sig_foreign(omega)[k] for an unknown foreign goal k, so the likelihood is a
         uniform mixture over k, maximised over omega.

  theta  enters neither likelihood. It gates what the observer CARRIES FORWARD between encounters,
         so it is invisible within a single artifact and only shows up in the trajectory of the
         prior across a sequence. Estimated by matching the observed mean prior drift.

**THE DISTINCTION THIS FORCES, AND IT IS NOT A TECHNICALITY.** For mu, recovery asks whether the
reader inside the model can identify the value. For the other three it asks whether an ideal analyst
holding the correct model could. Those are different questions and the second is strictly easier. For
omega the gap between them IS the version 4 reframe: the whole point of goal-foreign content is that
the observer cannot represent it, so a well-recovered omega would say the axis is identifiable in
principle while remaining invisible to every reader in every experiment. Every omega number this
module produces is reported with that attached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from .config import Config


# --------------------------------------------------------------------------- #
# Tapes.
# --------------------------------------------------------------------------- #
@dataclass
class Tape:
    """A fixed record of what a reader saw and did, so a candidate model can be scored on it.

    The actions are recorded and imposed on replay rather than re-chosen. A candidate parameter
    changes what the agent would decide, so letting it decide would score two different datasets
    against each other and the likelihood ratio would mean nothing.
    """
    steps: list = field(default_factory=list)      # [(observation, action), ...]
    features: list = field(default_factory=list)   # the DEEP feature indices only
    truth: dict = field(default_factory=dict)      # what generated it, for scoring


def record_tape(env, artifact, cfg: Config, rng: np.random.Generator, n_timesteps: int,
                attention: int = K.DEEP, initial_glance: bool = False) -> Tape:
    """Generate an observation tape at a known truth, with attention imposed.

    Attention is imposed rather than chosen so that the tape does not depend on the agent at all.
    That makes one tape scoreable by every candidate model, which is what a likelihood profile needs.
    """
    action = np.zeros(3)
    action[K.F_ATTENTION] = attention
    tape = Tape(truth={"provenance": artifact.provenance, "goal": artifact.goal,
                       "declared_signal": artifact.declared_signal,
                       "foreign_goal": artifact.foreign_goal, "attention": attention})
    if initial_glance:
        glance = [env.sample_feature(artifact, K.SKIM, rng), artifact.declared_signal, K.LOW_COST]
        tape.steps.append((glance, None))
    for _ in range(int(n_timesteps)):
        obs = env.observation(artifact, attention, rng)
        tape.steps.append((obs, action.copy()))
        if attention == K.DEEP:
            tape.features.append(int(obs[K.M_FEATURES] if hasattr(K, "M_FEATURES") else obs[0]))
    return tape


# --------------------------------------------------------------------------- #
# Grid maximisation, with the diagnostics a profile can supply and a point estimate cannot.
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    estimate: float
    grid: np.ndarray
    profile: np.ndarray                # log-likelihood at each grid point
    curvature: float = float("nan")     # second difference at the peak; flat means unidentifiable
    interval: tuple = (float("nan"), float("nan"))

    @property
    def is_flat(self) -> bool:
        """A profile with no curvature at its peak cannot locate the parameter, whatever it returns.

        This is the diagnostic a point estimate hides: an argmax always returns something, and a
        likelihood that is flat across the whole grid returns whichever cell won by rounding.
        """
        return not np.isfinite(self.curvature) or abs(self.curvature) < 1e-9


def fit_on_grid(grid, score) -> Fit:
    """Maximise ``score(value)`` over ``grid`` and report the whole profile.

    The interval is the set of grid points within 2 log-likelihood units of the peak, which is the
    conventional likelihood interval and is reported as an interval rather than a width so that a
    profile that never drops is visible as one spanning the entire grid.
    """
    g = np.asarray(grid, dtype=float)
    ll = np.array([float(score(v)) for v in g])
    best = int(np.nanargmax(ll))
    curvature = float("nan")
    if 0 < best < len(g) - 1:
        h1, h2 = g[best] - g[best - 1], g[best + 1] - g[best]
        if h1 > 0 and h2 > 0:
            curvature = float((ll[best - 1] - 2 * ll[best] + ll[best + 1]) / (0.5 * (h1 + h2)) ** 2)
    inside = g[ll >= ll[best] - 2.0]
    return Fit(estimate=float(g[best]), grid=g, profile=ll, curvature=curvature,
               interval=(float(inside.min()), float(inside.max())) if inside.size else
               (float("nan"), float("nan")))


# --------------------------------------------------------------------------- #
# kappa — from the observation tape, through the observer's own model.
# --------------------------------------------------------------------------- #
def fit_kappa(tape: Tape, cfg: Config, grid, D=None) -> Fit:
    """Estimate kappa by replaying ``tape`` through observers built at each candidate value.

    The prior D is held FIXED across candidates. Rebuilding it per candidate would draw fresh
    variates and add a per-candidate perturbation to the likelihood that has nothing to do with
    kappa, which is the kind of confound this project keeps finding.
    """
    from .exact import make_exact_agent
    from .generative_model import build_D, build_shared_model

    if D is None:
        D = build_D(cfg, np.random.default_rng(0))

    def score(kappa: float) -> float:
        c = cfg.copy()
        c.set("signal_model.kappa", float(kappa))
        c.set("inference.exact", True)
        gm = build_shared_model(c, kappa=float(kappa))
        agent = make_exact_agent(gm, D, c)
        return agent.replay(tape.steps)

    return fit_on_grid(grid, score)


# --------------------------------------------------------------------------- #
# omega — from the feature draws, through an analyst's model.
# --------------------------------------------------------------------------- #
def fit_omega(features, cfg: Config, grid, foreign_seed: int | None = None) -> Fit:
    """Estimate omega from DEEP feature draws off a foreign artifact of unknown foreign goal.

    The likelihood is a uniform mixture over which foreign goal produced the artifact, because the
    analyst is not told:

        L(omega) = log sum_k (1/K) prod_t sig_foreign(omega)[k][f_t]

    Computed in log space with a log-sum-exp over k. The analyst holds the correct family, including
    the same drawn foreign basis, so this is the most favourable case: any failure to recover omega
    here is structural rather than a shortage of information.
    """
    from . import foreign as FN

    f = np.asarray(features, dtype=int)
    sig_true = FN.build_human_signatures(cfg)
    basis = FN.build_foreign_basis(cfg, int(cfg.get("v4.foreign.draw_seed", 41))
                                   if foreign_seed is None else int(foreign_seed))
    counts = np.bincount(f, minlength=sig_true.shape[1]).astype(float)

    def score(omega: float) -> float:
        fam = FN.build_foreign_signatures(sig_true, basis, float(np.clip(omega, 0.0, 1.0)))
        per_k = np.array([float(np.sum(counts * np.log(np.clip(fam[k], 1e-300, None))))
                          for k in range(fam.shape[0])])
        m = per_k.max()
        return float(m + np.log(np.mean(np.exp(per_k - m))))

    return fit_on_grid(grid, score)


# --------------------------------------------------------------------------- #
# theta — from the trajectory of the carried-forward prior.
# --------------------------------------------------------------------------- #
def fit_lambda_from_drift(observed_drift: float, posteriors_by_encounter, prior0,
                          value_prior, theta_base: float, eta: float, grid) -> Fit:
    """Estimate the gate's sharpness from how far the prior moved over a sequence of encounters.

    theta appears in no likelihood, because it acts on what the observer carries forward rather than
    on what it perceives. So it is estimated by REPLAYING THE UPDATE RULE at each candidate lambda on
    the observed sequence of posteriors, and matching the resulting prior drift to the observed one.
    The score is the negative squared error, so ``fit_on_grid`` maximises it like any other.

    This makes the estimator exact in the absence of noise, which means the informative output is not
    the estimate but the PROFILE: if drift is insensitive to lambda over some range, the profile is
    flat there and the parameter is unidentifiable from this observable however precisely it is
    measured.
    """
    from .metrics import kl_divergence, normalize
    from .v4_5_model import theta_gate

    def replay(lam: float) -> float:
        prior = np.asarray(prior0, dtype=float).copy()
        for post in posteriors_by_encounter:
            th = theta_gate(post, value_prior, theta_base, float(lam))
            step = float(th) * float(eta)
            prior = normalize((1.0 - step) * prior + step * np.asarray(post, dtype=float))
        return float(kl_divergence(prior, np.asarray(prior0, dtype=float)))

    def score(lam: float) -> float:
        return -float((replay(lam) - float(observed_drift)) ** 2)

    return fit_on_grid(grid, score)


# --------------------------------------------------------------------------- #
# mu — the one parameter the model's own inference estimates.
# --------------------------------------------------------------------------- #
def recovered_mu_from_rollout(result, world) -> float:
    """The posterior mean of the depth level. The estimator N21 and E30 already use.

    Here for completeness, so P-1 reads all four estimates through one module and the asymmetry
    between them is visible in one place: this is the only one that is a posterior read-out rather
    than a fit.
    """
    from . import foreign as FN
    from .v5_model import recovered_mu

    return float(recovered_mu(np.asarray(result.final_goal_posterior, dtype=float),
                              len(world.mu_levels), FN.NUM_REAL_GOALS, world.n_subgoals))
