"""Recursive generations — the C4 loop, and the engine N11 gates (V2 spec §1 C4, §3 N11).

    Generation g:
      1. Observers learn A from a corpus with contamination fraction f.
      2. A subset become CREATORS for generation g+1. Their creator C is seeded from their own
         C_recovered, and their production likelihood is seeded from their learned A (the H4
         motor-model claim: what you absorbed shapes what you make).
      3. Generation g+1's corpus is those artifacts, plus fresh synthetic contamination at f.

-----------------------------------------------------------------------------------------
N11 IS THE ACCEPTANCE TEST FOR THIS MODULE, AND IT DROVE THE DESIGN.

    "N11 — Zero-contamination recursion. With f = 0, E8 must show NO degradation across
     generations. This is the most important new null. If generational decay appears without
     any contamination, the recursion loop itself is lossy and every E8 result is an artifact
     of the implementation rather than a finding."

Four ways this loop could leak, and what is done about each:

1. RESAMPLING THE CREATOR POPULATION each generation would make the goal distribution a
   random walk; at f=0 that alone produces drift that reads as decay. So generation g+1's
   creator goals are assigned by LARGEST-REMAINDER ALLOCATION from the mean C_recovered —
   deterministic given C_recovered, no sampling noise. At f=0 with C_recovered ~ C_true the
   allocation returns the same counts and the loop is a genuine fixed point, while a
   distorted C_recovered still shifts the allocation, so the real degradation channel that
   E8 measures is fully preserved.

2. COLLAPSING C_recovered TO AN ARGMAX when seeding the next generation would destroy the
   payload immediately. It is passed forward as a full distribution, always.

3. REUSING AN RNG STREAM across generations would correlate corpora. Each generation draws
   from an independently spawned stream.

4. SEEDING CREATORS FROM A MARGINALISED A (rejected Option B of decision D5) blurs
   goal-specific content even at f=0 and would make N11 fail by construction. Per D5 the
   creators stay REAL AGENTS with C = log(A_learned[:, CREATOR, goal, DEEP]), which preserves
   §4.2's load-bearing property that human artifacts are produced by a reward-optimising
   policy, and which returns sig_true exactly when A has been learned perfectly.

HONEST FORMULATION OF N11. Even a correct loop is a finite-sample estimator: each generation
learns A from finitely many artifacts, so a small amount of estimation noise is unavoidable
and strictly-zero drift is not achievable. N11 is therefore tested as "no SIGNIFICANT
degradation trend at f=0" (|t| on the per-generation slope < 2) AND "the f=0 slope is far
smaller than the f>0 slope". A loop that leaks structurally fails both; a loop that is merely
finite fails neither.
-----------------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from .config import Config
from .creators import HumanCreator
from .environment import Environment
from .generative_model import GenerativeModel, build_D, cards
from .observer import make_observer
from .preregistration import allocate_creator_goals
from . import learning as L
from . import metrics, regret as R

_LOG_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# The seeded creator (D5, Option C).
# --------------------------------------------------------------------------- #
class SeededCreator(HumanCreator):
    """A creator whose production likelihood comes from a LEARNED A rather than sig_true.

    This is the H4 motor-model claim made concrete: what you absorbed shapes what you make.
    It keeps the full ``HumanCreator`` POMDP — a real agent that SELECTS ACTIONS to realise a
    preference — because §4.2 names that as the entire theoretical content of the model. Only
    the preference changes: ``C = log(emission_target)`` where ``emission_target`` is the
    creator's own learned belief about what a CREATOR pursuing this goal looks like.

    With ``gamma=1`` the policy posterior equals ``emission_target`` exactly (the identity
    ``HumanCreator`` already relies on), so a perfectly-learned A reproduces ``sig_true`` and
    the f=0 loop is lossless — which is what N11 requires.
    """

    def __init__(self, cfg: Config, emission_target: np.ndarray, goal: int):
        target = np.asarray(emission_target, dtype=float)
        target = np.clip(target, _LOG_FLOOR, None)
        target = target / target.sum()
        # Reuse HumanCreator's machinery by handing it a one-row "signature" table.
        fake_sig = np.tile(target, (int(cfg.cardinalities.num_goals), 1))
        super().__init__(cfg, fake_sig, goal)
        self.emission_target = target


# --------------------------------------------------------------------------- #
# One generation.
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    generation: int
    c_recovered: np.ndarray
    c_true: np.ndarray
    mean_expertise: float
    mi_genuine: float          # MI(features; goal) of the learned CREATOR column
    kl_payload: float          # KL(C_recovered || C_true)
    mean_deep_genuine: float   # engagement on genuine artifacts
    learned_A0: np.ndarray = field(repr=False, default=None)
    panel: dict = field(default_factory=dict)


def run_generation(cfg: Config, gm: GenerativeModel, creator_bank: dict,
                   creator_goals: np.ndarray, c_true: np.ndarray,
                   contamination: float, signing_rate: float, honesty: float,
                   n_artifacts: int, n_observers: int, infer_steps: int,
                   d_i: float, rng: np.random.Generator, generation: int,
                   lr_pA: float | None = None,
                   use_param_info_gain: bool | None = None,
                   freeze_engagement: bool = False,
                   learn: bool = True) -> GenerationResult:
    """One generation: observers read a contaminated corpus, learn A, recover C.

    ``freeze_engagement`` forces DEEP throughout (E9's poisoning-only arm, where disuse must
    be prevented so the two corruption channels can be separated).
    """
    num_goals = int(cfg.cardinalities.num_goals)
    env = Environment(cfg, gm, rng_world=rng, honesty=honesty,
                      signing_rate=signing_rate, creator_bank=creator_bank)
    corpus = env.draw_corpus(n_artifacts, contamination=contamination,
                             creator_goals=creator_goals, rng=rng)
    if contamination == 0.0:
        assert all(a.provenance == K.CREATOR for a in corpus), \
            "f=0 corpus must be entirely genuine (N11 depends on it)"

    per_obs_c, mis, deeps = [], [], []
    learned_A0 = None
    for i in range(n_observers):
        r_obs = np.random.default_rng(rng.integers(2**63 - 1))
        D = build_D(cfg, r_obs)
        from .generative_model import build_observer_model
        from .observer import observer_sig_rng
        om = build_observer_model(gm, cfg, d_i,
                                  rng_sig=observer_sig_rng(r_obs) if d_i > 0 else None)
        agent = (L.make_learner_agent(om, D, cfg, lr_pA=lr_pA,
                                      use_param_info_gain=use_param_info_gain)
                 if learn else make_observer(gm, cfg, r_obs, d_i=d_i))

        accum = np.zeros(num_goals)
        n_deep_genuine = n_genuine = 0
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(rng.integers(2**63 - 1))
            from .observer import rollout_observer
            k_force = infer_steps if freeze_engagement else 0
            res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                                   force_deep_k=k_force, early_stop=False, learn=learn)
            p_ghost = float(res.prov_posterior[-1][K.GHOST])
            accum += (1.0 - p_ghost) * res.final_goal_posterior
            if art.provenance == K.CREATOR:
                n_genuine += 1
                n_deep_genuine += res.cum_deep
        c_rec = accum / accum.sum() if accum.sum() > 0 else np.full(num_goals, 1.0 / num_goals)
        per_obs_c.append(c_rec)
        mis.append(L.human_column_mi(agent.A[0], K.CREATOR) if learn
                   else metrics.mutual_information_features_goal(np.asarray(agent.A[0]),
                                                                 K.CREATOR, K.DEEP))
        deeps.append(n_deep_genuine / max(n_genuine, 1))
        learned_A0 = np.asarray(agent.A[0]).copy()   # last observer's, for the panel

    c_recovered = np.mean(per_obs_c, axis=0)
    c_recovered = c_recovered / c_recovered.sum()
    panel = R.behavioral_regret(c_recovered, c_true, cfg, seed=generation).flat()
    panel.update({"mi_genuine": float(np.mean(mis)),
                  "mean_deep_genuine": float(np.mean(deeps))})
    return GenerationResult(
        generation=generation, c_recovered=c_recovered, c_true=np.asarray(c_true),
        mean_expertise=1.0 - d_i, mi_genuine=float(np.mean(mis)),
        kl_payload=metrics.kl_divergence(c_recovered, c_true),
        mean_deep_genuine=float(np.mean(deeps)), learned_A0=learned_A0, panel=panel)


# --------------------------------------------------------------------------- #
# The chain.
# --------------------------------------------------------------------------- #
def run_chain(cfg: Config, gm: GenerativeModel, c_true_pop: np.ndarray,
              contamination: float, signing_rate: float, honesty: float,
              g_max: int, n_creators: int, n_artifacts: int, n_observers: int,
              infer_steps: int, d_i: float, base_seed: int,
              learn: bool = True, **kw) -> list[GenerationResult]:
    """Run ``g_max`` generations, seeding each from the last.

    ``c_true`` is held FIXED across generations: it is the original population's preference
    distribution, the thing we are asking whether the chain still transmits. Comparing each
    generation's C_recovered to a moving target would measure nothing.
    """
    num_goals = int(cfg.cardinalities.num_goals)
    seed_seq = np.random.SeedSequence(base_seed)
    gen_streams = [np.random.default_rng(s) for s in seed_seq.spawn(g_max + 1)]

    creator_goals = allocate_creator_goals(n_creators, c_true_pop[:num_goals],
                                           np.random.default_rng(base_seed * 13))
    c_true = np.bincount(creator_goals, minlength=num_goals).astype(float)
    c_true /= c_true.sum()

    from .creators import build_creator_bank
    bank = build_creator_bank(cfg, gm)          # generation 0: true human creators

    out = []
    for g in range(g_max):
        res = run_generation(cfg, gm, bank, creator_goals, c_true,
                             contamination, signing_rate, honesty, n_artifacts,
                             n_observers, infer_steps, d_i, gen_streams[g], g,
                             learn=learn, **kw)
        out.append(res)

        # ---- seed generation g+1 from what this generation absorbed (C4 step 2) ----
        if g + 1 < g_max:
            # Deterministic allocation from C_recovered: preserves the degradation channel
            # while removing the sampling random-walk that would break N11 (see module docs).
            creator_goals = allocate_creator_goals(n_creators, res.c_recovered,
                                                   np.random.default_rng(base_seed * 17 + g))
            if res.learned_A0 is not None:
                bank = {gl: SeededCreator(cfg, res.learned_A0[:, K.CREATOR, gl, K.DEEP], gl)
                        for gl in range(num_goals)}
    return out


def chain_trend(results: list[GenerationResult], key: str = "kl_payload") -> dict:
    """Per-generation slope with a t statistic, plus a quadratic term for the superlinearity
    prediction. E8 reports "a significant per-generation effect in the predicted direction",
    NOT an equilibrium, and does not extrapolate the curve (V2 spec §2)."""
    g = np.array([r.generation for r in results], dtype=float)
    y = np.array([getattr(r, key) if hasattr(r, key) else r.panel.get(key, np.nan)
                  for r in results], dtype=float)
    ok = np.isfinite(y)
    g, y = g[ok], y[ok]
    if len(g) < 3:
        return {"slope": float("nan"), "t": float("nan"), "quad": float("nan"), "n": len(g)}
    slope, intercept = np.polyfit(g, y, 1)
    resid = y - (slope * g + intercept)
    dof = max(len(g) - 2, 1)
    denom = np.sum((g - g.mean()) ** 2)
    se = float(np.sqrt((resid @ resid) / dof / denom)) if denom > 0 else float("inf")
    quad = float(np.polyfit(g, y, 2)[0]) if len(g) >= 3 else float("nan")
    return {"slope": float(slope), "se": se,
            "t": float(slope / se) if se > 0 else float("nan"),
            "quad": quad, "n": int(len(g))}
