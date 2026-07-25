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

V3 — WHAT LEAKED, AND THE FIX (V3 spec §1 C1).

V2's N11 failed in the HONEST-SIGNAL arm only: slope +0.0119 nats/generation (t = 3.75) at
f = 0, against +0.0003 (t = 0.09) unsigned. The asymmetry localises the leak precisely.
Every quantity in this loop was already population-averaged EXCEPT one:

    C_recovered   averaged over observers  (``np.mean(per_obs_c)``)      -> clean
    A_learned     taken from ONE observer  (``learned_A0``, the LAST)    -> the leak

An honest signal concentrates the provenance posterior on CREATOR, so Dirichlet updates land
almost entirely on the CREATOR column and it sharpens fast around a *finite-sample* estimate.
That estimate — one observer's — then seeded the next generation, and the per-observer
estimation error compounded generation over generation as a random walk. Without a signal,
updates smear across provenance columns and the column stays near its (correct) prior, which
is why the unsigned arm looked clean: it was accidentally regularised.

C1: seed generation g+1 from the POPULATION-AVERAGED learned column,

    A_seed[:, CREATOR, goal, DEEP] = mean_i( A_learned_i[:, CREATOR, goal, DEEP] )

which cancels the zero-mean per-observer error before it can compound.

THE FLOOR THIS DOES NOT REACH, stated plainly because E12 is built to measure it (D2). The
corpus is drawn ONCE per generation and shared by all M observers; only the feature emissions
are redrawn per observer. So averaging cancels the observer-side term but NOT the corpus's own
goal-composition and signal noise, which is common-mode across the population. The 1/M gain
therefore has a floor that only larger N can lower, and E12's M sweep measures where it sits.
This is why C2's sample-size sweep is not optional decoration on C1: it is the half of the fix
that addresses the part averaging structurally cannot.

D5 preserved: creators remain real ``HumanCreator`` POMDP agents seeded from the averaged
column. Averaging changes WHAT column seeds the creator, not THAT a reward-optimising policy
produces the artifact. Option B (marginalising over C_recovered) remains forbidden.
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
    kl_payload: float          # KL(C_recovered || C_true)   -- VALUE divergence (C3 primary)
    mean_deep_genuine: float   # engagement on genuine artifacts
    learned_A0: np.ndarray = field(repr=False, default=None)
    panel: dict = field(default_factory=dict)

    # ---- V3 additions ----------------------------------------------------- #
    seed_column: np.ndarray = field(repr=False, default=None)
    # (features, goals): the CREATOR/DEEP column that seeds generation g+1. Population-
    # averaged under C1; the last observer's under the V2 behaviour. This is the object the
    # whole V3 diagnosis is about, so it is returned rather than kept internal.

    creator_col_kl: float = float("nan")
    # THE SHARED QUANTITY (D7). Mean over goals of KL(seed column || true CREATOR/DEEP
    # column), in nats, via ``learning.column_kl``. E13 plots E9's starvation arm and this
    # recursion on exactly this axis, which is the only way "the same curve" is even defined
    # across two experiments that otherwise report in different spaces.

    eff_sample_count: float = float("nan")
    # Effective Dirichlet sample count on the CREATOR/DEEP columns: total concentration mass
    # minus the prior, averaged over goals and observers. E13's x-axis.

    col_change_rate: float = float("nan")
    # Mean L1 change per observation in the learned CREATOR/DEEP column over the generation's
    # final tracing window. The "rate of change collapses" signature C4 asks for.

    encoder: dict = field(default_factory=dict)
    # ENCODER divergence (C3 secondary), from ``probes.decode_probe_set``: free-engagement and
    # forced-DEEP decoding of a fixed, uncontaminated probe set. Empty when probes are off.


def creator_deep_column(A0: np.ndarray) -> np.ndarray:
    """The (features, goals) CREATOR/DEEP slice — the column the recursion carries forward."""
    return np.asarray(A0)[:, K.CREATOR, :, K.DEEP]


def average_seed_column(per_observer_columns: list[np.ndarray]) -> np.ndarray:
    """C1. Mean of the observers' learned CREATOR/DEEP columns, renormalised per goal.

    The V3 §3 invariant is asserted here rather than at the call site: every population-
    averaged seed column must remain a valid categorical distribution after averaging and
    renormalisation. A mean of stochastic columns is already stochastic up to floating-point
    error, so the renormalisation is defensive and the assertion is what proves it stayed so.

    At M = 1 there is nothing to average, and the column is returned UNTOUCHED. That is not a
    micro-optimisation: ``SeededCreator`` already clips and renormalises what it is handed, so
    normalising here first would hand it a column differing from V2's in the last bits (~5e-17),
    and a last-bit difference in an emission distribution can flip an ``rng.choice`` and send
    the whole chain down a different realisation. N14 asks for EXACT reduction to the V2
    seeding, so the M=1 path must be a literal pass-through rather than an equivalent one.
    """
    cols = [np.asarray(c, dtype=float) for c in per_observer_columns]
    if len(cols) == 1:
        col = cols[0]
    else:
        col = np.stack(cols, axis=0).mean(axis=0)
        col = np.clip(col, _LOG_FLOOR, None)
        col = col / col.sum(axis=0, keepdims=True)
    assert np.all(col >= 0.0) and np.allclose(col.sum(axis=0), 1.0, atol=1e-10), \
        "V3 §3 invariant: an averaged seed column is not a valid categorical distribution"
    return col


def _effective_sample_count(agent, cfg: Config) -> float:
    """Total Dirichlet concentration mass on the CREATOR/DEEP columns, minus the prior.

    E13's x-axis. ``build_pA`` seeds each column with ``prior_strength * sig[g]``, which sums
    to ``prior_strength``, so subtracting it leaves the mass actually deposited by experience.
    Returns NaN for a non-learner (there is no pA to read).
    """
    pA = getattr(agent, "pA", None)
    if pA is None:
        return float("nan")
    strength = float(cfg.get("learning.prior_strength", 1.0))
    mass = np.asarray(pA[0])[:, K.CREATOR, :, K.DEEP].sum(axis=0)   # per goal
    return float(np.mean(mass - strength))


def run_generation(cfg: Config, gm: GenerativeModel, creator_bank: dict,
                   creator_goals: np.ndarray, c_true: np.ndarray,
                   contamination: float, signing_rate: float, honesty: float,
                   n_artifacts: int, n_observers: int, infer_steps: int,
                   d_i: float, rng: np.random.Generator, generation: int,
                   lr_pA: float | None = None,
                   use_param_info_gain: bool | None = None,
                   freeze_engagement: bool = False,
                   learn: bool = True,
                   learn_mode: str = "online",
                   population_average_seed: bool = True,
                   probes: list | None = None,
                   probe_env: Environment | None = None,
                   probe_rng: np.random.Generator | None = None,
                   trace_sink: list | None = None,
                   trace_every: int = 0) -> GenerationResult:
    """One generation: observers read a contaminated corpus, learn A, recover C.

    ``freeze_engagement`` forces DEEP throughout (E9's poisoning-only arm, where disuse must
    be prevented so the two corruption channels can be separated).

    ``population_average_seed`` (V3 C1) selects the seeding rule: True averages every
    observer's learned CREATOR/DEEP column, False reproduces V2 by taking the last observer's.
    E12 runs both arms so the fix can be shown to be the averaging rather than an incidental
    change to the seeding code (N14).

    ``probes``/``probe_env``/``probe_rng`` enable C3's encoder-divergence channel. They are
    drawn ONCE for the whole chain and passed in, and ``probe_rng`` is an independent stream:
    probe decoding must not consume variates from ``rng``, or enabling the second channel
    would shift every downstream draw and the V2 comparison would move for a reason having
    nothing to do with the fix (the same hazard ``observer_sig_rng`` exists to avoid).

    ``trace_sink``/``trace_every`` enable E13's per-observation instrumentation.
    """
    num_goals = int(cfg.cardinalities.num_goals)
    env = Environment(cfg, gm, rng_world=rng, honesty=honesty,
                      signing_rate=signing_rate, creator_bank=creator_bank)
    corpus = env.draw_corpus(n_artifacts, contamination=contamination,
                             creator_goals=creator_goals, rng=rng)
    if contamination == 0.0:
        assert all(a.provenance == K.CREATOR for a in corpus), \
            "f=0 corpus must be entirely genuine (N11 depends on it)"

    A_true = np.asarray(gm.A[0])
    per_obs_c, mis, deeps = [], [], []
    per_obs_cols, per_obs_neff, per_obs_rate, per_obs_enc = [], [], [], []
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
        prev_col = creator_deep_column(agent.A[0]).copy()
        deltas: list[float] = []
        for j, art in enumerate(corpus):
            rr = np.random.default_rng(rng.integers(2**63 - 1))
            from .observer import rollout_observer
            k_force = infer_steps if freeze_engagement else 0
            res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                                   force_deep_k=k_force, early_stop=False, learn=learn,
                                   learn_mode=learn_mode)
            p_ghost = float(res.prov_posterior[-1][K.GHOST])
            accum += (1.0 - p_ghost) * res.final_goal_posterior
            if art.provenance == K.CREATOR:
                n_genuine += 1
                n_deep_genuine += res.cum_deep

            if trace_every and learn and (j + 1) % trace_every == 0:
                col = creator_deep_column(agent.A[0])
                delta = float(np.abs(col - prev_col).sum() / trace_every)
                deltas.append(delta)
                if trace_sink is not None:
                    trace_sink.append({
                        "generation": generation, "observer": i, "artifact_index": j + 1,
                        "eff_sample_count": _effective_sample_count(agent, cfg),
                        "creator_col_kl": float(np.mean(
                            [L.column_kl(agent.A[0], A_true, K.CREATOR, g)
                             for g in range(num_goals)])),
                        "col_change_rate": delta,
                        "deep_per_artifact": res.cum_deep / max(infer_steps, 1)})
                prev_col = col.copy()

        c_rec = accum / accum.sum() if accum.sum() > 0 else np.full(num_goals, 1.0 / num_goals)
        per_obs_c.append(c_rec)
        mis.append(L.human_column_mi(agent.A[0], K.CREATOR) if learn
                   else metrics.mutual_information_features_goal(np.asarray(agent.A[0]),
                                                                 K.CREATOR, K.DEEP))
        deeps.append(n_deep_genuine / max(n_genuine, 1))
        per_obs_cols.append(creator_deep_column(agent.A[0]).copy())
        per_obs_neff.append(_effective_sample_count(agent, cfg))
        # Late-window rate of change: the collapse C4 looks for is in the TAIL, so an average
        # over the whole generation would be dominated by the early, fast-learning phase.
        per_obs_rate.append(float(np.mean(deltas[len(deltas) // 2:])) if deltas else float("nan"))

        if probes:
            per_obs_enc.append(_decode_both_ways(agent, probes, probe_env or env, cfg,
                                                 probe_rng, infer_steps))
        learned_A0 = np.asarray(agent.A[0]).copy()   # last observer's, for the panel

    c_recovered = np.mean(per_obs_c, axis=0)
    c_recovered = c_recovered / c_recovered.sum()

    # C1: the seed column. ``[-1]`` is not a stand-in for "some observer" — it is exactly
    # what V2 carried forward (``learned_A0`` was the last observer's), so the without-
    # averaging arm reproduces V2 rather than approximating it.
    seed_column = average_seed_column(per_obs_cols if population_average_seed
                                      else per_obs_cols[-1:])
    creator_col_kl = float(np.mean(
        [metrics.kl_divergence(seed_column[:, g], A_true[:, K.CREATOR, g, K.DEEP])
         for g in range(num_goals)]))

    panel = R.behavioral_regret(c_recovered, c_true, cfg, seed=generation).flat()
    panel.update({"mi_genuine": float(np.mean(mis)),
                  "mean_deep_genuine": float(np.mean(deeps))})
    encoder = _mean_encoder(per_obs_enc)
    panel.update(encoder)
    return GenerationResult(
        generation=generation, c_recovered=c_recovered, c_true=np.asarray(c_true),
        mean_expertise=1.0 - d_i, mi_genuine=float(np.mean(mis)),
        kl_payload=metrics.kl_divergence(c_recovered, c_true),
        mean_deep_genuine=float(np.mean(deeps)), learned_A0=learned_A0, panel=panel,
        seed_column=seed_column, creator_col_kl=creator_col_kl,
        eff_sample_count=float(np.nanmean(per_obs_neff)) if per_obs_neff else float("nan"),
        col_change_rate=(float(np.nanmean(per_obs_rate))
                         if per_obs_rate and not all(np.isnan(per_obs_rate))
                         else float("nan")),
        encoder=encoder)


def _decode_both_ways(agent, probes, env, cfg, probe_rng, infer_steps) -> dict:
    """C3/D5: decode the probe set with engagement free AND with DEEP forced.

    The forced number isolates the observer's MODEL (the channel C3's shared-representation
    claim is about); the gap between them is what disengagement contributes. Reporting only
    one would conflate the two, and E9 already owns the engagement channel.
    """
    from .probes import decode_probe_set
    rng = probe_rng if probe_rng is not None else np.random.default_rng(0)
    free = decode_probe_set(agent, probes, env, cfg, rng, infer_steps, force_deep=False)
    forced = decode_probe_set(agent, probes, env, cfg, rng, infer_steps, force_deep=True)
    out = {f"enc_free_{k}": v for k, v in free.items()}
    out.update({f"enc_forced_{k}": v for k, v in forced.items()})
    out["probe_contaminated_count"] = 0      # N15, carried into every CSV row
    out["probe_n"] = free["probe_n"]
    return out


def _mean_encoder(per_obs: list[dict]) -> dict:
    """Average the per-observer encoder panels. Counts (probe_n, contaminated) are invariant
    across observers, so they pass through rather than being averaged into a float."""
    if not per_obs:
        return {}
    keys = per_obs[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in per_obs]
        out[k] = int(vals[0]) if k in ("probe_n", "probe_contaminated_count") \
            else float(np.mean(vals))
    return out


# --------------------------------------------------------------------------- #
# The chain.
# --------------------------------------------------------------------------- #
def run_chain(cfg: Config, gm: GenerativeModel, c_true_pop: np.ndarray,
              contamination: float, signing_rate: float, honesty: float,
              g_max: int, n_creators: int, n_artifacts: int, n_observers: int,
              infer_steps: int, d_i: float, base_seed: int,
              learn: bool = True, n_probes: int = 0,
              trace_sink: list | None = None, trace_every: int = 0,
              **kw) -> list[GenerationResult]:
    """Run ``g_max`` generations, seeding each from the last.

    ``c_true`` is held FIXED across generations: it is the original population's preference
    distribution, the thing we are asking whether the chain still transmits. Comparing each
    generation's C_recovered to a moving target would measure nothing.

    ``n_probes`` (V3 C3) turns on encoder divergence. The probe set is drawn ONCE, here, from
    the generation-0 creator bank, and every generation is scored against that same fixed set
    — for the same reason ``c_true`` is fixed. A probe set regenerated per generation would
    drift alongside the observer, and a flat result could then mean either a stable decoder or
    two drifts cancelling.
    """
    num_goals = int(cfg.cardinalities.num_goals)
    seed_seq = np.random.SeedSequence(base_seed)
    gen_streams = [np.random.default_rng(s) for s in seed_seq.spawn(g_max + 1)]
    # An independent stream for probe decoding: enabling C3 must not shift the main draws.
    probe_rng = np.random.default_rng(np.random.SeedSequence(base_seed + 104_729))

    creator_goals = allocate_creator_goals(n_creators, c_true_pop[:num_goals],
                                           np.random.default_rng(base_seed * 13))
    c_true = np.bincount(creator_goals, minlength=num_goals).astype(float)
    c_true /= c_true.sum()

    from .creators import build_creator_bank
    bank = build_creator_bank(cfg, gm)          # generation 0: true human creators

    probes, probe_env = None, None
    if n_probes > 0:
        from .probes import build_probe_env, draw_probe_set
        probe_env = build_probe_env(cfg, gm, np.random.default_rng(base_seed * 7717),
                                    honesty=honesty, signing_rate=signing_rate)
        probes = draw_probe_set(probe_env, int(n_probes), creator_goals,
                                np.random.default_rng(base_seed * 7717 + 1))

    out = []
    for g in range(g_max):
        res = run_generation(cfg, gm, bank, creator_goals, c_true,
                             contamination, signing_rate, honesty, n_artifacts,
                             n_observers, infer_steps, d_i, gen_streams[g], g,
                             learn=learn, probes=probes, probe_env=probe_env,
                             probe_rng=probe_rng, trace_sink=trace_sink,
                             trace_every=trace_every, **kw)
        out.append(res)

        # ---- seed generation g+1 from what this generation absorbed (C4 step 2) ----
        if g + 1 < g_max:
            # Deterministic allocation from C_recovered: preserves the degradation channel
            # while removing the sampling random-walk that would break N11 (see module docs).
            creator_goals = allocate_creator_goals(n_creators, res.c_recovered,
                                                   np.random.default_rng(base_seed * 17 + g))
            # C1: the POPULATION-AVERAGED column (or, in E12's control arm, the single
            # observer's, exactly as V2 did). ``seed_column`` already carries whichever.
            if res.seed_column is not None:
                bank = {gl: SeededCreator(cfg, res.seed_column[:, gl], gl)
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
