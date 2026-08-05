"""Shared machinery for the second batch (T-1 .. T-4).

THE SUPPLY CHANNEL, which is the one new idea in this batch and the reason T-1 is runnable at
all.

The obvious way to "supply ground truth at a vertex" is to clamp that vertex's prior. It is also
the wrong way, for a reason this repository has been bitten by before: clamping a prior changes
the reference the recovery measure is scored against, so the control and the treatment stop being
measured on the same axis. ``error_reduction`` is ``log post[truth] - log prior[truth]`` and a
clamped prior moves the second term directly.

Instead a vertex is supplied through an EXTRA OBSERVATION MODALITY -- the mechanism E33 already
uses for the creator's self-report. The reader's prior D is byte-identical in every arm; what
changes is how much evidence about one vertex arrives down a side channel. Three consequences,
all of them load-bearing:

  * every arm is scored against the same prior, so the numbers are commensurable;
  * the amount supplied is a CONTINUOUS dial (channel fidelity), so every edge has a
    dose-response curve rather than a single number, and a monotonicity check;
  * the dial can be converted to NATS, so an edge into a 3-valued vertex and an edge into a
    4-valued one can be compared at equal information rather than at equal fidelity. Comparing
    them at equal fidelity is the mistake that would have made every asymmetry finding an
    artifact of cardinality.

A channel at fidelity ``1/card`` is pure noise and carries zero nats. That is the control arm,
and it is deliberately NOT the same thing as "no channel": it keeps the modality count, the
observation count and the RNG draws identical, so a difference between the control and a
supplied arm cannot come from the harness.
"""
from __future__ import annotations

import numpy as np

from ... import constants as K
from ... import metrics
from ... import v6_model as V6
from ...environment import Artifact
from ...v5_model import F_ATTENTION, MU_LEVELS, build_v5_D, marginal_goal, marginal_mu, \
    marginal_subgoal
from ...v6 import harness as H

_EPS = 1e-12

#: The three vertices that actually exist as independent latents in this model.
VERTICES = ("goal", "process", "depth")


def card(vertex: str, n_mu: int, n_sub: int, ng: int) -> int:
    return {"goal": ng, "process": n_sub, "depth": n_mu}[vertex]


def vertex_value_by_state(vertex: str, n_mu: int, n_sub: int, ng: int) -> np.ndarray:
    """``v[mgs]`` -- the value of ``vertex`` in each merged (mu, goal, subgoal) state.

    The merged index is ``((mu_i * ng) + g) * n_sub + s``, which is ``build_v5_D``'s own
    reshape order. Asserted against ``mgs_index`` in the T-1 harness checks rather than
    trusted.
    """
    n = n_mu * ng * n_sub
    out = np.zeros(n, dtype=int)
    for mi in range(n_mu):
        for g in range(ng):
            for s in range(n_sub):
                idx = (mi * ng + g) * n_sub + s
                out[idx] = {"goal": g, "process": s, "depth": mi}[vertex]
    return out


def channel_matrix(vertex: str, fidelity: float, n_mu: int, n_sub: int, ng: int,
                   n_prov: int = 4, n_att: int = 2, duty: float = 1.0) -> np.ndarray:
    """A(o | mu, g, s) for one supply channel, broadcast over provenance and attention.

    ``duty`` < 1 adds a NULL SYMBOL at index ``c`` which the channel emits when it has nothing
    to say, and which is uninformative under the reader's own likelihood. This exists because of
    a confound the first T-1 run walked straight into.

    Goal and depth are ONE value per artifact; process is a NEW value every step. A channel at
    one nat per step therefore hands the reader one nat about the goal, twenty-four times over
    -- which saturates it -- against twenty-four independent nats about the process. Measured
    that way the process vertex is not a better source, it is simply a bigger one, and every
    asymmetry in the triangle would be a restatement of the state space's shape. Holding the
    DUTY CYCLE down equalises the total information delivered, so the edges can be compared at a
    matched budget as well as at a matched per-step rate.

    The channel is symmetric: it reports the true value with probability ``fidelity`` and
    spreads the remainder uniformly over the other values. Broadcasting over provenance and
    attention is what makes it a SIDE channel -- it is not something the reader can choose to
    look at harder, so it cannot change the attention policy through the epistemic term in a
    way that would confound the recovery measure with an engagement effect.
    """
    c = card(vertex, n_mu, n_sub, ng)
    f = float(fidelity)
    assert 1.0 / c - 1e-9 <= f <= 1.0, f"fidelity {f} outside [1/{c}, 1]"
    vals = vertex_value_by_state(vertex, n_mu, n_sub, ng)
    n_state = vals.size
    if float(duty) < 1.0 - 1e-12:
        d = float(duty)
        off = (1.0 - f) / max(c - 1, 1)
        M = np.full((c + 1, n_state), d * off)
        M[vals, np.arange(n_state)] = d * f
        M[c, :] = 1.0 - d
        return np.broadcast_to(M[:, None, :, None], (c + 1, n_prov, n_state, n_att)).copy()
    if abs(f - 1.0 / c) < 1e-9:
        # SNAPPED TO EXACTLY UNIFORM. At c = 3 the diagonal ``1/3`` and the off-diagonal
        # ``(1 - 1/3)/2`` differ in the last bits, which leaves the likelihood very slightly
        # non-constant. That is normally invisible, but the depth posterior sits close to a
        # three-way tie and an argmax on a near-tie flips on the last bit: the placebo arm moved
        # depth accuracy 0.417 -> 0.383 on a channel carrying no information. Exact uniformity
        # makes the zero-nat arm exactly the control, which is the only thing that makes it a
        # placebo.
        M = np.full((c, n_state), 1.0 / c)
        return np.broadcast_to(M[:, None, :, None], (c, n_prov, n_state, n_att)).copy()
    off = (1.0 - f) / max(c - 1, 1)
    M = np.full((c, n_state), off)
    M[vals, np.arange(n_state)] = f
    return np.broadcast_to(M[:, None, :, None], (c, n_prov, n_state, n_att)).copy()


def channel_nats(fidelity: float, c: int) -> float:
    """Mutual information of a symmetric channel on a uniform source, in nats.

    This is the axis every cross-vertex comparison in T-1 is made on. At ``f = 1/c`` it is
    exactly zero, which is what makes the control arm a zero-information arm rather than an
    absent one.
    """
    f = float(fidelity)
    c = int(c)
    if c <= 1:
        return 0.0
    off = (1.0 - f) / max(c - 1, 1)
    terms = f * np.log(max(f, _EPS)) + (c - 1) * off * np.log(max(off, _EPS))
    return float(np.log(c) + terms)


def fidelity_for_nats(target: float, c: int, tol: float = 1e-10) -> float:
    """Invert :func:`channel_nats` by bisection, so two vertices can be supplied equally.

    Monotone in ``f`` on ``[1/c, 1]``, so bisection is exact to tolerance and needs no guard
    beyond clipping the target to the channel's capacity ``log c``.
    """
    c = int(c)
    hi_cap = float(np.log(c))
    t = float(np.clip(target, 0.0, hi_cap - 1e-12))
    lo, hi = 1.0 / c, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if channel_nats(mid, c) < t:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


class SupplyEnvironment:
    """``Environment`` facade that appends one observation per supply channel.

    Implements only the three methods ``rollout_observer`` calls, for the reason
    ``TapedEnvironment`` gives: anything else must fail loudly rather than quietly sample
    content nobody asked for.

    ``true_modes`` is the creator's execution mode at each of the READER's timesteps, already
    off-by-one corrected by ``harness.creator_positions``. Getting that wrong would desynchronise
    the process channel by one step and show up as a plausible, small, entirely uninformative
    weakening of every edge out of process -- which is the failure mode this repository has
    named seven times.

    THE CHANNEL DRAWS FROM ITS OWN RNG STREAM, and that is not tidiness. Drawing from the
    rollout's generator was the first thing tried and the placebo check caught it within one
    run: a channel at chance fidelity carries zero nats and must be indistinguishable from no
    channel at all, and it was not, because every side-channel draw shifted the content stream
    underneath it. Depth recovery moved 0.163 -> 0.282 on nothing. The same hazard
    ``observer_sig_rng`` exists for, in a new place.
    """

    def __init__(self, env, channels: list, true_mu_index: int, true_goal: int,
                 true_modes: list, rng, donor: dict | None = None):
        self.env = env
        self.channels = channels          # list of (vertex, fidelity, card[, control_mode])
        self.mu_i = int(true_mu_index)
        self.g = int(true_goal)
        self.modes = list(true_modes)
        self.rng = rng
        self.donor = donor                # another artifact's truth, for the "swap" control
        self._t = 0

    def _true_value(self, vertex: str, t: int, mode=0, c: int = 1) -> int:
        """The value the channel reports on. Non-zero ``mode`` is a NEGATIVE CONTROL.

        Three of them, because the first one is not enough and the T-1 run proved it.

          ``1``        rotate the reported value by one. THIS ONE FAILED AS A CONTROL and the
                       failure is informative: a lying process channel helped goal recovery as
                       much as an honest one. The execution chains are per-goal CYCLES, so
                       relabelling every mode by a constant maps a goal's cycle onto itself and
                       preserves exactly the thing goal identity is carried by. A rotation is
                       not a lie in this model, it is a change of coordinates.
          ``"random"`` emit uniformly at random while the reader's likelihood still claims high
                       fidelity: a trusted channel carrying nothing. This is the real test of
                       whether an edge needs the channel's CONTENT.
          ``"swap"``   report a different artifact's true values -- for process, another
                       creator's whole mode trajectory. Preserves the marginal statistics and
                       the temporal structure and destroys only the correspondence to THIS
                       artifact, which is the strictest control of the three.
        """
        if mode == "random":
            return int(self.rng.integers(int(c)))
        src = self.donor if mode == "swap" and self.donor is not None else None
        if vertex == "goal":
            v = self.g if src is None else int(src["goal"])
        elif vertex == "depth":
            v = self.mu_i if src is None else int(src["mu_i"])
        else:
            seq = self.modes if src is None else src["modes"]
            v = int(seq[min(t, len(seq) - 1)])
        if isinstance(mode, int) and mode:
            v = (v + int(mode)) % int(c)
        return int(v)

    def _draw(self, vertex: str, f: float, c: int, t: int, mode=0, duty: float = 1.0) -> int:
        if float(duty) < 1.0 - 1e-12 and self.rng.random() >= float(duty):
            return int(c)                          # the null symbol: nothing to say this step
        if abs(f - 1.0 / c) < 1e-9:
            return int(self.rng.integers(c))       # exactly uniform, matching the A snap above
        v = self._true_value(vertex, t, mode, c)
        p = np.full(c, (1.0 - f) / max(c - 1, 1))
        p[v] = f
        return int(self.rng.choice(c, p=p / p.sum()))

    def sample_feature(self, artifact: Artifact, attention: int, rng) -> int:
        return self.env.sample_feature(artifact, attention, rng)

    def effort_obs(self, attention: int) -> int:
        return self.env.effort_obs(attention)

    def observation(self, artifact: Artifact, attention: int, rng) -> list:
        base = self.env.observation(artifact, attention, rng)
        t = self._t
        self._t += 1
        return list(base) + [self._draw(ch[0], ch[1], ch[2], t,
                                        ch[3] if len(ch) > 3 else 0,
                                        ch[4] if len(ch) > 4 else 1.0)
                             for ch in self.channels]


def make_supplied_observer(world, rng, channels: list, n_mu: int, n_sub: int, ng: int,
                           n_prov: int = 4, n_att: int = 2):
    """A V5 observer with one extra modality per supply channel.

    Mirrors ``make_v5_observer`` rather than calling it, because the extra modalities have to be
    present when the Agent is constructed -- pymdp caches the modality count and an A array grown
    afterwards raises inside the inference loop. D is built by the shipped ``build_v5_D`` with no
    arguments beyond the cardinalities, so the prior is identical in every arm including the
    control.
    """
    from pymdp.legacy import utils

    extras = [channel_matrix(ch[0], ch[1], n_mu, n_sub, ng, n_prov, n_att,
                             duty=(ch[4] if len(ch) > 4 else 1.0)) for ch in channels]
    D = build_v5_D(world.cfg, rng, n_mu, ng, n_sub)
    A, C_ = world.gm.A, world.gm.C
    if extras:
        A = utils.obj_array(len(world.gm.A) + len(extras))
        for m in range(len(world.gm.A)):
            A[m] = world.gm.A[m]
        C_ = utils.obj_array(len(world.gm.C) + len(extras))
        for m in range(len(world.gm.C)):
            C_[m] = world.gm.C[m]
        for j, ex in enumerate(extras):
            A[len(world.gm.A) + j] = ex
            # ZERO PREFERENCE, for null N7's reason: the reader must not be able to want a
            # particular answer down the side channel. A nonzero C here would let the pragmatic
            # term manufacture the edge.
            C_[len(world.gm.C) + j] = np.zeros(ex.shape[0])
    a = world.cfg.agent
    from ...exact import ExactAgent
    return ExactAgent(A=A, B=world.gm.B, C=C_, D=D, control_fac_idx=[F_ATTENTION],
                      policy_len=int(a.policy_len), gamma=float(a.gamma),
                      action_selection=str(a.action_selection),
                      use_utility=bool(a.use_utility),
                      use_states_info_gain=bool(a.use_states_info_gain), rng=rng)


def recoveries(res, D1, n_mu: int, n_sub: int, ng: int, true_goal: int, true_mu_i: int,
               true_modes: list) -> dict:
    """The three recovery numbers, each scored against the arm's own prior marginal.

    Goal and depth use ``error_reduction`` (log posterior minus log prior at the truth, in
    nats). Process uses ``process_error_reduction``, which is scored against CHANCE rather than
    against a prior -- that is the shipped measure and it is correct here only because the
    sub-goal prior is uniform in every arm, which ``t1_triangle`` asserts rather than assumes.
    """
    rows = np.asarray(res.goal_posterior, dtype=float)
    g_post = marginal_goal(np.asarray(res.final_goal_posterior, dtype=float), n_mu, ng, n_sub)
    g_prior = marginal_goal(np.asarray(D1, dtype=float), n_mu, ng, n_sub)
    mu_post = marginal_mu(np.asarray(res.final_goal_posterior, dtype=float), n_mu, ng, n_sub)
    mu_prior = marginal_mu(np.asarray(D1, dtype=float), n_mu, ng, n_sub)
    sub_posts = [marginal_subgoal(r, n_mu, ng, n_sub) for r in rows]
    proc = V6.process_recovery(sub_posts, true_modes, n_sub)
    return {
        "goal": float(metrics.error_reduction(g_post, g_prior, int(true_goal))),
        "depth": float(metrics.error_reduction(mu_post, mu_prior, int(true_mu_i))),
        "process": float(proc["process_error_reduction"]),
        "goal_acc": float(int(np.argmax(g_post)) == int(true_goal)),
        "depth_acc": float(int(np.argmax(mu_post)) == int(true_mu_i)),
        "process_acc": float(proc["process_accuracy"]),
        "goal_prior_at_truth": float(g_prior[int(true_goal)]),
        "sub_prior_entropy": float(metrics.within_observer_entropy(
            marginal_subgoal(np.asarray(D1, dtype=float), n_mu, ng, n_sub))),
    }


def boot_diff(a, b, rng, draws: int = 2000) -> dict:
    """Bootstrap difference of means, the interval every verdict in this batch is read off."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"mean_a": float("nan"), "mean_b": float("nan"), "difference": float("nan"),
                "interval": [float("nan"), float("nan")], "excludes_zero": False,
                "n_a": int(a.size), "n_b": int(b.size)}
    d = [float(np.mean(rng.choice(a, a.size, replace=True))
               - np.mean(rng.choice(b, b.size, replace=True))) for _ in range(draws)]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "difference": float(a.mean() - b.mean()), "interval": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_a": int(a.size), "n_b": int(b.size)}


def boot_paired(a, b, rng, draws: int = 2000) -> dict:
    """Paired bootstrap on ``a - b``, element by element.

    Every arm in a cell is run on the SAME seed, so row ``i`` of the supplied arm and row ``i``
    of the control are the same artifact, the same creator trajectory and the same content
    stream -- the side channel draws from an independently spawned generator precisely so that
    turning it on cannot disturb them. The comparison is therefore within-artifact, and an
    unpaired interval here would be wider than the design earns. It also makes the placebo check
    exact rather than statistical: a zero-nat channel must reproduce the control to the last bit,
    not merely to within an interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"mean_a": float("nan"), "mean_b": float("nan"), "difference": float("nan"),
                "interval": [float("nan"), float("nan")], "excludes_zero": False, "n": 0}
    d = a[:n] - b[:n]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean_a": float("nan"), "mean_b": float("nan"), "difference": float("nan"),
                "interval": [float("nan"), float("nan")], "excludes_zero": False, "n": 0}
    draws_ = [float(np.mean(rng.choice(d, d.size, replace=True))) for _ in range(draws)]
    lo, hi = float(np.percentile(draws_, 2.5)), float(np.percentile(draws_, 97.5))
    return {"mean_a": float(np.mean(a[:n])), "mean_b": float(np.mean(b[:n])),
            "difference": float(np.mean(d)), "interval": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(d.size),
            "max_abs_paired_deviation": float(np.max(np.abs(d)))}


def run_supplied_encounter(world, cfg_r, channels, mu: int, beta: float, g_true: int,
                           n_timesteps: int, forced_k: int, n_mu: int, n_sub: int, ng: int,
                           art_rng, obs_rng, provenance: int = K.CREATOR,
                           donor_seed: int | None = None) -> dict:
    """One encounter with zero or more vertices supplied down side channels.

    ``initial_glance`` is OFF in every arm. The shipped glance calls ``infer_states`` with three
    observations, which a four-modality agent cannot accept; turning it off in every arm keeps
    the arms comparable to each other, and the E36 comparison is made separately against the
    unmodified ``common.rollouts`` rather than against these.
    """
    from ...observer import rollout_observer

    creator, artifact, env = H.make_artifact_and_env(
        world, cfg_r, int(g_true), int(mu), float(beta), int(n_timesteps), art_rng,
        provenance=int(provenance))
    modes = H.creator_positions(creator, int(n_timesteps), initial_glance=False)
    mu_i = list(MU_LEVELS).index(int(mu))
    donor = None
    if donor_seed is not None:
        # A DIFFERENT artifact's truth, built from an independent stream so that constructing it
        # cannot perturb this one. Same depth and rationality, different goal and different
        # trajectory: the swap control has to differ in correspondence and in nothing else.
        d_rng = np.random.default_rng(int(donor_seed))
        d_goal = int((int(g_true) + 1 + d_rng.integers(max(ng - 1, 1))) % ng)
        d_creator, _a, _e = H.make_artifact_and_env(
            world, cfg_r, d_goal, int(mu), float(beta), int(n_timesteps), d_rng,
            provenance=int(provenance))
        # A DIFFERENT depth as well. The first version reused this artifact's own mu_i, which
        # made the swap control for the depth vertex bit-identical to the honest arm -- it
        # returned exactly the honest gain and looked like a passing control while testing
        # nothing.
        d_mu_i = int((mu_i + 1 + int(d_rng.integers(max(n_mu - 1, 1)))) % n_mu)
        donor = {"goal": d_goal, "mu_i": d_mu_i,
                 "modes": H.creator_positions(d_creator, int(n_timesteps),
                                              initial_glance=False)}
    chan_rng = np.random.default_rng(obs_rng.bit_generator.seed_seq.spawn(1)[0])
    senv = SupplyEnvironment(env, channels, mu_i, int(g_true), modes, chan_rng, donor=donor)
    agent = make_supplied_observer(world, obs_rng, channels, n_mu, n_sub, ng)
    D1 = np.asarray(agent.D[K.F_GOAL], dtype=float).copy()
    res = rollout_observer(agent, artifact, senv, cfg_r, obs_rng, n_timesteps=int(n_timesteps),
                           force_deep_k=int(forced_k), initial_glance=False, early_stop=False)
    out = recoveries(res, D1, n_mu, n_sub, ng, int(g_true), mu_i, modes)
    out["true_modes"] = modes
    return out
