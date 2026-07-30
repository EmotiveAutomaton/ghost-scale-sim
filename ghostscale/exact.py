"""V-1 — exact inference over the full joint state, as a drop-in for the pymdp agent.

WHY THIS FILE EXISTS, AND WHY IT IS THE HIGHEST-PRIORITY VALIDATION ITEM.

Every experiment before V5 ran on pymdp legacy's variational solver, which factorises the
posterior across hidden-state factors and updates each factor using the expected LOG likelihood
under the others' current marginals. By Jensen's inequality that systematically penalises any
hypothesis whose evidence depends on a *combination* of factors rather than on one factor at a
time. V5 caught this doing real damage: with model depth in its own factor, the mean-field agent
returned the shallow answer for every artifact and did so confidently (posterior entropy 0.047),
while exact filtering on the identical feature sequence recovered depth monotonically.

    true depth      exact filtering      mean-field agent
        1              1.52                  1.04
        2              2.20                  1.04
        3              2.27                  1.06

V5 fixed that one case by merging the coupled factors. It did not establish that the earlier
results are safe, and there is a specific reason to worry: "content whose structure the
observer's hypothesis space cannot express" is uncomfortably adjacent to "hypothesis carrying
latent structure", and that is exactly what the goal-foreign experiments are built out of.

So this module removes the factorisation entirely rather than working around it.

-----------------------------------------------------------------------------------------
WHAT IS EXACT HERE, STATED NARROWLY.

``ExactAgent`` carries a belief over the FULL JOINT of the hidden-state factors — a single
vector of length prod(num_states) — and:

* **Filtering is exact.** The predictive step multiplies by the joint transition, built as a
  Kronecker product of the per-factor transitions under the chosen action. The update step
  multiplies by prod_m A[m][o_m, s] pointwise over joint states and renormalises. This is Bayes'
  rule on the joint, with no independence assumption anywhere.
* **The expected free energy is exact over states.** The predictive state distribution under a
  policy is the joint propagated forward, and the information gain is the true mutual
  information I(s; o) under that joint, using the joint observation distribution rather than a
  per-modality sum. Utility is unchanged, because C factorises across modalities by construction
  and the expected-utility term is therefore already exact.

Everything else is deliberately identical to the pymdp path, because the point of the check is
to isolate one difference. Same policy set (built by pymdp's own ``construct_policies``), same
policy_len, same gamma, same softmax, same deterministic action selection, same free initial
glance, same early-stop rule. If a result moves, the factorisation moved it.

-----------------------------------------------------------------------------------------
WHERE THE TWO SOLVERS MUST AGREE, AND WHY THAT IS THE TEST THAT MATTERS.

When the joint posterior genuinely factorises, mean field is exact and the two agents must
return the same numbers. That happens in this model whenever the observation carries information
about at most one uncertain factor at a time. The V1 construction is close to that case: A[2]
pins attention deterministically, and A[1] depends on provenance alone. It is A[0] that couples
provenance and goal, and that coupling is exactly what alpha is.

``assert_agrees_on_factorised_case`` checks the agreement where it is provable, so a failure
there is a bug in this file rather than a finding about the model. The interesting output is
where they *disagree*, and that is measured rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymdp.legacy import control, utils

from . import constants as K
from .config import Config

_FLOOR = 1e-16


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


class ExactAgent:
    """A pymdp-legacy-compatible agent that keeps the joint belief instead of marginals.

    The public surface is exactly the part of ``pymdp.legacy.agent.Agent`` that
    ``observer.rollout_observer`` touches: ``reset``, ``infer_states``, ``infer_policies``,
    ``sample_action``, ``action``, ``policies``, ``D``, ``num_controls``, ``qs``. Anything else
    raises rather than silently doing something approximate, because a half-exact agent that
    quietly falls back is worse than no exact agent at all.
    """

    def __init__(self, A, B, C, D, control_fac_idx=None, policy_len: int = 2,
                 gamma: float = 16.0, action_selection: str = "deterministic",
                 use_utility: bool = True, use_states_info_gain: bool = True,
                 rng: np.random.Generator | None = None):
        self.A = utils.obj_array(len(A))
        for m in range(len(A)):
            self.A[m] = np.asarray(A[m], dtype=float)
        # B IS AN OBJECT ARRAY, NOT A LIST, and that is not cosmetic. pymdp's control helpers
        # inspect ``.dtype`` on the container, so a plain list makes ``metrics.policy_efe_terms``
        # raise AttributeError while ``metrics.epistemic_value`` succeeds and silently returns the
        # FACTORISED answer for an exact agent. One of those is a crash and the other is the exact
        # failure mode this whole file exists to remove. See ``efe_terms`` below for the fix to the
        # second half.
        self.B = utils.obj_array(len(B))
        for f in range(len(B)):
            self.B[f] = np.asarray(B[f], dtype=float)
        self.C = utils.obj_array(len(C))
        for m in range(len(C)):
            self.C[m] = np.asarray(C[m], dtype=float)
        self.D = D
        self.num_factors = len(self.B)
        self.num_modalities = len(self.A)
        self.num_states = [int(self.B[f].shape[0]) for f in range(self.num_factors)]
        self.num_controls = [int(self.B[f].shape[2]) for f in range(self.num_factors)]
        self.control_fac_idx = (list(range(self.num_factors)) if control_fac_idx is None
                                else list(control_fac_idx))
        self.policy_len = int(policy_len)
        self.gamma = float(gamma)
        self.action_selection = str(action_selection)
        self.use_utility = bool(use_utility)
        self.use_states_info_gain = bool(use_states_info_gain)
        self._rng = np.random.default_rng(0) if rng is None else rng

        self.policies = control.construct_policies(
            self.num_states, self.num_controls, policy_len=self.policy_len,
            control_fac_idx=self.control_fac_idx)

        # Flattened likelihoods: L[m] has shape (num_obs_m, prod(num_states)), C-ordered so
        # joint index = ravel_multi_index((s_0, ..., s_F), num_states).
        self._n_joint = int(np.prod(self.num_states))
        self._L = [a.reshape(a.shape[0], self._n_joint) for a in self.A]
        # Per-joint-state conditional observation entropy, summed over modalities. Constant in
        # the belief, so it is worth computing once: it is the E_s[H(o | s)] half of every
        # information-gain evaluation.
        self._condH = np.zeros(self._n_joint)
        for Lm in self._L:
            p = np.clip(Lm, _FLOOR, None)
            self._condH += -np.sum(p * np.log(p), axis=0)

        self._lnC = []
        for m in range(self.num_modalities):
            c = self.C[m]
            c = c[:, 0] if c.ndim > 1 else c
            self._lnC.append(np.log(_softmax(c) + _FLOOR))

        self._T_cache: dict[tuple, np.ndarray] = {}
        self.reset()

    # ------------------------------------------------------------------ #
    # Belief plumbing.
    # ------------------------------------------------------------------ #
    def _joint_prior(self) -> np.ndarray:
        b = np.asarray(self.D[0], dtype=float)
        for f in range(1, self.num_factors):
            b = np.kron(b, np.asarray(self.D[f], dtype=float))
        return b / b.sum()

    def _transition(self, action) -> np.ndarray:
        """Joint transition matrix for one action vector, as a Kronecker product.

        Cached per action: there are two of them in every model this is used on, and building
        one costs a 3072 x 3072 outer product in the V5 geometry.
        """
        key = tuple(int(a) for a in np.asarray(action).ravel())
        T = self._T_cache.get(key)
        if T is None:
            T = np.asarray(self.B[0][:, :, key[0]], dtype=float)
            for f in range(1, self.num_factors):
                T = np.kron(T, np.asarray(self.B[f][:, :, key[f]], dtype=float))
            self._T_cache[key] = T
        return T

    def _marginals(self, b: np.ndarray):
        """Per-factor marginals of a joint belief, in pymdp's object-array shape.

        These are a REPORTING view, never fed back into inference. That is the whole
        difference between this agent and the one it is checking.
        """
        cube = b.reshape(self.num_states)
        qs = utils.obj_array(self.num_factors)
        for f in range(self.num_factors):
            axes = tuple(i for i in range(self.num_factors) if i != f)
            qs[f] = cube.sum(axis=axes)
        return qs

    def reset(self):
        self.qs_joint = self._joint_prior()
        self.qs = self._marginals(self.qs_joint)
        self.action = None
        self.q_pi = None
        self.G = None
        self._n_steps = 0
        # THE SEQUENCE LOG-EVIDENCE, log P(o_1..o_T), accumulated as the sum of the one-step
        # predictive normalisers. The filter already computes each one, so this is free, and it is
        # what makes parameter fitting possible: a candidate parameter value is scored by replaying
        # a fixed observation tape and reading this number off. Nothing else in the project can
        # produce a likelihood for a parameter that is not a hidden state.
        self.log_evidence = 0.0
        return self.qs

    # ------------------------------------------------------------------ #
    # Exact filtering.
    # ------------------------------------------------------------------ #
    def infer_states(self, observation, distr_obs: bool = False):
        """Exact Bayes update on the joint, mirroring pymdp's empirical-prior convention.

        pymdp uses D as the prior while no action has been taken and the propagated posterior
        afterwards. That convention is copied rather than improved on, because the free initial
        glance depends on it: the glance is delivered before any action exists.
        """
        if distr_obs:
            raise NotImplementedError(
                "ExactAgent takes index observations only. The one caller that needs "
                "distributional observations is regret.py, which does not use this agent.")
        if self.action is None:
            prior = self._joint_prior()
        else:
            prior = self._transition(self.action) @ self.qs_joint

        lik = np.ones(self._n_joint)
        for m, o in enumerate(np.asarray(observation).ravel()):
            lik = lik * self._L[m][int(o)]
        post = prior * lik
        total = post.sum()
        if total <= 0.0:
            # An observation with zero probability under every joint state. Impossible with the
            # support floors this project asserts, so it is a bug rather than a regime.
            raise FloatingPointError(
                "exact update produced zero posterior mass; an observation is impossible under "
                "every joint state, which the synth_floor / sig_floor constraints forbid")
        self.qs_joint = post / total
        self.qs = self._marginals(self.qs_joint)
        self._n_steps += 1
        # ``total`` is exactly P(o_t | o_1..o_{t-1}) under this model, because ``prior`` was
        # normalised and the likelihood is a proper conditional. Summing the logs gives the
        # sequence log-evidence with no extra work.
        self.log_evidence += float(np.log(total))
        return self.qs

    # ------------------------------------------------------------------ #
    # Exact expected free energy.
    # ------------------------------------------------------------------ #
    def _policy_terms(self, policy) -> tuple[float, float]:
        """(utility, information gain) for one policy, summed over its horizon.

        The information gain is I(s; o) under the propagated joint — the exact quantity
        pymdp's ``spm_MDP_G`` estimates from an outer product of marginals.
        """
        pol = np.asarray(policy)
        b = self.qs_joint
        util = 0.0
        info = 0.0
        for tau in range(pol.shape[0]):
            b = self._transition(pol[tau]) @ b
            qo = [Lm @ b for Lm in self._L]
            if self.use_utility:
                for m in range(self.num_modalities):
                    util += float(np.dot(qo[m], self._lnC[m]))
            if self.use_states_info_gain:
                # H(o) over the JOINT observation, which factorises given the state:
                #   p(o) = sum_s b(s) prod_m A[m][o_m, s]
                # This is the same quantity pymdp's ``spm_MDP_G`` computes; the difference is
                # that b here is the true joint belief rather than an outer product of
                # marginals. Accumulated as a tensor contraction over modalities, with the
                # state axis carried until the end, so the 160-cell V4 observation space and
                # the V5 one both cost a handful of BLAS calls rather than a Python loop.
                W = b[None, :] * self._L[0]
                for Lm in self._L[1:]:
                    W = W[..., None, :] * Lm
                joint_o = W.sum(axis=-1)
                info += _entropy(joint_o.ravel()) - float(np.dot(b, self._condH))
        return util, info

    def infer_policies(self):
        G = np.zeros(len(self.policies))
        for i, pol in enumerate(self.policies):
            util, info = self._policy_terms(pol)
            G[i] = util + info
        self.G = G
        self.q_pi = _softmax(self.gamma * G)
        return self.q_pi, G

    def sample_action(self):
        """Marginal action selection over the first step of each policy, as pymdp does."""
        if self.q_pi is None:
            self.infer_policies()
        action = np.zeros(self.num_factors)
        for f in range(self.num_factors):
            if f not in self.control_fac_idx:
                continue
            marg = np.zeros(self.num_controls[f])
            for i, pol in enumerate(self.policies):
                marg[int(np.asarray(pol)[0, f])] += self.q_pi[i]
            if self.action_selection == "deterministic":
                a = int(np.argmax(marg))
            else:
                a = int(self._rng.choice(len(marg), p=marg / marg.sum()))
            action[f] = a
        self.action = action
        return action

    # ------------------------------------------------------------------ #
    # Exact counterparts of the reported EFE quantities (D-4).
    # ------------------------------------------------------------------ #
    def efe_terms(self, policy) -> tuple[float, float]:
        """Exact (pragmatic, epistemic-total) for one policy.

        The exact counterpart of ``metrics.policy_efe_terms``, which cannot be used on this agent:
        it builds its predictive states from the factorised marginals, so on an exact agent it
        would report the mean-field answer under an exact agent's name. E1 and E5 are the
        experiments that record these, and this is what they would need to run under exact
        inference.
        """
        return self._policy_terms(policy)

    def epistemic_value_about(self, policy, factor: int = K.F_GOAL) -> float:
        """Exact expected information gain about ONE factor, I(s_factor ; o), over the horizon.

        This is Spec section 6's ``epistemic_value``, the quantity whose collapse to zero on
        machine-made content is E1's claim. ``metrics.epistemic_value`` computes it from an outer
        product of marginals; here the predictive state is the true joint, so the mutual
        information is the real one.
        """
        pol = np.asarray(policy)
        b = self.qs_joint
        total = 0.0
        cube_axes = tuple(i for i in range(self.num_factors) if i != factor)
        for tau in range(pol.shape[0]):
            b = self._transition(pol[tau]) @ b
            prior_marg = b.reshape(self.num_states).sum(axis=cube_axes)
            h_prior = _entropy(prior_marg)
            # Joint predictive over observations, carrying the state axis, then the posterior
            # marginal over the target factor for each observation cell.
            W = b[None, :] * self._L[0]
            for Lm in self._L[1:]:
                W = W[..., None, :] * Lm
            n_obs_cells = int(np.prod([Lm.shape[0] for Lm in self._L]))
            flat = W.reshape(n_obs_cells, self._n_joint)
            p_o = flat.sum(axis=1)
            exp_post_h = 0.0
            for c in np.nonzero(p_o > 1e-15)[0]:
                post = flat[c] / p_o[c]
                marg = post.reshape(self.num_states).sum(axis=cube_axes)
                exp_post_h += p_o[c] * _entropy(marg)
            total += max(h_prior - exp_post_h, 0.0)
        return float(total)

    def replay(self, tape) -> float:
        """Score a FIXED (observation, action) tape under this agent's model; returns log-evidence.

        The action sequence is imposed rather than chosen, which is the whole point. A candidate
        parameter value changes what the agent would decide, so letting it decide would compare two
        different observation sequences and the likelihood ratio would mean nothing. Fixing the
        actions makes the comparison a likelihood over one dataset, which is what a fit needs.
        """
        self.reset()
        for obs, action in tape:
            self.infer_states(obs)
            self.action = None if action is None else np.asarray(action, dtype=float)
        return float(self.log_evidence)

    # Anything that would silently be approximate.
    def infer_parameters(self, *a, **k):
        raise NotImplementedError("ExactAgent does not implement Dirichlet learning; the "
                                 "experiments that learn are not in V-1's scope")

    update_A = infer_parameters


# --------------------------------------------------------------------------- #
# Construction helpers mirroring the pymdp make_*_observer functions.
# --------------------------------------------------------------------------- #
def make_exact_agent(gm, D, cfg: Config, rng: np.random.Generator | None = None) -> ExactAgent:
    """Exact counterpart of ``generative_model.make_agent``, same agent-config block."""
    a = cfg.agent
    return ExactAgent(A=gm.A, B=gm.B, C=gm.C, D=D,
                      control_fac_idx=[K.F_ATTENTION],
                      policy_len=int(a.policy_len),
                      gamma=float(a.gamma),
                      action_selection=str(a.action_selection),
                      use_utility=bool(a.use_utility),
                      use_states_info_gain=bool(a.use_states_info_gain),
                      rng=rng)


def make_exact_v4_observer(world, rng: np.random.Generator, kappa: float | None = None,
                           d_i: float = 0.0) -> ExactAgent:
    """Exact counterpart of ``v4_model.make_v4_observer``.

    The RNG is consumed in exactly the same order as the approximate path — ``build_D`` first,
    then the spawned signature stream only when inexpertise is nonzero — so a matched pair of
    agents differs in the solver and in nothing else. The prior draw is a per-observer source
    of variance easily large enough to swamp the effect being measured, so letting it differ
    would make the whole comparison meaningless.
    """
    from .generative_model import build_D, build_observer_model
    from .observer import observer_sig_rng

    D = build_D(world.cfg, rng)
    gm = world.gm
    if float(d_i) > 0.0:
        gm = build_observer_model(world.gm, world.cfg, float(d_i),
                                  rng_sig=observer_sig_rng(rng), kappa=kappa)
    return make_exact_agent(gm, D, world.cfg, rng=rng)


def make_exact_v5_observer(world, rng: np.random.Generator,
                           mu_prior: np.ndarray | None = None) -> ExactAgent:
    """Exact counterpart of ``v5_model.make_v5_observer`` (without extra modalities)."""
    from . import foreign as FN
    from .v5_model import build_v5_D
    D = build_v5_D(world.cfg, rng, len(world.mu_levels), FN.NUM_REAL_GOALS,
                   world.n_subgoals, mu_prior=mu_prior)
    return make_exact_agent(world.gm, D, world.cfg, rng=rng)


# --------------------------------------------------------------------------- #
# The agreement check.
# --------------------------------------------------------------------------- #
def assert_agrees_on_factorised_case(tol: float = 1e-9) -> dict:
    """Where mean field is provably exact, the two agents must return identical beliefs.

    The construction used here is the one case in this model family where the joint posterior
    genuinely factorises: alpha = 1 for every provenance tier, so A[0] depends on the goal and
    not on provenance, A[1] depends on provenance and not on the goal, and A[2] pins attention.
    With no cross-factor coupling left in any modality, exact filtering and mean field agree —
    and if they do not, this file is wrong.

    Returns the measured discrepancies so a caller can report them rather than only assert.
    """
    from .config import load_config
    from .environment import Artifact, Environment
    from .generative_model import build_D, build_shared_model, make_agent

    cfg = load_config()
    gm = build_shared_model(cfg, alpha_overrides={name: 1.0 for name in K.PROVENANCE_NAMES})
    rng = np.random.default_rng(7)
    D = build_D(cfg, rng)

    approx = make_agent(gm, D, cfg)
    exact = make_exact_agent(gm, D, cfg)
    approx.reset()
    exact.reset()

    env = Environment(cfg, gm, np.random.default_rng(11), honesty=1.0, signing_rate=1.0)
    art = Artifact(provenance=K.CREATOR, goal=1, declared_signal=K.TRUTHFUL_SIGNAL[K.CREATOR])

    worst_goal = worst_prov = 0.0
    obs_rng = np.random.default_rng(13)
    for _ in range(8):
        obs = env.observation(art, K.DEEP, obs_rng)
        qa = approx.infer_states(obs)
        qe = exact.infer_states(obs)
        approx.action = exact.action = _deep_action_vec(approx)
        worst_goal = max(worst_goal, float(np.max(np.abs(
            np.asarray(qa[K.F_GOAL]) - np.asarray(qe[K.F_GOAL])))))
        worst_prov = max(worst_prov, float(np.max(np.abs(
            np.asarray(qa[K.F_PROVENANCE]) - np.asarray(qe[K.F_PROVENANCE])))))

    assert worst_goal < tol and worst_prov < tol, (
        f"exact and mean-field disagree on a construction where mean field is provably exact "
        f"(goal {worst_goal:.2e}, provenance {worst_prov:.2e}); this is a bug in exact.py, not "
        f"a finding about the model")
    return {"max_goal_discrepancy": worst_goal, "max_provenance_discrepancy": worst_prov,
            "tolerance": tol}


def _deep_action_vec(agent) -> np.ndarray:
    a = np.zeros(len(agent.num_controls))
    a[K.F_ATTENTION] = K.DEEP
    return a
