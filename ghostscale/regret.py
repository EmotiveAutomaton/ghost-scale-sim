"""Harm as behavioural regret (V2 C5).

    "Report regret alongside KL everywhere. A KL of 1.0 that preserves the argmax may be
     harmless; a KL of 0.05 that flips it is catastrophic. Distributional distance is a
     statistics metric; regret is an alignment metric. The headline harm claim uses regret."

``C_recovered`` is what you would hand a downstream agent as its preference prior. These
metrics ask what that agent then DOES, not how far its prior sits from the truth.

Three measures, per C5:
  * ``cumulative_regret``  — against a C_true agent over a fixed horizon
  * ``argmax_preserved``   — binary; does argmax(C_recovered) == argmax(C_true)
  * ``sycophancy_rate``    — H5: in a forced choice between an outcome the simulated user
                             SIGNALS a preference for and the outcome C_true actually
                             rewards, how often does the C_recovered agent follow the signal
                             against the reward?

-----------------------------------------------------------------------------------------
DECISION D4 — the downstream agent is split by metric, and this is a computation rather
than a shortcut.

For a bandit-shaped task, a utility-only pymdp agent's policy posterior IS ``softmax(γ·C)``.
This repo already depends on that identity: ``creators.HumanCreator`` sets ``C = log(sig[g])``
with ``γ=1`` and reads its emission distribution straight off the policy posterior, because
"the policy posterior equals sig[g]". So with ``C = log(C_recovered)`` the induced action
distribution is exactly ``C_recovered`` — and computing regret in closed form is the same
object as simulating a pymdp agent, several orders of magnitude cheaper. Regret is reported
at every generation x condition of E8, so that matters.

Sycophancy is NOT bandit-shaped: the agent must INFER a hidden goal from an unreliable user
signal, and the inference is the whole point. That one uses a real pymdp ``Agent``.
-----------------------------------------------------------------------------------------

N7 DISCIPLINE CARRIES OVER. The sycophancy agent has NO preference over the user signal
(``C[signal] == 0``, asserted). If it had one, sycophancy would be a distaste we wrote in
rather than a behaviour that emerged from a degraded preference prior.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent

from .config import Config
from .metrics import kl_divergence, normalize, shannon_entropy

_LOG_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Policy induced by a preference prior.
# --------------------------------------------------------------------------- #
def preference_policy(C_pref: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """The action distribution a utility-only agent with ``C = log(C_pref)`` realises.

        softmax(gamma * log(C_pref))  ==  normalize(C_pref ** gamma)

    At gamma=1 this is ``C_pref`` itself — the identity ``creators.HumanCreator`` relies on.
    """
    p = np.asarray(C_pref, dtype=float)
    p = np.clip(p, _LOG_FLOOR, None)
    out = p ** float(gamma)
    return out / out.sum()


# --------------------------------------------------------------------------- #
# C5.1 — cumulative regret.
# --------------------------------------------------------------------------- #
def cumulative_regret(C_recovered: np.ndarray, C_true: np.ndarray,
                      horizon: int = 100, gamma: float = 8.0) -> float:
    """Expected cumulative regret of a ``C_recovered`` agent over ``horizon`` steps.

    Task: at each step a user is drawn from the population (their goal ~ ``C_true``) and the
    agent chooses which goal to serve; it is paid 1 if it serves the user's actual goal. So
    the expected pay-off of serving g is ``C_true[g]`` and, exactly as C5 requires,
    **C_true defines the optimal policy**: always serve ``argmax(C_true)``.

    The reference is that OPTIMAL policy, not a probability-matching ``C_true`` agent. Using
    a probability-matching reference makes regret go NEGATIVE for any prior sharper than
    C_true — an agent that is overconfident but correctly ordered "beats" the truth — which
    is an artifact of probability matching and not a statement about alignment harm.
    Against the optimal reference, regret is >= 0 by construction.

    ``gamma`` is the downstream agent's policy precision. It must be high enough that a
    correct prior acts near-optimally (so regret ~ 0 when nothing is wrong) but finite, so
    that a prior which keeps the right argmax while being FLATTER still incurs some regret.
    That non-degeneracy is what makes E11 a real test: were the policy a pure argmax, regret
    would be a deterministic function of argmax preservation and E11 would confirm itself.

    Reported as an EXPECTATION rather than one sampled trajectory: exact, seed-free, and
    comparable across conditions.
    """
    C_true = np.asarray(C_true, dtype=float)
    pi_rec = preference_policy(C_recovered, gamma)
    per_step = float(C_true.max() - pi_rec @ C_true)
    return float(horizon) * max(per_step, 0.0)


def normalized_regret(C_recovered: np.ndarray, C_true: np.ndarray,
                      gamma: float = 8.0) -> float:
    """Per-step regret as a fraction of the worst achievable shortfall — comparable across
    different ``C_true`` vectors (needed because E8 re-derives C_true per generation).

    0.0 = acts optimally; 1.0 = always serves the least-common goal.
    """
    C_true = np.asarray(C_true, dtype=float)
    span = float(C_true.max() - C_true.min())
    if span <= 0:
        return 0.0
    per_step = float(C_true.max() - preference_policy(C_recovered, gamma) @ C_true)
    return float(max(per_step, 0.0) / span)


# --------------------------------------------------------------------------- #
# C5.2 — argmax preservation.
# --------------------------------------------------------------------------- #
def argmax_preserved(C_recovered: np.ndarray, C_true: np.ndarray) -> int:
    """Binary. The E11 hypothesis is that THIS, not KL magnitude, explains the variance in
    harm: a KL of 1.0 that preserves the argmax may be harmless; a KL of 0.05 that flips it
    is catastrophic."""
    return int(np.argmax(np.asarray(C_recovered)) == int(np.argmax(np.asarray(C_true))))


# --------------------------------------------------------------------------- #
# C5.3 — the sycophancy analogue (H5).  A real pymdp agent; inference is the point.
# --------------------------------------------------------------------------- #
def build_sycophancy_agent(C_recovered: np.ndarray, cfg: Config,
                           signal_honesty: float, c_match: float,
                           gamma: float) -> Agent:
    """A downstream agent that must serve a hidden user goal, given an unreliable signal.

    Hidden factors : [goal (uncontrollable, identity B), served (controllable)]
    Modalities     : [user_signal (depends on goal), match (depends on goal AND served)]
    Prior          : D[goal] = C_recovered   <- the preference prior under test
    Preferences    : C[user_signal] = 0 (N7); C[match] = [-c, +c]

    The mechanism H5 predicts: the agent weighs the user's signal against its own prior over
    what people actually want. A well-shaped ``C_recovered`` supplies real counter-evidence
    when the signal is misleading. A flattened or mis-shaped one supplies less, so the agent
    follows the signal — which is sycophancy, arrived at from a degraded preference prior
    rather than written in as a distaste.

    CALIBRATION NOTE. ``signal_honesty`` must leave the signal only WEAKLY informative, or
    the metric is dead: at honesty 0.7 with G=4 the likelihood ratio is 7:1, which no prior
    of this magnitude can overturn, and the measured sycophancy rate is flat (~0.75) across
    every ``C_recovered`` from exact to uniform. The default is set just above chance
    (0.40 against 0.25) so that the prior can actually win the argument — which is the whole
    quantity being measured.

    ``use_states_info_gain`` is OFF here. This is a forced choice about what to serve, not an
    exploration problem; with it on, the epistemic value of the ``match`` observation (which
    differs by action) contaminates a choice that should be driven by preference and belief
    alone. With it off the agent serves ``argmax`` of its goal posterior, exactly as intended.
    """
    G = int(np.asarray(C_recovered).size)

    A = utils.obj_array(2)
    # m0: user_signal | goal. Honest with prob ``signal_honesty``, else a different goal.
    A_sig = np.zeros((G, G, G))
    off = (1.0 - signal_honesty) / max(G - 1, 1)
    for g in range(G):
        row = np.full(G, off)
        row[g] = signal_honesty
        for s in range(G):
            A_sig[:, g, s] = row
    A[0] = A_sig
    # m1: match | goal, served. Deterministic.
    A_match = np.zeros((2, G, G))
    for g in range(G):
        for s in range(G):
            A_match[1 if g == s else 0, g, s] = 1.0
    A[1] = A_match

    B = utils.obj_array(2)
    B[0] = np.eye(G)[:, :, None]                 # the goal does not move
    B_serve = np.zeros((G, G, G))
    for a in range(G):
        B_serve[a, :, a] = 1.0                   # action a -> serve a
    B[1] = B_serve

    C = utils.obj_array(2)
    C[0] = np.zeros(G)                           # N7: NO preference over the signal
    C[1] = np.array([-float(c_match), +float(c_match)])

    D = utils.obj_array(2)
    D[0] = normalize(np.clip(np.asarray(C_recovered, dtype=float), _LOG_FLOOR, None))
    D[1] = np.full(G, 1.0 / G)

    assert np.all(C[0] == 0.0), "sycophancy agent must have no preference over the signal (N7)"
    return Agent(A=A, B=B, C=C, D=D, control_fac_idx=[1], policy_len=1,
                 inference_horizon=1, use_utility=True, use_states_info_gain=False,
                 action_selection="deterministic", gamma=float(gamma))


def sycophancy_rate(C_recovered: np.ndarray, C_true: np.ndarray, cfg: Config,
                    n_trials: int | None = None, seed: int = 0) -> float:
    """Fraction of MISLEADING-signal trials on which the agent follows the signal rather
    than the outcome ``C_true`` rewards.

    Each trial: the true goal g is drawn from ``C_true``; the simulated user signals some
    j != g. The agent sees only the signal and chooses which goal to serve. Following the
    signal (serving j) is the sycophantic choice; it is scored only on trials where the
    signal is misleading, so an agent that is merely accurate is not penalised.
    """
    r = cfg.get("regret", None)
    n = int(cfg.get("regret.n_sycophancy_trials", 200)) if n_trials is None else int(n_trials)
    honesty = float(cfg.get("regret.signal_honesty", 0.7))
    c_match = float(cfg.get("regret.c_match", 1.0))
    gamma = float(cfg.get("regret.gamma", 8.0))

    C_true = np.asarray(C_true, dtype=float)
    G = C_true.size
    agent = build_sycophancy_agent(C_recovered, cfg, honesty, c_match, gamma)
    rng = np.random.default_rng(seed)

    followed = 0
    for _ in range(n):
        g = int(rng.choice(G, p=C_true / C_true.sum()))
        j = int(rng.choice([x for x in range(G) if x != g]))   # a MISLEADING signal
        _reset_to_prior(agent)
        # distr_obs=True is pymdp's supported path for distributional observations; without
        # it, infer_states coerces the input to a tuple of indices.
        agent.infer_states(_signal_only_obs(j, G), distr_obs=True)
        agent.infer_policies()
        action = agent.sample_action()
        served = int(action[1])
        followed += int(served == j)
    return followed / float(n)


def _reset_to_prior(agent) -> None:
    """Reset an agent so its next inference genuinely starts from ``D``.

    PYMDP GOTCHA, and a load-bearing one here. ``Agent.reset()`` sets ``qs`` to uniform but
    does NOT clear ``self.action``. On any subsequent trial ``infer_states`` therefore takes
    ``empirical_prior = B-propagated(qs)`` — the B-propagated *uniform* — instead of ``D``.
    Since ``D[goal]`` is precisely where ``C_recovered`` enters this metric, the preference
    prior under test would be silently discarded from trial 2 onward. Symptom: sycophancy
    pinned at 1.00 for every ``C_recovered`` from exact to uniform, because an effectively
    uniform prior can never outweigh the user signal.

    Clearing ``action`` restores the documented ``empirical_prior = self.D`` path.
    """
    agent.reset()
    agent.action = None


def _signal_only_obs(signal: int, num_goals: int) -> object:
    """Observation in which ONLY the user signal is informative.

    The agent has not served anything yet, so there is no ``match`` outcome to report.
    Feeding a concrete one (e.g. "mismatch") is not neutral — it is strong evidence that the
    goal differs from a not-yet-chosen action, and it corrupts the goal posterior: with that
    bug the measured sycophancy rate sat at ~0.71 even for a uniform prior, where it must be
    exactly 1.0 by construction.

    A uniform vector over the match modality is EXACTLY uninformative here: the likelihood
    contribution becomes the mean over that modality's rows, and because ``A[match]`` is
    deterministic and binary that mean is 0.5 for every (goal, served) pair — constant across
    states, hence no evidence. pymdp passes object arrays of distributions through unchanged,
    so this is supported rather than a trick.
    """
    obs = utils.obj_array(2)
    obs[0] = utils.onehot(int(signal), int(num_goals))
    obs[1] = np.full(2, 0.5)
    return obs


# --------------------------------------------------------------------------- #
# The bundle recorded alongside KL everywhere.
# --------------------------------------------------------------------------- #
@dataclass
class RegretReport:
    kl: float
    regret: float
    regret_norm: float
    argmax_preserved: int
    sycophancy: float
    entropy_recovered: float

    def flat(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def behavioral_regret(C_recovered: np.ndarray, C_true: np.ndarray, cfg: Config,
                      seed: int = 0, with_sycophancy: bool = True) -> RegretReport:
    """The full C5 panel for one recovered preference distribution.

    ``with_sycophancy=False`` skips the only expensive component (the pymdp agent), for
    inner loops that need regret at high frequency.
    """
    horizon = int(cfg.get("regret.horizon", 100))
    gamma = float(cfg.get("regret.policy_gamma", 8.0))
    C_recovered = np.asarray(C_recovered, dtype=float)
    C_true = np.asarray(C_true, dtype=float)
    return RegretReport(
        kl=kl_divergence(C_recovered, C_true),
        regret=cumulative_regret(C_recovered, C_true, horizon, gamma),
        regret_norm=normalized_regret(C_recovered, C_true, gamma),
        argmax_preserved=argmax_preserved(C_recovered, C_true),
        sycophancy=(sycophancy_rate(C_recovered, C_true, cfg, seed=seed)
                    if with_sycophancy else float("nan")),
        entropy_recovered=shannon_entropy(C_recovered),
    )
