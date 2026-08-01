"""V6 — the Intent Extraction Limit, implemented.

WHAT THIS MODULE IS FOR, AND WHY IT IS DIFFERENT FROM V1-V5.

Every previous version asked a new question about the world. This one asks whether the
simulation and the theory it claims to implement are the same object. Reading the preprint's
formal model against the shipped code found three terms in the equation with no counterpart in
the code, and one of them is a MECHANISM DISAGREEMENT rather than an omission.

    Psi = sigmoid( k * (omega - theta_EC(kappa)) ) * [ -ln(1 - kappa) ] * D_KL( Q(R|tau) || P0 )
    theta_EC(kappa) = theta_base(E) + lambda * D_KL( Q(R|tau) || Pc(R) )

    term                              in the shipped code?
    -ln(1-kappa)                      yes, metrics.trust_factor
    D_KL(Q || P0)                     yes; V6 reports it beside the signed measure
    lambda * D_KL(Q || Pc)            yes, the acceptance gate
    theta_base(E)                     NO. No depletion state exists anywhere.
    sigmoid(k * ...)                  NO. Replaced by a binary engagement decision.
    kappa -> 1  =>  theta_EC -> 0     NO. kappa and theta never interact.

THE THIRD ONE IS THE INTERESTING ONE. The preprint's trust exploit works by trust SUPPRESSING
the disgust threshold, so the integration gate is held open while the bottom-up signal
collapses. The code's trust exploit works by the label channel OUT-ARGUING the content channel,
with an analytic crossover at kappa = 0.538 (diagnostics D-1). Both produce the phenomenon and
they are not the same claim: the coupled version predicts an exploit on a reader that CORRECTLY
BELIEVES the content is machine-made, which the channel race structurally cannot produce.

So the coupling is built as a switch rather than a replacement, defaulting OFF, and the
comparison is the experiment. That is the whole design principle of this module: every addition
is independently switchable and off by default, so that a V6 run with everything off reproduces
V5 elementwise (null N23) and any single addition is attributable on its own.

-----------------------------------------------------------------------------------------
WHAT IS NOT HERE, RECORDED SO THE BOUNDARY IS A COMMITMENT RATHER THAN AN OUTCOME.

No creator agent. Zahavian signalling and the whole reputational-cost security argument need a
maker that chooses when to lie, which is an agent with its own objective. The preprint gives
that a formal appendix and the simulation does not test one line of it. Named as an open hole.

No recursion. The claim that the same extraction runs at every scale needs an unbounded
self-similar structure, and the state space does not survive it. What IS built is the scaled
version: the same extraction on a sub-window of one artifact (``subwindow_recovery``), which
tests scale-invariance without recursion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as K
from . import foreign as FN
from .config import Config
from .metrics import kl_divergence, normalize, trust_factor

_EPS = 1e-12


# =========================================================================== #
# §4.1  Metabolic reserve.
# =========================================================================== #
@dataclass
class MetabolicReserve:
    """The reader's energy budget, carried ACROSS encounters. The equation's ``E``.

    THIS IS THE LARGEST GAP BETWEEN THE PREPRINT AND THE SHIPPED CODE. ``theta_base(E)`` is in
    the formal model, with its own symbol, and nothing in V1-V5 implements it: effort cost is a
    constant, every reader arrives fresh, and nothing that happens in one encounter changes what
    the reader will spend on the next. So the essay's central cultural claim -- that the damage
    ACCUMULATES IN THE READER and ends in apathy -- was not untested. It was unrepresentable.

    DEPLETION IS DRIVEN BY UNRESOLVED ENGAGEMENT, NOT BY EXPOSURE, and that distinction is the
    whole mechanism. A reader that looks closely and works the artifact out has spent effort and
    got something for it; it is not depleted. A reader that looks closely and gets nothing has
    paid and been disappointed, which is the essay's "your brain keeps being disappointed after
    its search for meaning". Making depletion a function of exposure alone would produce H6.1 by
    construction and the null would be unfalsifiable.

    That is also what makes N22 a real null rather than a formality: on a corpus where everything
    resolves, a reader may look as much as it likes and must not deplete.
    """

    e: float = 1.0
    rate: float = 0.08              # fall per fully-unresolved engaged encounter
    recovery: float = 0.02          # return per encounter toward the ceiling
    floor: float = 0.20             # a reader is never entirely spent
    scale: float = 1.0              # nats of threshold per unit of (1-E)/E
    history: list = field(default_factory=list)

    def update(self, engaged_fraction: float, resolved: float) -> float:
        """One encounter. ``resolved`` in [0,1] is how much of the goal the reader got.

        The cost is the PRODUCT of how much it looked and how little it got, which is the only
        form under which a reader that never looks and a reader that always succeeds are both
        undepleted, and those are exactly the two cases the null needs to hold for.
        """
        engaged = float(np.clip(engaged_fraction, 0.0, 1.0))
        got = float(np.clip(resolved, 0.0, 1.0))
        disappointment = engaged * (1.0 - got)
        self.e = float(np.clip(
            self.e - self.rate * disappointment + self.recovery * (1.0 - disappointment),
            self.floor, 1.0))
        self.history.append({"engaged": engaged, "resolved": got,
                             "disappointment": disappointment, "e": self.e})
        return self.e

    def theta_base(self, theta_0: float = 0.0) -> float:
        """The reserve's contribution to the disgust threshold: ``theta_base(E)``.

        Rises as the reserve falls, so a depleted reader disengages sooner. Written as
        ``theta_0 + scale * (1 - E) / E`` so that a full reserve contributes exactly ``theta_0``
        and the penalty grows as the reserve is spent, which is the shape the preprint's "scales
        inversely with available metabolic reserves" asks for.

        The FLOOR is what keeps this bounded, and it is a modelling claim rather than a
        convenience: a reader is never entirely spent, so the threshold is never infinite and
        even a maximally fatigued reader can still be moved by a sufficiently legible artifact.
        A model without the floor predicts terminal apathy, which is stronger than anything the
        theory claims.
        """
        e = max(float(self.e), float(self.floor))
        return float(theta_0 + float(self.scale) * (1.0 - e) / e)


# =========================================================================== #
# §4.2  The gate, with the coupling switch.
# =========================================================================== #
def disgust_threshold(value_divergence: float, kappa: float, reserve: MetabolicReserve | None,
                      lam: float = 1.0, theta_0: float = 0.0,
                      coupling: float = 0.0) -> float:
    """``theta_EC(kappa) = (theta_base(E) + lambda * D_KL(Q || Pc)) * (1 - kappa)^c``.

    ``coupling`` is ``c``. THE DEFAULT IS ZERO AND THAT IS DELIBERATE: at c = 0 this returns
    exactly what the V1-V5 gate returns, kappa does not enter, and the V5 reduction null holds.
    At c = 1 it is the preprint's stated behaviour, kappa -> 1 driving theta_EC -> 0.

    Building it as a switch rather than a replacement is the point. The two are different claims
    about why the trust exploit happens and the comparison is H6.2, so replacing one with the
    other would destroy the experiment.
    """
    base = float(theta_0) if reserve is None else reserve.theta_base(theta_0)
    raw = base + float(lam) * float(value_divergence)
    c = float(coupling)
    if c <= 0.0:
        return float(raw)
    k = float(np.clip(kappa, 0.0, 1.0 - 1e-9))
    return float(raw * (1.0 - k) ** c)


def gate(omega_precision: float, theta: float, k_gain: float = 8.0,
         leak: float = 0.0) -> float:
    """``leak + (1 - leak) * sigmoid(k * (omega - theta_EC))``.

    THE LEAK IS A V7 ADDITION AND THE THEORY ALWAYS HAD IT. The gate as V1-V6 built it can shut
    completely: a reader that rejects material integrates exactly none of it, which measured out at
    0.00 in E42. The preprint says otherwise, in as many words -- computing the value disagreement
    *itself* requires simulating the thing, and that simulation drives learning through gating
    imperfections, "likely the mechanism for indoctrination and propaganda".

    So a perfectly closed gate is not a strong reader. It is a missing term. You cannot look at
    something without taking a little of it on, because looking IS partly running it.

    DEFAULTS TO ZERO, which reproduces the V6 gate exactly (null N31). This is not adopted as the
    model's standing behaviour: turning it on changes every result in the repository, and an
    addition that silently rewrites the record is the accretion problem the repair pass was written
    against. What it changes is measured and reported; it is not switched on by fiat.
    """
    lk = float(np.clip(leak, 0.0, 1.0))
    return lk + (1.0 - lk) * _sigmoid_gate(omega_precision, theta, k_gain)


def _sigmoid_gate(omega_precision: float, theta: float, k_gain: float = 8.0) -> float:
    """``sigmoid(k * (omega - theta_EC))`` -- the graded gate the code never had.

    V1-V5 replaced this with a binary engagement decision, declared as a deviation. The
    consequence was not declared and matters here: a binary gate cannot express hesitating,
    half-looking, or a gate that is nearly closed, and it throws away most of what makes an
    ADDITIVE combination of cues interesting (§4.6), because any sum immediately collapses to a
    side of a threshold.

    As ``k_gain`` grows this reproduces the binary decision, which is null N24.
    """
    z = float(k_gain) * (float(omega_precision) - float(theta))
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


def psi_intent_extraction_limit(posterior: np.ndarray, prior: np.ndarray, kappa: float,
                                omega_precision: float, value_divergence: float,
                                reserve: MetabolicReserve | None = None,
                                lam: float = 1.0, theta_0: float = 0.0,
                                coupling: float = 0.0, k_gain: float = 8.0) -> dict:
    """The full Intent Extraction Limit, every term reported separately.

    Computed as a MEASURE and reported beside ``metrics.psi_analogue``, which is retained
    unchanged. The standing rule for this project is that no committed value is silently
    superseded, and psi_analogue is what every V1-V5 number was computed with.

    The terms are returned individually because the whole argument of this version is about
    WHICH TERM does the work, and a single scalar hides that.
    """
    theta = disgust_threshold(value_divergence, kappa, reserve, lam, theta_0, coupling)
    g = gate(omega_precision, theta, k_gain)
    tf = trust_factor(kappa)
    movement = kl_divergence(np.asarray(posterior, dtype=float),
                             np.asarray(prior, dtype=float))
    return {"psi": float(g * tf * movement), "gate": g, "theta_EC": theta,
            "trust_factor": tf, "movement": movement,
            "omega_precision": float(omega_precision),
            "value_divergence": float(value_divergence),
            "reserve": None if reserve is None else float(reserve.e)}


# =========================================================================== #
# §4.7  The values layer.
# =========================================================================== #
def build_values_map(n_goals: int, n_values: int = 2, rng=None) -> np.ndarray:
    """``M[value, goal]`` -- what values each goal implies. NON-INJECTIVE by construction.

    The gate in V1-V5 compares the recovered GOAL to the reader's value prior. The theory says
    something one step removed: you infer the goal, the goal implies values, and the VALUES
    decide whether the process is allowed to integrate. Read Mein Kampf and the goal is to
    persuade; the values are what stop the convergence.

    NON-INJECTIVE IS THE WHOLE POINT AND IT IS ASSERTED (null N26). If the map were a bijection
    on goals it would be the goal relabelled and the layer would buy nothing. What it buys is
    that TWO DIFFERENT GOALS CAN IMPLY THE SAME VALUES, so the gate opens for both -- which is
    "I disagree with what you were doing but we want the same things", a distinction the code
    could not previously represent at all.
    """
    ng = int(n_goals)
    nv = int(n_values)
    assert nv < ng, (
        f"a values map with {nv} values for {ng} goals cannot be non-injective. The layer "
        f"would be the goal renamed and null N26 exists to catch exactly that.")
    M = np.zeros((nv, ng))
    for g in range(ng):
        M[g % nv, g] = 1.0
    return M


def implied_values(goal_posterior: np.ndarray, values_map: np.ndarray) -> np.ndarray:
    """The value distribution a goal belief implies."""
    v = np.asarray(values_map, dtype=float) @ np.asarray(goal_posterior, dtype=float)
    return normalize(v)


def value_divergence_via_values(goal_posterior: np.ndarray, value_prior: np.ndarray,
                                values_map: np.ndarray) -> float:
    """The gate's divergence term, computed on VALUES rather than on goals (§4.7)."""
    return kl_divergence(implied_values(goal_posterior, values_map),
                         np.asarray(value_prior, dtype=float))


# =========================================================================== #
# §4.6  Aesthetic and social cues.
# =========================================================================== #
@dataclass
class CueChannels:
    """Two pre-inference cues that feed the engagement decision and nothing else.

    The reader in V1-V5 decides to look closely on expected information gain alone. The theory
    says two other things feed that decision before any inference has happened:

      * AESTHETIC SALIENCE -- the honeypot. A surface property that, in an honest world,
        predicts that depth is present. "It is something attention-grabbing, but not
        necessarily attention-keeping."
      * SOCIAL ENDORSEMENT -- presumed density. Extracting process is so expensive that you
        often cannot see it yourself, so you take someone's word that it is there.

    ADDITIVE IS PRE-REGISTERED, MULTIPLICATIVE IS BUILT ALONGSIDE (H6.8). The two disagree
    exactly at the corners -- ugly-but-endorsed and beautiful-but-unendorsed -- and those
    corners are where the interesting cases live, so the comparison is worth running rather
    than assuming.

    BOTH CHANNELS CARRY ZERO GOAL INFORMATION (null N29). If a cue told the reader anything
    about WHICH goal, it would be a second legibility channel and would quietly re-derive the
    label effect through the back door.
    """

    w_aesthetic: float = 0.0
    w_social: float = 0.0
    mode: str = "additive"          # or "multiplicative"

    def combine(self, information_gain: float, aesthetic: float, social: float) -> float:
        ig = float(information_gain)
        a = self.w_aesthetic * float(aesthetic)
        s = self.w_social * float(social)
        if self.mode == "multiplicative":
            return float(ig * (1.0 + a) * (1.0 + s))
        return float(ig + a + s)


def learned_salience_map(depth_levels, honest: bool = True, rng=None,
                         decouple: bool = False) -> dict:
    """What aesthetic salience a reader has LEARNED to expect at each depth.

    In the honest regime salience rises with depth, so the cue is a valid predictor and a reader
    that uses it is not making a mistake. ``decouple=True`` is the RLHF arm: salience is
    maximised at EVERY depth, including zero, so the cue keeps firing while the thing it
    predicted is gone.

    That is the alignment argument in miniature. A cue learned as a predictor of depth, then
    optimised directly, decouples from what it predicted -- and the reader keeps spending on a
    signal that no longer carries information. The prediction is over-engagement with no uptake,
    which is a THIRD failure mode: not the crash (disengaged, unresolved) and not the exploit
    (engaged, confidently wrong about provenance), but engaged, correct about provenance, and
    still empty-handed.
    """
    levels = list(depth_levels)
    if decouple:
        return {int(m): 1.0 for m in levels}
    lo, hi = float(min(levels)), float(max(levels))
    span = max(hi - lo, _EPS)
    return {int(m): float((float(m) - lo) / span) for m in levels}


# =========================================================================== #
# §4.10  Process recovery.
# =========================================================================== #
def process_recovery(subgoal_posteriors: list, true_modes: list, n_sub: int) -> dict:
    """How much of the maker's EXECUTION CHAIN the reader recovered.

    THE MEASURE THE PROJECT NEVER HAD, AND THE REASON E30 CAME BACK NULL. Every uptake measure
    in V1-V5 scores GOAL recovery. Depth is constructed so the goal is EXACTLY as recoverable at
    every depth -- that is the design commitment that stops depth being legibility renamed -- so
    a goal-uptake measure could not have moved with depth whatever was true of the reader, which
    E30's own write-up conceded and the repair pass then bounded near zero.

    Under the theory the experiment was measuring the wrong quantity. Depth should change how
    much of the PROCESS transfers, not how much of the goal. The reader has carried a posterior
    over the sub-goal chain in every V5 run ever made and nobody ever scored it.

    Returns accuracy against the true mode at each step, and the mean log-probability the reader
    assigned to the truth, which is the process analogue of error reduction and can go negative.
    """
    accs, lps = [], []
    for q, s_true in zip(subgoal_posteriors, true_modes):
        p = np.asarray(q, dtype=float)
        p = p / max(p.sum(), _EPS)
        accs.append(float(int(np.argmax(p)) == int(s_true)))
        lps.append(float(np.log(max(p[int(s_true)], _EPS))))
    chance = np.log(1.0 / max(int(n_sub), 1))
    return {"process_accuracy": float(np.mean(accs)) if accs else float("nan"),
            "process_logprob": float(np.mean(lps)) if lps else float("nan"),
            "process_error_reduction": float(np.mean(lps) - chance) if lps else float("nan"),
            "n_steps": len(accs)}


def subwindow_recovery(subgoal_posteriors: list, true_modes: list, n_sub: int,
                       fraction: float = 0.25) -> dict:
    """§4.8 -- the same extraction run on a SLICE of the artifact.

    The scaled version of the fractal claim. The full claim is that the extraction runs at every
    scale -- perspective inside a building inside a painting -- and needs an unbounded
    self-similar structure that this state space does not survive. What can be asked without
    recursion is whether recovery on a WINDOW returns the same SHAPE of answer as recovery on
    the whole, at reduced precision. If it does not, the extraction is tied to the artifact
    boundary and the fractal claim does not hold even in the small.
    """
    n = len(subgoal_posteriors)
    w = max(2, int(round(n * float(fraction))))
    whole = process_recovery(subgoal_posteriors, true_modes, n_sub)
    windows = []
    for start in range(0, max(n - w + 1, 1), max(w // 2, 1)):
        windows.append(process_recovery(subgoal_posteriors[start:start + w],
                                        true_modes[start:start + w], n_sub))
    acc = [x["process_accuracy"] for x in windows if np.isfinite(x["process_accuracy"])]
    return {"whole_accuracy": whole["process_accuracy"],
            "window_accuracy_mean": float(np.mean(acc)) if acc else float("nan"),
            "window_accuracy_sd": float(np.std(acc)) if acc else float("nan"),
            "n_windows": len(windows), "window_size": w}


# =========================================================================== #
# §4.9  Graded self-report.
# =========================================================================== #
def self_report_accuracy(mu: int, mu_levels=(1, 2, 3), base: float = 0.95,
                         decay: float = 0.30) -> float:
    """How reliably a maker can state its own goal, as a function of depth.

    THE AUTHOR'S CORRECTION, and it is a real one. The working note said the subconscious holds
    the process goals. It holds the PRACTISED ones: recently acquired, poorly modelled skills
    are held consciously and can be reported, and it is the tightly compressed, heavily
    automatised structure that becomes inaccessible to its own author.

    So self-report accuracy falls with depth. A master cannot tell you why the perspective
    works; a novice can tell you exactly which rule they were following, because they are still
    following it deliberately. That is the automaticity story running in the direction the
    theory actually needs -- and E33 has already shown the reader is unaffected, recovering the
    latent goal at 0.88 while the declared goal collapses.
    """
    lo, hi = float(min(mu_levels)), float(max(mu_levels))
    frac = (float(mu) - lo) / max(hi - lo, _EPS)
    return float(np.clip(base - decay * frac, 0.0, 1.0))


# =========================================================================== #
# §4.3  The wall: a non-invertible generator.
# =========================================================================== #
def build_noninvertible_family(sig_true: np.ndarray, n_states: int = 4,
                               collapse_to: int = 2) -> dict:
    """Machine content on the HUMAN block whose maker-state cannot be inverted from the surface.

    THE AUTHOR'S SHARPEST CRITIQUE OF THE CONSTRUCTION, and it lands. Readability and reader
    inexpertise are both built as scalars on a shared-vocabulary axis, differing only in whether
    the content moved or the reader moved. E32 found they behave OPPOSITELY, which is real
    evidence they are not the same thing, but the objection was about the build rather than the
    result: the wall in front of generated content is not a degree of vocabulary overlap.

    Humans cheat when reading other humans because they assume the maker's decisions bottom out
    in human-shaped sensory experience -- he felt sad, so the colours went that way -- and that
    assumption is what makes the inversion tractable. With a generative model there is no
    mapping translation. Not less overlap. NO INVERTIBLE ROUTE from the surface back to a state
    you could occupy.

    Built as a MANY-TO-ONE map: ``n_states`` maker states emit only ``collapse_to`` distinct
    surface distributions, on features the reader knows perfectly well. The reader has full
    vocabulary, sees familiar structure, and still cannot recover which state produced it.

    That predicts a signature neither existing condition produces: LEGIBLE AND EMPTY. Low
    uncertainty about what is on the surface, no recovery of what is behind it -- which is the
    complaint people actually make about generated text, and is not "I cannot parse this".

    Stays on the human block (null N30). Without that this is foreign content in new vocabulary
    and H6.5 is unfalsifiable.
    """
    sig = np.asarray(sig_true, dtype=float)
    ng, nf = sig.shape
    assert int(collapse_to) < int(n_states), (
        "a non-invertible family needs strictly fewer surfaces than maker states; with as many "
        "surfaces as states the map is a bijection and there is no wall")
    human = np.asarray(FN.HUMAN_FEATURES, dtype=int)

    surfaces = np.zeros((int(collapse_to), nf))
    for j in range(int(collapse_to)):
        members = [g for g in range(ng) if g % int(collapse_to) == j]
        surfaces[j] = normalize(sig[members].mean(axis=0))

    foreign_mass = float(surfaces[:, np.asarray(FN.FOREIGN_FEATURES, dtype=int)].sum(axis=1).max())
    assert foreign_mass < 0.10, (
        f"the non-invertible family carries {foreign_mass:.3f} mass on the FOREIGN block. That "
        f"makes it foreign content wearing a new name and H6.5 unfalsifiable; see null N30.")

    state_to_surface = {s: s % int(collapse_to) for s in range(int(n_states))}
    return {"surfaces": surfaces, "state_to_surface": state_to_surface,
            "n_states": int(n_states), "collapse_to": int(collapse_to),
            "human_support": human,
            "invertible": False,
            "max_foreign_mass": foreign_mass}


def noninvertible_sample(family: dict, maker_state: int, rng) -> int:
    """One feature from the non-invertible generator."""
    j = family["state_to_surface"][int(maker_state) % family["n_states"]]
    return int(rng.choice(family["surfaces"].shape[1], p=family["surfaces"][j]))


# =========================================================================== #
# §4.5  NO_MAKER as a first-class hypothesis.
# =========================================================================== #
def build_no_maker_signature(noise_free_synth: np.ndarray) -> np.ndarray:
    """The likelihood of "there is no maker here", matched to intent-empty content.

    V1-V5 gave the reader a machine-made HYPOTHESIS only in the sense that it could fail to
    settle on any human goal. There was never a hypothesis that SAYS there is no maker, so the
    reader could not conclude one -- it could only keep failing.

    That is not what the Ghost Scale is supposed to do. Its job is not to make you distrust a
    label; it is to let your brain RELAX. The preprint says so directly: the affordance lets the
    viewer "save energy calculating theta_EC by quickly feeding the user the answer of a
    low-value kappa", and to climb "into higher-order reasoning, such as guessing a prompt's
    motivation". A reader that has concluded there is no maker should stop cleanly: resolved,
    disengaged, and NOT fabricating.

    Structurally this is the move V4 made with EXPLORE, pointed at a different target, so it
    inherits EXPLORE's failure mode and its null. If NO_MAKER absorbs human work it is a uniform
    fallback and it destroys every result in the project (null N27).
    """
    return normalize(np.asarray(noise_free_synth, dtype=float))


def assert_no_maker_does_not_absorb(no_maker_sig: np.ndarray, sig_true: np.ndarray,
                                    floor: float = 0.20) -> dict:
    """Null N27 -- the same check V4 applied to EXPLORE, for the same reason.

    A hypothesis that sits close to every human goal absorbs human work and trivially destroys
    the results. V4 caught exactly this: at V1-V3 cardinality ``sig_EXPLORE`` was flat over the
    whole feature space and sat four times closer to synthetic content than to any goal, so E19
    could only ever have returned one answer.
    """
    from .metrics import js_divergence
    d = [js_divergence(no_maker_sig, np.asarray(sig_true, dtype=float)[g])
         for g in range(np.asarray(sig_true).shape[0])]
    worst = float(np.min(d))
    assert worst >= floor, (
        f"NO_MAKER sits {worst:.4f} nats from the nearest human goal (floor {floor}). It would "
        f"absorb human work, which is null N27 and is the failure V4 caught with EXPLORE.")
    return {"min_js_to_human_goal": worst, "floor": floor,
            "js_to_each_goal": [float(x) for x in d]}


# =========================================================================== #
# §4.4  The machine-matched reader.
# =========================================================================== #
def build_machine_matched_signatures(family: dict, n_goals: int) -> np.ndarray:
    """A reader whose hypothesis family matches the MACHINE generator rather than the human one.

    The author's prediction, and it is a genuine forward one: someone expert in diffusion could
    ratchet through generated work using AI skill instead of art skill, so the people most
    familiar with these systems would be least affected.

    The darker second-order prediction is the one worth measuring. If expertise SUBSTITUTES
    rather than stacks, this reader recovers machine content AND LOSES HUMAN CONTENT by a
    comparable margin -- a crossover, not a dominance. The adaptation that protects you from the
    crash is the same adaptation that costs you the human channel.

    Everything else about the reader is identical, so the 2x2 is clean and the interesting cell
    is the machine-matched reader on human work.
    """
    surfaces = np.asarray(family["surfaces"], dtype=float)
    out = np.zeros((int(n_goals), surfaces.shape[1]))
    for g in range(int(n_goals)):
        out[g] = surfaces[g % surfaces.shape[0]]
    return out


# =========================================================================== #
# Config plumbing.
# =========================================================================== #
@dataclass
class V6Switches:
    """Every V6 addition, independently switchable, ALL OFF BY DEFAULT.

    This is the pre-mortem's answer to "too many additions at once, so nothing is attributable".
    With every switch off, a V6 run reproduces V5 elementwise, which is null N23, and any single
    addition can be turned on alone.
    """

    depletion: bool = False
    coupling: float = 0.0
    k_gain: float = 8.0
    cues: bool = False
    cue_mode: str = "additive"
    values_layer: bool = False
    no_maker: bool = False
    noninvertible: bool = False
    machine_matched: bool = False
    graded_self_report: bool = False

    @classmethod
    def from_config(cls, cfg: Config) -> "V6Switches":
        g = cfg.get
        return cls(
            depletion=bool(g("v6.depletion.enabled", False)),
            coupling=float(g("v6.gate.coupling", 0.0)),
            k_gain=float(g("v6.gate.k_gain", 8.0)),
            cues=bool(g("v6.cues.enabled", False)),
            cue_mode=str(g("v6.cues.mode", "additive")),
            values_layer=bool(g("v6.values.enabled", False)),
            no_maker=bool(g("v6.no_maker.enabled", False)),
            noninvertible=bool(g("v6.noninvertible.enabled", False)),
            machine_matched=bool(g("v6.machine_matched.enabled", False)),
            graded_self_report=bool(g("v6.self_report.graded", False)),
        )

    def all_off(self) -> bool:
        return not (self.depletion or self.coupling > 0 or self.cues or self.values_layer
                    or self.no_maker or self.noninvertible or self.machine_matched
                    or self.graded_self_report)
