"""Affect ownership and strategic communication (spec §3.3, trunk A).

A SOURCE holds a belief B about a world state, faces objective content support S, and speaks
with a desired audience appraisal A^a, an evidence-selection policy, a willingness to correct,
and a private/off-audience action. "Honest warning", "sincere fanatic", "strategic propagandist"
and "neutral report" name REGIONS of that factorial and nothing else: a discriminator is valid
only if it predicts a counterfactual action (the correction event, the private action) where
the regions diverge.

Owners are separate by construction: the reader's own induced response is a planted reader-side
appraisal function of the artifact's intensity; the maker's appraisal is a function of its
belief; the intended audience appraisal is the source's A^a. None is written into another.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from . import common as C

BELIEFS = ("threat", "benefit", "neutral")
SUPPORT = ("supports", "contradicts", "none")
APPRAISALS = ("alarm", "calm", "admire")
POLICIES = ("full", "cherry_pick", "fabricate")
CORRECT = ("corrects", "doubles_down")
PRIVATE = ("consistent", "inconsistent")
INTENSITY = ("low", "high")
N_EVID = 8                 # evidence tokens per artifact
REGIONS = {
    "honest_warning": {"belief": "threat", "support": "supports", "appraisal": "alarm", "policy": "full", "correct": "corrects", "private": "consistent"},
    "sincere_fanatic": {"belief": "threat", "support": "contradicts", "appraisal": "alarm", "policy": "cherry_pick", "correct": "doubles_down", "private": "consistent"},
    "strategic_propagandist": {"belief": "neutral", "support": "contradicts", "appraisal": "alarm", "policy": "fabricate", "correct": "doubles_down", "private": "inconsistent"},
    "neutral_report": {"belief": "neutral", "support": "none", "appraisal": "calm", "policy": "full", "correct": "corrects", "private": "consistent"},
}
SOURCE_STATES = list(product(range(len(BELIEFS)), range(len(SUPPORT)), range(len(APPRAISALS)), range(len(POLICIES)),
                             range(len(CORRECT)), range(len(PRIVATE))))
N_SOURCE_STATES = len(SOURCE_STATES)
_SIDX = {s: i for i, s in enumerate(SOURCE_STATES)}


def source(rng: np.random.Generator, region: str | None = None, **overrides) -> dict:
    if region is not None:
        s = dict(REGIONS[region])
    else:
        s = {"belief": BELIEFS[int(rng.integers(3))], "support": SUPPORT[int(rng.integers(3))], "appraisal": APPRAISALS[int(rng.integers(3))],
             "policy": POLICIES[int(rng.integers(3))], "correct": CORRECT[int(rng.integers(2))], "private": PRIVATE[int(rng.integers(2))]}
    s.update(overrides)
    s["intensity"] = s.get("intensity", INTENSITY[int(rng.integers(2))])
    s["reliability"] = float(s.get("reliability", rng.uniform(0.5, 0.95)))
    return s


def state_index(s: dict) -> int:
    return _SIDX[(BELIEFS.index(s["belief"]), SUPPORT.index(s["support"]), APPRAISALS.index(s["appraisal"]),
                  POLICIES.index(s["policy"]), CORRECT.index(s["correct"]), PRIVATE.index(s["private"]))]


# --------------------------------------------------------------------------- #
# Emission: assertion, evidence tokens, intensity tokens, and the counterfactual probes.
# --------------------------------------------------------------------------- #
def _evidence_pool(support: str) -> np.ndarray:
    """Token polarity distribution of the world's available evidence: +1 supports the claim, -1
    contradicts, 0 is neutral."""
    return {"supports": np.array([0.7, 0.1, 0.2]), "contradicts": np.array([0.1, 0.7, 0.2]), "none": np.array([0.2, 0.2, 0.6])}[support]


SUPPORTING = np.array([0.7, 0.1, 0.2])


def emission_policy(support: str, policy: str) -> np.ndarray:
    """What the source EMITS. Full selection shows the pool; cherry-picking and fabrication both
    reproduce the supporting pool's distribution exactly, so an honest warning, a sincere fanatic
    and a strategic propagandist collide on the artifact by construction (spec §3.3: a valid
    discriminator predicts a counterfactual action; the artifact is not one)."""
    if policy == "full":
        return _evidence_pool(support)
    return SUPPORTING


def assertion_of(s: dict) -> str:
    """The claim the source makes: a source seeking alarm asserts a threat whatever it believes;
    otherwise it asserts what it believes."""
    return "threat" if s["appraisal"] == "alarm" else s["belief"]


P_HIGH = {"alarm": 0.85, "calm": 0.15, "admire": 0.5}       # the planted intensity law: sought alarm speaks loud, sought calm quiet


def speak(s: dict, rng: np.random.Generator) -> dict:
    """One artifact: the assertion, N_EVID evidence tokens (+1/-1/0 polarity) under the emission
    policy, and an intensity that follows the DESIRED appraisal (the intended-effect owner)."""
    pol = emission_policy(s["support"], s["policy"])
    tokens = [int(rng.choice(3, p=pol)) for _ in range(N_EVID)]      # 0:+1 supports, 1:-1 contradicts, 2:neutral
    high = bool(rng.random() < P_HIGH[s["appraisal"]])
    inten = 3 if high else 1
    return {"assertion": assertion_of(s), "evidence": tokens, "intensity": inten, "reliability_tag": s["reliability"]}


def correction_event(s: dict, rng: np.random.Generator) -> int:
    """The source meets contradicting evidence: 1 = corrects, 0 = doubles down (with noise)."""
    base = 0.85 if s["correct"] == "corrects" else 0.15
    return int(rng.random() < base)


def private_action(s: dict, rng: np.random.Generator) -> dict:
    """Off-audience action: whether it is consistent with the belief (with noise), and which
    belief it acts on - the held belief with 0.9 fidelity when consistent, any belief when not.
    A strategic assertion can mask the belief; the private action cannot."""
    base = 0.85 if s["private"] == "consistent" else 0.15
    consistent = int(rng.random() < base)
    if consistent:
        acted = BELIEFS.index(s["belief"]) if rng.random() < 0.9 else int(rng.integers(3))
    else:
        acted = int(rng.integers(3))
    return {"consistent": consistent, "acted": acted}


# --------------------------------------------------------------------------- #
# Owners.
# --------------------------------------------------------------------------- #
def reader_response(art: dict, sensitivity: float = 1.0) -> float:
    """The reader's own induced appraisal: a planted function of intensity (0..1)."""
    return float(1.0 - np.exp(-sensitivity * art["intensity"] / 3.0))


def maker_appraisal(s: dict) -> str:
    return {"threat": "alarm", "benefit": "admire", "neutral": "calm"}[s["belief"]]


def intended_effect(s: dict) -> str:
    return s["appraisal"]


# --------------------------------------------------------------------------- #
# The exact reader over the source factorial.
# --------------------------------------------------------------------------- #
def loglik_artifact(art: dict, assume_full: bool = False) -> np.ndarray:
    """log P(assertion, evidence | source state) for every state, mirroring ``speak``. A reader
    that ``assume_full`` takes evidence at face value: every policy is read as full selection."""
    out = np.zeros(N_SOURCE_STATES)
    for i, (b, sup, ap, pol, co, pv) in enumerate(SOURCE_STATES):
        s = {"belief": BELIEFS[b], "appraisal": APPRAISALS[ap]}
        p = emission_policy(SUPPORT[sup], "full" if assume_full else POLICIES[pol])
        ll = np.log(0.9 if art["assertion"] == assertion_of(s) else 0.05)
        ll += sum(np.log(p[t]) for t in art["evidence"])
        out[i] = ll
    return out


def loglik_correction(obs: int) -> np.ndarray:
    out = np.zeros(N_SOURCE_STATES)
    for i, (_, _, _, _, co, _) in enumerate(SOURCE_STATES):
        base = 0.85 if CORRECT[co] == "corrects" else 0.15
        out[i] = np.log(base if obs == 1 else 1 - base)
    return out


def loglik_private(obs) -> np.ndarray:
    consistent = obs["consistent"] if isinstance(obs, dict) else int(obs)
    acted = obs.get("acted") if isinstance(obs, dict) else None
    out = np.zeros(N_SOURCE_STATES)
    for i, (b, _, _, _, _, pv) in enumerate(SOURCE_STATES):
        base = 0.85 if PRIVATE[pv] == "consistent" else 0.15
        out[i] = np.log(base if consistent == 1 else 1 - base)
        if acted is not None:
            if consistent == 1:
                out[i] += np.log((0.9 + 0.1 / 3) if acted == b else 0.1 / 3)
            else:
                out[i] += np.log(1.0 / 3)
    return out


def loglik_appraisal_cue(art: dict, cue: str) -> np.ndarray:
    """A weak cue about the intended appraisal carried by intensity: high intensity is more likely
    under 'alarm'."""
    out = np.zeros(N_SOURCE_STATES)
    for i, (_, _, ap, _, _, _) in enumerate(SOURCE_STATES):
        p_high = P_HIGH[APPRAISALS[ap]]
        out[i] = np.log(p_high if art["intensity"] == 3 else 1 - p_high)
    return out


def posterior(ll: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    pr = np.full(N_SOURCE_STATES, 1.0 / N_SOURCE_STATES) if prior is None else prior
    return C.softmax(np.log(pr) + ll)


def marginal(post: np.ndarray, which: str) -> np.ndarray:
    pos, names = {"belief": (0, BELIEFS), "support": (1, SUPPORT), "appraisal": (2, APPRAISALS), "policy": (3, POLICIES),
                  "correct": (4, CORRECT), "private": (5, PRIVATE)}[which]
    out = np.zeros(len(names))
    for i, s in enumerate(SOURCE_STATES):
        out[s[pos]] += post[i]
    return out


def region_posterior(post: np.ndarray) -> dict:
    """Mass on each named region (exact state match on the region's defining variables)."""
    out = {}
    for name, spec in REGIONS.items():
        mass = 0.0
        for i, (b, sup, ap, pol, co, pv) in enumerate(SOURCE_STATES):
            if (BELIEFS[b], SUPPORT[sup], APPRAISALS[ap], POLICIES[pol], CORRECT[co], PRIVATE[pv]) == (spec["belief"], spec["support"], spec["appraisal"], spec["policy"], spec["correct"], spec["private"]):
                mass += post[i]
        out[name] = float(mass)
    return out


def region_prior(floor: float = 0.02) -> np.ndarray:
    """A prior that knows the source population is structured: most mass on the four named
    regions, a floor elsewhere. This is what couples a counterfactual probe (a correction, a
    private action) to content support; under a factorial prior the probes say nothing about it."""
    prior = np.full(N_SOURCE_STATES, floor / N_SOURCE_STATES)
    for spec in REGIONS.values():
        for i, (b, sup, ap, pol, co, pv) in enumerate(SOURCE_STATES):
            if (BELIEFS[b], SUPPORT[sup], APPRAISALS[ap], POLICIES[pol], CORRECT[co], PRIVATE[pv]) == (spec["belief"], spec["support"], spec["appraisal"], spec["policy"], spec["correct"], spec["private"]):
                prior[i] += (1.0 - floor) / len(REGIONS)
    return prior / prior.sum()


# --------------------------------------------------------------------------- #
# The inverse-inverse maker: evidence selected to steer an audience model.
# --------------------------------------------------------------------------- #
def audience_modelling_speak(s: dict, rng: np.random.Generator, reader_prior: np.ndarray | None = None, candidates: int = 6) -> dict:
    """Among candidate artifacts, emit the one that maximizes the audience's posterior mass on the
    source's desired appraisal under an exact audience model."""
    best, best_mass = None, -1.0
    for _ in range(candidates):
        art = speak(dict(s, policy="cherry_pick"), rng)
        # the audience model: a plain reader that takes evidence at face value; pick what most convinces it of support
        post = posterior(loglik_artifact(art, assume_full=True), reader_prior)
        mass = marginal(post, "support")[SUPPORT.index("supports")]
        if mass > best_mass:
            best, best_mass = art, mass
    best["audience_modelled"] = True
    return best


# --------------------------------------------------------------------------- #
# Uptake: policy change gated by reliability after reconstruction.
# --------------------------------------------------------------------------- #
def uptake(post: np.ndarray, reliability_posterior: float, content_truth_p: float, gate: str = "factored") -> dict:
    """Belief update follows content support; policy adoption is gated by reliability; inferred goal
    is untouched by either. ``scalar`` collapses the gate into one number."""
    if gate == "factored":
        belief = content_truth_p
        policy = content_truth_p * reliability_posterior
    elif gate == "scalar":
        belief = policy = 0.5 * (content_truth_p + reliability_posterior)
    else:                                                       # suppress: distrust everything
        belief = policy = 0.0
    return {"belief_update": float(belief), "policy_uptake": float(policy),
            "goal_posterior": marginal(post, "appraisal").tolist()}


def habituated_response(exposures: int, tau: float = 2.0) -> float:
    """Acute response decays with repeated exposure (fast channel)."""
    return float(np.exp(-exposures / tau))
