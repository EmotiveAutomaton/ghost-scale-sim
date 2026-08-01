"""V8 — the mechanisms that give the reader a mind of its own.

FOUR ADDITIONS, ALL OFF BY DEFAULT (null N35), for the reason every version since six has kept:
an addition that silently rewrites the record is the accretion problem the repair pass was written
against. What each one changes is measured and reported; none is adopted by fiat.

    reader depth        the reader's OWN compiled hierarchy, as against how well-aimed its
                        templates are. The theory's expertise; the code only ever had calibration.
    integration cost    being changed costs something. The theory puts the metabolic cost on the
                        updating, not on the search, and the code had it entirely on the search.
    forgetting          associations weaken toward a floor and never vanish.
    density             artfulness as hierarchy invoked per unit of observable extent, which is
                        what lets a readymade be dense rather than empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import kl_divergence, normalize

_EPS = 1e-12


# =========================================================================== #
# §2.1  Reader depth — the theory's expertise, which the code never had.
# =========================================================================== #
@dataclass
class ReaderHierarchy:
    """How many levels of compiled structure the reader has built for itself.

    THE DISTINCTION THIS EXISTS TO MAKE, because the code has been conflating two things since
    version 1. Reader inexpertise in V1-V7 is a perturbation of the reader's goal templates toward
    noise: it makes a reader WRONG. The theory's expertise is something else entirely -- the
    compressed hierarchy you built by having made those decisions yourself, which is what lets you
    SEE them in someone else's work. That makes a reader DEEP.

    A badly calibrated expert and a well-calibrated novice are different people and the model could
    not tell them apart.

    The two are kept separate and named plainly:

        calibration    how well-aimed the templates are   -> whether the reader is right
        depth          how many levels it has itself      -> how far up someone else's it can see

    Every V1-V7 result was measured on calibration and is NOT reinterpreted, which is the same
    discipline applied when effort was replaced by depth.
    """

    levels: int = 3                 # how many levels this reader has compiled
    max_levels: int = 3
    acquired: float = 0.0           # growth from exposure, which is what H8.3 measures

    @property
    def effective(self) -> float:
        return float(min(self.levels + self.acquired, self.max_levels))

    def ceiling_on(self, maker_depth: int) -> float:
        """How much of a maker's hierarchy this reader can resolve.

        A READER CANNOT SEE FURTHER UP A HIERARCHY THAN IT HAS BUILT. That is the whole claim, and
        it is a ceiling rather than a scaling: a deep reader looking at shallow work is not
        penalised, because there is nothing above it to miss. Only the shortfall costs.
        """
        return float(min(self.effective, float(maker_depth)) / max(float(maker_depth), _EPS))

    def observe(self, maker_depth: int, resolved: float, rate: float = 0.05) -> float:
        """Exposure to work deeper than yourself grows your own hierarchy.

        H8.3'S MECHANISM, AND IT IS THE AUTHOR'S HYPOTHESIS. Under active inference, perception and
        action are the same computation -- both minimise free energy, one by changing beliefs and
        one by changing the world. Taken seriously, a compiled motor routine and a compiled
        perceptual schema are the same kind of object, so the hierarchy a reader uses to READ a
        maker is the hierarchy it would use to MAKE.

        If that is right, appreciation is not merely recognition. It is acquisition: reading
        something deeper than you are should leave you deeper, and the growth should show up in
        what you could PRODUCE rather than only in what you can recognise.

        Growth is gated on having actually resolved the work, which is what stops this being a
        counter of exposures. You do not get a master's hierarchy by staring at a master.
        """
        shortfall = max(float(maker_depth) - self.effective, 0.0)
        gain = float(rate) * shortfall * float(np.clip(resolved, 0.0, 1.0))
        self.acquired += gain
        return gain


def depth_reading(true_depth: int, reader: ReaderHierarchy, noise: float = 0.0,
                  rng=None) -> float:
    """What a reader of a given depth reports about a maker of a given depth.

    Predicts H8.1's interaction directly: the reading is the truth scaled by how much of it the
    reader can reach. A matched reader reads it straight; a shallow one compresses it toward its
    own ceiling; a deep one is not rewarded for depth it did not need.
    """
    seen = reader.ceiling_on(true_depth) * float(true_depth)
    if noise and rng is not None:
        seen += float(rng.normal(0.0, noise))
    return float(np.clip(seen, 0.0, float(reader.max_levels)))


# =========================================================================== #
# §2.2  Integration costs something.
# =========================================================================== #
def integration_cost(posterior: np.ndarray, prior: np.ndarray, rate: float = 1.0) -> float:
    """What it costs to be changed by what you just read.

    The only non-zero preference in V1-V7 is effort: looking is expensive and skimming is cheap.
    Integration is free, so a reader that takes on a colossal update pays exactly what one that
    takes on none pays.

    The theory says the opposite is the main cost. The senate metaphor is about UPDATING LEDGERS
    burning the twenty watts -- "the more wrong they were, the more they have to change" -- so
    surprise is expensive precisely BECAUSE rewiring is expensive.

    Charging for it makes a prediction that is not tidiness: THE TRUST EXPLOIT GETS WORSE. A fooled
    reader currently absorbs a large false update for nothing. Charge for it and the victim pays
    real resources for a belief that is wrong, which is a sharper statement of the harm than the
    one on record.
    """
    return float(rate) * float(kl_divergence(np.asarray(posterior, dtype=float),
                                             np.asarray(prior, dtype=float)))


# =========================================================================== #
# §2.3  Forgetting — associations weaken, and never vanish.
# =========================================================================== #
@dataclass
class DecayingBelief:
    """A carried belief that fades toward a floor rather than toward nothing.

    THE SHAPE IS THE AUTHOR'S SPECIFICATION AND IT MATTERS. "All we do is build new associations
    and old associations weaken, but they don't seem to ever quite disappear." So decay is
    asymptotic toward a residual, not erasure: what was learned becomes weak, not absent.

    WHAT THIS CHANGES ABOUT AN EXISTING RESULT. E46 showed a reader drifting under repeated
    exposure to material it rejects every time, and the drift was UNBOUNDED because nothing pulled
    it back. With forgetting the same mechanism gives an EQUILIBRIUM, set by exposure rate against
    decay rate.

    That is a better claim in both directions. "Propaganda moves you without limit" is weak and
    slightly silly. "Propaganda moves you to a level set by how often you meet it against how fast
    you forget it" says FREQUENCY MATTERS MORE THAN INTENSITY, which is non-obvious, testable, and
    has a policy reading.
    """

    belief: np.ndarray
    baseline: np.ndarray
    rate: float = 0.05              # how fast an association weakens per encounter
    floor: float = 0.15             # the share that never fades
    history: list = field(default_factory=list)

    def decay(self) -> np.ndarray:
        """One step of forgetting. Pulls toward the baseline but never all the way."""
        keep = self.floor + (1.0 - self.floor) * (1.0 - float(self.rate))
        self.belief = normalize(keep * self.belief + (1.0 - keep) * self.baseline)
        return self.belief

    def integrate(self, evidence: np.ndarray, weight: float) -> np.ndarray:
        w = float(np.clip(weight, 0.0, 1.0))
        self.belief = normalize((1.0 - w) * self.belief + w * np.asarray(evidence, dtype=float))
        return self.belief

    def drift(self) -> float:
        return float(kl_divergence(self.belief, self.baseline))


# =========================================================================== #
# §3 H8.4  Density — what lets a readymade be dense rather than empty.
# =========================================================================== #
def density(hierarchy_levels_invoked: float, observable_extent: int,
            floor_extent: int = 1) -> float:
    """Artfulness as hierarchy invoked PER UNIT of observable extent.

    THE CASE THE THEORY SPENDS THE MOST WORDS DEFENDING AND THE MODEL COULD NOT REPRESENT. Depth in
    V5-V7 is measured over a sequence of observations, so an artifact with almost no observable
    extent carries almost nothing to read, and a readymade scores as empty. But the essay's whole
    argument about Duchamp is that the fabrication is near zero and the compressed context is
    enormous: "simplicity without a dense underlying decision tree is just empty data; simplicity
    born from extreme compression is a masterpiece."

    So the quantity is a RATIO. One act that requires three levels of hierarchy to explain is
    maximally dense; three hundred acts that require one level are not.

    NULL N39 GUARDS THE OBVIOUS FAILURE: this must not simply reward shortness. A brief artifact
    with no hierarchy behind it scores at the floor, because the numerator is what carries the
    claim and a short empty thing has a numerator of nothing.
    """
    extent = max(int(observable_extent), int(floor_extent))
    return float(hierarchy_levels_invoked) / float(extent)


def bimodality(readings: np.ndarray) -> dict:
    """How split a population's readings are, as against merely uncertain.

    H8.4'S INTERESTING HALF. If density is what a readymade has, then a reader with a matched
    hierarchy sees a great deal in it and a reader without sees nothing -- and there is no middle,
    because the thing that separates them is whether they possess a structure, not how much
    evidence they gathered. The population should go BIMODAL rather than uniformly unimpressed.

    Which is exactly what conceptual art does to a room, and it is a different prediction from "a
    readymade is weak work", which would give a low mean and one mode.

    Scored with a bimodality coefficient: values above about 5/9 indicate a split population, which
    is the standard threshold for the statistic and is used here rather than invented.
    """
    x = np.asarray([v for v in readings if np.isfinite(v)], dtype=float)
    n = x.size
    if n < 4 or np.allclose(x, x[0]):
        return {"coefficient": float("nan"), "split": False, "n": int(n)}
    m = x.mean()
    s = x.std(ddof=1)
    if s <= _EPS:
        return {"coefficient": float("nan"), "split": False, "n": int(n)}
    skew = float(np.mean(((x - m) / s) ** 3))
    kurt = float(np.mean(((x - m) / s) ** 4)) - 3.0
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    coeff = (skew ** 2 + 1.0) / denom if abs(denom) > _EPS else float("nan")
    return {"coefficient": float(coeff), "split": bool(np.isfinite(coeff) and coeff > 5.0 / 9.0),
            "n": int(n), "mean": float(m), "sd": float(s)}


# =========================================================================== #
# §3 H8.5  Capture and sustain, as two decisions.
# =========================================================================== #
@dataclass
class TwoStageAttention:
    """Grabbing attention and keeping it are different, and the model had one decision repeated.

    The essay is precise about this: aesthetics is "something attention-grabbing, but not
    necessarily attention-keeping", and shock art is "raw aesthetic attentional capture without the
    follow-up in observable decision-making required to maintain that attention".

    With a single repeated engagement decision the model cannot tell SLOP from SHOCK ART. Both end
    with nothing recovered, so both score identically, and the theory says they are different
    objects with different signatures: one never got looked at, the other got looked at hard and
    then abandoned.
    """

    capture_weight: float = 1.0
    sustain_weight: float = 1.0

    def capture(self, salience: float, social: float) -> float:
        """Whether it gets looked at at all. Surface properties only; no inference has happened."""
        return float(np.clip(self.capture_weight * (0.5 * float(salience) + 0.5 * float(social)),
                             0.0, 1.0))

    def sustain(self, information_gain: float) -> float:
        """Whether looking continues. This is where density has to do the work."""
        return float(np.clip(self.sustain_weight * float(information_gain), 0.0, 1.0))

    def trace(self, salience: float, social: float, information_gain: float,
              n_steps: int = 12) -> dict:
        """The attention trace, which is what separates the two failures.

        Shock art: capture high, sustain nothing -> a spike and a collapse.
        Slop:      capture low, sustain nothing  -> a flat line at the bottom.
        Both recover nothing, and only the trace tells them apart.
        """
        c = self.capture(salience, social)
        s = self.sustain(information_gain)
        series = [float(c * (s ** t)) if s < 1.0 else float(c) for t in range(int(n_steps))]
        peak = float(max(series))
        tail = float(np.mean(series[len(series) // 2:]))
        return {"series": series, "peak": peak, "tail": tail,
                "collapse": float(peak - tail),
                "captured": bool(peak >= 0.5), "sustained": bool(tail >= 0.25)}


@dataclass
class V8Switches:
    """Every V8 addition, independently switchable, ALL OFF BY DEFAULT (null N35)."""

    reader_depth: bool = False
    integration_cost: bool = False
    forgetting: bool = False
    density: bool = False
    two_stage_attention: bool = False

    @classmethod
    def from_config(cls, cfg) -> "V8Switches":
        g = cfg.get
        return cls(
            reader_depth=bool(g("v8.reader_depth.enabled", False)),
            integration_cost=bool(g("v8.integration_cost.enabled", False)),
            forgetting=bool(g("v8.forgetting.enabled", False)),
            density=bool(g("v8.density.enabled", False)),
            two_stage_attention=bool(g("v8.attention.two_stage", False)),
        )

    def all_off(self) -> bool:
        return not (self.reader_depth or self.integration_cost or self.forgetting
                    or self.density or self.two_stage_attention)
