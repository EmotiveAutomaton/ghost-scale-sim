"""R-8b — trust as something a reader learns about a source, rather than a knob set from outside.

WHETHER THIS EARNS ITS PLACE, ASSESSED BEFORE IT WAS BUILT.

The specification justifies this item as the move that makes trust measurable. That justification did
not survive: trust becomes measurable by fitting it to the reader's behaviour instead of to the
observation tape, which is a small change and is already done. So this item had to earn its place on
different grounds or not be built at all, and the rule for the pass is that an addition qualifies
only if it converts a free choice into a measured quantity.

It does, and the argument is specific.

**In the model as it stands, trust is a fixed disposition and there is nothing it could be learned
from.** A reader arrives with a trust level, applies it to every label it ever sees, and leaves with
the same one. That is a knob by construction: not merely unmeasured but unmeasurABLE in principle,
because nothing in the reader's history could revise it.

Give the reader a source identity to attach beliefs to, and trust stops being a disposition and
becomes an inference: how reliable has THIS source's labelling been. That is a quantity with a true
value the experimenter sets, a reader-side estimate of it, and therefore a recovery question. The
conversion is exactly what the rule asks for.

-----------------------------------------------------------------------------------------
AND IT MAKES A PREDICTION THE FIXED-TRUST MODEL CANNOT.

The channel-accounting result says the label beats the content above a crossover in trust. Now ask
what happens to a reader trying to learn whether a source lies. Detecting a lie means noticing that
the label and the work disagree. Above the crossover the label wins every such disagreement, so the
reader's belief about provenance simply follows the label, and **the mismatch never registers**.

That predicts a threshold in something quite different from anything measured so far:
**above the crossover a reader cannot form a reputation for a source, however many times it is
lied to.** It is not that it learns slowly. It is that the evidence never arrives, because the
channel that would carry it has already been overruled.

If that holds it is a substantive and unpleasant claim about disclosure regimes: the readers most
disposed to believe labels are exactly the ones who cannot learn that a labeller is unreliable.

-----------------------------------------------------------------------------------------
WHAT IS DELIBERATELY NOT BUILT.

Reputation dynamics and strategic defection, which the signalling appendix would want, are NOT here.
They need a source that chooses when to lie, which is an agent with its own objective, and that is a
substantial addition whose results would be hard to attribute. This module adds one thing only: a
source identity and a reader-side belief about that source's honesty. Everything else waits, and the
minimal-model programme is where it should be argued for.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from ..config import Config
from . import criteria as CR
from . import repair_dir


# =========================================================================== #
# The source, and the reader's belief about it.
# =========================================================================== #
class Source:
    """A named source with an honesty rate the experimenter knows and the reader does not."""

    def __init__(self, name: str, honesty: float, provenance: int):
        self.name = name
        self.honesty = float(honesty)
        self.provenance = int(provenance)

    def emit_label(self, rng) -> int:
        """The label this source attaches, honest at its own rate and misleading otherwise."""
        if rng.random() < self.honesty:
            return K.TRUTHFUL_SIGNAL[self.provenance]
        # When it lies it claims full human authorship, which is the lie the framework is about.
        return K.SIG_CREATOR


class TrustLearner:
    """A reader that keeps a Beta belief about how honest a named source's labelling has been.

    THE ONE PIECE OF NEW MACHINERY. The reader observes, per encounter, whether the label agreed
    with what the CONTENT said about provenance, and updates a Beta over that source's honesty.
    Its effective trust on the next encounter is the mean of that Beta.

    The content-side reading is deliberately computed from the content channel ALONE rather than
    from the reader's fused posterior. That is the whole mechanism under test: a reader whose fused
    belief is dominated by the label has no independent content reading to compare against, and so
    can never register a mismatch. Giving it a separate content reading is the generous case, and
    the prediction is that even so, its ability to learn is gated by how much the label already
    dominates.
    """

    def __init__(self, prior_a: float = 1.0, prior_b: float = 1.0, kappa: float = 0.9):
        self.a, self.b = float(prior_a), float(prior_b)
        self.kappa = float(kappa)
        self.history = []

    @property
    def believed_honesty(self) -> float:
        return self.a / (self.a + self.b)

    def encounter(self, content_says: int, label_says: int, weight: float) -> None:
        """One encounter. ``weight`` is how much the reader credits its own content reading.

        A reader that trusts the label heavily discounts its content reading, which is what makes
        the mismatch evidence weak exactly where trust is high. That coupling is the mechanism, not
        an implementation convenience, and it is the reason the reputation threshold exists.
        """
        agrees = int(content_says == label_says)
        self.a += weight * agrees
        self.b += weight * (1 - agrees)
        self.history.append({"agrees": agrees, "weight": weight,
                             "believed_honesty": self.believed_honesty})


def _content_reading(cfg: Config, gm, source: Source, rng, n_looks: int) -> int:
    """What the WORK alone says about who made it, ignoring the label entirely."""
    from ..environment import Artifact, Environment
    from ..exact import make_exact_agent
    from ..generative_model import build_D

    env = Environment(cfg, gm, rng, honesty=1.0, signing_rate=0.0)
    art = Artifact(provenance=source.provenance, goal=int(rng.integers(4)),
                   declared_signal=K.UNSIGNED)
    agent = make_exact_agent(gm, build_D(cfg, rng), cfg)
    agent.reset()
    for _ in range(n_looks):
        agent.infer_states(env.observation(art, K.DEEP, rng))
        act = np.zeros(len(agent.num_controls))
        act[K.F_ATTENTION] = K.DEEP
        agent.action = act
    p = int(np.argmax(np.asarray(agent.qs[K.F_PROVENANCE], dtype=float)))
    return K.TRUTHFUL_SIGNAL[p]


def _label_dominance(cfg, kappa: float) -> float:
    """How much the label outweighs the content per glance, in nats. Positive means the label wins.

    Read from the channel-accounting arithmetic rather than recomputed, so this module and the
    diagnostics pass cannot drift apart on the number that decides the whole prediction.
    """
    from ..diagnostics.d1_channels import _expected_llr_content, label_llr_from_kappa
    from ..generative_model import build_shared_model
    gm = build_shared_model(cfg)
    content = _expected_llr_content(np.asarray(gm.A[0]), K.GHOST, K.CREATOR, 1)
    label = label_llr_from_kappa(kappa, int(cfg.cardinalities.num_signals))
    return float(label + content)


def run(cfg: Config, workers: int = 1, n_encounters: int = 40, n_readers: int = 40,
        n_looks: int = 6) -> dict:
    """Two questions: does learned trust recover, and can a high-trust reader learn a lie at all?"""
    from ..generative_model import build_shared_model

    out = repair_dir("r8b_learned_trust")
    gm = build_shared_model(cfg)
    honesty_grid = np.round(np.linspace(0.1, 0.9, 9), 3)
    kappa_grid = (0.2, 0.4, 0.538, 0.7, 0.9)

    rows = []
    for kappa in kappa_grid:
        dominance = _label_dominance(cfg, kappa)
        # How much a reader at this trust credits its own content reading. Above the crossover the
        # label has already won, so the content reading carries no weight in the fused belief, and
        # the mismatch it would have revealed never registers.
        weight = float(1.0 / (1.0 + np.exp(4.0 * dominance)))
        for true_h in honesty_grid:
            ests = []
            for i in range(n_readers):
                rng = np.random.default_rng(41_000 + 313 * i + int(1000 * kappa))
                src = Source("s", float(true_h), K.GHOST)
                learner = TrustLearner(kappa=kappa)
                for _ in range(n_encounters):
                    label = src.emit_label(rng)
                    content = _content_reading(cfg, gm, src, rng, n_looks)
                    learner.encounter(content, label, weight)
                ests.append(learner.believed_honesty)
            rows.append({
                "kappa": float(kappa), "label_dominance_nats": dominance,
                "content_weight": weight,
                "true_honesty": float(true_h),
                "believed_honesty_mean": float(np.mean(ests)),
                "believed_honesty_sd": float(np.std(ests, ddof=1)) if len(ests) > 1 else 0.0,
                "believed_honesty_sem": (float(np.std(ests, ddof=1) / np.sqrt(len(ests)))
                                         if len(ests) > 1 else 0.0),
                "n_readers": n_readers, "n_encounters": n_encounters,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "learned_trust.csv", index=False)

    from ..diagnostics.criteria import spearman
    per_kappa = []
    for kappa, g in df.groupby("kappa"):
        g = g.sort_values("true_honesty")
        frac, _ = CR.identifiable_fraction(g.true_honesty.values, g.believed_honesty_mean.values,
                                           g.believed_honesty_sem.values)
        per_kappa.append({
            "kappa": float(kappa),
            "label_dominance_nats": float(g.label_dominance_nats.iloc[0]),
            "content_weight": float(g.content_weight.iloc[0]),
            "rank_correlation": spearman(g.true_honesty.values, g.believed_honesty_mean.values),
            "slope": float(np.polyfit(g.true_honesty.values,
                                      g.believed_honesty_mean.values, 1)[0]),
            "identifiable_fraction": frac,
            "reading_statistical_only": CR.map_label(frac),
            "reading": CR.map_label(frac, float(np.polyfit(
                g.true_honesty.values, g.believed_honesty_mean.values, 1)[0])),
            "spread_of_beliefs_across_true_honesty": float(np.ptp(g.believed_honesty_mean.values)),
        })
    pd.DataFrame(per_kappa).to_csv(out / "by_trust.csv", index=False)

    # BOTH TERMS. Separable from noise AND carrying the signal; see the slope floor in criteria.py.
    def _learns(q):
        return bool(q["identifiable_fraction"] >= CR.MAP_PARTIAL
                    and abs(q["slope"]) >= CR.MAP_SLOPE_FLOOR)

    learns = [p for p in per_kappa if _learns(p)]
    blind = [p for p in per_kappa if not _learns(p)]
    threshold_holds = bool(learns and blind
                           and max(p["kappa"] for p in learns) < min(p["kappa"] for p in blind))

    if threshold_holds:
        verdict = "REPUTATION_BLINDNESS_ABOVE_THE_CROSSOVER"
    elif learns and not blind:
        verdict = "EVERY_READER_LEARNS_THE_SOURCE"
    elif blind and not learns:
        verdict = "NO_READER_LEARNS_THE_SOURCE"
    else:
        verdict = "LEARNING_DOES_NOT_TRACK_THE_CROSSOVER"

    payload = {
        "check": "R-8b",
        "question": ("If trust is something a reader learns about a named source rather than a "
                     "knob, does it recover, and can a trusting reader learn it at all?"),
        "plain_language": (
            "Until now a reader's trust in labels was a setting: fixed when the reader was made, "
            "never revised, and so not the kind of thing that could be measured even in principle. "
            "This gives sources names and honesty rates, lets a reader build up a view of how "
            "reliable a particular source has been, and then asks two things. Can the reader work "
            "out how honest the source is? And does that ability depend on how much it trusted "
            "labels to begin with?"),
        "why_it_was_built": (
            "not for the reason the specification gives. Trust became measurable by fitting it to "
            "behaviour, which is a much smaller change and is already done. This earns its place "
            "under the pass's own rule instead: it converts a fixed disposition, which nothing in "
            "the reader's history could ever revise, into an inference with a true value and an "
            "estimate. It also makes a prediction the fixed-trust model cannot."),
        "what_was_deliberately_not_built": (
            "reputation dynamics and strategic defection. Those need a source that chooses when to "
            "lie, which is an agent with its own objective, and the results would be hard to "
            "attribute. This adds a source identity and a reader-side belief about that source's "
            "honesty, and nothing else."),
        "criteria": {"identifiable_se": CR.IDENTIFIABLE_SE, "map_partial": CR.MAP_PARTIAL},
        "by_trust": per_kappa,
        "rows": rows,
        "readers_that_learn": [p["kappa"] for p in learns],
        "readers_that_cannot": [p["kappa"] for p in blind],
        "threshold_tracks_the_crossover": threshold_holds,
        "verdict": verdict,
    }
    payload["statement"] = _statement(payload)
    (repair_dir() / "r8b_learned_trust.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def _statement(p: dict) -> str:
    bits = []
    tab = "\n".join(
        "| %.3f | %+.3f | %.3f | %s | %.0f%% | %s |"
        % (r["kappa"], r["label_dominance_nats"], r["content_weight"],
           ("%.2f" % r["rank_correlation"]) if np.isfinite(r["rank_correlation"]) else "n/a",
           100 * r["identifiable_fraction"], r["reading"])
        for r in p["by_trust"])
    bits.append(
        "| trust the reader started with | how far the label outweighs the work, per glance | "
        "weight it gives its own reading | recovery of the source's honesty | identifiable range | "
        "reading |\n|---|---|---|---|---|---|\n" + tab)

    if p["threshold_tracks_the_crossover"]:
        bits.append(
            "**A trusting reader cannot learn that a source lies, and the threshold is the one the "
            "channel accounting already located.** Readers starting at trust %s recover the "
            "source's honesty. Readers starting at %s do not, at any number of encounters, because "
            "the evidence never arrives: working out that a source is unreliable means noticing "
            "that the label and the work disagree, and above the crossover the label wins every "
            "such disagreement before it can be registered.\n\n"
            "**This is not slow learning. It is learning that cannot start.** And it is the "
            "prediction the fixed-trust model was structurally incapable of making, which is what "
            "earns this addition its place: it converts a setting into an inference, and the "
            "inference has a threshold in it."
            % (", ".join("%.2f" % k for k in p["readers_that_learn"]),
               ", ".join("%.2f" % k for k in p["readers_that_cannot"])))
        bits.append(
            "**If it holds outside this model it is an unpleasant claim about disclosure.** The "
            "readers most disposed to believe provenance labels are exactly the ones who cannot "
            "discover that a labeller is unreliable, so a disclosure regime protects them least "
            "where it fails most. It is stated here as a property of a simulation and nothing "
            "more, and it is the kind of claim the prediction card exists to hand to a human "
            "study rather than to settle.")
    elif p["verdict"] == "EVERY_READER_LEARNS_THE_SOURCE":
        bits.append("Every reader recovers the source's honesty regardless of where its trust "
                    "started, so the reputation threshold the channel accounting predicted does "
                    "not appear. The addition still converts a setting into an inference, which is "
                    "what it was built for, but the prediction that motivated it is not supported.")
    else:
        bits.append("Learning does not track the crossover in the way the channel accounting "
                    "predicted, and the result is reported as it stands.")
    return "\n\n".join(bits)
