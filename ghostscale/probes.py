"""The encoder-divergence probe set — V3 spec §1 C3, guarded by N15.

C3's second channel asks: *can this generation's observer still read intent from clean
human work?* That is a question about the OBSERVER, so the probe set has to be held
absolutely constant while the observer drifts.

-----------------------------------------------------------------------------------------
DECISION D5 — encoder divergence is a DECODING measure, not a corpus statistic.

V3 §1 C3 words it as "``MI(features; goal)`` on a held-out set of known-genuine probe
artifacts". Read literally, ``MI(features; goal)`` over a corpus measures **the corpus**:
it is a property of how those artifacts were generated, and it would return the same number
for a perfect decoder and a destroyed one. That is precisely the failure N15 exists to
prevent, arrived at from the other direction.

So the measured quantity is the observer's DECODING performance on the probes:

    encoder_mi   = I(modal inferred goal ; true goal)      [nats, discrete, over the probe set]
    encoder_soft = mean posterior mass placed on the true goal

``encoder_mi`` is primary (it is the information-theoretic quantity C3 asks for, computed on
the channel that actually matters — artifact in, inferred goal out). ``encoder_soft`` is
reported alongside because a 4x4 confusion matrix over a few hundred probes is a noisy MI
estimate, and the soft measure moves before the argmax does. ``learning.human_column_mi``
— the learned-A statistic V2 already records — stays in the CSVs as the cheap proxy, so all
three can be compared.

TWO SOURCES, DELIBERATELY SEPARATED. Decoding can fail because the observer's model has
drifted, or because the observer no longer bothers to look. Both are real, but C3's claim is
about the model (encoder divergence downstream of value divergence "through the shared
representation"), and E9 already owns the engagement channel. So every probe set is decoded
TWICE — once with engagement free, once with DEEP forced — and both are reported. The forced
number isolates the model; the gap between them is the engagement contribution.

WHAT THE PROBES ARE MADE OF. Generation-0 creators, always. The probe artifacts are drawn
from the ORIGINAL ``HumanCreator`` bank, never from the generation's own seeded creators:
"clean human work" must not itself drift, or the measurement would be a comparison of two
moving objects and a flat result could mean either a stable decoder or two drifts cancelling.

N15 — PURITY. Zero contaminated and zero synthetic artifacts, asserted in the worker at
construction AND verified in the CSV (every row carries ``probe_contaminated_count``, which
must be 0). Asserted rather than merely arranged, because "the probe set is clean" is the
assumption the entire second channel rests on.
-----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np

from . import constants as K
from .config import Config
from .creators import build_creator_bank
from .environment import Artifact, Environment
from .generative_model import GenerativeModel
from .metrics import shannon_entropy

_EPS = 1e-12


def build_probe_env(cfg: Config, gm: GenerativeModel, rng: np.random.Generator,
                    honesty: float = 1.0, signing_rate: float = 1.0) -> Environment:
    """An environment whose creators are the ORIGINAL generation-0 humans.

    Built fresh from ``gm`` rather than handed the chain's current bank: the probe corpus is
    the fixed reference against which drift is measured, so it may not inherit drift.
    """
    return Environment(cfg, gm, rng_world=rng, honesty=honesty,
                       signing_rate=signing_rate,
                       creator_bank=build_creator_bank(cfg, gm))


def draw_probe_set(env: Environment, n_probes: int, creator_goals: np.ndarray,
                   rng: np.random.Generator) -> list[Artifact]:
    """A held-out corpus of exclusively genuine CREATOR artifacts (N15).

    Uses ``draw_corpus`` at ``contamination=0.0`` rather than a bespoke loop, so the probes
    are constructed by the same code path as every other corpus in the suite and cannot
    quietly differ from it.
    """
    probes = env.draw_corpus(n_probes, contamination=0.0,
                             creator_goals=creator_goals, rng=rng)
    assert_probe_purity(probes)
    return probes


def assert_probe_purity(probes: list[Artifact]) -> int:
    """N15, at construction. Returns the contaminated count (always 0, or it raises)."""
    bad = sum(1 for a in probes if a.provenance != K.CREATOR)
    assert bad == 0, (
        f"N15: the encoder-divergence probe set contains {bad} non-CREATOR artifacts. "
        "Encoder divergence would then be measuring the probe corpus rather than the "
        "observer (V3 spec §3 N15).")
    return 0


def discrete_mutual_information(joint_counts: np.ndarray) -> float:
    """I(X; Y) in nats from a joint count table. Empty rows/columns contribute nothing."""
    c = np.asarray(joint_counts, dtype=float)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (px @ py)[mask] + _EPS)))


def decode_probe_set(agent, probes: list[Artifact], env: Environment, cfg: Config,
                     rng: np.random.Generator, infer_steps: int,
                     force_deep: bool) -> dict:
    """Roll ``agent`` over the probe set and score its intent decoding.

    LEARNING IS OFF. A probe that updated the model would make the measurement change the
    thing it measures, and would hand the observer free corrective evidence that the
    experiment's own corpus is supposed to be the only source of.
    """
    from .observer import rollout_observer   # local import: observer imports learning

    num_goals = int(cfg.cardinalities.num_goals)
    joint = np.zeros((num_goals, num_goals))   # [modal inferred, true]
    soft, deep = 0.0, 0
    for j, art in enumerate(probes):
        rr = np.random.default_rng(rng.integers(2**63 - 1))
        res = rollout_observer(agent, art, env, cfg, rr, infer_steps,
                               force_deep_k=(infer_steps if force_deep else 0),
                               early_stop=False, learn=False)
        post = np.asarray(res.final_goal_posterior, dtype=float)
        joint[int(np.argmax(post)), int(art.goal)] += 1.0
        soft += float(post[int(art.goal)])
        deep += int(res.cum_deep)

    n = max(len(probes), 1)
    return {"encoder_mi": discrete_mutual_information(joint),
            "encoder_soft": soft / n,
            "encoder_accuracy": float(np.trace(joint) / n),
            "probe_deep_per_artifact": deep / n,
            "probe_n": int(len(probes)),
            "probe_contaminated_count": 0,
            "probe_true_goal_entropy": shannon_entropy(joint.sum(axis=0) / n)}
