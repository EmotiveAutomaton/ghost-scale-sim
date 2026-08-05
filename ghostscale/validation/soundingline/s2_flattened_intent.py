"""S-2 — does flattened intent show up as posterior concentration, at matched decision density?

**WITHDRAWN 2026-08-05. THIS MODULE'S MANIPULATION NEVER REACHES THE READER. DO NOT QUOTE IT.**

The construction below draws a per-position active goal and emits from it by passing a fresh
``Artifact`` into ``env.sample_feature``. ``V5Environment.sample_feature`` never reads
``artifact.goal`` once a creator is bound -- it returns ``self.creator.next_feature(rng)``, and the
creator holds ONE fixed goal for the whole artifact. Only ``artifact.provenance`` is consulted, for
the alpha lookup. So ``actives`` is drawn and discarded.

``scripts/audit_s2_mixture.py`` shows the feature streams are bit-identical with the mixture forced
off, in 100% of artifacts, and that the reported ``purpose_breadth`` separation of -0.108
reproduces exactly under that switch. What the two arms actually differ by is the artifact's TRUE
goal -- flattened is always goal 0, layered is a uniform draw -- plus the RNG-stream shift from the
extra ``integers()`` call.

A working emitter is ``t2_automaticity.mixed_deep_features``, which builds the emission from
``world.subsig[mu, active_goal, mode]`` directly. S-2's actual question -- concentration rather
than diversity -- has still not been tested and should be re-run there.

THE CURATOR'S CLAIM, in his own correction of a stronger earlier version:

    It is unfair to say corporate goals are singular... the SHARE of the goal is
    disproportionately large. You'd expect that to be a larger piece, NOT the whole pie.

So the construct is not *thinner*, it is *more concentrated*, and the two are confounded in every
real corpus. Here they can be separated, which is the entire reason this test belongs in a
simulation.

THE CONSTRUCTION. An artifact is built position by position, and at each position an active
terminal goal is drawn from a weight vector:

    layered    weights flat over all goals
    flattened  one goal carrying most of the weight, the rest sharing the remainder

**Decision density is held constant by construction**: same number of positions, same depth, same
number of execution-mode decisions. Only the concentration of the goal mixture differs.

THREE THINGS MEASURED, and two of them are traps rather than outcomes:

  purpose_breadth   normalised entropy of the reader's final goal posterior. The claim lives here.
  goal accuracy     should NOT fall for flattened work. A flattened maker is easier to read, not
                    harder. If accuracy drops, the construct has become 'harder' rather than
                    'more concentrated'.
  process recovery  should NOT fall either. If it does, 'flattened' is quietly 'shallower' and the
                    whole construct is confounded in exactly the way the curator was warning about.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ... import constants as K
from ... import metrics
from ...baselines import ObservationTape, TapedEnvironment
from ...config import Config
from ...environment import Artifact
from ...prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ...v5_model import make_v5_observer
from ...v6 import SEED_OFFSET, harness as H
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from . import sl_dir
from .common import build

MU = 3                 # practised work, where there is a process to recover at all
BETA = 1.0             # the goal is legible; concentration is the manipulation, not legibility
DOMINANT = 0.70        # the flattened maker's share for its main terminal value

# READ AT THE CURATOR TIER WITH A SHORT LOOK, for the reason E45 had to. On fully transparent work
# with a long look this reader scores 1.00 on the goal and its posterior entropy collapses to
# ~1e-10 in BOTH arms -- pinned at the floor, so breadth cannot separate anything and the test
# measures nothing. Partial transmission leaves the headroom the comparison needs.
READ_TIER = K.CURATOR
READ_GLANCES = 3


def _weights(kind: str, ng: int) -> np.ndarray:
    if kind == "layered":
        return np.full(ng, 1.0 / ng)
    w = np.full(ng, (1.0 - DOMINANT) / max(ng - 1, 1))
    w[0] = DOMINANT
    return w


def _mixture_off_deviation(world, cfg_r, n_mu, n_sub, ng, n_obs: int = 24,
                           n_timesteps: int = 24) -> float:
    """How much this module's output changes when the mixture is forced off. It is zero.

    The live gate needs a number, and the number is the whole diagnosis. This runs the module's
    own inner loop twice per artifact -- once with the per-position mixture drawn as designed,
    once with every position forced to the modal goal -- and returns the largest absolute change
    in ``purpose_breadth`` across artifacts.

    A working manipulation returns something well away from zero here. This returns exactly 0.0,
    because ``V5Environment.sample_feature`` never reads ``artifact.goal`` once a creator is
    bound, so the mixture is drawn and discarded. Same finding as
    ``scripts/audit_s2_mixture.py``, compressed to one number so it can be a standing gate rather
    than a script somebody has to remember to run.
    """
    worst = 0.0
    for kind in ("layered", "flattened"):
        w = _weights(kind, ng)
        for i in range(int(n_obs)):
            vals = []
            for force_off in (False, True):
                base = 62_000 + (0 if kind == "layered" else 1) * 977
                art_rng = np.random.default_rng(base * 31 + i)
                modal = int(np.argmax(w)) if kind == "flattened" else int(art_rng.integers(ng))
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, modal, MU, BETA, n_timesteps, art_rng, provenance=READ_TIER)
                tape = ObservationTape(env, artifact,
                                       np.random.default_rng(base * 104729 + i), n_timesteps)
                actives = art_rng.choice(ng, size=n_timesteps, p=w)
                if force_off:
                    actives = np.full(n_timesteps, modal, dtype=int)
                tape.deep = np.asarray(
                    [int(env.sample_feature(
                        Artifact(provenance=READ_TIER, goal=int(actives[t]),
                                 declared_signal=K.UNSIGNED), K.DEEP, art_rng))
                     for t in range(n_timesteps)], dtype=int)
                agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
                enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent,
                                      creator, np.random.default_rng(base * 7907 + i),
                                      n_timesteps, READ_GLANCES, n_sub, n_mu, ng,
                                      float(world.cfg.signal_model.kappa), true_goal=modal)
                vals.append(float(metrics.within_observer_entropy(enc.goal_posterior)))
            worst = max(worst, abs(vals[0] - vals[1]))
    return float(worst)


def run(cfg: Config, n_obs: int = 120, n_timesteps: int = 24, forced_k: int = 24) -> dict:
    world, _cfg_b, cfg_r, n_mu, n_sub, ng = build(cfg)
    hmax = float(np.log(max(ng, 2)))
    rows = []

    for kind in ("layered", "flattened"):
        w = _weights(kind, ng)
        for i in range(int(n_obs)):
            base = 62_000 + (0 if kind == "layered" else 1) * 977
            art_rng = np.random.default_rng(base * 31 + i)
            # The modal goal is what a reader is scored against: it is what the work is MOSTLY for.
            modal = int(np.argmax(w)) if kind == "flattened" else int(art_rng.integers(ng))
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, modal, MU, BETA, n_timesteps, art_rng, provenance=READ_TIER)

            # Position by position, draw the active terminal goal from the mixture and emit from
            # it. Same number of positions in both arms, so density is matched by construction.
            tape = ObservationTape(env, artifact, np.random.default_rng(base * 104729 + i),
                                   n_timesteps)
            actives = art_rng.choice(ng, size=n_timesteps, p=w)
            deep = []
            for t in range(n_timesteps):
                a = Artifact(provenance=READ_TIER, goal=int(actives[t]),
                             declared_signal=K.UNSIGNED)
                deep.append(int(env.sample_feature(a, K.DEEP, art_rng)))
            tape.deep = np.asarray(deep, dtype=int)

            agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
            enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent, creator,
                                  np.random.default_rng(base * 7907 + i),
                                  n_timesteps, READ_GLANCES, n_sub, n_mu, ng,
                                  float(world.cfg.signal_model.kappa), true_goal=modal)
            rows.append({
                "kind": kind, "i": i,
                "purpose_breadth": float(metrics.within_observer_entropy(enc.goal_posterior))
                / hmax,
                "goal_correct": int(enc.correct),
                "process": float(enc.process["process_error_reduction"]),
                "distinct_goals_used": int(len(set(actives.tolist()))),
            })

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "s2_flattened_intent_points.csv", index=False)
    rng = np.random.default_rng(SEED_OFFSET + 90_200)

    def contrast(col: str) -> dict:
        a = df[df.kind == "flattened"][col].to_numpy(dtype=float)
        b = df[df.kind == "layered"][col].to_numpy(dtype=float)
        draws = [float(np.mean(rng.choice(a, a.size, replace=True))
                       - np.mean(rng.choice(b, b.size, replace=True)))
                 for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        return {"flattened": float(a.mean()), "layered": float(b.mean()),
                "difference": float(a.mean() - b.mean()), "interval": [lo, hi],
                "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}

    breadth = contrast("purpose_breadth")
    acc = contrast("goal_correct")
    proc = contrast("process")
    separates = bool(breadth["excludes_zero"] and breadth["difference"] < 0)
    confounded = bool(proc["excludes_zero"] and proc["difference"] < 0)

    # ---- the gate that would have caught this module before it shipped -----------------------
    # A LIVE GATE THAT FAILS ON PURPOSE. The manipulation is drawn and discarded, so forcing the
    # mixture off changes nothing: max deviation is exactly 0.0 and the feature streams are
    # bit-identical (scripts/audit_s2_mixture.py). Recording the failure here rather than
    # deleting the module makes the withdrawal auditable -- the evidence travels with the result
    # instead of living in a commit message. tests/test_gates.py allows a failing gate only on a
    # verdict explicitly marked withdrawn, which is what makes this honest rather than a
    # suppressed error.
    gr = G.GateReport()
    _off = _mixture_off_deviation(world, cfg_r, n_mu, n_sub, ng, n_obs=min(int(n_obs), 24))
    gr.live("mixture_reaches_the_reader", _off, 1e-9, expected_to_fail=True,
            detail="forcing the per-position goal mixture to a constant must change the reader's "
                   "posterior. It does not: V5Environment.sample_feature ignores artifact.goal "
                   "once a creator is bound, so `actives` never enters the observations. This "
                   "gate fails by design and is why the module is withdrawn.")

    verdict = {
        "WITHDRAWN": (
            "2026-08-05. The manipulation never reaches the reader; see the failing live gate "
            "below and scripts/audit_s2_mixture.py. Superseded by "
            "t2_automaticity.mixed_deep_features, which emits from world.subsig directly."),
        "withdrawn": True,
        "test": "S-2 — does flattened intent read as posterior concentration at matched density?",
        "for": "Sounding Line, C-22 and purpose_breadth",
        "construction": {"mu": MU, "beta": BETA, "dominant_share": DOMINANT, "n_goals": int(ng),
                         "positions_per_artifact": int(n_timesteps),
                         "note": "decision density identical in both arms by construction"},
        "purpose_breadth": breadth,
        "goal_accuracy_should_not_fall": acc,
        "process_recovery_should_not_fall": proc,
        "outcome": ("BREADTH_SEPARATES_THE_TWO" if separates
                    else "BREADTH_DOES_NOT_SEPARATE_THE_TWO"),
        "construct_is_confounded": confounded,
        "how_to_read": (
            "the claim needs breadth to fall for flattened work while accuracy and process "
            "recovery hold. If process recovery falls too, 'flattened' has become 'shallower' "
            "and the measure is reading depth rather than concentration."),
        "what_would_have_falsified_it": (
            "identical purpose_breadth across the two arms. Then concentration of terminal values "
            "is invisible to a goal posterior and Sounding Line's purpose_breadth is measuring "
            "something else."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "s2_flattened_intent.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
