"""Does S-2's goal mixture ever reach the reader? No. This is the check.

S-2 builds a per-position mixture of terminal goals and emits from it:

    a = Artifact(provenance=READ_TIER, goal=int(actives[t]), declared_signal=K.UNSIGNED)
    deep.append(int(env.sample_feature(a, K.DEEP, art_rng)))

``env`` is a ``V5Environment`` with a ``HierarchicalCreator`` bound, and that method never reads
``artifact.goal``:

    def sample_feature(self, artifact, attention, rng):
        if self.creator is None: ...
        a = float(self.alpha[artifact.provenance])
        if rng.random() < a:
            return self.creator.next_feature(rng)

Only ``artifact.provenance`` is consulted, for the alpha lookup. Every feature comes from the one
creator, which holds one fixed goal for the whole artifact. So ``actives`` steers nothing and the
manipulation S-2 believes it is running is not in the observations.

What is left between S-2's two arms is ``modal``: the flattened arm always uses goal 0, the
layered arm draws a goal uniformly per artifact. That is a contrast between "always goal 0" and
"a random goal", plus the RNG-stream shift from the extra ``integers`` draw. This script measures
how much of S-2's reported separation that accounts for.

    python scripts/audit_s2_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostscale import constants as K                                    # noqa: E402
from ghostscale import metrics                                           # noqa: E402
from ghostscale.baselines import ObservationTape, TapedEnvironment       # noqa: E402
from ghostscale.config import load_config                                # noqa: E402
from ghostscale.environment import Artifact                              # noqa: E402
from ghostscale.v5_model import make_v5_observer                         # noqa: E402
from ghostscale.v6 import harness as H                                   # noqa: E402
from ghostscale.validation.soundingline import sl_dir                    # noqa: E402
from ghostscale.validation.soundingline.common import build              # noqa: E402
from ghostscale.validation.soundingline.s2_flattened_intent import (     # noqa: E402
    DOMINANT, MU, READ_GLANCES, READ_TIER, _weights)

N_TIMESTEPS = 24


def one(world, cfg_r, n_mu, n_sub, ng, kind, i, force_actives_to_modal: bool):
    """S-2's own inner loop, with one switch: whether the mixture is drawn at all."""
    base = 62_000 + (0 if kind == "layered" else 1) * 977
    art_rng = np.random.default_rng(base * 31 + i)
    modal = int(np.argmax(_weights(kind, ng))) if kind == "flattened" \
        else int(art_rng.integers(ng))
    creator, artifact, env = H.make_artifact_and_env(
        world, cfg_r, modal, MU, 1.0, N_TIMESTEPS, art_rng, provenance=READ_TIER)
    tape = ObservationTape(env, artifact, np.random.default_rng(base * 104729 + i), N_TIMESTEPS)
    actives = art_rng.choice(ng, size=N_TIMESTEPS, p=_weights(kind, ng))
    if force_actives_to_modal:
        actives = np.full(N_TIMESTEPS, modal, dtype=int)
    tape.deep = np.asarray(
        [int(env.sample_feature(Artifact(provenance=READ_TIER, goal=int(actives[t]),
                                         declared_signal=K.UNSIGNED), K.DEEP, art_rng))
         for t in range(N_TIMESTEPS)], dtype=int)
    agent = make_v5_observer(world, np.random.default_rng(base * 7907 + i))
    enc = H.run_encounter(world, cfg_r, artifact, TapedEnvironment(tape), agent, creator,
                          np.random.default_rng(base * 7907 + i), N_TIMESTEPS, READ_GLANCES,
                          n_sub, n_mu, ng, float(world.cfg.signal_model.kappa), true_goal=modal)
    hmax = float(np.log(max(ng, 2)))
    return {"breadth": float(metrics.within_observer_entropy(enc.goal_posterior)) / hmax,
            "correct": int(enc.correct), "modal": modal,
            "features": tape.deep.tolist()}


def main() -> None:
    cfg = load_config()
    world, _b, cfg_r, n_mu, n_sub, ng = build(cfg)
    n = 120
    out = {}

    # 1. Bit-identity: forcing the mixture off must change nothing at all.
    ident, feats_same = [], []
    for kind in ("layered", "flattened"):
        for i in range(n):
            a = one(world, cfg_r, n_mu, n_sub, ng, kind, i, force_actives_to_modal=False)
            b = one(world, cfg_r, n_mu, n_sub, ng, kind, i, force_actives_to_modal=True)
            ident.append(abs(a["breadth"] - b["breadth"]))
            feats_same.append(a["features"] == b["features"])
    out["mixture_is_inert"] = {
        "max_abs_breadth_difference_with_mixture_off": float(np.max(ident)),
        "feature_streams_identical_fraction": float(np.mean(feats_same)),
        "verdict": ("MIXTURE_NEVER_REACHES_THE_READER" if float(np.max(ident)) < 1e-12
                    else "MIXTURE_HAS_SOME_EFFECT"),
    }

    # 2. What the arms actually differ by: the modal goal, and nothing else.
    by_kind = {}
    for kind in ("layered", "flattened"):
        rs = [one(world, cfg_r, n_mu, n_sub, ng, kind, i, False) for i in range(n)]
        by_kind[kind] = rs
        out[f"{kind}_modal_goal_distribution"] = np.bincount(
            [r["modal"] for r in rs], minlength=ng).tolist()
    diff = (float(np.mean([r["breadth"] for r in by_kind["flattened"]]))
            - float(np.mean([r["breadth"] for r in by_kind["layered"]])))
    out["reproduced_s2_breadth_difference"] = diff

    # 3. The confound isolated: score the LAYERED arm restricted to artifacts whose modal goal
    #    happened to be 0, which is the only goal the flattened arm ever uses.
    lay0 = [r["breadth"] for r in by_kind["layered"] if r["modal"] == 0]
    fla = [r["breadth"] for r in by_kind["flattened"]]
    out["confound_isolated"] = {
        "layered_restricted_to_modal_goal_0_breadth": float(np.mean(lay0)) if lay0 else None,
        "n_layered_with_modal_0": len(lay0),
        "flattened_breadth": float(np.mean(fla)),
        "residual_difference_after_matching_the_goal": (
            float(np.mean(fla) - np.mean(lay0)) if lay0 else None),
        "how_to_read": (
            "S-2's arms differ only in which TRUE goal the artifact has: flattened is always "
            "goal 0, layered is a uniform draw. Matching on the goal removes the manipulation "
            "entirely, because the mixture was never in the observations. Whatever is left is "
            "the residual RNG-stream shift from the extra integers() draw."),
    }

    out["what_this_means"] = (
        "S-2's reported purpose_breadth separation is not a measurement of goal-mixture "
        "concentration. The mixture is drawn and then discarded by the environment. The arms "
        "differ in the artifact's true goal and in RNG stream position. S-2 should be withdrawn "
        "and re-run against an emitter that actually mixes drives at the emission.")

    p = sl_dir() / "audit_s2_mixture.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
