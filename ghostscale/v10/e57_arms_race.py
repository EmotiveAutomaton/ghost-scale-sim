"""E57 — the arms race. E53 done in a world that has an adversary in it.

E53 swept detector training against FIXED content and found the misfiring gets rarer as the
detector sharpens: false alarms fell from 0.43 to 0.23 between 8 examples and 512. That was
reported as the author's prediction failing.

The objection, and it is a design flaw rather than a disagreement: **the real world sweeps both
sides.** Detection improves, evasion improves in response, and makers begin explicitly optimising
against the reflex -- *make sure it doesn't look like AI*. E53 measured a world with no adversary
in it and then drew a conclusion about a world that has one.

It changes the sign of the argument rather than its precision. What separates machine from human at
depth in this model is EFFORT, so evasion means machine content emitting human-typical surfaces.
The detector's discriminating feature erodes as fast as it sharpens -- while its CONFIDENCE does
not, because confidence comes from sample size and sample size only grows.

H10.7  With evasion tracking detection, the false-alarm rate is NON-MONOTONE in training, rather
       than the clean decline E53 reported.

H10.8  A STALE detector -- trained on an older content distribution and never updated -- fails
       ASYMMETRICALLY. It does not become noisy. It becomes confidently wrong in one direction.
       Aggregate false-alarm rates conceal this completely, which is what E53's reporting did.

N49    With evasion switched off, this harness reproduces E53's monotone decline. Otherwise the
       harness changed the old result and no comparison between them is valid.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from ..config import Config
from ..v6 import harness as H
from ..v9.e53_e54 import DETECTOR_TRAINING
from . import SEED_OFFSET, v10_dir

EVASION = (0.0, 0.25, 0.5, 0.75)


class EvadingDetector:
    """E53's detector, trained against content that is trying not to be caught.

    ``evasion`` is the probability that a machine artifact emits a HUMAN-typical feature instead
    of its own. That is what "make sure it doesn't look like AI" means when the only thing
    separating the two populations is effort: you spend the effort, or you imitate its traces.

    The detector is a log-likelihood ratio and nothing more. It has no access to intent -- that is
    what the rest of this project is for -- so evasion is not something it can see through.
    """

    def __init__(self, world, cfg_r, n_features: int, n_train: int, ng: int,
                 evasion: float, rng):
        self.n_features = int(n_features)
        self.n_train = int(n_train)
        self.evasion = float(evasion)
        self.log_ratio = np.zeros(self.n_features, dtype=float)
        if n_train <= 0:
            return
        mach = np.ones(self.n_features, dtype=float)
        human = np.ones(self.n_features, dtype=float)
        for i in range(int(n_train)):
            g = int(rng.integers(ng))
            for j, (prov, beta, tally) in enumerate(((K.GHOST, 0.0, mach),
                                                     (K.CREATOR, 1.0, human))):
                r = np.random.default_rng(SEED_OFFSET + 20_000 + i * 7 + j * 3_001)
                use_beta = beta
                if prov == K.GHOST and r.random() < self.evasion:
                    use_beta = 1.0           # imitate the effortful surface
                    prov_used = K.CREATOR
                else:
                    prov_used = prov
                _, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, use_beta, 4, r, provenance=prov_used)
                tally[int(env.sample_feature(art, K.DEEP, r))] += 1.0
        self.log_ratio = np.log(mach / mach.sum()) - np.log(human / human.sum())

    def score(self, features) -> float:
        if self.n_train == 0 or not len(features):
            return 0.0
        return float(np.mean([self.log_ratio[int(f)] for f in features]))

    def fires(self, features, threshold: float = 0.0) -> bool:
        return bool(self.score(features) > threshold)


def _evaluate(det, world, cfg_r, ng, n_obs, n_timesteps, evasion_at_test, tag) -> list:
    """Run the detector over machine and human content at a given level of evasion."""
    rows = []
    arms = (("machine", K.GHOST, 0.0), ("human_careful", K.CREATOR, 1.0),
            ("human_fast", K.CREATOR, 0.25))
    for arm, prov, beta in arms:
        for i in range(int(n_obs)):
            rng = np.random.default_rng(SEED_OFFSET + 21_000 + i)
            g = int(rng.integers(ng))
            use_beta, use_prov = beta, prov
            evaded = False
            if prov == K.GHOST and rng.random() < float(evasion_at_test):
                use_beta, use_prov, evaded = 1.0, K.CREATOR, True
            _, art, env = H.make_artifact_and_env(
                world, cfg_r, g, 2, use_beta, n_timesteps, rng, provenance=use_prov)
            glance = [int(env.sample_feature(art, K.DEEP, rng)) for _ in range(3)]
            rows.append({"regime": tag, "arm": arm, "observer": i,
                         "n_train": det.n_train, "evasion": float(evasion_at_test),
                         "trained_at": det.evasion, "evaded": int(evaded),
                         "fired": int(det.fires(glance)), "score": det.score(glance)})
    return rows


def run(cfg: Config, n_obs: int = 40, n_timesteps: int = 16, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    nf = int(cfg_b.cardinalities.num_features)

    rows = []
    # --- CO-EVOLUTION: both sides advance together ---------------------------
    for n_train, ev in zip(DETECTOR_TRAINING, (0.0,) + EVASION):
        det = EvadingDetector(world, cfg_r, nf, n_train, ng, ev,
                              np.random.default_rng(SEED_OFFSET + 22_000))
        rows += _evaluate(det, world, cfg_r, ng, n_obs, n_timesteps, ev, "co_evolution")

    # --- CONTROL (N49): evasion off, detector sharpening alone ---------------
    for n_train in DETECTOR_TRAINING:
        det = EvadingDetector(world, cfg_r, nf, n_train, ng, 0.0,
                              np.random.default_rng(SEED_OFFSET + 22_000))
        rows += _evaluate(det, world, cfg_r, ng, n_obs, n_timesteps, 0.0, "no_evasion")

    # --- STALE: trained once at evasion 0, never updated ---------------------
    stale = EvadingDetector(world, cfg_r, nf, DETECTOR_TRAINING[-1], ng, 0.0,
                            np.random.default_rng(SEED_OFFSET + 22_000))
    for ev in (0.0,) + EVASION[1:]:
        rows += _evaluate(stale, world, cfg_r, ng, n_obs, n_timesteps, ev, "stale")

    df = pd.DataFrame(rows)
    out = v10_dir("e57_arms_race")
    df.to_csv(out / "e57_arms_race.csv", index=False)

    def _curve(regime, key):
        g = (df[df.regime == regime].groupby([key, "arm"])["fired"].mean()
             .unstack("arm").reset_index())
        return g.to_dict(orient="records")

    co = _curve("co_evolution", "n_train")
    ctrl = _curve("no_evasion", "n_train")
    st = _curve("stale", "evasion")

    def _fa(recs, col="human_careful"):
        return [float(r[col]) for r in recs]

    # H10.7 -- non-monotone under co-evolution, where the control declines cleanly.
    co_fa = _fa(co)
    diffs = np.diff(co_fa)
    h107 = bool(np.any(diffs > 0) and np.any(diffs < 0))

    # N49 -- with evasion off, the decline must reproduce.
    ctrl_fa = _fa(ctrl)
    n49 = bool(ctrl_fa[-1] < ctrl_fa[1]) if len(ctrl_fa) > 2 else False

    # H10.8 -- the stale detector fails ASYMMETRICALLY: hits collapse while false alarms do not,
    # so what survives is a detector that mostly says "human" and is confident when it does not.
    hit0, hitN = float(st[0]["machine"]), float(st[-1]["machine"])
    fa0, faN = float(st[0]["human_careful"]), float(st[-1]["human_careful"])
    d_hit, d_fa = hitN - hit0, faN - fa0
    h108 = bool(abs(d_hit) > 2.0 * abs(d_fa) if abs(d_fa) > 1e-9 else abs(d_hit) > 0.05)

    verdict = {
        "experiment": "E57",
        "hypotheses": ["H10.7", "H10.8"],
        "question": ("What happens to a learned detector when the content is trying not to be "
                     "caught -- and to a reader whose detector stopped updating?"),
        "plain_language": (
            "E53 measured a world with no adversary in it. Detection improves, but so does "
            "evasion, because once people know the reflex exists they optimise against it. And "
            "some readers stop updating."),
        "co_evolution": co,
        "control_no_evasion": ctrl,
        "stale_detector": st,
        "H10.7": {
            "outcome": ("MISFIRING_IS_NON_MONOTONE_ONCE_CONTENT_FIGHTS_BACK" if h107
                        else "MISFIRING_STILL_DECLINES_UNDER_CO_EVOLUTION"),
            "false_alarm_by_step_co_evolution": co_fa,
            "false_alarm_by_step_control": ctrl_fa,
        },
        "H10.8": {
            "outcome": ("THE_STALE_DETECTOR_FAILS_ASYMMETRICALLY" if h108
                        else "THE_STALE_DETECTOR_JUST_GETS_NOISIER"),
            "hit_rate_change": d_hit, "false_alarm_change": d_fa,
            "how_to_read": (
                "a detector that merely decayed would lose hits and gain false alarms in "
                "proportion. One that loses hits while holding its false-alarm rate has not become "
                "unreliable -- it has become a detector that mostly says HUMAN and is confident "
                "when it is wrong. Aggregate rates hide this, which is what E53 reported."),
        },
        "null_n49": {
            "statement": "with evasion off, this harness reproduces E53's monotone decline",
            "passed": n49,
            "why": "otherwise the harness changed the old result and no comparison is valid",
        },
        "n_obs": int(n_obs),
    }
    (v10_dir() / "e57_arms_race.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
