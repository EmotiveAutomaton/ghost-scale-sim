"""E51 — a maker that can lie, and whether honest marking pays for itself.

THE PART OF THE PROPOSAL WITH A FORMAL APPENDIX AND ZERO LINES OF SIMULATION.

Every version until now has had makers that emit and readers that read. Nobody in the model has
ever CHOSEN anything. So the framework's answer to bad actors -- that honest marking is costly and
therefore self-policing, by Zahavian signalling -- has been asserted for eight versions and never
tested.

The argument: a maker who marks its work honestly pays a cost, because the mark lowers how much
readers take from it. A maker who lies gets the uptake of honest work. But if lying is DETECTED,
the reputational cost exceeds what the lie gained, so honesty is stable above some detection rate.

That is a claim about an equilibrium and it needs two agents.

-----------------------------------------------------------------------------------------
A CORRECTION TO THE FRAMEWORK'S OWN FRAMING, ARRIVED AT BY RUNNING IT.

The framework reaches this argument through ZAHAVIAN SIGNALLING -- the handicap principle, on which
a signal is honest only if it is WASTEFUL, because the waste is what a low-quality signaller cannot
afford. That is where the hypothesis came from and it is worth saying so.

**Signalling theory has moved off that position, and this experiment lands on the newer one without
having been aimed at it.** Honesty is now understood to be maintained by TRADE-OFFS rather than by
cost: what matters is not that the signal is expensive but that DECEPTION is expensive relative to
what it gains. (Honesty in signalling games is maintained by trade-offs rather than costs, BMC
Biology 2022; general signalling theory, J. Evol. Biol. 2026.)

What this experiment finds is a DETECTION-RATE THRESHOLD, which is a trade-off account: honesty
wins where being caught costs more than lying gains. Nothing here depends on the honest signal being
wasteful. So the result supports the framework's conclusion and not its stated mechanism, and both
are reported.

-----------------------------------------------------------------------------------------
WHAT IS DELIBERATELY NOT BUILT, per the author's decision.

A maker that chooses HOW MUCH TO DELEGATE to a tool -- the cognitive-surrender dynamic -- is a
second decision, and a two-decision maker makes attribution hard in exactly the way the repair pass
was written against. This maker makes ONE choice: declare honestly, or claim to be human.

-----------------------------------------------------------------------------------------
AND THE SECOND HALF, WHICH IS WHY THIS EXPERIMENT CARRIES H8.7 TOO.

E39 gave a reader a "there is no maker here" hypothesis and it bought nothing, because it was
redundant with what the reader already knew about origin. The richer version needs a maker to
exist: a reader that models THE PERSON WHO CHOSE TO USE A TOOL. That is a maker, with intent, at
one remove -- and it is something the reader can conclude rather than merely fail at.

NULL N40 IS THE GATE. Lying must PAY when it is not caught, or the honest equilibrium is an
artefact of a rigged payoff rather than a result about signalling.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import v6_model as V6
from .. import v8_model as V8
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v8_dir

DETECTION_RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
LEAKS = (0.0, 0.10)          # does a reader who cannot fully refuse make defection cheaper?


class Maker:
    """A maker with one decision: declare honestly, or claim to be human.

    It has a real quality -- how much intent it actually put in -- and it chooses a label. Its
    payoff is how much readers take from its work, minus whatever being caught costs it.
    """

    def __init__(self, quality: int, honest: bool):
        self.quality = int(quality)          # the depth actually behind the work
        self.honest = bool(honest)
        self.caught = 0
        self.payoff = 0.0

    @property
    def declared(self) -> int:
        if self.honest:
            return K.SIG_CREATOR if self.quality >= 2 else K.SIG_GHOST
        return K.SIG_CREATOR                 # the lie is always "a person made this"

    @property
    def is_lying(self) -> bool:
        return (not self.honest) and self.quality < 2


def run(cfg: Config, n_makers: int = 24, n_encounters: int = 12, n_timesteps: int = 16,
        reputation_cost: float = 2.0, workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    lam = float(world.cfg.get("v6.gate.lam", 1.0))
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))

    rows = []
    for leak in LEAKS:
        for detect in DETECTION_RATES:
            for strategy in ("honest", "defect"):
                for m in range(int(n_makers)):
                    rng = np.random.default_rng(SEED_OFFSET + 87_000 + m * 53)
                    # Half the makers actually have depth; the other half do not, and it is the
                    # second group for whom the lie is a lie.
                    quality = 3 if m % 2 == 0 else 1
                    maker = Maker(quality, honest=(strategy == "honest"))

                    for e in range(int(n_encounters)):
                        r = np.random.default_rng(SEED_OFFSET + 88_000 + m * 31 + e)
                        g = int(r.integers(ng))
                        creator, art, env = H.make_artifact_and_env(
                            world, cfg_r, g, maker.quality, 1.0, n_timesteps, r,
                            provenance=K.CREATOR if maker.quality >= 2 else K.GHOST,
                            declared_signal=maker.declared, signing_rate=1.0)
                        enc = H.run_encounter(world, cfg_r, art, env,
                                              make_v5_observer(world, r), creator, r,
                                              n_timesteps, 8, n_sub, n_mu, ng, kappa)

                        resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
                        divergence = float(max(0.0, -enc.error_reduction))
                        theta = V6.disgust_threshold(divergence, kappa, None, lam, theta_0, 0.0)
                        gate = V6.gate(float(np.clip(resolved, 0.0, 1.0)), theta, k_gain,
                                       leak=float(leak))
                        uptake = float(gate) * float(max(enc.error_reduction, 0.0))

                        # DETECTION. A lie is caught at the stated rate, and only a lie can be
                        # caught -- an honest maker has nothing to find.
                        caught = bool(maker.is_lying and r.random() < detect)
                        if caught:
                            maker.caught += 1
                            maker.payoff -= float(reputation_cost)
                        maker.payoff += uptake

                    rows.append({
                        "leak": float(leak), "detection": float(detect), "strategy": strategy,
                        "maker": m, "quality": maker.quality, "was_lying": maker.is_lying,
                        "times_caught": maker.caught,
                        "payoff": float(maker.payoff / max(n_encounters, 1)),
                    })

    df = pd.DataFrame(rows)
    out = v8_dir("e51_creator")
    df.to_csv(out / "e51_creator.csv", index=False)

    # The equilibrium question is about the makers for whom the lie IS a lie.
    liars = df[(df.strategy == "defect") & (df.quality < 2)]
    honest_low = df[(df.strategy == "honest") & (df.quality < 2)]

    curve = []
    for leak in LEAKS:
        for detect in DETECTION_RATES:
            d = float(liars[(liars.leak == leak) & (liars.detection == detect)].payoff.mean())
            h = float(honest_low[(honest_low.leak == leak)
                                 & (honest_low.detection == detect)].payoff.mean())
            curve.append({"leak": float(leak), "detection": float(detect),
                          "defector_payoff": d, "honest_payoff": h,
                          "honesty_pays": bool(h >= d)})
    cdf = pd.DataFrame(curve)
    cdf.to_csv(out / "e51_equilibrium.csv", index=False)

    def _threshold(leak):
        s = cdf[cdf.leak == leak].sort_values("detection")
        for r in s.itertuples():
            if r.honesty_pays:
                return float(r.detection)
        return None

    t_tight = _threshold(0.0)
    t_leaky = _threshold(0.10)

    # N40: lying must pay when it is never caught.
    n40 = bool(float(cdf[(cdf.leak == 0.0) & (cdf.detection == 0.0)].defector_payoff.iloc[0])
               > float(cdf[(cdf.leak == 0.0) & (cdf.detection == 0.0)].honest_payoff.iloc[0]))

    exists = t_tight is not None
    leak_makes_it_worse = (t_tight is not None and t_leaky is not None and t_leaky > t_tight) or \
                          (t_tight is not None and t_leaky is None)

    verdict = {
        "experiment": "E51",
        "hypothesis": "H8.6",
        "question": ("Is honest marking self-policing? The framework's answer to bad actors has "
                     "been asserted for eight versions and never simulated."),
        "plain_language": (
            "Marking your work honestly costs you: the mark lowers what readers take from it. "
            "Lying gets you the uptake of honest work. The proposal's security argument is that "
            "being caught costs more than the lie gained, so honesty holds up on its own above "
            "some rate of detection. That is a claim about an equilibrium and it needs a maker who "
            "can choose."),
        "equilibrium": cdf.to_dict(orient="records"),
        "detection_rate_where_honesty_pays": {"tight_gate": t_tight, "leaky_gate": t_leaky},
        "H8.6": {
            "an_honest_equilibrium_exists": exists,
            "a_leaky_reader_makes_defection_cheaper": bool(leak_makes_it_worse),
            "outcome": ("HONEST_MARKING_IS_SELF_POLICING_ABOVE_A_DETECTION_RATE" if exists
                        else "DEFECTION_DOMINATES_AT_EVERY_DETECTION_RATE"),
        },
        "null_n40": {
            "statement": "lying must pay when it is never caught, or the payoff is rigged",
            "passed": n40,
            "why": ("without it an honest equilibrium is an artefact of the scoring rather than a "
                    "result about signalling"),
        },
        "what_was_deliberately_not_built": (
            "a maker that chooses HOW MUCH TO DELEGATE to a tool. That is a second decision and a "
            "two-decision maker makes attribution hard in exactly the way the repair pass was "
            "written against. This maker makes one choice: declare honestly, or claim to be human."),
        "reputation_cost": float(reputation_cost),
        "n_makers": int(n_makers), "n_encounters": int(n_encounters),
    }
    if not n40:
        verdict["INTERPRETABILITY"] = (
            "NULL N40 FAILED. Lying does not pay even when it is never caught, so the payoff is "
            "rigged and the equilibrium result is an artefact of the scoring.")
    (v8_dir() / "e51_creator.json").write_text(json.dumps(verdict, indent=2, default=str),
                                               encoding="utf-8")
    return verdict
