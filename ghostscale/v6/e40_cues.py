"""E40 — aesthetics, social endorsement, and the RLHF decoupling.

WHAT THE READER HAS BEEN MISSING. In V1-V5 the decision to look closely is driven by expected
information gain and nothing else. The theory says two other things feed that decision BEFORE any
inference has happened:

  * AESTHETIC SALIENCE -- the honeypot. A surface property that, in an honest world, predicts
    that depth is there. Attention-GRABBING, and not necessarily attention-KEEPING.
  * SOCIAL ENDORSEMENT -- presumed density. Extracting a process is so expensive that you often
    cannot see it yourself, so you take someone's word that it is there at all.

TWO QUESTIONS, AND THE SECOND IS THE ONE THAT MATTERS.

H6.8 asks HOW they combine. Additive is pre-registered, on the author's instinct that being told
something is good does not scale with how it looks -- the two are judged separately. The rules
disagree exactly where the model's own information gain is near zero: an additive rule lets a cue
drive engagement on content that offers nothing, and a multiplicative rule cannot, because it
scales the gain. That corner IS the test, and it is why the comparison is worth running rather
than assuming.

H6.9 is the RLHF decoupling and it is the alignment argument in miniature. If salience is LEARNED
as a predictor of depth, and generation then optimises the predictor directly, the cue decouples
from the thing it predicted. The reader keeps spending on a signal that no longer carries
information.

    prediction: engagement ABOVE the honest baseline, with error reduction at or below zero.
    the reader pays MORE and gets LESS.

That is a THIRD failure mode and it is worth being precise about why it is not one of the two the
project already has. It is not the crash: the reader is engaged, not disengaged. It is not the
trust exploit: the reader is not wrong about provenance, and no label has lied to it. It is a
reader correctly reading an artifact that has been shaped to trip its own heuristic for deciding
what is worth reading.

NULL N29. Both channels must carry ZERO goal information. A cue that told the reader anything
about WHICH goal would be a second legibility channel and would re-derive the label effect
through the back door.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import v6_model as V6
from ..config import Config
from ..prereg_v6 import h68_verdict, h69_verdict
from ..v5_model import MU_LEVELS, make_v5_observer
from . import harness as H
from . import SEED_OFFSET, v6_dir

# The four corners of the cue space. The two off-diagonal ones are where the combination rules
# disagree, which is why they are named rather than swept.
CORNERS = (("plain", 0.0, 0.0), ("beautiful_unendorsed", 1.0, 0.0),
           ("ugly_endorsed", 0.0, 1.0), ("beautiful_endorsed", 1.0, 1.0))


def run(cfg: Config, n_obs: int = 50, n_timesteps: int = 24, forced_k: int = 6,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)

    w_a = float(cfg.get("v6.cues.w_aesthetic", 0.35))
    w_s = float(cfg.get("v6.cues.w_social", 0.35))

    honest_map = V6.learned_salience_map(MU_LEVELS, decouple=False)
    decoupled_map = V6.learned_salience_map(MU_LEVELS, decouple=True)

    recs = []
    # ---- H6.8: the four corners, under both combination rules --------------
    for name, aesthetic, social in CORNERS:
        for i in range(int(n_obs)):
            art_rng = np.random.default_rng(SEED_OFFSET + 11_000 + i)
            # DEPTH ZERO CONTENT for the corner test: the model's own information gain is near
            # nothing, which is exactly where the two rules part company.
            creator, artifact, env = H.make_artifact_and_env(
                world, cfg_r, int(art_rng.integers(ng)), 1, 0.0, n_timesteps, art_rng)
            agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 12_000 + i))
            enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                  np.random.default_rng(SEED_OFFSET + 12_000 + i),
                                  n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
            ig = float(enc.engaged_fraction)
            add = V6.CueChannels(w_a, w_s, "additive").combine(ig, aesthetic, social)
            mul = V6.CueChannels(w_a, w_s, "multiplicative").combine(ig, aesthetic, social)
            recs.append({"block": "corners", "corner": name, "observer": i,
                         "aesthetic": aesthetic, "social": social,
                         "information_gain": ig,
                         "engagement_additive": min(add, 1.0),
                         "engagement_multiplicative": min(mul, 1.0),
                         "error_reduction": float(enc.error_reduction),
                         "goal_correct": int(enc.correct)})

    # ---- H6.9: the decoupling -----------------------------------------------
    for regime, smap in (("honest", honest_map), ("decoupled", decoupled_map)):
        for mu in MU_LEVELS:
            # In the decoupled regime the cue is maximised at EVERY depth including the
            # shallowest. In the honest regime it tracks depth, so a reader using it is not
            # making a mistake.
            for i in range(int(n_obs)):
                art_rng = np.random.default_rng(SEED_OFFSET + 13_000 + mu * 97 + i)
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, int(art_rng.integers(ng)), int(mu),
                    1.0 if mu > 1 else 0.0, n_timesteps, art_rng)
                agent = make_v5_observer(world, np.random.default_rng(SEED_OFFSET + 14_000 + mu * 89 + i))
                enc = H.run_encounter(world, cfg_r, artifact, env, agent, creator,
                                      np.random.default_rng(SEED_OFFSET + 14_000 + mu * 89 + i),
                                      n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
                sal = float(smap[int(mu)])
                eng = V6.CueChannels(w_a, w_s, "additive").combine(
                    float(enc.engaged_fraction), sal, 0.0)
                recs.append({"block": "decoupling", "regime": regime, "mu": int(mu),
                             "observer": i, "salience": sal,
                             "information_gain": float(enc.engaged_fraction),
                             "engagement": min(eng, 1.0),
                             "error_reduction": float(enc.error_reduction),
                             "process_error_reduction":
                                 float(enc.process["process_error_reduction"]),
                             "goal_correct": int(enc.correct)})

    df = pd.DataFrame(recs)
    out = v6_dir("e40_cues")
    df.to_csv(out / "e40_cues.csv", index=False)

    corners = df[df.block == "corners"]
    corner_means = corners.groupby("corner").agg(
        engagement_additive=("engagement_additive", "mean"),
        engagement_multiplicative=("engagement_multiplicative", "mean"),
        information_gain=("information_gain", "mean")).reset_index()
    corner_means.to_csv(out / "e40_corners.csv", index=False)

    def _c(name, col):
        return float(corner_means[corner_means.corner == name][col].iloc[0])

    # The empty corner: a cue present, the content offering nothing.
    h68 = h68_verdict(_c("ugly_endorsed", "engagement_additive"),
                      _c("ugly_endorsed", "engagement_multiplicative"),
                      _c("plain", "engagement_additive"))

    dec = df[df.block == "decoupling"]
    dec_means = dec.groupby(["regime", "mu"]).agg(
        engagement=("engagement", "mean"),
        error_reduction=("error_reduction", "mean"),
        process_error_reduction=("process_error_reduction", "mean"),
        information_gain=("information_gain", "mean")).reset_index()
    dec_means.to_csv(out / "e40_decoupling.csv", index=False)

    # The decoupled reader meeting SHALLOW content with the cue maximised, against the honest
    # reader meeting the same shallow content with the cue correctly low.
    d_shallow = dec_means[(dec_means.regime == "decoupled") & (dec_means.mu == 1)]
    h_shallow = dec_means[(dec_means.regime == "honest") & (dec_means.mu == 1)]
    h69 = h69_verdict(float(d_shallow.engagement.iloc[0]),
                      float(h_shallow.engagement.iloc[0]),
                      float(d_shallow.error_reduction.iloc[0]))

    # ---- N29: the cues must carry no goal information -----------------------
    # Measured directly: does goal accuracy differ across corners, which vary only in cue value?
    acc_by_corner = corners.groupby("corner").goal_correct.mean()
    n29_spread = float(acc_by_corner.max() - acc_by_corner.min())
    n29_passed = bool(n29_spread <= 0.05)

    verdict = {
        "experiment": "E40",
        "hypotheses": ["H6.8", "H6.9"],
        "question": ("How do surface appeal and social endorsement combine into the decision to "
                     "look closely, and what happens when the surface cue is optimised directly?"),
        "plain_language": (
            "The reader in this model decides to look closely on one basis: how much it expects "
            "to learn. People do not work that way. Something catches your eye, and someone "
            "tells you it is worth your time, and both of those happen before you have learned "
            "anything at all. This adds those two channels, and then asks what happens when the "
            "eye-catching one is optimised on purpose."),
        "corners": corner_means.to_dict(orient="records"),
        "H6.8": h68,
        "decoupling": dec_means.to_dict(orient="records"),
        "H6.9": h69,
        "null_n29": {
            "statement": "the cue channels carry no goal information",
            "goal_accuracy_spread_across_corners": n29_spread,
            "passed": n29_passed,
            "why": ("a cue that told the reader which goal would be a second legibility channel "
                    "and would re-derive the label effect through the back door"),
        },
        "weights": {"aesthetic": w_a, "social": w_s},
        "why_this_is_a_third_failure_mode": (
            "not the crash, because the reader is engaged rather than disengaged; not the trust "
            "exploit, because no label has lied and the reader is not wrong about provenance. It "
            "is a reader correctly reading an artifact shaped to trip its own heuristic for "
            "deciding what is worth reading."),
        "n_obs": int(n_obs),
    }
    if not n29_passed:
        verdict["INTERPRETABILITY"] = (
            "NULL N29 FAILED. The cue channels move goal accuracy, so they are carrying goal "
            "information and are a second legibility channel. Every number above is "
            "uninterpretable.")
    (v6_dir() / "e40_cues.json").write_text(json.dumps(verdict, indent=2, default=str),
                                            encoding="utf-8")
    return verdict
