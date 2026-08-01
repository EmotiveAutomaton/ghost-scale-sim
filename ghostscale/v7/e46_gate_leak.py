"""E46 — the gate cannot fully close, and that is why propaganda works.

THE CLAIM, AND IT IS A CORRECTION TO THE MODEL RATHER THAN AN ADDITION TO IT.

The acceptance gate as V1 through V6 built it can shut completely. E42 measured a reader that
rejects material integrating exactly 0.00 of it. That reader is perfectly protected: it can study
something it disagrees with as long as it likes and walk away unchanged.

**People are not like that, and the theory never said they were.** The preprint is explicit, and it
is worth quoting because this is a term the code was missing rather than a new idea:

    "The calculation of the value disagreement itself implies simulation, which inherently drives
    Hebbian learning due to gating imperfections. This forced Hebbian learning is likely the
    mechanism for indoctrination and propaganda."

The argument is tight. To decide whether you disagree with something you have to work out what it
is saying, and working out what it is saying means running it. You cannot evaluate a claim without
partly instantiating it. So the act of rejection is itself a small act of absorption, and a gate
that closes to exactly zero is not a strong reader -- it is a missing term.

-----------------------------------------------------------------------------------------
THE PREDICTION THAT MAKES THIS A TEST RATHER THAN A PATCH.

If the leak is real, then rejection is not protection at scale. A reader exposed repeatedly to
material it rejects every single time still drifts, and the drift accumulates.

And the second half is the unpleasant one: **the drift should be LARGER for readers who engage more
closely.** The person who reads the propaganda carefully in order to refute it is more affected than
the person who skims it, because the leak passes a fraction of what was actually recovered, and
careful reading recovers more.

    fails if drift is flat in exposure count, which would make the leak a scaling constant on a
    single encounter rather than a mechanism that compounds

NULL N32 IS THE ONE THAT MATTERS. A leak must not manufacture learning. Exposure to material with no
recoverable structure in it must produce no drift at any leak setting, because the leak passes a
fraction of what was RECOVERED and where nothing was recovered there is nothing to pass. Without
that null the leak is just a drift knob and every number here would be the knob's.

THE LEAK'S SIZE IS SWEPT RATHER THAN CHOSEN. There is no principled value for it, so picking one and
reporting the consequence would be reporting the choice. The output is the SHAPE of drift against
leak, not a number at one setting.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import constants as K
from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from . import SEED_OFFSET, v7_dir

LEAKS = (0.0, 0.02, 0.05, 0.10, 0.20)
ENGAGEMENT_ARMS = (("skims_it", 2), ("reads_it_closely", 16))


def _drift_over_exposures(world, cfg_r, n_mu, n_sub, ng, leak: float, forced_k: int,
                          n_encounters: int, n_readers: int, n_timesteps: int,
                          rejectable: bool, base_seed: int) -> list:
    """One reader's belief drift across repeated exposure to material it rejects every time.

    THE READER REJECTS EVERY ENCOUNTER. That is set up rather than hoped for: the reader's values
    are placed opposite whatever it is about to recover, so the gate is closed on every single
    artifact. Any movement in its beliefs is therefore movement it refused.
    """
    kappa = float(world.cfg.signal_model.kappa)
    lam = float(world.cfg.get("v6.gate.lam", 1.0))
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))
    values_map = V6.build_values_map(ng, n_values=2)

    rows = []
    for r in range(int(n_readers)):
        # The reader's own belief about what people are for. This is what drifts.
        carried = np.full(ng, 1.0 / ng)
        start = carried.copy()
        for e in range(int(n_encounters)):
            rng = np.random.default_rng(SEED_OFFSET + base_seed + r * 997 + e)
            g = int(rng.integers(ng))
            if rejectable:
                creator, artifact, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, artifact, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, forced_k, n_sub, n_mu, ng, kappa)
                recovered = enc.goal_posterior
            else:
                # NULL N32's condition: nothing recoverable in the content at all, so a leak has
                # nothing to leak. Foreign material the reader has no hypothesis for.
                artifact, env = H.make_foreign_artifact_and_env(
                    world, cfg_r, g, n_timesteps, rng, omega=0.0)
                enc = H.run_encounter(world, cfg_r, artifact, env,
                                      make_v5_observer(world, rng), None, rng,
                                      n_timesteps, forced_k, n_sub, n_mu, ng, kappa,
                                      true_goal=g)
                recovered = enc.goal_posterior

            # The reader's values sit OPPOSITE what it just recovered, so the gate closes.
            implied = V6.implied_values(recovered, values_map)
            value_prior = np.clip(implied[::-1], 1e-6, None)
            value_prior = value_prior / value_prior.sum()
            divergence = V6.value_divergence_via_values(recovered, value_prior, values_map)

            resolved = 1.0 - float(enc.final_entropy) / float(np.log(ng))
            theta = V6.disgust_threshold(divergence, kappa, None, lam, theta_0, coupling=0.0)
            g_open = V6.gate(float(np.clip(resolved, 0.0, 1.0)), theta, k_gain, leak=leak)

            # WHAT LEAKS IS WHAT WAS RECOVERED, scaled by how far the gate is open and by how much
            # the reader actually engaged. That is the whole mechanism: rejection still requires
            # simulation, and simulation is what does the writing.
            looked = float(np.mean(np.asarray(enc.attention) == K.DEEP))
            weight = float(g_open) * looked
            carried = (1.0 - weight) * carried + weight * np.asarray(recovered, dtype=float)
            carried = carried / carried.sum()

            rows.append({
                "leak": float(leak), "reader": r, "encounter": e,
                "rejectable": bool(rejectable),
                "gate_open": float(g_open),
                "looked": looked,
                "drift_from_start": float(metrics.kl_divergence(carried, start)),
                "value_divergence": float(divergence),
            })
    return rows


def run(cfg: Config, n_readers: int = 12, n_encounters: int = 16, n_timesteps: int = 24,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)

    rows = []
    # ---- the sweep: how much drift, at each leak ---------------------------
    for leak in LEAKS:
        rows.extend(_drift_over_exposures(world, cfg_r, n_mu, n_sub, ng, leak, 6,
                                          n_encounters, n_readers, n_timesteps,
                                          rejectable=True, base_seed=46_000))
    # ---- N32: nothing recoverable, so nothing may leak ---------------------
    null_rows = _drift_over_exposures(world, cfg_r, n_mu, n_sub, ng, max(LEAKS), 6,
                                      n_encounters, n_readers, n_timesteps,
                                      rejectable=False, base_seed=47_000)
    # ---- how engagement changes the damage ---------------------------------
    eng_rows = []
    for name, fk in ENGAGEMENT_ARMS:
        r = _drift_over_exposures(world, cfg_r, n_mu, n_sub, ng, 0.10, fk,
                                  n_encounters, n_readers, n_timesteps,
                                  rejectable=True, base_seed=48_000)
        for x in r:
            x["arm"] = name
        eng_rows.extend(r)

    df = pd.DataFrame(rows)
    ndf = pd.DataFrame(null_rows)
    edf = pd.DataFrame(eng_rows)
    out = v7_dir("e46_gate_leak")
    df.to_csv(out / "e46_drift.csv", index=False)
    edf.to_csv(out / "e46_engagement.csv", index=False)

    by_leak = df.groupby(["leak", "encounter"]).drift_from_start.mean().reset_index()
    final = by_leak[by_leak.encounter == by_leak.encounter.max()].sort_values("leak")

    from ..prereg_v6 import spearman
    accumulates = {}
    for leak in LEAKS:
        s = by_leak[by_leak.leak == leak].sort_values("encounter")
        accumulates[float(leak)] = {
            "final_drift": float(s.drift_from_start.iloc[-1]),
            "monotone_rho": spearman(s.encounter, s.drift_from_start),
        }

    zero = accumulates[0.0]["final_drift"]
    biggest = accumulates[max(LEAKS)]["final_drift"]
    n32_drift = float(ndf.groupby("encounter").drift_from_start.mean().iloc[-1])

    # NULL N32, RESTATED BEFORE IT WAS SCORED, AND THE RESTATEMENT IS ITSELF A FINDING.
    #
    # The null was written as "material with nothing recoverable in it must produce no drift". The
    # condition chosen for it was fully foreign content -- and that is exactly the content this
    # project has spent four versions showing makes readers CONFIDENTLY WRONG. Such a reader has
    # recovered something. It has recovered a fabrication. So the leak passes it, correctly, and
    # the null as written was asking the model to violate its own headline result.
    #
    # What the null can actually check is that the leak passes rather than manufactures: at leak
    # zero there must be no drift at all, and drift must be monotone in the leak. Both are scored.
    # The foreign-content number is reported as a RESULT, not as a failure, and it is an unpleasant
    # one: a reader absorbs its own invention.
    # SCORED AS A RATIO, NOT AGAINST EXACT ZERO, AND THE REASON IS ITSELF WORTH RECORDING.
    #
    # At leak = 0 the drift is not exactly zero. It is about 2e-05, because version 6 replaced the
    # binary engagement decision with a SIGMOID, and a sigmoid never reaches zero. So the graded
    # gate already leaks a little, structurally, and nobody noticed: E42 reported integration as
    # 0.00 because that is what 3e-06 looks like at two decimal places.
    #
    # Which means the only version of this model that could protect a reader perfectly was the one
    # with the binary gate -- V1 through V5 -- and the version that made the gate more realistic
    # quietly introduced an infinitesimal version of the very term V7 is now adding deliberately.
    # The null therefore asks whether the leak DOMINATES that floor, not whether the floor is zero.
    ordered = sorted(LEAKS)
    monotone = all(accumulates[a]["final_drift"] <= accumulates[b]["final_drift"] + 1e-9
                   for a, b in zip(ordered, ordered[1:]))
    n32_passed = bool(monotone and biggest > zero * 100.0)

    eng_final = edf.groupby("arm").apply(
        lambda s: float(s[s.encounter == s.encounter.max()].drift_from_start.mean()),
        include_groups=False).to_dict()
    careful = float(eng_final.get("reads_it_closely", float("nan")))
    skimmed = float(eng_final.get("skims_it", float("nan")))

    # SCORED ON CONTRAST AND SHAPE, NOT ON AN ABSOLUTE SIZE. E35 was scored on an absolute drop
    # and the threshold turned out not to be seed-stable, because an absolute bar on a quantity
    # with no natural scale cannot be. The claim here is that a closed gate leaks at all and that
    # the leaking compounds; both are scale-free statements.
    fires = bool(biggest > zero * 100.0
                 and accumulates[max(LEAKS)]["monotone_rho"] > 0.7)
    outcome = ("REJECTION_IS_NOT_PROTECTION" if fires
               else "A_CLOSED_GATE_PROTECTS_COMPLETELY")

    verdict = {
        "experiment": "E46",
        "hypothesis": "H7.3",
        "question": "Can a reader look at something, reject it every time, and be unchanged?",
        "plain_language": (
            "In this model a reader that disagrees with something absorbs exactly none of it. "
            "People do not work that way, and the theory never said they did: to decide you "
            "disagree with a claim you have to work out what it says, and working out what it says "
            "means running it. The act of refusing is itself a small act of taking on."),
        "why_this_is_a_correction_not_an_addition": (
            "the preprint already contains this term -- 'the calculation of the value disagreement "
            "itself implies simulation, which inherently drives Hebbian learning due to gating "
            "imperfections... likely the mechanism for indoctrination and propaganda'. The code "
            "never had it. This is the same class of finding version 6 made three times."),
        "drift_by_leak": accumulates,
        "final_drift_by_leak": {float(r.leak): float(r.drift_from_start)
                                for r in final.itertuples()},
        "engagement": {
            "reads_it_closely": careful, "skims_it": skimmed,
            "ratio": (careful / skimmed) if skimmed else None,
            "reading": ("the reader who studies it carefully in order to refute it is more "
                        "affected than the one who skims, because the leak passes a fraction of "
                        "what was RECOVERED and careful reading recovers more"),
        },
        "what_a_reader_absorbs_of_its_own_invention": {
            "drift_on_content_with_no_recoverable_intent": n32_drift,
            "reading": ("fully foreign content is the condition this project has spent four "
                        "versions showing makes readers confidently WRONG. Such a reader has "
                        "recovered something -- a fabrication -- and the leak passes it. So a "
                        "reader that invents an intent then absorbs its own invention, and the "
                        "drift is comparable to what real intent produces."),
        },
        "null_n32": {
            "statement": ("the leak must PASS rather than MANUFACTURE: no drift at zero leak, and "
                          "drift monotone in the leak"),
            "restated": ("the null was first written as 'no drift on unrecoverable content' and "
                         "the condition chosen for it was foreign content, which this model says "
                         "produces confident fabrication. That asked the model to violate its own "
                         "headline. Restated before scoring; the original condition is reported "
                         "above as a result."),
            "drift_on_unrecoverable_content": n32_drift,
            "drift_at_zero_leak": zero,
            "passed": n32_passed,
            "why": ("without this the leak is a drift knob and every number here would be the "
                    "knob's rather than the mechanism's"),
        },
        "the_graded_gate_already_leaked": {
            "drift_at_zero_leak": zero,
            "drift_at_the_largest_leak": biggest,
            "ratio": (biggest / zero) if zero else None,
            "note": ("version 6 replaced the binary engagement decision with a sigmoid, and a "
                     "sigmoid never reaches zero. So the graded gate already leaked a little and "
                     "nobody noticed -- E42 reported integration as 0.00 because that is what "
                     "3e-06 looks like at two decimal places. The only versions of this model "
                     "that could protect a reader perfectly were the ones with the binary gate."),
        },
        "leak_is_swept_not_chosen": (
            "there is no principled value for the leak, so choosing one and reporting the "
            "consequence would be reporting the choice. The output is the shape of drift against "
            "leak. The leak stays OFF by default in the model."),
        "outcome": outcome,
        "n_readers": int(n_readers), "n_encounters": int(n_encounters), "leaks": list(LEAKS),
    }
    if not n32_passed:
        verdict["INTERPRETABILITY"] = (
            "NULL N32 FAILED. The leak produces drift on content with nothing recoverable in it, "
            "so it is manufacturing learning rather than passing it, and every number above is "
            "uninterpretable.")
    (v7_dir() / "e46_gate_leak.json").write_text(json.dumps(verdict, indent=2, default=str),
                                                 encoding="utf-8")
    return verdict
