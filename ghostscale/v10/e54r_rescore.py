"""E54-R — E54 rescored on error rather than movement.

Not a new experiment. E54 scored DRIFT: how far a reader's belief moved from where it started. That
measure counts being carried TOWARD the truth exactly the same as being misled, which this project
separates everywhere else and did not separate there.

If the adversarial stance selectively blocks WRONG uptake, an error-based measure can separate arms
that drift did not. If it does not, E54's verdict stands on a second measure and is that much
harder to argue with.

THE ORIGINAL SCORING IS RETAINED AND REPORTED BESIDE THIS ONE, AND THE ORIGINAL DECIDES. That is
the standing rule for every restated criterion in this project, and it exists precisely so that a
rescore cannot quietly become a rescue.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import metrics
from .. import v6_model as V6
from ..config import Config
from ..v5_model import make_v5_observer
from ..v6 import harness as H
from ..v9.e53_e54 import LABELS, STANCES, _absorb, _dismissed, _paired_diff
from . import SEED_OFFSET, v10_dir
from .. import constants as K


def run(cfg: Config, n_readers: int = 24, n_encounters: int = 16, n_timesteps: int = 16,
        workers: int = 1) -> dict:
    cfg = cfg.copy()
    cfg.set("inference.exact", True)
    world, cfg_b, cfg_r, n_mu, n_sub, ng = H.build_world_and_config(cfg)
    kappa = float(world.cfg.signal_model.kappa)
    k_gain = float(world.cfg.get("v6.gate.k_gain", 8.0))
    theta_0 = float(world.cfg.get("v6.gate.theta_0", 0.35))
    lam, leak = 0.25, 0.10
    values_map = V6.build_values_map(ng, n_values=2)

    rows = []
    for stance in STANCES:
        for r in range(int(n_readers)):
            carried = np.full(ng, 1.0 / ng)
            start = carried.copy()
            signed = []
            for e in range(int(n_encounters)):
                rng = np.random.default_rng(SEED_OFFSET + 96_000 + r * 131 + e)
                g = int(rng.integers(ng))
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 1.0, n_timesteps, rng)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 8, n_sub, n_mu, ng, kappa)
                before = carried.copy()
                carried, _, _ = _absorb(enc, stance, carried, values_map,
                                        kappa, lam, theta_0, k_gain, leak, ng)
                # SIGNED uptake: positive means this encounter carried the reader toward the
                # truth about who made it, negative means away. Drift cannot tell the two apart.
                signed.append(float(metrics.error_reduction(carried, before, int(enc.true_goal))))
            rows.append({"stance": stance, "reader": r,
                         "drift": float(metrics.kl_divergence(carried, start)),
                         "signed_uptake": float(np.mean(signed))})

    sdf = pd.DataFrame(rows)

    label_rows = []
    for label in LABELS:
        for r in range(int(n_readers)):
            carried = np.full(ng, 1.0 / ng)
            start = carried.copy()
            mis, gen = [], []
            for e in range(int(n_encounters)):
                rng = np.random.default_rng(SEED_OFFSET + 97_000 + r * 137 + e)
                g = int(rng.integers(ng))
                misleading = (e % 2 == 0)
                creator, art, env = H.make_artifact_and_env(
                    world, cfg_r, g, 2, 0.0 if misleading else 1.0, n_timesteps, rng,
                    provenance=K.GHOST if misleading else K.CREATOR,
                    declared_signal=K.SIG_CREATOR, signing_rate=1.0)
                enc = H.run_encounter(world, cfg_r, art, env,
                                      make_v5_observer(world, rng), creator, rng,
                                      n_timesteps, 8, n_sub, n_mu, ng, kappa)
                if misleading and label == "do_not_read":
                    enc, stance = _dismissed(enc), "sympathetic"
                elif misleading and label == "read_differently":
                    stance = "adversarial"
                else:
                    stance = "sympathetic"
                before = carried.copy()
                carried, _, _ = _absorb(enc, stance, carried, values_map,
                                        kappa, lam, theta_0, k_gain, leak, ng)
                d = float(metrics.error_reduction(carried, before, int(enc.true_goal)))
                (mis if misleading else gen).append(d)
            label_rows.append({
                "label": label, "reader": r,
                "drift": float(metrics.kl_divergence(carried, start)),
                "harm_from_misleading": -float(np.mean(mis)) if mis else float("nan"),
                "gain_from_genuine": float(np.mean(gen)) if gen else float("nan")})

    ldf = pd.DataFrame(label_rows)
    out = v10_dir("e54r_rescore")
    sdf.to_csv(out / "e54r_stances.csv", index=False)
    ldf.to_csv(out / "e54r_labels.csv", index=False)

    st = sdf.groupby("stance").mean(numeric_only=True).reset_index()
    lb = ldf.groupby("label").mean(numeric_only=True).reset_index()

    d_signed = _paired_diff(sdf, "signed_uptake", "stance", "adversarial", "sympathetic")
    d_drift = _paired_diff(sdf, "drift", "stance", "sympathetic", "adversarial")
    l_harm = _paired_diff(ldf, "harm_from_misleading", "label", "none", "read_differently")
    l_harm_dnr = _paired_diff(ldf, "harm_from_misleading", "label", "none", "do_not_read")

    verdict = {
        "experiment": "E54-R",
        "what_this_is": ("a rescoring of E54 on signed error rather than unsigned movement. Not a "
                         "new experiment, and not a second chance -- the original scoring is "
                         "reported beside it and the original decides."),
        "stances": st.to_dict(orient="records"),
        "labels": lb.to_dict(orient="records"),
        "stance_on_error": {
            "adversarial_minus_sympathetic_signed_uptake": d_signed,
            "outcome": ("THE_STANCE_SEPARATES_ON_ERROR_TOO" if d_signed["separated_from_zero"]
                        else "THE_STANCE_DOES_NOT_SEPARATE_ON_ERROR"),
        },
        "stance_on_drift_original": {
            "sympathetic_minus_adversarial_drift": d_drift,
            "note": "the pre-registered measure; reproduced here for comparability",
        },
        "labels_on_error": {
            "no_label_minus_read_differently_harm": l_harm,
            "no_label_minus_do_not_read_harm": l_harm_dnr,
            "outcome": ("A_LABEL_REDUCES_HARM_EVEN_THOUGH_IT_DID_NOT_REDUCE_DRIFT"
                        if (l_harm["separated_from_zero"] or l_harm_dnr["separated_from_zero"])
                        else "NO_LABEL_SEPARATES_ON_HARM_EITHER"),
        },
        "n_readers": int(n_readers), "n_encounters": int(n_encounters),
    }
    (v10_dir() / "e54r_rescore.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
