"""V5 pre-registration for C2, C3 and C4 — E31, E32, E33.

A SEPARATE FILE FROM ``prereg_v5.py``, and for V4.5's reason rather than for tidiness: that
file is hash-locked and its experiments (N21, E30) have reported. A pre-registration that
acquires new content after its experiments run is not a pre-registration. V4.5 learned this
with E20 and wrote it its own file; C2/C3/C4 get the same treatment.

Everything here is fixed before any of E31, E32 or E33 runs, including E32's matching curve,
which is a free parameter that would otherwise decide its own result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config

# --------------------------------------------------------------------------- #
# E31 — two gates, provenance as evidence about depth.
# --------------------------------------------------------------------------- #
# C2's claim in one line: the generative crash is a CORRECTLY low mu, the trust exploit is a
# FALSELY high mu, and they are one mechanism with opposite signs. That makes the decisive
# quantity the RECOVERED DEPTH, not the content and not the label — so the pre-registered
# criterion is that update magnitude tracks recovered mu REGARDLESS OF WHICH CHANNEL MOVED IT.
E31_UPDATE_TRACKS_MU_RHO = 0.70    # Spearman(recovered mu, prior drift) across all cells
# The two headline cells must separate: honest ghost (crash) and dishonest ghost (exploit)
# read the SAME content and must differ in recovered mu by at least this much.
E31_EXPLOIT_MU_GAP = 0.15
# Fabrication: confident and wrong. The exploit cell must exceed the crash cell.
E31_FABRICATION_GAP = 0.05
# The mu/theta dissociation, inherited from E29's decisive contrast in its corrected form.
E31_UPDATE_NONE = 0.05
E31_RESOLVED_ENTROPY = 0.50

# --------------------------------------------------------------------------- #
# E33 — the latent goal.
# --------------------------------------------------------------------------- #
E33_DIVERGENCE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
E33_ARMS = ("human_unaware", "deceptive", "generative")
# The observer beats the creator's self-model if it recovers the LATENT goal more often than
# the declaration names it. At divergence p the declaration is right (1 - p) of the time, so
# this is a comparison against a moving baseline rather than a fixed threshold.
E33_BEATS_SELF_MODEL_MARGIN = 0.05
# C4's categorical claim, in its corrected form (V5 decision 18): what a generative system
# cannot produce is not a divergence, it is a divergence that LEAVES A TRACE in the artifact.
# So the human arm must be separable from BOTH other arms on content alone, and the deceptive
# and generative arms must not be separable from each other on content alone.
E33_TRACE_SEPARATION = 0.10

# --------------------------------------------------------------------------- #
# E32 — the omega/d equivalence test.
# --------------------------------------------------------------------------- #
E32_OMEGA_GRID = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70)
# Two arms count as indistinguishable on a measure if their mean absolute gap is under this
# fraction of the measure's own range across the grid. A fraction rather than a bare number,
# so the criterion means the same thing for entropies in nats and for accuracies in [0, 1].
E32_INDISTINGUISHABLE_FRAC = 0.10


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def e31_verdict(update_tracks_mu_rho: float, exploit_mu_gap: float,
                fabrication_gap: float, dissociates: bool) -> dict:
    """C2's outcome. Every branch written before the run.

    * ONE_MECHANISM_OPPOSITE_SIGNS — update tracks recovered depth whatever moved it, AND the
      dishonest label inflates depth on content that has none. C2's claim, supported.
    * DEPTH_IS_NOT_THE_COMMON_PATH — the exploit works but update does not track depth, so
      provenance is doing something the depth estimate does not carry and the collapse from
      three channels to two has not been earned.
    * NO_EXPLOIT — the label does not inflate the depth estimate at all, so provenance is not
      evidence about depth in this model and C2's re-architecture is wrong.
    * NEITHER — both fail.
    """
    tracks = bool(update_tracks_mu_rho >= E31_UPDATE_TRACKS_MU_RHO)
    exploit = bool(exploit_mu_gap >= E31_EXPLOIT_MU_GAP)
    if tracks and exploit:
        v = "ONE_MECHANISM_OPPOSITE_SIGNS"
    elif exploit:
        v = "DEPTH_IS_NOT_THE_COMMON_PATH"
    elif tracks:
        v = "NO_EXPLOIT"
    else:
        v = "NEITHER"
    return {
        "verdict": v,
        "update_tracks_recovered_depth": tracks,
        "update_tracks_mu_rho": float(update_tracks_mu_rho),
        "update_tracks_mu_rho_required": E31_UPDATE_TRACKS_MU_RHO,
        "dishonest_label_inflates_depth": exploit,
        "exploit_mu_gap": float(exploit_mu_gap),
        "exploit_mu_gap_required": E31_EXPLOIT_MU_GAP,
        "fabrication_gap": float(fabrication_gap),
        "fabrication_gap_required": E31_FABRICATION_GAP,
        "mu_theta_dissociate_on_behaviour": bool(dissociates),
    }


def e32_verdict(per_measure: dict) -> dict:
    """C3's outcome, with the divergence branch named rather than treated as a failure."""
    same = [m for m, d in per_measure.items() if d["indistinguishable"]]
    if len(same) == len(per_measure):
        v = "ONE_VARIABLE"
    elif not same:
        v = "TWO_DIMENSIONS"
    else:
        v = "PARTLY_DISTINGUISHABLE"
    return {
        "verdict": v,
        "indistinguishable_measures": same,
        "reading": {
            "ONE_VARIABLE": ("foreign content and an unskilled reader are behaviourally the "
                             "same at matched overlap; the model should carry one variable "
                             "and say so."),
            "TWO_DIMENSIONS": ("they differ on every measure. The second dimension is whether "
                               "the failure is DETECTABLE TO THE OBSERVER: a mis-aimed "
                               "template fails silently and confidently, an out-of-range one "
                               "fails loudly and stays unresolved."),
            "PARTLY_DISTINGUISHABLE": ("they agree on some measures and not others; the "
                                       "measures on which they differ name what the second "
                                       "dimension is about."),
        }[v],
    }


def e33_verdict(latent_recovery, declared_recovery, self_model_accuracy,
                depth_by_divergence, arm_separation: dict) -> dict:
    """C4's outcome. The categorical claim is scored on the TRACE, not on the divergence.

    V5 decision 18: a generative system cannot produce a divergence that MARKS THE WORK.
    Deception produces declared != latent with no trace; unconsciousness produces one with a
    trace. So the human arm must separate from both others on content alone, and the deceptive
    and generative arms must not separate from each other — that pairing is what shows the
    measure is reading the trace rather than the divergence.
    """
    lat = np.asarray(latent_recovery, dtype=float)
    dec = np.asarray(declared_recovery, dtype=float)
    self_acc = np.asarray(self_model_accuracy, dtype=float)
    beats = bool(np.any(lat > self_acc + E33_BEATS_SELF_MODEL_MARGIN))

    rho_depth = spearman(range(len(depth_by_divergence)), depth_by_divergence)
    human_vs_dec = float(arm_separation.get("human_vs_deceptive", 0.0))
    human_vs_gen = float(arm_separation.get("human_vs_generative", 0.0))
    dec_vs_gen = float(arm_separation.get("deceptive_vs_generative", 0.0))
    trace = bool(human_vs_dec >= E33_TRACE_SEPARATION
                 and human_vs_gen >= E33_TRACE_SEPARATION
                 and dec_vs_gen < E33_TRACE_SEPARATION)

    if trace and beats:
        v = "TRACE_IS_REAL_AND_READABLE"
    elif trace:
        v = "TRACE_IS_REAL_BUT_UNREAD"
    elif beats:
        v = "READS_THE_LATENT_WITHOUT_A_TRACE"
    else:
        v = "NO_CATEGORICAL_DIFFERENCE"
    return {
        "verdict": v,
        "observer_beats_the_self_model": beats,
        "latent_recovery": lat.tolist(),
        "declared_recovery": dec.tolist(),
        "self_model_accuracy": self_acc.tolist(),
        "divergence_raises_recovered_depth": bool(rho_depth > 0),
        "depth_vs_divergence_rho": float(rho_depth),
        "trace_is_separable": trace,
        "arm_separation": arm_separation,
        "trace_separation_required": E33_TRACE_SEPARATION,
        "reading": ("the categorical claim is about the TRACE, not the divergence. Deception "
                    "and unconsciousness both produce declared != latent; only unconsciousness "
                    "is supposed to mark the work. So the human arm must separate from both "
                    "others on content alone WHILE the deceptive and generative arms do not "
                    "separate from each other — if all three separate, the measure is reading "
                    "something else."),
    }


# --------------------------------------------------------------------------- #
# The locked payload.
# --------------------------------------------------------------------------- #
def build_preregistration_c234(cfg: Config, matching_curve=None) -> dict:
    payload = {
        "version": "V5-C234",
        "scope": "C2/E31, C3/E32, C4/E33. C1 is covered by v5_preregistration.json.",
        "E31": {
            "claim": ("C2: the generative crash is a CORRECTLY low mu and the trust exploit "
                      "is a FALSELY high mu — one mechanism, opposite signs."),
            "design": ("2 content (human / foreign) x 3 label (unsigned / SIG_CREATOR / "
                       "SIG_GHOST) x 2 theta (open / closed), over 4 sequential encounters "
                       "with the gated update carried between them. Sequential for E29's "
                       "reason: theta gates the update, so in a single-artifact design it "
                       "changes nothing observable and would appear to dissociate on the "
                       "strength of its own definition."),
            "provenance_path": ("a mu prior CONDITIONED ON THE OBSERVED LABEL, which is where "
                                "decision 10 landed after the factor merge; see v5_model."),
            "primary": "Spearman(recovered mu, prior drift) across all cells",
            "thresholds": {"update_tracks_mu_rho": E31_UPDATE_TRACKS_MU_RHO,
                           "exploit_mu_gap": E31_EXPLOIT_MU_GAP,
                           "fabrication_gap": E31_FABRICATION_GAP},
            "outcomes": ["ONE_MECHANISM_OPPOSITE_SIGNS", "DEPTH_IS_NOT_THE_COMMON_PATH",
                         "NO_EXPLOIT", "NEITHER"],
        },
        "E32": {
            "claim": "C3: omega and observer inexpertise d may be one channel.",
            "omega_grid": list(E32_OMEGA_GRID),
            "matching": ("MI(features; goal) under the observer's OWN likelihood on the "
                         "content it sees, in nats; d solved by bisection. COMPUTED AND "
                         "LOCKED BEFORE ANY ROLLOUT, because it is a free parameter that "
                         "would otherwise decide the result."),
            "matching_curve": matching_curve,
            "indistinguishable_frac": E32_INDISTINGUISHABLE_FRAC,
            "prediction": ("they DIVERGE. A mis-aimed template fails silently — confident, "
                           "wrong, idiosyncratic. An out-of-range one fails loudly — "
                           "uncertain and sustained. The second dimension is whether the "
                           "failure is detectable to the observer."),
            "outcomes": ["ONE_VARIABLE", "TWO_DIMENSIONS", "PARTLY_DISTINGUISHABLE"],
        },
        "E33": {
            "claim": ("C4: a large part of what makes something art is a goal that shapes the "
                      "creator's behaviour while the creator has no access to it, and an "
                      "observer can sometimes extract it anyway."),
            "arms": list(E33_ARMS),
            "divergence_grid": list(E33_DIVERGENCE_GRID),
            "construction": {
                "goal_latent": "drives the creator's policy; not represented in its own model",
                "goal_declared": ("what the creator believes and would report; delivered as a "
                                  "fourth observation modality with its own precision, "
                                  "parallel to the provenance signal's kappa"),
                "divergence": ("UNIFORM over the other goals. The assumption-free choice; a "
                               "systematic or flattering variant is a separate arm and is "
                               "not run here."),
                "productive_conflict": (
                    "when latent != declared the creator SUPPRESSES the declared goal's "
                    "features while pursuing the latent one, so the artifact carries a "
                    "signature that is neither. The trace is present or absent, and its "
                    "STRENGTH does not scale with the divergence RATE — otherwise 'divergence "
                    "increases depth' would be true by construction and E33 would prove "
                    "nothing."),
            },
            "categorical_claim": (
                "V5 decision 18, 'no conscious' rather than 'no unconscious': a generative "
                "system emits a plausible self-report produced by the same process as its "
                "output rather than by introspection, so its declaration is CONFABULATED and "
                "uncorrelated with what drove emission. What it cannot produce is not a "
                "divergence — it is a divergence that leaves a TRACE. The deceptive arm is "
                "the control that stops the generative arm from being true by definition."),
            "thresholds": {"beats_self_model_margin": E33_BEATS_SELF_MODEL_MARGIN,
                           "trace_separation": E33_TRACE_SEPARATION},
            "outcomes": ["TRACE_IS_REAL_AND_READABLE", "TRACE_IS_REAL_BUT_UNREAD",
                         "READS_THE_LATENT_WITHOUT_A_TRACE", "NO_CATEGORICAL_DIFFERENCE"],
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "content_hash"},
                      sort_keys=True, separators=(",", ":"), default=float)


def write_preregistration_c234(cfg: Config, path: Path, matching_curve=None,
                               force: bool = False) -> dict:
    payload = build_preregistration_c234(cfg, matching_curve=matching_curve)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n  now:     {payload['content_hash']}\n"
                "V5's C2/C3/C4 criteria were pre-registered and must not change after the "
                "fact. Investigate, or pass force=True and record it as a deviation.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def ensure_preregistration_c234(cfg: Config, path: Path) -> dict:
    """Write the C2/C3/C4 pre-registration if it does not exist, then assert it is intact.

    THE ONE ENTRY POINT ALL THREE EXPERIMENTS USE, and it exists because they did not have
    one. E32 wrote the payload WITH its matching curve; E31 wrote it WITHOUT, because E31 has
    no matching curve of its own to pass. Same criteria, different payload, different hash —
    so whichever experiment ran second refused to start, reporting that V5's criteria had been
    changed after the fact when nothing of the sort had happened.

    The lock behaved exactly as designed; the call sites were inconsistent. Building the curve
    here means every experiment produces the identical payload, and the hash means what it is
    supposed to mean: that the CRITERIA have not moved, not that the caller happened to
    assemble them the same way.
    """
    from .experiments.e32_omega_d import build_matching_curve

    path = Path(path)
    if not path.exists():
        write_preregistration_c234(cfg, path,
                                   matching_curve=build_matching_curve(cfg, E32_OMEGA_GRID))
    return assert_prereg_locked_c234(path)


def assert_prereg_locked_c234(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} not found. No V5 C2/C3/C4 experiment may run before its "
                           "criteria are pre-registered.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written ({stated} != {recomputed}). "
            "The pre-registered criteria are not trustworthy; the programme will not run.")
    return payload
