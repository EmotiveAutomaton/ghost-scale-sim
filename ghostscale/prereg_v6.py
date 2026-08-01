"""V6 pre-registration — criteria as executable code, content-hash locked before any run.

Same machinery as V3, V4, V4.5, V5, and the three audit passes, and for the same reason: the
written criterion and the applied criterion are ONE OBJECT, so a criterion cannot drift from the
thing the experiment actually computes.

WHAT MAKES THIS ONE DIFFERENT. V6's hypotheses come in two kinds and they are locked separately
below, because they carry different weight:

  * PREDICTIONS about the world, in the ordinary sense (H6.1, H6.3-H6.7, H6.9-H6.12).
  * A COMPARISON between two mechanisms that both claim to explain a result already in the
    record (H6.2). That one cannot "fail" in the usual way -- one of the two mechanisms wins, or
    they turn out behaviourally identical, and all three outcomes are informative. Its outcome
    branches are named here so that the answer cannot be reframed after it arrives.

Every threshold below was fixed before a single V6 cell ran.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config

# --------------------------------------------------------------------------- #
# Shared statistics. Tie-aware ranks, because the diagnostics pass found the naive
# argsort-of-argsort version scoring a CONSTANT estimator at rho = 1.00.
# --------------------------------------------------------------------------- #
def _tied_ranks(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _tied_ranks(x), _tied_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def percentile_interval(samples, lo: float = 2.5, hi: float = 97.5) -> tuple:
    s = np.asarray([x for x in samples if np.isfinite(x)], dtype=float)
    if s.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(s, lo)), float(np.percentile(s, hi)))


BOOTSTRAP_DRAWS = 4000

# =========================================================================== #
# H6.1 — depletion produces carryover damage with no contamination.
# =========================================================================== #
# The PROBE is the criterion, not the exposed condition. A depletion term that only lowers
# engagement on the content that caused it is a knob doing what it was pointed at; the claim is
# that it carries to work the reader has never seen.
H61_PROBE_DROP = 0.10           # engagement on the fixed human probe must fall by this much
H61_MONOTONE_RHO = 0.70         # and fall monotonically with exposure count

# N22, the null that comes first: on a fully resolvable corpus, the reserve must not move.
N22_RESERVE_TOL = 0.02


def h61_verdict(probe_engagement_by_exposure, reserve_by_exposure,
                null_reserve_start: float, null_reserve_end: float) -> dict:
    xs = list(range(len(probe_engagement_by_exposure)))
    ys = list(probe_engagement_by_exposure)
    drop = float(ys[0] - ys[-1]) if len(ys) >= 2 else float("nan")
    rho = spearman(xs, ys)
    null_ok = abs(float(null_reserve_end) - float(null_reserve_start)) <= N22_RESERVE_TOL
    carried = bool(np.isfinite(drop) and drop >= H61_PROBE_DROP
                   and np.isfinite(rho) and rho <= -H61_MONOTONE_RHO)
    if not null_ok:
        outcome = "NULL_N22_FAILED_DEPLETION_UNINTERPRETABLE"
    elif carried:
        outcome = "DEPLETION_CARRIES_TO_UNSEEN_HUMAN_WORK"
    else:
        outcome = "DEPLETION_DOES_NOT_CARRY"
    return {"probe_drop": drop, "probe_monotone_rho": rho,
            "null_n22_passed": bool(null_ok),
            "reserve_first": float(reserve_by_exposure[0]) if len(reserve_by_exposure) else float("nan"),
            "reserve_last": float(reserve_by_exposure[-1]) if len(reserve_by_exposure) else float("nan"),
            "outcome": outcome}


# =========================================================================== #
# H6.2 — the two trust mechanisms, compared.
# =========================================================================== #
# The discriminating cell: HONEST label on machine content, high trust in the SOURCE. Under the
# channel race the reader's provenance belief is correct and there is nothing to exploit. Under
# the coupled gate the threshold is suppressed and the reader integrates anyway.
H62_INTEGRATION_GAP = 0.20      # coupled minus uncoupled, on the honest-label cell
H62_OUTCOMES = ("COUPLING_PREDICTS_AN_EXPLOIT_THE_RACE_CANNOT",
                "MECHANISMS_BEHAVIOURALLY_IDENTICAL",
                "RACE_ALONE_ACCOUNTS_FOR_THE_EXPLOIT")


def h62_verdict(integration_coupled, integration_uncoupled, kappa_grid) -> dict:
    a = np.asarray(integration_coupled, dtype=float)
    b = np.asarray(integration_uncoupled, dtype=float)
    gap = float(np.nanmax(a - b)) if a.size else float("nan")
    mean_abs = float(np.nanmean(np.abs(a - b))) if a.size else float("nan")
    if np.isfinite(gap) and gap >= H62_INTEGRATION_GAP:
        outcome = H62_OUTCOMES[0]
    elif np.isfinite(mean_abs) and mean_abs < 0.02:
        outcome = H62_OUTCOMES[1]
    else:
        outcome = H62_OUTCOMES[2]
    return {"max_integration_gap": gap, "mean_abs_difference": mean_abs,
            "kappa_grid": [float(k) for k in kappa_grid], "outcome": outcome}


# =========================================================================== #
# H6.3 / H6.4 — goal gates process; depth moves process uptake.
# =========================================================================== #
H63_ORDERING_GAP = 0.15         # process recovery, goal-correct minus goal-incorrect
H64_DEPTH_CONTRAST = 0.0        # deepest minus shallowest, interval must EXCLUDE this


def h63_verdict(process_when_goal_right, process_when_goal_wrong) -> dict:
    a = float(np.nanmean(process_when_goal_right)) if len(process_when_goal_right) else float("nan")
    b = float(np.nanmean(process_when_goal_wrong)) if len(process_when_goal_wrong) else float("nan")
    gap = a - b
    return {"process_recovery_goal_right": a, "process_recovery_goal_wrong": b,
            "ordering_gap": gap,
            "outcome": ("GOAL_GATES_PROCESS" if np.isfinite(gap) and gap >= H63_ORDERING_GAP
                        else "PROCESS_IS_INDEPENDENT_OF_GOAL")}


def h64_verdict(contrast: float, interval: tuple) -> dict:
    lo, hi = float(interval[0]), float(interval[1])
    excludes = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > H64_DEPTH_CONTRAST))
    return {"deepest_minus_shallowest": float(contrast), "interval": [lo, hi],
            "excludes_zero_positive": excludes,
            "outcome": ("DEPTH_MOVES_PROCESS_UPTAKE" if excludes
                        else "DEPTH_MOVES_NOTHING_ON_PROCESS_EITHER")}


# =========================================================================== #
# H6.5 — the wall is a distinct failure.
# =========================================================================== #
# LEGIBLE AND EMPTY: low uncertainty about what is on the surface, no recovery of intent.
# Against foreign content's high uncertainty and sustained search.
H65_SEPARATION = 0.30           # L1 distance between the two (entropy, engagement) signatures


def h65_verdict(noninvertible: dict, foreign: dict) -> dict:
    d_ent = abs(float(noninvertible["final_entropy"]) - float(foreign["final_entropy"]))
    d_eng = abs(float(noninvertible["engaged_fraction"]) - float(foreign["engaged_fraction"]))
    sep = d_ent + d_eng
    legible_and_empty = bool(float(noninvertible["final_entropy"]) < float(foreign["final_entropy"])
                             and float(noninvertible["goal_accuracy"]) <= 0.40)
    return {"signature_separation": float(sep), "entropy_gap": float(d_ent),
            "engagement_gap": float(d_eng), "legible_and_empty": legible_and_empty,
            "outcome": ("THE_WALL_IS_A_DISTINCT_FAILURE" if sep >= H65_SEPARATION
                        else "THE_WALL_IS_A_VOCABULARY_DEFICIT")}


# =========================================================================== #
# H6.6 — expertise substitutes rather than stacks.
# =========================================================================== #
H66_CROSSOVER = 0.10            # the machine-matched reader must LOSE this much on human work


def h66_verdict(cells: dict) -> dict:
    """``cells[(reader, content)] = accuracy``, reader in {human,machine}, content likewise."""
    hh = float(cells[("human", "human")])
    hm = float(cells[("human", "machine")])
    mh = float(cells[("machine", "human")])
    mm = float(cells[("machine", "machine")])
    gain = mm - hm                 # what the machine-matched reader gains on machine content
    loss = hh - mh                 # what it gives up on human content
    substitutes = bool(gain >= H66_CROSSOVER and loss >= H66_CROSSOVER)
    return {"human_reader_human_content": hh, "human_reader_machine_content": hm,
            "machine_reader_human_content": mh, "machine_reader_machine_content": mm,
            "gain_on_machine": gain, "loss_on_human": loss,
            "outcome": ("EXPERTISE_SUBSTITUTES" if substitutes
                        else ("EXPERTISE_STACKS" if gain >= H66_CROSSOVER
                              else "NO_MACHINE_ADVANTAGE"))}


# =========================================================================== #
# H6.7 — the tool hypothesis produces a third signature.
# =========================================================================== #
H67_RESOLVED = 0.50             # final entropy at or below this
H67_DISENGAGED = 0.50           # engaged fraction at or below this
H67_NO_INVENTION = 0.10         # fabrication index at or below this


def h67_verdict(tool_cell: dict, crash_cell: dict) -> dict:
    resolved = float(tool_cell["final_entropy"]) <= H67_RESOLVED
    disengaged = float(tool_cell["engaged_fraction"]) <= H67_DISENGAGED
    quiet = float(tool_cell["fabrication_index"]) <= H67_NO_INVENTION
    distinct = bool(resolved and disengaged and quiet)
    return {"final_entropy": float(tool_cell["final_entropy"]),
            "engaged_fraction": float(tool_cell["engaged_fraction"]),
            "fabrication_index": float(tool_cell["fabrication_index"]),
            "crash_final_entropy": float(crash_cell["final_entropy"]),
            "crash_engaged_fraction": float(crash_cell["engaged_fraction"]),
            "resolved": resolved, "disengaged": disengaged, "no_invention": quiet,
            "outcome": ("TOOL_HYPOTHESIS_RELAXES_THE_READER" if distinct
                        else "TOOL_HYPOTHESIS_REDIRECTS_RATHER_THAN_RELAXES")}


# =========================================================================== #
# H6.8 / H6.9 — cue combination, and the decoupling.
# =========================================================================== #
# ADDITIVE IS THE PRE-REGISTERED FORM. The two rules disagree where information gain is near
# zero: additive lets a cue drive engagement on content that offers nothing, multiplicative
# cannot, because it scales the gain. That corner IS the test.
H68_PREREGISTERED_FORM = "additive"
H69_OVERENGAGEMENT = 0.10       # engagement above the honest-depth baseline
H69_NO_UPTAKE = 0.0             # error reduction at or below this


def h68_verdict(additive_corner: float, multiplicative_corner: float,
                baseline_corner: float) -> dict:
    a_lift = float(additive_corner) - float(baseline_corner)
    m_lift = float(multiplicative_corner) - float(baseline_corner)
    return {"additive_lift_at_empty_corner": a_lift,
            "multiplicative_lift_at_empty_corner": m_lift,
            "preregistered": H68_PREREGISTERED_FORM,
            "outcome": ("ADDITIVE_FITS" if a_lift > m_lift else "MULTIPLICATIVE_FITS")}


def h69_verdict(engagement_decoupled: float, engagement_honest: float,
                error_reduction_decoupled: float) -> dict:
    over = float(engagement_decoupled) - float(engagement_honest)
    empty = float(error_reduction_decoupled) <= H69_NO_UPTAKE
    fired = bool(over >= H69_OVERENGAGEMENT and empty)
    return {"engagement_lift": over, "error_reduction": float(error_reduction_decoupled),
            "pays_more_gets_less": fired,
            "outcome": ("A_THIRD_FAILURE_MODE_OVERENGAGEMENT" if fired
                        else "CUE_DID_NOT_BECOME_LOAD_BEARING")}


# =========================================================================== #
# H6.10 — vulnerability is the gate, not the engagement decision.
# =========================================================================== #
H610_ENGAGED = 0.50             # high engagement
H610_INTEGRATION = 0.10         # with near-zero integration


def h610_verdict(cells: list) -> dict:
    """Any cell with high engagement AND a closed gate dissociates the two."""
    hits = [c for c in cells
            if float(c["engaged_fraction"]) >= H610_ENGAGED
            and float(c["integration"]) <= H610_INTEGRATION]
    return {"n_cells": len(cells), "n_dissociating": len(hits),
            "dissociating_cells": [c.get("name") for c in hits],
            "outcome": ("ENGAGEMENT_AND_INTEGRATION_DISSOCIATE" if hits
                        else "ENGAGEMENT_IS_INTEGRATION")}


# =========================================================================== #
# H6.11 / H6.12 — graded self-report, and scale invariance.
# =========================================================================== #
H611_DECLINE = 0.10             # declared-goal accuracy must fall this much across the mu range
H611_READER_FLAT = 0.10         # while the reader's latent recovery moves less than this
H612_SHAPE_TOL = 0.25           # window accuracy within this of whole-artifact accuracy


def h611_verdict(declared_by_mu, latent_by_mu) -> dict:
    d = np.asarray(declared_by_mu, dtype=float)
    l = np.asarray(latent_by_mu, dtype=float)
    decline = float(d[0] - d[-1]) if d.size >= 2 else float("nan")
    reader_move = float(np.nanmax(l) - np.nanmin(l)) if l.size else float("nan")
    ok = bool(np.isfinite(decline) and decline >= H611_DECLINE
              and np.isfinite(reader_move) and reader_move <= H611_READER_FLAT)
    return {"declared_decline": decline, "reader_movement": reader_move,
            "outcome": ("AUTOMATICITY_HIDES_THE_GOAL_FROM_ITS_OWN_AUTHOR" if ok
                        else "SELF_REPORT_IS_FLAT_IN_DEPTH")}


def h612_verdict(whole_acc: float, window_acc: float) -> dict:
    gap = abs(float(whole_acc) - float(window_acc))
    return {"whole_accuracy": float(whole_acc), "window_accuracy": float(window_acc),
            "gap": gap,
            "outcome": ("EXTRACTION_IS_SCALE_INVARIANT" if gap <= H612_SHAPE_TOL
                        else "EXTRACTION_IS_TIED_TO_THE_ARTIFACT_BOUNDARY")}


# =========================================================================== #
# The payload.
# =========================================================================== #
def build_preregistration_v6(cfg: Config) -> dict:
    payload = {
        "version": "V6",
        "scope": ("Alignment of the simulation to the Intent Extraction Limit, plus six theory "
                  "extensions and two author corrections. Every addition is independently "
                  "switchable and OFF by default; with all switches off V6 must reproduce V5 "
                  "(null N23)."),
        "what_is_deliberately_not_built": [
            "no creator agent, so Zahavian signalling and the whole reputational-cost security "
            "argument remain UNTESTED IN SIMULATION. Named as an open hole, not discovered later.",
            "no recursive/fractal hierarchy; the scaled version (sub-window recovery) is built "
            "instead",
            "no affective/mirror channel",
            "no creator-side cognitive-surrender dynamics",
        ],
        "terms_of_the_formal_model_now_implemented": [
            "theta_base(E): metabolic reserve, absent from V1-V5 entirely",
            "sigmoid(k * ...): the graded gate, replaced in V1-V5 by a binary decision",
            "kappa -> theta coupling: ABSENT from V1-V5, and a different mechanism from the "
            "channel race the code uses. Built as a SWITCH so the two can be compared (H6.2).",
        ],
        "H6.1": {"probe_drop": H61_PROBE_DROP, "monotone_rho": H61_MONOTONE_RHO,
                 "null_first": "N22 reserve tolerance %.3f on a fully resolvable corpus"
                               % N22_RESERVE_TOL,
                 "criterion_is_the_probe": "engagement on human work the reader has NEVER SEEN"},
        "H6.2": {"integration_gap": H62_INTEGRATION_GAP, "outcomes": list(H62_OUTCOMES),
                 "note": "a comparison, not a prediction: all three outcomes are informative"},
        "H6.3": {"ordering_gap": H63_ORDERING_GAP},
        "H6.4": {"contrast_must_exceed": H64_DEPTH_CONTRAST,
                 "bootstrap_draws": BOOTSTRAP_DRAWS,
                 "note": "E30 measured GOAL uptake, which depth is constructed to hold constant"},
        "H6.5": {"signature_separation": H65_SEPARATION},
        "H6.6": {"crossover": H66_CROSSOVER},
        "H6.7": {"resolved": H67_RESOLVED, "disengaged": H67_DISENGAGED,
                 "no_invention": H67_NO_INVENTION},
        "H6.8": {"preregistered_form": H68_PREREGISTERED_FORM},
        "H6.9": {"overengagement": H69_OVERENGAGEMENT, "no_uptake": H69_NO_UPTAKE},
        "H6.10": {"engaged": H610_ENGAGED, "integration": H610_INTEGRATION,
                  "origin": "an author correction: the top engages deeply with the gate shut"},
        "H6.11": {"decline": H611_DECLINE, "reader_flat": H611_READER_FLAT,
                  "origin": "an author correction: the subconscious holds the PRACTISED goals"},
        "H6.12": {"shape_tolerance": H612_SHAPE_TOL},
        "nulls": {
            "N22": "depletion must not accumulate on a fully resolvable corpus",
            "N23": "with every switch off, V6 reproduces V5",
            "N24": "as k -> infinity the graded gate reproduces the binary decision",
            "N25": "no preference over provenance; every new channel gets zero preference",
            "N26": "the values map is non-injective, or it is the goal renamed",
            "N27": "NO_MAKER must not absorb human work (the EXPLORE failure V4 caught)",
            "N28": "at mu = 1 process recovery is at chance whatever goal recovery is",
            "N29": "cue channels carry no goal information",
            "N30": "non-invertible content stays on the human block",
        },
        "inference": "exact by default; V6 is the first version for which that is true",
        "unchanged": {
            "E8": "stays withheld with its xfail(strict) marker",
            "E27": "the V3 residual stays open",
            "retention": "every original measure retained and reported beside its replacement",
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "content_hash"},
                      sort_keys=True, separators=(",", ":"))


def write_preregistration_v6(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_v6(cfg)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} already exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n"
                f"  now:     {payload['content_hash']}\n"
                "V6's criteria were pre-registered and must not change after the fact.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v6(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V6 experiment may run before its criteria are "
            "pre-registered.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written (hash {stated} != recomputed "
            f"{recomputed}). The pre-registered criteria are not trustworthy; the V6 "
            f"programme will not run.")
    return payload
