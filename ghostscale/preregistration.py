"""Pre-registration for E6b (V2 spec §2, §5, §6).

    "Mandatory pre-registration. Before running, compute and write to
     results/e6b_preregistration.json:  bound(f, k) = KL( (1-f)*C_true + f*onehot(k) || C_true )"

    "The pre-registered bound for E6b, computed and written to disk BEFORE the run."
    -- §6, things that may not change.

This module is what makes that a mechanism rather than an intention. The bound file is
written first and content-hashed; ``assert_prereg_locked`` refuses to let E6b proceed if a
file already exists whose content differs. You cannot quietly recompute the bound after
seeing the result.

-----------------------------------------------------------------------------------------
DECISION D3a — the biased arm sweeps four synthetic draws, stratified by favoured goal.

At the committed V1 seed the un-symmetrized draw favours k=3, which is the RAREST goal in
C_true = [0.40, 0.30, 0.20, 0.10] and therefore yields the LARGEST of the four possible
bounds (0.50 / 0.68 / 0.95 / 1.44 nats at f=0.8). Nothing was rigged, but a reader cannot
distinguish that from rigging, and E6b is the experiment §5 says a skeptical reader checks
first. So the arm runs four draws, one favouring each goal, and the reported claim becomes
"KL scales with the realized lean" rather than "one draw gave a large number".

As directed, the FULL SEED SCAN is written to the pre-registration file -- every seed
examined with its favoured k and lean, not merely the four selected -- so the stratification
is auditable end to end.
-----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config
from .generative_model import build_goal_signatures, build_noise_free_synth, synth_lean
from .metrics import kl_divergence, shannon_entropy

# The population goal distribution E6/E6b recover. Kept in one place; imported by e6b.
POP_GOAL_DIST = np.array([0.40, 0.30, 0.20, 0.10])


def allocate_creator_goals(n_creators: int, dist: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Assign goals to creators by largest-remainder allocation, then shuffle.

    WHY NOT AN I.I.D. DRAW (a V2 deviation from V1's E6, and a fix rather than a change of
    subject). V1 sampled creator goals i.i.d. from the population distribution and took
    ``C_true`` to be the realized empirical distribution. With a finite creator pool that
    empirical vector can contain a ZERO — at 10 creators with p(G3)=0.10 it happens 35% of
    the time, and even at 50 it happens ~0.5% — and ``KL(C_recovered || C_true)`` against a
    zero-support reference diverges. Symptom when this bit: E6b reported KL ~5.1 nats in BOTH
    arms, exceeding its own pre-registered bound, which the prereg file itself names as the
    signature of an accumulation bug rather than a strong effect.

    Largest-remainder allocation makes the realized ``C_true`` the closest achievable
    approximation to the population distribution at that creator count, with full support by
    construction. At the spec's 50 creators it is exactly [20, 15, 10, 5] = the population
    distribution, so the pre-registered bound (defined against that distribution) describes
    the quantity actually being measured. Randomness stays where it belongs — in which
    creator holds which goal, and in every downstream draw.
    """
    dist = np.asarray(dist, dtype=float)
    dist = dist / dist.sum()
    exact = dist * int(n_creators)
    counts = np.floor(exact).astype(int)
    remainder = int(n_creators) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(exact - counts))
        counts[order[:remainder]] += 1
    goals = np.repeat(np.arange(dist.size), counts)
    rng.shuffle(goals)
    return goals


# --------------------------------------------------------------------------- #
# The bound.
# --------------------------------------------------------------------------- #
def contamination_bound(C_true: np.ndarray, f: float, k: int) -> float:
    """bound(f, k) = KL( (1-f)*C_true + f*onehot(k) || C_true ), in nats.

    The worst case for an aggregator that treated every synthetic artifact as a hard vote
    for goal ``k``. The observed value must fall BELOW this, because the aggregator weights
    soft posteriors rather than deltas — so the bound is a sanity ceiling, and an observed
    value above it would mean the accumulation logic is wrong, not that the effect is strong.
    """
    C_true = np.asarray(C_true, dtype=float)
    mix = (1.0 - f) * C_true + f * np.eye(C_true.size)[int(k)]
    return kl_divergence(mix, C_true)


# --------------------------------------------------------------------------- #
# The seed scan and the stratified selection.
# --------------------------------------------------------------------------- #
def scan_synth_seeds(cfg: Config, n_seeds: int, seed_start: int) -> list[dict]:
    """Characterise every candidate synthetic draw. Recorded in full in the prereg file."""
    pairs = cfg.artifact_model.goal_feature_pairs
    rows = []
    for s in range(seed_start, seed_start + n_seeds):
        synth = build_noise_free_synth(cfg, goal_symmetric=False, synth_draw_seed=s)
        k, lean = synth_lean(synth, pairs)
        rows.append({"seed": int(s), "favoured_k": int(k), "lean": float(lean),
                     "entropy": float(shannon_entropy(synth)),
                     "min_mass": float(synth.min())})
    return rows


def select_stratified(scan: list[dict], num_goals: int) -> tuple[list[dict], str]:
    """Pick one draw per favoured goal, with COMPARABLE lean magnitude across goals.

    Rule (stated here, recorded in the prereg file, applied without exception): among all
    scanned seeds whose favoured goal is k, take the one whose lean is closest to the MEDIAN
    lean of the whole scan. Holding lean roughly constant is what makes k the only thing
    varying across the four draws — otherwise a difference between goals could just be a
    difference in how biased their draws happened to be.
    """
    leans = np.array([r["lean"] for r in scan], dtype=float)
    target = float(np.median(leans))
    chosen = []
    for k in range(num_goals):
        cands = [r for r in scan if r["favoured_k"] == k]
        if not cands:
            continue
        best = min(cands, key=lambda r: abs(r["lean"] - target))
        chosen.append(dict(best))
    rule = (f"For each favoured goal k, the scanned seed whose lean is closest to the scan "
            f"median lean ({target:.4f}). Holds bias magnitude ~constant across k so that k "
            f"is the only quantity varying between the four draws.")
    return chosen, rule


# --------------------------------------------------------------------------- #
# Writing and locking.
# --------------------------------------------------------------------------- #
def _canonical(payload: dict) -> str:
    scrubbed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


def build_preregistration(cfg: Config) -> dict:
    """Compute the whole pre-registration payload. No rollouts, no randomness beyond the
    named synth seeds — this is cheap and fully determined by the config."""
    e6b = cfg.get("experiments.e6b", None)
    n_scan = int(cfg.get("experiments.e6b.seed_scan_n", 200))
    scan_start = int(cfg.get("experiments.e6b.seed_scan_start", 1))
    f_sweep = list(cfg.get("experiments.e6b.contamination_sweep",
                           [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    falsify_f = float(cfg.get("experiments.e6b.falsification_f", 0.8))
    falsify_thresh = float(cfg.get("experiments.e6b.falsification_kl", 0.10))
    pred_lo = float(cfg.get("experiments.e6b.predicted_kl_low", 0.15))
    pred_hi = float(cfg.get("experiments.e6b.predicted_kl_high", 0.60))
    num_goals = int(cfg.cardinalities.num_goals)
    C_true = POP_GOAL_DIST[:num_goals]

    scan = scan_synth_seeds(cfg, n_scan, scan_start)
    chosen, rule = select_stratified(scan, num_goals)

    draws = []
    for row in chosen:
        k = int(row["favoured_k"])
        draws.append({
            **row,
            "bounds": {f"{f:g}": contamination_bound(C_true, f, k) for f in f_sweep},
        })

    payload = {
        "experiment": "E6b",
        "written_before_run": True,
        "spec": "V2 §2 (E6b), §5 (pre-registration compliance), §6 (may not change)",
        "decision": "D3a — four draws stratified by favoured goal; full seed scan recorded",
        "C_true": C_true.tolist(),
        "contamination_sweep": f_sweep,
        "bound_formula": "KL( (1-f)*C_true + f*onehot(k) || C_true )  [nats]",
        "bound_is_a_ceiling_because":
            "the aggregator weights soft posteriors, not deltas, so the observed value must "
            "fall below the bound; an observed value ABOVE it indicates an accumulation bug, "
            "not a strong effect",
        "selection_rule": rule,
        "seed_scan": {"start": scan_start, "n": n_scan, "rows": scan},
        "selected_draws": draws,
        "prediction": {
            "quantity": "naive-aggregator KL(C_recovered || C_true)",
            "at_f": falsify_f,
            "predicted_low": pred_lo,
            "predicted_high": pred_hi,
            "v1_reference": 0.066,
        },
        "falsification": {
            "criterion": f"naive KL at f={falsify_f} stays BELOW {falsify_thresh}",
            "threshold": falsify_thresh,
            "consequence":
                "the noise-cancellation diagnosis is wrong, symmetrization was not what "
                "suppressed the V1 effect, and every downstream V2 experiment needs "
                "rethinking before it is run. STOP and report loudly.",
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def write_preregistration(cfg: Config, path: Path, force: bool = False) -> dict:
    """Write the pre-registration, refusing to silently overwrite a differing one.

    This is the enforcement §6 asks for. If a prereg file already exists and its content
    hash differs from what we would write now, that means the bound changed after it was
    first committed — exactly the failure mode pre-registration exists to prevent — so we
    raise rather than overwrite. ``force`` is available for a deliberate, visible reset.
    """
    payload = build_preregistration(cfg)
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
                "The E6b bound was pre-registered and must not change after the fact "
                "(V2 spec §6). Investigate the config change, or pass force=True to reset "
                "deliberately and record it as a deviation in RESULTS_V2.md.")
        if existing is not None:
            return existing   # identical; leave the original file and its mtime alone
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked(path: Path) -> dict:
    """Load the pre-registration and verify it has not been edited since it was written.

    Called by E6b before it does any inference. A tampered or missing file stops the run.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. E6b may not run before its bound is pre-registered "
            "(V2 spec §6). Run: python -m ghostscale.experiments.e6b_corpus_biased --prereg-only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written "
            f"(hash {stated} != recomputed {recomputed}). The pre-registered bound is not "
            "trustworthy; E6b will not run.")
    return payload
