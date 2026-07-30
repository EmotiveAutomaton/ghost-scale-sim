"""Pre-registered acceptance criteria for V3 (V3 spec §3, §5, §6).

V2 pre-registered a *bound* (E6b). V3 pre-registers *decision rules*, because the V3
programme is a chain of gates and every one of them is a place where a criterion could be
quietly fitted to the result it is supposed to judge:

  * **N11** decides whether E8 is reportable at all.
  * **E12's threshold** decides the sample size N11 is then evaluated at — so the criterion
    is choosing its own operating point.
  * **N13** decides whether the finite-sample diagnosis behind the whole of V3 is correct.
  * **E13's classification** decides which of C4's three outcomes gets written down.

All four are written to ``results/v3_preregistration.json`` and content-hashed BEFORE any V3
experiment runs, and ``assert_prereg_locked_v3`` refuses to let the programme proceed against
a file that has been edited since. This is the same mechanism as V2's E6b bound, applied to
the thing V3 can actually get wrong.

-----------------------------------------------------------------------------------------
DECISION D1 — N11 is CONJUNCTIVE: a t statistic AND an absolute effect-size ceiling.

V2's N11 used ``|t| < 2`` alone. That is power-dependent in a direction that works against
the experiment: raising per-generation sample size N shrinks the leak slope, but raising
replications shrinks the standard error, so "insignificant at the scale of the experiment"
is not monotone in N and the verdict can be moved by the replication count alone. V3 spec
§3 then defines the minimum viable sample size AS the point where that criterion is crossed,
which closes the loop: the criterion picks its own operating point.

So N11 passes only if BOTH hold at the E12-determined scale:

    |t| < 2                            (no detectable trend), AND
    |slope| < slope_ceiling            (and the trend, if any, is negligible)

``slope_ceiling`` is 0.001 nats/generation: ~12x below V2's measured leak (+0.0119) and ~40x
below the f=0 baseline KL that leak sits on (0.042). ``n_replications`` is FIXED here and
every E12 cell must use it, so that the standard error is not a free parameter.

DECISION D8 — N13 tests the 1/N PREDICTION, not monotonicity.

The V3 spec states N13 as "the leak slope must be monotonically non-increasing in sample
size". Five noisy slope estimates break strict monotonicity by luck, and N13 blocks E8 — so
as stated it fails for statistical rather than substantive reasons. But the fix is not merely
a more robust null. Monotonicity says "the leak did not get worse". The DIAGNOSIS says the
leak is finite-sample estimation error, and finite-sample estimation error has a specific
signature: KL error falls as 1/N. So N13 regresses

    log |slope|  =  a + b * log N

and tests b. Monotonicity would pass a leak shrinking as 1/log N, which is not the claim.
This null therefore CONFIRMS OR REFUTES the finite-sample story rather than guarding it:

    b significantly < 0                    -> the leak shrinks with data (gate passes)
    b within [-1.5, -0.5] (predicted ~ -1) -> consistent with 1/N finite-sample error
    b not significantly < 0                -> V3's diagnosis is REFUTED; E8 does not run

The middle band is reported, not gated: a b of -0.5 would pass the gate while arguing the
error is not the 1/N kind, and that tension is a finding to state, not to suppress.

DECISION D7 — E13 compares the two experiments on ONE quantity, and expects them to differ.

The E9 freeze is reported in feature space (per-column KL of the learned A) and the E8 leak
in goal space (payload KL). "The same curve" is not defined across those. Both experiments
therefore report the IDENTICAL quantity here — divergence of the learned CREATOR/DEEP column
from the true one — plotted against effective Dirichlet sample count.

And the framing is softened, per the author's direction: the hypothesis is no longer that
the freeze and the leak are *the same effect*, but that they lie on **a shared finite-sample
axis, on which they may sit at opposite ends**. E9's starvation arm never departs from its
(informative, D1-seeded) prior because engagement collapses — LOW effective sample count.
E8's honest-signal arm sharpens fast around a finite-sample estimate — HIGH effective sample
count. On the proposed axis those are opposite signatures, so **C4 outcome 2 (two distinct
effects) is the expected result, and is recorded here as the prior expectation before the
run**. That is not a failure of the redo: it is the redo establishing that the framework has
two finite-sample failure modes where it assumed one.
-----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config

# ---- D1: the repaired N11 criterion ---------------------------------------- #
N11_T_THRESHOLD = 2.0
N11_SLOPE_CEILING = 0.001        # nats/generation
# ---- D8: the N13 criterion -------------------------------------------------- #
N13_PREDICTED_EXPONENT = -1.0
N13_CONSISTENT_BAND = (-1.5, -0.5)
# ---- D7: the E13 classification criterion ----------------------------------- #
E13_TOLERANCE_FACTOR = 2.0       # E9's point must sit within this factor of E8's fitted curve


def _canonical(payload: dict) -> str:
    scrubbed = {k: v for k, v in payload.items() if k != "content_hash"}
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# The criteria, as executable functions. The prereg file records their parameters;
# these functions are what the experiments and nulls actually call, so the written
# criterion and the applied criterion cannot drift apart.
# --------------------------------------------------------------------------- #
def n11_verdict(slope: float, t: float) -> dict:
    """D1. Conjunctive: no detectable trend AND no material trend."""
    t_ok = bool(np.isfinite(t) and abs(float(t)) < N11_T_THRESHOLD)
    slope_ok = bool(np.isfinite(slope) and abs(float(slope)) < N11_SLOPE_CEILING)
    return {"slope": float(slope), "t": float(t),
            "t_ok": t_ok, "slope_ok": slope_ok, "passed": bool(t_ok and slope_ok),
            "criterion": f"|t| < {N11_T_THRESHOLD} AND |slope| < {N11_SLOPE_CEILING}"}


def loglog_slope_fit(n_values, slopes) -> dict:
    """D8. Regress log|slope| on log N and return the coefficient with a t statistic.

    Cells whose slope is exactly zero or non-finite are dropped (log undefined) and counted,
    because silently dropping them would flatter the fit.
    """
    n = np.asarray(n_values, dtype=float)
    s = np.abs(np.asarray(slopes, dtype=float))
    ok = np.isfinite(n) & np.isfinite(s) & (n > 0) & (s > 0)
    dropped = int((~ok).sum())
    n, s = n[ok], s[ok]
    if len(n) < 3:
        return {"b": float("nan"), "se": float("nan"), "t": float("nan"),
                "n_points": int(len(n)), "dropped": dropped}
    x, y = np.log(n), np.log(s)
    b, a = np.polyfit(x, y, 1)
    resid = y - (b * x + a)
    dof = max(len(x) - 2, 1)
    denom = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt((resid @ resid) / dof / denom)) if denom > 0 else float("inf")
    return {"b": float(b), "intercept": float(a), "se": se,
            "t": float(b / se) if se > 0 else float("nan"),
            "n_points": int(len(n)), "dropped": dropped}


def n13_verdict(fit: dict) -> dict:
    """D8. The gate is 'significantly negative'; the 1/N band is reported, not gated."""
    b, t = fit.get("b", float("nan")), fit.get("t", float("nan"))
    significant_decline = bool(np.isfinite(t) and t < -N11_T_THRESHOLD)
    lo, hi = N13_CONSISTENT_BAND
    consistent = bool(np.isfinite(b) and lo <= b <= hi)
    return {**fit, "passed": significant_decline,
            "consistent_with_one_over_N": consistent,
            "predicted_exponent": N13_PREDICTED_EXPONENT,
            "consistent_band": list(N13_CONSISTENT_BAND),
            "criterion": (f"gate: log|slope| ~ log N coefficient b significantly < 0 "
                          f"(t < -{N11_T_THRESHOLD}); reported: b in "
                          f"[{lo}, {hi}] means 1/N finite-sample error"),
            "if_refuted": ("the leak is not finite-sample estimation error; C1 is not the "
                           "right fix, the V3 diagnosis is wrong, and E8 must not run")}


def select_sample_size(rows: list[dict]) -> dict:
    """E12 -> E8. The smallest swept N whose AVERAGED f=0 honest arm passes D1's N11.

    'Smallest passing' rather than 'largest swept' on purpose: the threshold is a claim about
    where the loop becomes clean, and taking the largest cell would hide a failure to find one.
    """
    passing = sorted((r for r in rows if r.get("averaging") and r.get("n11_passed")),
                     key=lambda r: r["n_artifacts"])
    if not passing:
        return {"found": False, "n_artifacts": None,
                "reason": "no swept sample size passed the pre-registered N11 criterion "
                          "with averaging; E8 does not run"}
    chosen = passing[0]
    return {"found": True, "n_artifacts": int(chosen["n_artifacts"]),
            "slope": float(chosen["slope"]), "t": float(chosen["t"]),
            "rule": "smallest swept per-generation sample size whose averaged f=0 "
                    "honest-signal arm satisfies |t| < 2 AND |slope| < 0.001"}


def e13_classify(fit_c: float, fit_b: float, freeze_n_eff: float, freeze_kl: float,
                 control_kl: float, freeze_vanished_tol: float = 0.05) -> dict:
    """D7. Which of C4's three outcomes obtained.

    ``fit_c``/``fit_b`` are the power-law fit ``kl = c * n_eff ** b`` from the E8/E12 honest
    recursion. ``freeze_*`` is E9's starvation arm on the SAME quantity.
    """
    if np.isfinite(freeze_kl) and abs(freeze_kl - control_kl) < freeze_vanished_tol:
        return {"outcome": 3, "label": "freeze_vanished",
                "statement": "under C1 and the E12-determined sample size the starvation arm "
                             "no longer freezes; the freeze was never real, only under-sampled "
                             "— the cleanest confirmation of D4"}
    predicted = float(fit_c * (freeze_n_eff ** fit_b)) if freeze_n_eff > 0 else float("nan")
    ratio = float(freeze_kl / predicted) if predicted > 0 else float("nan")
    on_curve = bool(np.isfinite(ratio)
                    and 1.0 / E13_TOLERANCE_FACTOR <= ratio <= E13_TOLERANCE_FACTOR)
    if on_curve:
        return {"outcome": 1, "label": "shared_axis", "predicted_kl": predicted, "ratio": ratio,
                "statement": "the freeze and the leak fall on one finite-sample axis; the E9 "
                             "freeze is an artifact of premature convergence, which is what D4 "
                             "predicts a correctly specified model's freeze would turn out to be"}
    return {"outcome": 2, "label": "distinct_effects", "predicted_kl": predicted, "ratio": ratio,
            "statement": "the E9 freeze does not lie on the E8 leak's finite-sample curve. "
                         "There are TWO distinct finite-sample failure modes where the "
                         "framework assumed one. This is an OPEN PROBLEM in the framework and "
                         "is reported as such, not explained away.",
            "note": "recorded before the run as the EXPECTED outcome (D7)"}


# --------------------------------------------------------------------------- #
# The pre-registration payload.
# --------------------------------------------------------------------------- #
def build_preregistration_v3(cfg: Config) -> dict:
    e12 = cfg.get("experiments.e12", None)
    n_sweep = list(cfg.get("experiments.e12.n_artifacts_sweep", [100, 300, 1000, 3000, 10000]))
    m_sweep = list(cfg.get("experiments.e12.m_observers_sweep", [1, 5, 20]))
    n_reps = int(cfg.get("experiments.e12.n_replications", 3))
    g_max = int(cfg.get("experiments.e12.g_max", 8))

    payload = {
        "programme": "V3",
        "written_before_run": True,
        "spec": "V3 §1 (C2, C4), §3 (N11 repaired, N13), §5 (reporting), §6 (may not change)",
        "N11": {
            "decision": "D1 — conjunctive criterion, replications fixed in advance",
            "why": ("|t| alone is power-dependent: larger N shrinks the leak while more "
                    "replications shrink the standard error, so 'insignificant at the scale "
                    "of the experiment' is not monotone in N. V3 §1 C2 then defines the "
                    "sample-size threshold as the crossing of that criterion, so the "
                    "criterion would be selecting its own operating point."),
            "criterion": f"|t| < {N11_T_THRESHOLD} AND |slope| < {N11_SLOPE_CEILING} nats/gen",
            "t_threshold": N11_T_THRESHOLD,
            "slope_ceiling_nats_per_generation": N11_SLOPE_CEILING,
            "n_replications_fixed_at": n_reps,
            "evaluated_at": "the E12-determined per-generation sample size, at full E8 scale",
            "v2_reference": {"slope": 0.0119, "t": 3.75, "baseline_kl_at_f0": 0.042},
            "consequence_if_failed": "E8 is not reported and is excluded from E11",
        },
        "N13": {
            "decision": "D8 — test the 1/N prediction, not monotonicity",
            "why": ("monotonicity says only 'the leak did not get worse'. The diagnosis says "
                    "the leak is finite-sample estimation error, whose signature is KL "
                    "falling as 1/N. Testing the exponent confirms or refutes the diagnosis "
                    "rather than merely guarding it, and does not fail by luck when five "
                    "noisy slope estimates happen to be non-monotone."),
            "model": "log|slope| = a + b * log N, over the without-averaging arm",
            "gate": f"b significantly negative (t < -{N11_T_THRESHOLD})",
            "predicted_exponent": N13_PREDICTED_EXPONENT,
            "consistent_band": list(N13_CONSISTENT_BAND),
            "band_is_reported_not_gated": True,
            "consequence_if_refuted": ("the leak is not finite-sample estimation error; the "
                                       "entire V3 diagnosis is wrong and E8 must not be run "
                                       "until the structural reason is found (V3 §1 C2)"),
        },
        "E12": {
            "n_artifacts_sweep": n_sweep,
            "m_observers_sweep": m_sweep,
            "g_max": g_max,
            "n_replications": n_reps,
            "condition": "f = 0 with an honest signal — the exact condition that failed N11 in V2",
            "arms": ["with C1 population averaging", "without (V2 single-observer seeding)"],
            "threshold_rule": ("smallest swept per-generation sample size whose AVERAGED arm "
                               "satisfies the N11 criterion above; 'smallest passing' rather "
                               "than 'largest swept' so that a failure to find one is visible"),
            "m_sweep_purpose": ("D2 — the corpus is drawn once per generation and shared by "
                                "all M observers, so averaging cannot cancel the corpus's own "
                                "goal-composition noise. The 1/M gain therefore has a floor "
                                "that only N can lower, and the M sweep measures where it sits."),
        },
        "E13": {
            "decision": "D7 — one shared quantity, and outcome 2 expected",
            "shared_quantity": ("mean over goals of KL(learned CREATOR/DEEP column || true "
                                "CREATOR/DEEP column), in nats — computed by the identical "
                                "code path (learning.column_kl) in both experiments"),
            "shared_axis": "effective Dirichlet sample count = concentration mass - prior",
            "fit": "kl = c * n_eff ** b, fitted on the honest-signal recursion",
            "classification": {
                "outcome_1_shared_axis": (f"E9's starvation point falls within a factor of "
                                          f"{E13_TOLERANCE_FACTOR} of the fitted curve"),
                "outcome_2_distinct_effects": "it does not — two distinct finite-sample failure modes",
                "outcome_3_freeze_vanished": "the starvation arm no longer freezes at all",
            },
            "tolerance_factor": E13_TOLERANCE_FACTOR,
            "prior_expectation": {
                "expected_outcome": 2,
                "reasoning": ("E9's starvation arm never departs from its informative D1 prior "
                              "because engagement collapses (1.87 of 6 DEEP steps) — LOW "
                              "effective sample count. E8's honest arm sharpens fast around a "
                              "finite-sample estimate — HIGH effective sample count. On this "
                              "axis those are opposite signatures."),
                "framing": ("the V3 §0 hypothesis is softened accordingly: not 'the same "
                            "effect' but 'a shared finite-sample axis, on which they may sit "
                            "at opposite ends'"),
                "why_this_is_not_a_failure": ("two distinct finite-sample failure modes where "
                                              "the framework assumed one is a finding, and V3 "
                                              "§6 requires E13 to be capable of returning it"),
            },
        },
        "C3_regression": {
            "decision": "D6 — partial correlation controlling for generation",
            "why": ("value divergence and encoder divergence both trend with generation by "
                    "construction, so a pooled regression across generations would come out "
                    "significant even if the two channels were independent — it would confirm "
                    "the shared-mechanism claim automatically and could not refute it"),
            "reported": ["partial correlation of encoder divergence with value divergence, "
                         "controlling for generation",
                         "within-generation correlation across conditions and replications",
                         "lag-1 cross-correlation, for the 'lags and tracks' claim"],
            "pooled_regression_is_reported_but_not_evidence": True,
        },
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def write_preregistration_v3(cfg: Config, path: Path, force: bool = False) -> dict:
    """Write the V3 criteria, refusing to silently overwrite a differing set."""
    payload = build_preregistration_v3(cfg)
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
                "V3's acceptance criteria were pre-registered and must not change after the "
                "fact (V3 spec §6). Investigate the config change, or pass force=True to "
                "reset deliberately and record it as a deviation in RESULTS_V3.md.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v3(path: Path) -> dict:
    """Load the V3 criteria and verify they have not been edited since being written."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V3 experiment may run before its acceptance criteria are "
            "pre-registered (V3 spec §6). Run: python scripts/run_all_v3.py --prereg-only")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written "
            f"(hash {stated} != recomputed {recomputed}). The pre-registered criteria are not "
            "trustworthy; the V3 programme will not run.")
    return payload
