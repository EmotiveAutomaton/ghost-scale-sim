"""V-8 and V-9 — the independent reimplementation, and the one forward test.

V-8 REIMPLEMENTS THE STRONGEST RESULT FROM THE PROSE ALONE. The code lives in
``scripts/independent_two_gates.py``, imports nothing from this package, and shares no parameter
with it. This module runs it and scores it against criteria committed before it was written. If it
replicates, that is the strongest evidence available anywhere in this project; if it does not,
finding out why is worth more than any new experiment.

**WHAT "REPLICATES" WAS ALLOWED TO MEAN, decided in advance.** A reimplementation from prose cannot
be held to a magnitude: it shares no feature count, no goal count, no opacity values, no trust
parameter and no random stream, so a multiple matching to two decimal places would mean the two
implementations had accidentally chosen the same arbitrary constants. What it CAN be held to is the
direction and the order of magnitude, and those are what ``criteria.py`` pre-registers. Anything
stricter would be a test of coincidence.

-----------------------------------------------------------------------------------------
V-9 IS THE ONLY TRUE FORWARD TEST THE PROJECT CAN GENERATE FOR ITSELF, and the reason it is needed
is stated bluntly in the spec: the literature check is retrospective, every simulation was designed
by people who knew the theory, and there is currently no forward test anywhere in the body of work.

So a full prediction is pre-registered — outcome, direction, magnitude, and named failure branches
— for an experiment that HAS NOT BEEN BUILT, and it is hash-locked. The candidate is the obvious
one and the natural next experiment anyway: **the reader equipped with a hypothesis for what a maker
was AVOIDING.**

Writing the prediction now and building the experiment later is the whole mechanism. It is the one
arrangement under which this project can be wrong about something in advance.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..experiments._common import RESULTS_DIR
from . import criteria as CR
from . import validation_dir


# =========================================================================== #
# V-8.
# =========================================================================== #
def run_v8(cfg: Config, readers: int = 400) -> dict:
    out = validation_dir() / "v8"
    out.mkdir(parents=True, exist_ok=True)
    script = RESULTS_DIR.parent / "scripts" / "independent_two_gates.py"
    payload_path = out / "independent_reimplementation.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--readers", str(readers), "--out", str(payload_path)],
        capture_output=True, text=True, cwd=str(RESULTS_DIR.parent))
    if proc.returncode != 0 or not payload_path.exists():
        verdict = {
            "check": "V-8", "verdict": "REIMPLEMENTATION_DID_NOT_RUN",
            "statement": ("The independent reimplementation failed to run, so nothing is claimed "
                          "from it either way."),
            "stderr": proc.stderr[-2000:],
        }
        (validation_dir() / "v8_reimplementation.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")
        return verdict

    got = json.loads(payload_path.read_text(encoding="utf-8"))
    original = _original_two_gates()
    reimpl_multiple = float(got["uptake_multiple"])
    orig_multiple = original["uptake_multiple"]

    direction_ok = bool(got["direction_matches"])
    mechanism_ok = bool(np.isfinite(got["update_tracks_believed_depth_rho"])
                        and got["update_tracks_believed_depth_rho"]
                        >= original["update_tracks_mu_rho_required"])
    if np.isfinite(orig_multiple) and orig_multiple > 0 and reimpl_multiple > 0:
        ratio = max(orig_multiple / reimpl_multiple, reimpl_multiple / orig_multiple)
    else:
        ratio = float("inf")
    magnitude_ok = bool(ratio <= CR.V8_ORDER_OF_MAGNITUDE_FACTOR)

    if direction_ok and mechanism_ok and magnitude_ok:
        v = "REPLICATES"
        statement = (
            f"The result replicates in separate code written from the prose alone. The direction "
            f"holds, the mechanism holds, in that how far the reader moves tracks how much "
            f"thinking it "
            f"believes went in whichever channel put that belief there, and the multiple "
            f"({reimpl_multiple:.1f}x) is within an order of magnitude of the original's "
            f"({orig_multiple:.1f}x). This is the strongest evidence in the project.")
    elif direction_ok and mechanism_ok:
        v = "MECHANISM_REPLICATES_MAGNITUDE_DOES_NOT"
        statement = (
            f"The MECHANISM replicates and the MAGNITUDE does not. In separate code written from "
            f"the prose alone, a false label still inflates the reader's estimate of how much "
            f"thinking went into machine-made work, and how far the reader moves still tracks that "
            f"estimate whichever channel produced it, and the rank correlation across all six "
            f"conditions is {got['update_tracks_believed_depth_rho']:.2f}. But the uptake multiple "
            f"is {reimpl_multiple:.1f}x against the original's {orig_multiple:.1f}x, a factor of "
            f"{ratio:.0f} apart and outside the order of magnitude committed before the "
            f"reimplementation was written.\n\n"
            f"**The honest reading, and it is the more useful one.** The {orig_multiple:.0f}-fold "
            f"figure is a property of this model's particular dimensions and is not the claim. The "
            f"claim that survives independent construction is directional: a false provenance "
            f"label makes a reader take on substantially more from machine-made work than an "
            f"honest one does, through an inflated estimate of the thinking behind it. Every "
            f"public-facing use of the multiple has to be stated that way, as a direction with "
            f"a "
            f"model-specific size, and the specific number must not be quoted as though it "
            f"transfers.")
    elif direction_ok:
        v = "DIRECTION_REPLICATES_MECHANISM_DOES_NOT"
        statement = (
            "The direction replicates but the proposed mechanism does not: the reimplemented "
            "reader takes on more under a false label without its uptake tracking its depth "
            "estimate. That separates the finding from its explanation, and the explanation is "
            "the part the framework uses.")
    else:
        v = "DOES_NOT_REPLICATE"
        statement = (
            "The result does not replicate in independent code. Per the spec, finding out why is "
            "worth more than any new experiment, and until that is done the claim is withdrawn "
            "from the public-facing material.")

    verdict = {
        "check": "V-8",
        "question": ("Does the strongest result survive being rebuilt from its own description, "
                     "in code that shares nothing with the original?"),
        "plain_language": (
            "The single most legible finding in the project was rewritten from scratch, using only "
            "the plain-English description of it and none of the original code, settings or random "
            "seeds. If a finding only exists in the code that produced it, it is not a finding "
            "about anything. This is the check that tells them apart."),
        "criteria": {"direction_required": CR.V8_DIRECTION_REQUIRED,
                     "order_of_magnitude_factor": CR.V8_ORDER_OF_MAGNITUDE_FACTOR,
                     "why_not_stricter": ("the reimplementation shares no parameter with the "
                                          "original, so a matching magnitude would mean two "
                                          "people picked the same arbitrary constants")},
        "original": original,
        "reimplementation": got,
        "direction_matches": direction_ok,
        "mechanism_matches": mechanism_ok,
        "magnitude_within_an_order_of_magnitude": magnitude_ok,
        "multiples_factor_apart": ratio,
        "independence": ("scripts/independent_two_gates.py imports numpy and the standard library "
                         "only. It does not import ghostscale, does not read results/, and every "
                         "parameter in it is declared at the top of the file."),
        "verdict": v,
        "statement": statement,
    }
    (validation_dir() / "v8_reimplementation.json").write_text(
        json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    return verdict


def _original_two_gates() -> dict:
    """The original's own numbers, read from the committed E31 verdict."""
    p = RESULTS_DIR / "e31_verdict.json"
    if not p.exists():
        return {"uptake_multiple": float("nan"), "update_tracks_mu_rho_required": 0.70}
    v = json.loads(p.read_text(encoding="utf-8"))
    cells = v.get("headline_cells", {})
    crash = (cells.get("crash_honest_ghost_label") or {}).get("prior_drift")
    exploit = (cells.get("exploit_dishonest_creator_label") or {}).get("prior_drift")
    multiple = (float(exploit) / float(crash)
                if crash not in (None, 0) and exploit is not None else float("nan"))
    return {
        "source": "results/e31_verdict.json",
        "uptake_multiple": multiple,
        "view_shift_under_false_label": exploit,
        "view_shift_under_honest_label": crash,
        "update_tracks_mu_rho": v.get("update_tracks_mu_rho"),
        "update_tracks_mu_rho_required": float(v.get("update_tracks_mu_rho_required", 0.70)),
        "limitation": v.get("limitation"),
    }


# =========================================================================== #
# V-9 — the pre-registration.
# =========================================================================== #
PREDICTION = {
    "title": "The reader equipped with a hypothesis for what the maker was AVOIDING",
    "status": "PRE-REGISTERED. The experiment is not built. Nothing has been run.",
    "why_this_one": (
        "It is the natural next experiment regardless of validation, and it is the only candidate "
        "where the theory makes a prediction that does not follow from anything already measured. "
        "Every existing hypothesis space in this project is a space of things a maker might have "
        "been TRYING TO DO. Avoidance is a different shape of intention: the maker's purpose is "
        "specified by what is absent rather than by what is present, which means the evidence for "
        "it is a hole in the distribution rather than a peak in it."),
    "the_setup": (
        "The maker holds an avoidance goal: some region of the feature space it will not enter. "
        "Otherwise it behaves exactly as an existing maker does. Two readers see its work. The "
        "FIRST holds only the existing hypothesis space, meaning purposes as things being pursued. The "
        "SECOND additionally holds avoidance hypotheses, one per region, in the same likelihood "
        "family. Both are otherwise identical, including their priors and their random stream."),
    "the_prediction": {
        "primary_outcome": "recovered intent on avoidance-driven work, reader two against reader one",
        "direction": ("reader two recovers the maker's actual constraint and reader one does not, "
                      "so the accuracy gap is positive"),
        "magnitude": ("reader two's accuracy on avoidance-driven work exceeds reader one's by at "
                      "least 0.30 in absolute terms, on a four-alternative task where chance is "
                      "0.25. Stated as an absolute gap rather than a ratio because reader one is "
                      "predicted to be near chance and a ratio against chance is unstable."),
        "secondary_outcome_and_it_is_the_interesting_one": (
            "reader one is predicted to be CONFIDENT and WRONG rather than uncertain: within-reader "
            "doubt below 0.5 nats with accuracy at or below chance. Avoidance leaves a hole, the "
            "existing hypotheses all have support inside that hole, and the one whose peak sits "
            "furthest from it wins by default. That is the same failure mode as the partial-overlap "
            "peak, arriving by a different route, and if it holds it is the second independent "
            "instance of one mechanism."),
        "cost_prediction": ("reader two pays MORE attention, not less, because an avoidance "
                            "hypothesis is confirmed by continued absence and absence accumulates "
                            "slowly. Predicted at least 1.5x the deep looks of reader one."),
    },
    "named_failure_branches": {
        "NO_GAP": ("reader two does no better. Avoidance is not recoverable from a hole in this "
                   "geometry, and the framework's claim that intention can be read from what is "
                   "absent is unsupported in simulation. This is the outcome that costs the most "
                   "and it is a real possibility: the hole may simply be too weak a signal at "
                   "this feature count, in which case the result is about the geometry and has to "
                   "be reported as inconclusive rather than negative."),
        "GAP_BUT_READER_ONE_IS_UNCERTAIN": ("reader two wins and reader one is appropriately "
                                            "unsure rather than confidently wrong. The primary "
                                            "prediction holds and the interesting secondary one "
                                            "fails, which would separate the two mechanisms this "
                                            "prediction claims are the same."),
        "GAP_REVERSES": ("reader one does BETTER. The avoidance hypotheses act as an absorbing "
                         "fallback the way a uniform EXPLORE signature would, and the extra "
                         "hypotheses hurt. This would be a direct hit on the framework and would "
                         "be reported as one."),
        "READER_TWO_PAYS_LESS": ("the cost prediction inverts. An avoidance hypothesis resolves "
                                 "faster than a pursuit hypothesis, which would contradict the "
                                 "metabolic account the whole engagement story rests on."),
    },
    "what_is_committed_and_what_is_not": (
        "COMMITTED: the direction, the 0.30 absolute accuracy gap, the 0.5 nat confidence "
        "threshold on reader one, the 1.5x attention ratio, and the four branches above. NOT "
        "COMMITTED: the feature count, the number of avoidance regions, the observer count and the "
        "seeds, because those are scale decisions and fixing them now without having built "
        "anything would be pre-registering guesses as though they were design."),
    "how_this_gets_scored": (
        "The experiment is built and run once. Its outcome is matched against the branches above "
        "by their stated thresholds, and the branch that fires is reported whatever it is. If none "
        "fires cleanly, that is recorded as the prediction having been under-specified, which is "
        "itself a result about how well this framework can be made to commit in advance."),
}


# Keys that describe the LOCK rather than the prediction. They are written after the hash is taken
# and must be excluded from it, or the file fails its own check the first time it is re-read — which
# is exactly what happened on the first run of this pass, and it is why the exclusion list is
# explicit rather than "everything except content_hash".
_STATUS_KEYS = frozenset({"content_hash", "lock_intact", "verdict", "statement"})


def _canonical(payload: dict) -> str:
    trimmed = {k: v for k, v in payload.items() if k not in _STATUS_KEYS}
    return json.dumps(trimmed, sort_keys=True, separators=(",", ":"), default=str)


def run_v9(cfg: Config) -> dict:
    path = validation_dir() / "v9_out_of_sample_prediction.json"
    payload = {
        "check": "V-9",
        "question": "Can this project be wrong about something in advance?",
        "plain_language": (
            "Everything in this repository was predicted by people who already knew the theory, "
            "and the literature check happened afterwards. That means nothing here is a forward "
            "test. This is one: a complete prediction covering which way, how big, and four "
            "named ways it "
            "could fail, for an experiment that does not exist yet, written down and sealed with "
            "a hash before anybody builds it."),
        "prediction": PREDICTION,
        "locked_before": "the experiment exists",
    }
    if path.exists():
        # ALREADY LOCKED. A pre-registration that can be rewritten is not one, so an existing file
        # is verified and returned rather than overwritten.
        existing = json.loads(path.read_text(encoding="utf-8"))
        stated = existing.get("content_hash")
        recomputed = hashlib.sha256(_canonical(existing).encode()).hexdigest()
        existing["lock_intact"] = bool(stated == recomputed)
        existing["verdict"] = ("PREDICTION_LOCKED" if existing["lock_intact"]
                              else "PREDICTION_FILE_HAS_BEEN_MODIFIED")
        if not existing["lock_intact"]:
            existing["statement"] = (
                "The pre-registration file has changed since it was written. Its contents cannot "
                "be treated as having been fixed in advance and it is worthless as a forward "
                "test until re-locked from a clean state.")
        return existing

    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    payload["lock_intact"] = True
    payload["verdict"] = "PREDICTION_LOCKED"
    payload["statement"] = (
        "A full prediction for the avoidance-reader experiment is written down and hash-locked "
        "before the experiment exists: the direction, the magnitude, and four named ways it can "
        "fail, including one that would be a direct hit on the framework. This is the only "
        "genuinely out-of-sample test the project can generate for itself, and it is now the "
        "largest thing it owes.")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run(cfg: Config, workers: int = 1) -> dict:
    return {"v8": run_v8(cfg), "v9": run_v9(cfg)}
