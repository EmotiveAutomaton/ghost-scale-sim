"""V7 pre-registration — criteria as executable code, content-hash locked before any run.

Same machinery and the same reason as every version since V3: the written criterion and the applied
criterion are ONE OBJECT, so a criterion cannot drift from the thing the experiment computes.

ONE THING IS DIFFERENT HERE AND IT IS DELIBERATE. Two of V7's criteria are scored on CONTRAST AND
SHAPE rather than on an absolute magnitude, because absolute thresholds have now failed twice in
this project on quantities with no natural scale -- E35's probe drop, which passed on one seed block
of three while its mechanism reproduced on all three, and E46's drift, which has no natural unit at
all. A bar of the form "at least 0.10" is only meaningful when 0.10 means something fixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Config

# C-1 / C-2: the bars these criteria were originally written against, unchanged.
E31_RHO_BAR = 0.70
N21_DOMINANCE = 3.0

# H7.1 / H7.2: what counts as an efficiency advantage, and as reading an unseen intent.
H71_COMPETENCE = 0.80
H72_FLOOR = 0.60            # the simulator must clear this on an intention it has never seen

# H7.3: scored on contrast and shape. No absolute magnitude.
H73_MONOTONE = 0.70

# H7.4: how clean a picture has to be to count as protected.
H74_CLEAN = 0.25


def build_preregistration_v7(cfg: Config) -> dict:
    payload = {
        "version": "V7",
        "scope": ("Two jobs: close the four results version 6 would not draw, and attack E21 on "
                  "the axis it was never tested on."),
        "C-1": {
            "experiment": "E31", "bar": E31_RHO_BAR,
            "scored_on": "process uptake, on E31's own design rather than a reconstruction",
            "history": {"approximate": 0.886, "exact": 0.600, "retrofit": 0.833},
        },
        "C-2": {
            "experiment": "N21", "bar": N21_DOMINANCE,
            "note": ("the pre-registered quantity still decides; the method measure is reported "
                     "beside it and decides nothing"),
        },
        "C-3": {
            "experiment": "E20",
            "decision_rule": ("establish the co-location under exact inference, or retire it from "
                              "the README and the prediction card"),
        },
        "C-4": {
            "experiment": "E35", "scored_on": "relative drop and monotonicity",
            "original_clause_retained": ("an absolute drop of 0.10, reported as failing wherever "
                                         "it fails"),
        },
        "H7.1": {
            "competence": H71_COMPETENCE,
            "scored_quantity": ("evidence needed to reach a fixed competence, not competence at a "
                                "fixed evidence level; the second confounds the ceiling with the "
                                "rate and the claim is about the rate"),
        },
        "H7.2": {
            "floor": H72_FLOOR, "n_held_out": 2,
            "why_two": ("with one held out and disjoint feature supports the counter identifies it "
                        "by elimination, which is the partition answering rather than the reader"),
        },
        "H7.3": {
            "monotone": H73_MONOTONE,
            "scored_on": ("contrast against zero leak and monotonicity in exposure; no absolute "
                          "magnitude"),
        },
        "H7.4": {
            "clean_picture": H74_CLEAN,
            "conditional": ("reports what the policy number would be IF the coupled mechanism is "
                            "right; nothing here establishes that it is"),
        },
        "nulls": {
            "N31": "at leak zero, V7 reproduces the V6 gate exactly",
            "N32": ("the leak PASSES rather than MANUFACTURES: no drift at zero leak, drift "
                    "monotone in leak"),
            "N33": ("both readers in E45 see the same artifacts from the same tape with the same "
                    "priors"),
            "N34": "the held-out intentions have zero training examples",
        },
        "not_built": [
            ("no follow-up to the tool-hypothesis negative. Looking and absorbing are not "
             "separable in the way it would need, which is H7.3's whole point, and an affordance "
             "that reads as 'do not look at this' would never be adopted by the people who have "
             "to apply it. It has to read as 'interact with this differently'."),
            "no creator agent, so the Zahavian security argument stays untested",
            "no recursion",
        ],
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "content_hash"},
                      sort_keys=True, separators=(",", ":"))


def write_preregistration_v7(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_v7(cfg)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n"
                f"  now:     {payload['content_hash']}\n"
                "V7's criteria were pre-registered and must not change after the fact.")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v7(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. No V7 experiment may run before its criteria are pre-registered.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written ({stated} != {recomputed}). "
            f"The pre-registered criteria are not trustworthy; the V7 programme will not run.")
    return payload
