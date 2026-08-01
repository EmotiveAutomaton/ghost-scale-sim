"""Pre-registration for V9. Hash-locked, same as every version before it.

The card is written from SPEC_V9 and hashed over canonical JSON. Two of the four hypotheses on it
did not survive contact, and the card is what makes that statement checkable rather than a claim.
"""
from __future__ import annotations

import hashlib
import json

from .v9 import PREREG_PATH, v9_dir

CARD = {
    "version": "v9",
    "title": "The minimal-model programme, and the two experiments the literature asked for",
    "written_before_code": True,
    "hypotheses": {
        "minimal_models": (
            "Four findings x six structural ablations. A finding's minimal model is the set of "
            "commitments whose REMOVAL kills it. Reported as a set, not a score."),
        "H9.1": ("A learned surface detector sitting in front of the intent machinery reconciles "
                 "the eye-tracking with E19 -- and it misfires on human work sharing the surface."),
        "H9.2": ("The misfiring gets WORSE as the detector gets better, because confidence on a "
                 "correlate is what produces false accusation."),
        "H9.3": ("A pre-shut gate protects where a reactively-shut one does not. At matched "
                 "engagement, adversarial reading drifts less than sympathetic reading."),
        "H9.4": ("A 'read this differently' label protects better than a 'do not read this' "
                 "label, without costing accuracy on genuine work."),
    },
    "nulls": {
        "N41": "the label effect survives at least one ablation",
        "N42": "at least one finding dies to at least one ablation",
        "N43": "the surface detector carries no goal information",
        "N44": "adversarial mode does not protect merely by reducing engagement",
    },
    "fails_if": {
        "minimal_models": "every ablation kills everything (N41) or nothing (N42)",
        "H9.2": "detector accuracy and false-alarm rate move in the same direction",
        "H9.4": "the two labels are equivalent, in which case the practical objection to the "
                "affordance -- nobody applies a 'do not look' label to their own work -- stands",
    },
    "not_built": [
        "the human acquisition test -- named as the top external priority, needs subjects and money",
        "distributed authorship, tool delegation, recursion",
    ],
}


def write_card() -> dict:
    v9_dir()
    body = json.dumps(CARD, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = dict(CARD)
    payload["sha256"] = hashlib.sha256(body).hexdigest()
    PREREG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify() -> bool:
    """Recompute the hash over everything but the hash field."""
    if not PREREG_PATH.exists():
        return False
    stored = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    digest = stored.pop("sha256", None)
    body = json.dumps(stored, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return bool(digest == hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    print(write_card()["sha256"])
