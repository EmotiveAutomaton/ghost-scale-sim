"""V8 pre-registration — criteria as executable code, content-hash locked before any run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Config

H81_INTERACTION = 0.25       # reading gap across readers, on deep work
H82_GAP_RHO = -0.50          # compression against reader/maker gap
H83_GROWTH = 0.10            # growth from exposure to deeper work
H84_SPLIT = 5.0 / 9.0        # the standard bimodality threshold, not invented here
H85_SEPARATION = 0.30        # shock art against slop, on the trace
H86_REPUTATION_COST = 2.0

# Carried verbatim from the validation pass's sealed file, whose STATUS is downgraded in E52.
SEALED_ACCURACY_GAP = 0.30
SEALED_CONFIDENCE = 0.5
SEALED_ATTENTION_RATIO = 1.5


def build_preregistration_v8(cfg: Config) -> dict:
    payload = {
        "version": "V8",
        "scope": ("The reader gets a hierarchy of its own, a cost for being changed, and a memory "
                  "that fades. Plus the severity check the programme was missing, a maker that "
                  "can lie, and the readymade."),
        "S-1": {
            "what": ("false-positive rate: keep the model's shape, throw its settings away, count "
                     "how often the finding survives"),
            "reference": {"finding": "the label effect", "rate": 0.64,
                          "source": "the validation pass"},
            "reported": "every rate, whatever it is",
        },
        "H8.1": {"interaction": H81_INTERACTION},
        "H8.2": {"gap_rho": H82_GAP_RHO,
                 "explains": "the unexplained depth compression, measured at about a third"},
        "H8.3": {"growth": H83_GROWTH,
                 "note": "the author's hypothesis: reading and making are the same machinery"},
        "H8.4": {"bimodality": H84_SPLIT},
        "H8.5": {"separation": H85_SEPARATION},
        "H8.6": {"reputation_cost": H86_REPUTATION_COST},
        "E52": {
            "sealed_criteria": {"accuracy_gap": SEALED_ACCURACY_GAP,
                                "confidence": SEALED_CONFIDENCE,
                                "attention_ratio": SEALED_ATTENTION_RATIO},
            "status": ("the sealed prediction's FORWARD-TEST status is withdrawn: the author does "
                       "not recognise authoring it. The experiment is run on its merits and the "
                       "project is recorded as having no forward test."),
        },
        "nulls": {
            "N35": "every V8 addition off by default; all off reproduces V7",
            "N36": "forgetting approaches a floor and never reaches zero",
            "N37": "reader depth buys nothing where there is no hierarchy to see",
            "N38": "integration cost does not become a preference over provenance",
            "N39": "density does not reward noise",
            "N40": "lying must pay when it is never caught",
        },
        "not_built": [
            "distributed authorship",
            "a maker that chooses how much to delegate to a tool",
            "recursion",
        ],
    }
    payload["content_hash"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return payload


def _canonical(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k != "content_hash"},
                      sort_keys=True, separators=(",", ":"))


def write_preregistration_v8(cfg: Config, path: Path, force: bool = False) -> dict:
    payload = build_preregistration_v8(cfg)
    path = Path(path)
    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing is not None and existing.get("content_hash") != payload["content_hash"]:
            raise RuntimeError(
                f"{path.name} exists with a DIFFERENT content hash.\n"
                f"  on disk: {existing.get('content_hash')}\n  now: {payload['content_hash']}")
        if existing is not None:
            return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def assert_prereg_locked_v8(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} not found. No V8 experiment may run before its criteria are locked.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_hash")
    recomputed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if stated != recomputed:
        raise RuntimeError(
            f"{path.name} has been modified since it was written ({stated} != {recomputed}).")
    return payload
