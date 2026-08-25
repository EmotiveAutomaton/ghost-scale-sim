"""Every committed V12 verdict must carry provenance, gates, environment, runtime, and a claim
ceiling; the manifest may never hold a forbidden state; lineages may never overlap.

Recursive counterpart of tests/test_gates.py for the nested v12 verdict directory. Reads
committed JSON only, never re-runs anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V12 = REPO / "results" / "validation" / "soundingline" / "v12"
MANIFEST = REPO / "results" / "v12" / "QUEUE_MANIFEST.json"


def verdicts():
    if not V12.exists():
        return []
    out = []
    for p in sorted(V12.rglob("*.json")):
        try:
            out.append((p.stem, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            out.append((p.stem, None))
    return out


ALL = verdicts()
pytestmark = pytest.mark.skipif(not ALL and not MANIFEST.exists(),
                                reason="no V12 artefacts on disk yet")


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_verdict_is_complete(name, v):
    assert v is not None, f"{name}.json does not parse"
    for key in ("produced_by", "gates", "environment", "runtime_seconds", "claim_ceiling", "state"):
        assert key in v, f"{name} lacks {key}"
    assert v["produced_by"].get("sha256"), f"{name} has no source hash"
    assert v["claim_ceiling"] in ("METHOD", "CONSTRUCTED_MECHANISM", "BOUNDARY",
                                  "INSTRUMENT_FAILURE", "VOID")


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_no_control_broke(name, v):
    if v is None or not v.get("gates"):
        pytest.skip("no gates")
    failed = v["gates"].get("failed_names") or []
    # A card whose instrument failed records it as INSTRUMENT_FAILED; a LANDED card may not
    # carry a broken control.
    if v.get("state") == "LANDED":
        assert not failed, f"{name} LANDED with failed controls {failed}"
    assert not (v["gates"].get("unexpected_passes") or []), f"{name}: documented defect passed"


@pytest.mark.skipif(not MANIFEST.exists(), reason="no manifest")
def test_manifest_states_and_lineages():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for d in doc["cards"]:
        assert d["status"] != "DONE", f"{d['id']}: DONE is forbidden"
        assert d["status"] in doc["allowed_states"], d["id"]
    disc, conf = set(doc["lineages"]["discovery"]), set(doc["lineages"]["confirmation"])
    assert not (disc & conf), "discovery and confirmation lineages overlap"
    ids = [d["id"] for d in doc["cards"]]
    assert len(ids) == len(set(ids)), "duplicate card ids"
