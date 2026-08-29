"""Every committed V14 verdict must carry its plain-language record, provenance, gates,
environment, runtime, cells, receipt and claim ceiling; a LANDED verdict may not carry a broken
control; the manifest may never hold a forbidden state; lane lineages never overlap; the
completion ledger's hashes must match the files on disk.

Reads committed JSON only, never re-runs anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V14 = REPO / "results" / "validation" / "soundingline" / "v14"
RESULTS = REPO / "results" / "v14"
MANIFEST = RESULTS / "QUEUE_MANIFEST.json"
COMPLETION = RESULTS / "COMPLETION.json"


def verdicts():
    if not V14.exists():
        return []
    out = []
    for p in sorted(V14.rglob("*.json")):
        try:
            out.append((p.relative_to(V14).as_posix(), p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            out.append((p.relative_to(V14).as_posix(), p, None))
    return out


ALL = verdicts()
pytestmark = pytest.mark.skipif(not ALL and not MANIFEST.exists(), reason="no V14 artefacts on disk yet")


@pytest.mark.parametrize("name,path,v", ALL, ids=[n for n, _, _ in ALL])
def test_verdict_is_complete(name, path, v):
    assert v is not None, f"{name} does not parse"
    for key in ("record", "produced_by", "gates", "environment", "runtime_seconds", "claim_ceiling", "state", "cells", "expected_cell_receipt"):
        assert key in v, f"{name} lacks {key}"
    assert v["produced_by"].get("sha256"), f"{name} has no source hash"
    assert v["claim_ceiling"] in ("METHOD", "CONSTRUCTED_MECHANISM", "BOUNDARY", "INSTRUMENT_FAILURE", "VOID", "RESOURCE_BLOCKED")
    rec = v["record"]
    for key in ("question", "what_happened", "claim_ceiling"):
        assert rec.get(key), f"{name} plain-language record lacks {key}"


@pytest.mark.parametrize("name,path,v", ALL, ids=[n for n, _, _ in ALL])
def test_no_control_broke(name, path, v):
    if v is None or not v.get("gates"):
        pytest.skip("no gates")
    failed = v["gates"].get("failed_names") or []
    if v.get("state") == "LANDED":
        assert not failed, f"{name} LANDED with failed controls {failed}"
    assert not (v["gates"].get("unexpected_passes") or []), f"{name}: documented defect passed"


@pytest.mark.skipif(not MANIFEST.exists(), reason="no manifest")
def test_manifest_states_and_lineages():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for d in doc["cards"]:
        assert d["status"] != "DONE", f"{d['id']}: DONE is forbidden"
        assert d["status"] in doc["allowed_states"], d["id"]
    lanes = {k: set(v) for k, v in doc["lineages"].items() if isinstance(v, list)}
    names = list(lanes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (lanes[a] & lanes[b]), f"lineages {a} and {b} overlap"
    ids = [d["id"] for d in doc["cards"]]
    assert len(ids) == len(set(ids)), "duplicate card ids"


@pytest.mark.skipif(not COMPLETION.exists(), reason="no completion ledger")
def test_completion_ledger_hashes_match():
    import hashlib
    doc = json.loads(COMPLETION.read_text(encoding="utf-8"))
    bad = []
    for key, e in doc.get("entries", {}).items():
        p = REPO / e["verdict_path"]
        if not p.exists():
            bad.append(f"{key}: file missing")
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != e["verdict_sha256"]:
            bad.append(f"{key}: hash mismatch")
    assert not bad, bad[:10]
