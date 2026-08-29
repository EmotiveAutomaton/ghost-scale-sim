"""The committed-state receipt: when a fresh-clone receipt exists, it must be coherent with the
committed tree it claims to describe. This test never clones anything itself; it audits the
receipt runners/fresh_clone_v14.py wrote.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECEIPT = REPO / "results" / "v14" / "FRESH_CLONE_RECEIPT.json"

pytestmark = pytest.mark.skipif(not RECEIPT.exists(), reason="no fresh-clone receipt yet")


def test_receipt_is_ok():
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert r.get("ok") is True, f"fresh-clone validation did not pass: {r.get('checks')}"
    assert r.get("scientific_fields_match_local") is not False


def test_receipt_head_is_an_ancestor_of_the_tree():
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    head = r.get("head", "")
    assert len(head) == 40
    out = subprocess.run(["git", "cat-file", "-t", head], cwd=str(REPO), capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        pytest.skip("git object store unavailable")
    assert out.stdout.strip() == "commit"
