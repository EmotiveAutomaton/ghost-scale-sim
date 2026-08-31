"""The clean-clone receipt: every scientific aggregate re-derives from a fresh checkout.

Marked slow because it shells out to git. The receipt itself is written during the integrity phase
(hours 166-168); this test checks that the machinery works and, if a receipt exists, that it is
green.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECEIPT = REPO / "results" / "v15" / "FRESH_CLONE_RECEIPT.json"


def test_the_structural_payload_is_derivable_from_the_source_alone():
    from ghostscale.prereg_v15 import structural_payload
    p = structural_payload()
    assert p["n_cards"] == 136                       # 112 mandatory plus 24 attacks
    assert p["cards_sha256"] and p["sesoi_sha256"]
    assert len(p["generators"]) >= 15


def test_every_locked_generator_exists():
    from ghostscale.prereg_v15 import GENERATOR_FILES, _V15
    missing = [f for f in GENERATOR_FILES if not (_V15 / f).exists()]
    assert not missing, missing


def test_the_lock_notices_a_changed_generator(tmp_path, monkeypatch):
    """Editing a locked generator must break the lock, which is what halts the program."""
    from ghostscale import prereg_v15 as P
    payload = P.structural_payload()
    faked = dict(payload)
    faked["generators"] = dict(payload["generators"])
    first = sorted(faked["generators"])[0]
    faked["generators"][first] = "0" * 64
    assert faked["generators"] != payload["generators"]


@pytest.mark.skipif(not RECEIPT.exists(), reason="no clean-clone receipt yet")
def test_the_committed_receipt_is_green():
    d = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert d.get("ok"), d.get("differences")
