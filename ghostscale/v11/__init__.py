"""V11 — The Maker. A persistent maker with a value profile, and the experiments it unblocks.

The world lives in ``maker.py``; the experiment modules live in
``ghostscale/validation/soundingline/`` (s12, s14, s15) because they answer batch-four requests
from the sibling project and that is where S-modules live. Nothing here calls a versioned
``run()`` and nothing here writes outside ``results/v11/`` or
``results/validation/soundingline/``.

Spec: ``docs/versions/v11-the-maker/SPEC.md``, written before this package.
"""
from __future__ import annotations

import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Seed block for V11. House rule: seeds derive from zlib.crc32 of named strings, never hash().
SEED_OFFSET_V11 = 1_100_000


def seed(name: str) -> int:
    """A stable seed from a name. crc32 is deterministic across processes and platforms."""
    return SEED_OFFSET_V11 + (zlib.crc32(name.encode("utf-8")) % 1_000_000)


def v11_dir(sub: str | None = None) -> Path:
    d = REPO / "results" / "v11"
    if sub:
        d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


PREREG_PATH = v11_dir() / "prereg_v11_lock.json"
