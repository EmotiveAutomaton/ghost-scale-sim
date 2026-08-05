"""Which file, at which content, produced this result.

The gap this closes is small and it bites a month later rather than today. Every verdict JSON in
``results/`` says what it tested and who it was for. None of them says WHICH MODULE wrote it, or
what that module contained at the time. After a refactor -- and this repository has had several --
a result and the code that produced it can no longer be matched up except by memory.

So every verdict carries::

    "produced_by": {
      "module": "ghostscale/validation/soundingline/t1_triangle.py",
      "sha256": "3f2a...",           # of the module's source, at run time
      "git_commit": "b4f5612",       # HEAD when it ran, or null outside a repo
      "git_dirty": true              # whether the tree had uncommitted changes
    }

``git_dirty`` is the field that earns its place. A result produced from a dirty tree cannot be
reconstructed from the commit alone, and knowing that at read time is the difference between "I
can check this" and "I think I remember what that was".

No dependency on GitPython: this shells out to git and returns ``None`` on any failure, because a
provenance stamp must never be the reason a run dies.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                   # noqa: BLE001
        return None


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def produced_by(module_file: str) -> dict:
    """Provenance for the module that is writing a verdict. Pass ``__file__``."""
    p = Path(module_file)
    try:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        sha = None
    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "module": _rel(p),
        "sha256": sha,
        "git_commit": commit,
        "git_dirty": (bool(status) if status is not None else None),
    }


def stamp(verdict: dict, module_file: str, report=None) -> dict:
    """Attach provenance and, if given, a gate block. Returns the same dict for chaining.

    Ordering is deliberate: ``produced_by`` and ``gates`` go in LAST so they sort to the bottom of
    the JSON and never push the actual result off the first screen of a diff.
    """
    from .gates import gate_block

    verdict["produced_by"] = produced_by(module_file)
    if report is not None:
        verdict["gates"] = gate_block(report)
    return verdict
