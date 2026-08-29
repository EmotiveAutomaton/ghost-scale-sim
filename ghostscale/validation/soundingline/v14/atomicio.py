"""Atomic JSON publication for V14 working state.

Correctness-critical JSON in this program is rewritten in full on every card: the queue
manifest is ~350 KB and the completion ledger ~75 KB, and both are rewritten dozens of times
per run. A plain ``write_text`` truncates the file first, so a crash, a kill, or a full disk
between truncate and flush leaves a half-written file that no longer parses -- losing the
record of work that was actually done.

``save_ckpt`` in ``common.py`` already uses the tmp-then-rename pattern. This module is that
same pattern, factored out so the writers *outside* the structural lock can share it.

Scope note: this module is deliberately NOT one of the hash-locked generator files. It carries
no scientific machinery -- no seeds, no lineages, no scores, no gates. It is execution
provenance only. ``common.py``'s ``write_verdict`` and ``record_completion`` are still
non-atomic; moving them onto this helper edits a locked generator and needs an explicit
structural-lock amendment (see GHOST_SCALE_AGENT_HARDENING.md, H4).

Atomicity here is per-writer, not mutual exclusion. ``os.replace`` is atomic on Windows and
POSIX, so a reader never sees a torn file, but two concurrent writers still race to be last.
The single-coordinator model (one runner, a watchdog that refuses to start a second) is what
prevents that, and this module does not change it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_text_atomic(path, text: str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``text`` so readers see either the old file or the whole new one, never a torn one.

    The temporary file is created beside the target: ``os.replace`` is only atomic within a
    single filesystem, so a shared temp directory would silently degrade to a copy.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")         # pid-tagged: two writers never share a temp path
    try:
        with tmp.open("w", encoding=encoding, newline=newline) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())                             # the rename is atomic; the bytes still have to be on disk first
        # Windows: an indexer or editor holding the target open makes os.replace raise
        # PermissionError for a few milliseconds; retry with backoff before giving up
        for attempt in range(8):
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return p


def write_json_atomic(path, obj, *, indent: int = 2, sort_keys: bool = False,
                      default=str, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Serialise first, then publish atomically.

    Serialising before opening the temp file matters: if ``obj`` is not serialisable, the
    error is raised while the old file is still intact and untouched.
    """
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, default=default)
    return write_text_atomic(path, text, encoding=encoding, newline=newline)
