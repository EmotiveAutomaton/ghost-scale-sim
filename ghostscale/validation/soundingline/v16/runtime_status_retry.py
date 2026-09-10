"""Bounded Windows status-publication retry with original supervisor bytecode.

Only mutable RUNNER_STATUS.json writes retry transient Windows denial/sharing
errors. Scientific records, sources, clocks, ownership and unit bytecode remain
unchanged. Permanent denial fails after a finite allowance and stays recorded.
"""
from contextlib import contextmanager
from types import FunctionType
import os
import time
from . import runtime
from .records import write as _write, canonical, now

MAX_ATTEMPTS = 40


def retry_status_write(path, value, *, immutable=True):
    if immutable or path.name != "RUNNER_STATUS.json":
        return _write(path, value, immutable=immutable)
    for attempt in range(1, MAX_ATTEMPTS+1):
        try:
            return _write(path, value, immutable=False)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33}:
                raise
            record = {"recorded_at": now(), "pid": os.getpid(), "attempt": attempt,
                "maximum_attempts": MAX_ATTEMPTS, "windows_error": error.winerror,
                "operation": "mutable status publication", "scientific_unit_reexecuted": False,
                "will_retry": attempt < MAX_ATTEMPTS}
            log = path.parent/"operations/status-publication-retries.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as stream:
                stream.write(canonical(record)+b"\n")
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(.25, .05*attempt))


original = runtime.supervisor.__wrapped__
bindings = dict(original.__globals__, write=retry_status_write)
copied = FunctionType(original.__code__, bindings, original.__name__, original.__defaults__, original.__closure__)
copied.__kwdefaults__ = original.__kwdefaults__
supervisor = contextmanager(copied)
