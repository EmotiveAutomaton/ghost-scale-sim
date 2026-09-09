"""Single-owner status and independently frozen implementation packets."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import os
import platform
import sys
from .records import read, write, file_digest, now, digest

REPO = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent


@contextmanager
def supervisor(root: Path, stage: str):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "OWNER.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("another supervisor owns this campaign")
    state = {"schema_version": "v16.status.1", "pid": os.getpid(), "stage": stage,
             "started_at": now(), "heartbeat": now(), "execution_state": "running"}
    def heartbeat(**updates):
        state.update(updates)
        state["heartbeat"] = now()
        write(root / "RUNNER_STATUS.json", state, immutable=False)
    heartbeat()
    try:
        yield heartbeat
    except BaseException as error:
        heartbeat(execution_state="checkpointed" if isinstance(error, KeyboardInterrupt) else "failed",
                  error=f"{type(error).__name__}: {error}")
        raise
    finally:
        handle.close()


def campaign(root: Path):
    accepted = read(root / "CAMPAIGN.json")
    spec = REPO / "docs/versions/v16-acquired-craft/CODING_PACKAGE.md"
    if file_digest(spec) != accepted["commission_sha256"]:
        raise ValueError("commission hash mismatch")
    return accepted


def remaining_seconds(root: Path):
    deadline = datetime.fromisoformat(campaign(root)["deadline"].replace("Z", "+00:00"))
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def freeze(root: Path, packet_id: str, files: list[Path], design: dict):
    accepted = campaign(root)
    contents = {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in files}
    identity = {"packet_id": packet_id, "commission_hash": accepted["commission_sha256"],
                "files": contents, "design": design,
                "environment": {"python": sys.version, "platform": platform.platform()}}
    path = root / "packets" / f"{packet_id}.json"
    if path.exists():
        previous = read(path)
        if previous["identity"] != identity:
            raise ValueError("frozen packet changed; use an explicit new packet amendment")
        return previous
    record = {"identity": identity, "packet_hash": digest(identity), "frozen_at": now(),
              "accepted_at": accepted["accepted_at"], "deadline": accepted["deadline"]}
    write(path, record)
    return record
