"""Single-owner status and independently frozen implementation packets."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import os
import platform
import sys
import threading
import time
import signal
from .records import read, write, file_digest, now, digest
from .campaign_ownership import campaign_owner,acceptance

REPO = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent


@contextmanager
def local_owner(root: Path):
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
    try:
        yield
    finally:
        handle.close()


@contextmanager
def supervisor(root: Path, stage: str, *, heartbeat_interval=15.0):
    if heartbeat_interval<=0:
        raise ValueError("heartbeat interval must be positive")
    root.mkdir(parents=True,exist_ok=True)
    accepted=campaign(root) if (root/"CAMPAIGN.json").exists() else None
    if accepted and stage in {"pilot","discovery","transfer","confirmation"} and remaining_seconds(root)<=0:
        raise RuntimeError("immutable ceiling forbids new scientific dispatch")
    with campaign_owner(root) as owner_key,local_owner(root):
        state={"schema_version":"v16.status.2","pid":os.getpid(),"stage":stage,"started_at":now(),
            "heartbeat":now(),"execution_state":"running","campaign_owner_key":owner_key,
            "ownership_scope":"cross-checkout OS ownership, with the per-directory lock retained"}
        started=time.monotonic();cpu=time.process_time()
        serialized=threading.RLock();stop=threading.Event();failures=[]
        def emit(updates,source):
            with serialized:
                state.update(updates)
                state["heartbeat"]=now();state["heartbeat_source"]=source
                state["supervisor_elapsed_seconds"]=time.monotonic()-started
                state["supervisor_cpu_seconds"]=time.process_time()-cpu
                if "completed_units" in updates:
                    state["last_unit_report_at"]=state["heartbeat"]
                state["unit_loop_complete"]=bool(state.get("planned_units") and state.get("completed_units")==state["planned_units"])
                if accepted:
                    deadline=datetime.fromisoformat(accepted["deadline"].replace("Z","+00:00"))
                    state["ceiling_exceeded"]=(datetime.now(timezone.utc)>deadline)
                write(root/"RUNNER_STATUS.json",state,immutable=False)
        def heartbeat(**updates):
            if failures:
                raise RuntimeError("supervisor heartbeat failed") from failures[0]
            emit(updates,"supervisor")
        def pulse():
            while not stop.wait(heartbeat_interval):
                try:
                    emit({},"liveness timer; no implied new scientific completion")
                except BaseException as error:
                    failures.append(error);return
        def checkpoint_signal(signum,frame):
            raise KeyboardInterrupt(f"checkpoint requested by signal {signum}")
        previous=None
        if threading.current_thread() is threading.main_thread():
            previous=signal.signal(signal.SIGTERM,checkpoint_signal)
        heartbeat()
        timer=threading.Thread(target=pulse,name="v16-supervisor-heartbeat",daemon=True)
        timer.start()
        try:
            yield heartbeat
            if failures:
                raise RuntimeError("supervisor heartbeat failed") from failures[0]
        except BaseException as error:
            emit({"execution_state":"checkpointed" if isinstance(error,KeyboardInterrupt) else "failed",
                  "error":f"{type(error).__name__}: {error}"},"supervisor exception")
            raise
        finally:
            stop.set();timer.join()
            if previous is not None:
                signal.signal(signal.SIGTERM,previous)


def campaign(root: Path):
    return acceptance(root,REPO)


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
