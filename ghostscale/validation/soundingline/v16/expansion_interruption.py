"""Interrupt actual expansion workers and preserve their first saved predictions."""
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from .runtime import REPO
from .records import read, write, file_digest, now


def fixture_campaign(root):
    accepted = read(REPO/"results/v16/CAMPAIGN.json")
    started = datetime.now(timezone.utc)
    accepted.update(campaign_id="fixture-"+uuid.uuid4().hex, accepted_at=started.isoformat(),
        deadline=(started+timedelta(hours=120)).isoformat(),
        clock_scope="separately identified operational fixture; never modifies or extends scientific campaign clocks")
    write(root/"CAMPAIGN.json", accepted)


def control(directory, card):
    root = directory/"campaign"
    if root.exists():
        raise ValueError("previous interruption fixture retained; choose a new attempt directory")
    fixture_campaign(root)
    clock = (root/"CAMPAIGN.json").read_bytes()
    command = [sys.executable, "-B", "-m", "runners.run_v16_expansion", "--root", str(root), "--fixture", card]
    options = {"cwd": REPO, "env": dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    base = root/("expansion-fixture-"+card)/card
    with (directory/"interrupt.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, **options)
        try:
            deadline = time.monotonic()+45
            while time.monotonic() < deadline:
                saved = sorted((base/"units").glob("*_points.json"))
                if saved:
                    break
                if child.poll() is not None:
                    raise ValueError("expansion child exited before a retained unit")
                time.sleep(.01)
            else:
                raise ValueError("expansion fixture made no bounded progress")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
    before = read(root/"RUNNER_STATUS.json")
    write(directory/"INTERRUPTED_STATUS.json", before)
    retained = {path.relative_to(base).as_posix(): path.read_bytes()
        for folder in ["public", "private", "predictions", "units"] for path in (base/folder).glob("*.json")}
    lock = root/"packets"/("expansion-fixture-"+card+".json")
    lock_bytes = lock.read_bytes()
    with (directory/"resume.log").open("wb") as log:
        resumed = subprocess.run(command+["--resume"], stdout=log, stderr=log, timeout=120, **options)
    if resumed.returncode:
        raise ValueError("actual expansion resume failed; retained resume.log")
    completion = read(base/"COMPLETION.json")
    checks = {"actual_child_interrupted": child.returncode != 0 and bool(retained),
        "interruption_not_complete": before["execution_state"] != "completed",
        "same_packet_and_clock": lock.read_bytes() == lock_bytes and (root/"CAMPAIGN.json").read_bytes() == clock,
        "all_saved_bytes_preserved": all((base/name).read_bytes() == content for name,content in retained.items()),
        "independent_physics_and_summary_passed": completion["execution_state"] == "completed" and completion["instrument_state"] == "valid",
        "no_campaign_completion_from_job": completion["campaign_complete"] is False}
    result = {"instrument_state": "valid" if all(checks.values()) else "failed", "checks": checks,
        "card_id": card, "completed_at": now(), "packet_hash": completion["packet_hash"],
        "files": {path.relative_to(directory).as_posix(): file_digest(path) for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".tmp"}}, "scientific_maker_count": "not applicable; operational fixture"}
    write(directory/"RECEIPT.json", result)
    return result
