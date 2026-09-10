"""Actual general-CLI termination and continuation of a finite frozen fixture."""
import json
import os
import subprocess
import sys
import time
from .runtime import REPO, PACKAGE, freeze
from .records import read, write, file_digest, now
from .expansion_interruption import fixture_campaign
from .operational_entry import job
from .operational_queue import freeze_plan

FILES = ["operational_queue.py", "operational_entry.py", "active_binding.py", "operational_fixture.py", "operational_interruption.py"]


def stable_bytes(path):
    for attempt in range(20):
        try:
            return path.read_bytes()
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def control(directory):
    root = directory/"campaign"
    if root.exists():
        raise ValueError("prior operational fixture retained; a new attempt needs a new directory")
    fixture_campaign(root)
    files = [PACKAGE/name for name in FILES]+[REPO/"runners/run_v16.py", REPO/"tests/test_v16_operational_interruption.py"]
    packet = freeze(root, "operations-runtime-fixture-1", files, {"scope": "known operational fixture", "complete_units": 192})
    definition = {"jobs": [job("known-construction", "discovery", 192, "KNOWN_ADMISSION.json",
        "expansion-fixture-K01/COMPLETION.json", [], "Verify actual general-CLI continuation", "operational_fixture")]}
    freeze_plan(root, definition)
    write(root/"KNOWN_ADMISSION.json", {"instrument_state": "valid", "scope": "known fixture using already validated original construction"})
    command = [sys.executable, "-B", "-m", "runners.run_v16", "--root", str(root)]
    options = {"cwd": REPO, "env": dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    base = root/"expansion-fixture-K01/K01"
    with (directory/"interrupt.log").open("wb") as log:
        child = subprocess.Popen(command+["--stage", "discovery"], stdout=log, stderr=log, **options)
        try:
            end = time.monotonic()+45
            while time.monotonic() < end:
                if list((base/"units").glob("*_points.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("general CLI exited before first fixture unit")
                time.sleep(.01)
            else:
                raise ValueError("general CLI made no bounded fixture progress")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
    before = json.loads(stable_bytes(root/"RUNNER_STATUS.json"))
    write(directory/"INTERRUPTED_STATUS.json", before)
    saved = {path.relative_to(root).as_posix(): stable_bytes(path) for path in root.rglob("*.json") if path.name != "RUNNER_STATUS.json"}
    with (directory/"resume.log").open("wb") as log:
        resumed = subprocess.run(command+["--stage", "resume"], stdout=log, stderr=log, timeout=120, **options)
    if resumed.returncode:
        raise ValueError("general CLI did not resume; retain its resume log")
    completed = read(root/"expansion-fixture-K01/COMPLETION.json")
    checks = {"real_general_cli_interrupted": child.returncode != 0 and bool(saved),
        "no_false_completion_at_interruption": before["execution_state"] != "completed",
        "saved_units_predictions_source_locks_and_clocks_unchanged": all(stable_bytes(root/name) == content for name,content in saved.items()),
        "finite_bound_reached_without_duplicate_units": completed["completed_condition_records"] == 192 and len(list((base/"units").glob("*_points.json"))) == 192,
        "job_completion_does_not_close_campaign": completed["campaign_complete"] is False,
        "same_active_job_resumed": read(root/"operations/ACTIVE_JOB.json")["job_id"] == "known-construction"}
    result = {"instrument_state": "valid" if all(checks.values()) else "failed", "checks": checks,
        "packet_hash": packet["packet_hash"], "recorded_at": now(), "scope": "known operational fixture; not scientific expansion makers",
        "files": {path.relative_to(directory).as_posix(): file_digest(path) for path in directory.rglob("*") if path.is_file() and path.suffix in {".json", ".tmp"}}}
    write(directory/"RECEIPT.json", result)
    return result
