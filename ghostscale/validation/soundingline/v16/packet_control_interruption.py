"""Terminate and resume the actual source-control CLI on known retained inputs."""
import os
import subprocess
import sys
import time
from .runtime import REPO
from .records import read, write, file_digest, now
from .expansion_interruption import fixture_campaign


def control(directory, source_root, source_name="constructor-expansion-1"):
    root = directory/"campaign"
    if root.exists():
        raise ValueError("prior control interruption evidence retained; use a new attempt")
    fixture_campaign(root)
    command = [sys.executable, "-B", "-m", "runners.control_v16_expansion", "--root", str(root),
               "--source-root", str(source_root), "--source-name", source_name]
    options = {"cwd": REPO, "env": dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
               "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    with (directory/"interrupt.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, **options)
        try:
            until = time.monotonic()+90
            while time.monotonic() < until:
                if list((root/"constructor-controls-1").glob("*/conditions/*/RECEIPT.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("source-control CLI exited before first condition; see interrupt.log")
                time.sleep(.005)
            else:
                raise ValueError("source-control CLI made no bounded progress")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    before = read(root/"RUNNER_STATUS.json")
    write(directory/"INTERRUPTED_STATUS.json", before)
    saved = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*.json") if path.name != "RUNNER_STATUS.json"}
    with (directory/"resume.log").open("wb") as log:
        resumed = subprocess.run(command+["--resume"], stdout=log, stderr=log, timeout=300, **options)
    if resumed.returncode:
        raise ValueError("source-control CLI resume failed; see resume.log")
    completion = read(root/"constructor-controls-1/COMPLETION.json")
    spec = read(root/"packets/constructor-controls-1.json")["identity"]["design"]
    checks = {"actual_cli_terminated": child.returncode != 0,
              "no_false_completion": before["execution_state"] != "completed",
              "saved_units_partial_attempts_clocks_and_locks_unchanged": all((root/name).read_bytes() == data for name,data in saved.items()),
              "finite_all_condition_bound_reached": completion["condition_controls"] == spec["planned_condition_controls"],
              "completed_source_controls_valid": completion["instrument_state"] == "valid",
              "job_does_not_close_campaign": completion["campaign_complete"] is False}
    result = {"execution_state": "completed", "instrument_state": "valid" if all(checks.values()) else "failed",
              "checks": checks, "completed_at": now(), "packet_hash": completion["packet_hash"],
              "completed_condition_controls": completion["condition_controls"],
              "source_packet_hash": completion["source_packet_hash"],
              "scope": "Actual control CLI on retained known fixtures; no scientific makers generated",
              "files": {path.relative_to(directory).as_posix(): file_digest(path) for path in directory.rglob("*.json")}}
    write(directory/"RECEIPT.json", result)
    return result
