"""Known actual confirmation CLI interruption and frozen continuation."""
import os
import subprocess
import sys
import time
from .runtime import REPO
from .records import read, write, file_digest, now
from .expansion_interruption import fixture_campaign


def control(directory):
    root = directory/"campaign"
    if root.exists():
        raise ValueError("prior confirmation fixture retained")
    fixture_campaign(root)
    command = [sys.executable, "-B", "-m", "runners.confirm_v16", "--root", str(root), "--fixture"]
    options = {"cwd": REPO, "env": dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    with (directory/"interrupt.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, **options)
        try:
            until = time.monotonic()+60
            while time.monotonic() < until:
                if list((root/"confirmation/K01/units").glob("*_points.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("known confirmation CLI exited before first unit; see interrupt.log")
                time.sleep(.01)
            else:
                raise ValueError("known confirmation CLI made no bounded progress")
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
        result = subprocess.run(command+["--resume"], stdout=log, stderr=log, timeout=180, **options)
    if result.returncode:
        raise ValueError("known confirmation resume failed; see resume.log")
    completed = read(root/"confirmation/COMPLETION.json")
    finished_files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*.json") if path.name != "RUNNER_STATUS.json"}
    with (directory/"completed-resume.log").open("wb") as log:
        repeated = subprocess.run(command+["--resume"], stdout=log, stderr=log, timeout=180, **options)
    checks = {"actual_confirmation_cli_interrupted": child.returncode != 0,
        "partial_work_did_not_claim_completion": before["execution_state"] != "completed",
        "all_saved_inputs_units_and_clocks_preserved": all((root/name).read_bytes() == value for name, value in saved.items()),
        "full_known_allocation_and_source_controls": completed["condition_records"] == 192 and completed["source_controls_completed"],
        "exactly_two_frozen_primary_claims": len(completed["claims"]) == 2,
        "completed_resume_preserves_all_prior_evidence": repeated.returncode == 0 and all((root/name).read_bytes() == value for name, value in finished_files.items()),
        "job_completion_does_not_close_campaign": completed["campaign_complete"] is False}
    receipt = {"execution_state": "completed", "instrument_state": "valid" if all(checks.values()) else "failed",
        "checks": checks, "completed_at": now(), "packet_hash": completed["packet_hash"],
        "files": {path.relative_to(directory).as_posix(): file_digest(path) for path in directory.rglob("*.json")},
        "scope": "Known confirmation construction/self-monitor fixtures; no scientific reserve data opened"}
    write(directory/"RECEIPT.json", receipt)
    return receipt
