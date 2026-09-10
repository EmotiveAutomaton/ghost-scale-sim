"""X08 executes the actual B02 search/follow-up process, interrupts and resumes it."""
import os
import subprocess
import sys
import time
import uuid
from .records import read, write, file_digest, now
from .runtime import REPO


def control(directory, source_identity):
    receipt = directory/"RECEIPT.json"
    if receipt.exists():
        result = read(receipt)
        if result["source_identity"] != source_identity:
            raise ValueError("archive interruption source changed")
        for name, expected in result["files"].items():
            if file_digest(directory/name) != expected:
                raise ValueError("archive interruption evidence changed")
        return result
    root = directory/"campaign"
    if root.exists():
        raise ValueError("incomplete archive interruption attempt retained")
    campaign = read(REPO/"results/v16/CAMPAIGN.json")
    campaign["campaign_id"] = "fixture-"+uuid.uuid4().hex
    write(root/"CAMPAIGN.json", campaign)
    clock_bytes = (root/"CAMPAIGN.json").read_bytes()
    command = [sys.executable, "-B", "-m", "runners.run_v16_archive", "--fixture", "--root", str(root)]
    env = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    output = root/"archive-known-fixture-1"
    options = {"cwd": REPO, "env": env, "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    with (directory/"interrupt.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, **options)
        try:
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                if list((output/"units").glob("*_points.json")):
                    break
                if child.poll() is not None:
                    raise ValueError("archive fixture exited before first retained journal")
                time.sleep(0.01)
            else:
                raise ValueError("archive fixture made no bounded progress")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
    before = read(root/"RUNNER_STATUS.json")
    write(directory/"INTERRUPTED_STATUS.json", before)
    saved = {path.relative_to(output).as_posix(): path.read_bytes()
             for folder in ["units", "private"] for path in (output/folder).rglob("*.json")}
    packet_path = root/"packets/archive-known-fixture-1.json"
    packet_bytes = packet_path.read_bytes()
    with (directory/"resume.log").open("wb") as log:
        resumed = subprocess.run(command+["--resume"], stdout=log, stderr=log, timeout=60, **options)
    completion = read(output/"COMPLETION.json")
    checks = {"owned_archive_child_interrupted": child.returncode != 0 and bool(saved),
        "interruption_not_completion": before["execution_state"] != "completed" and not before.get("campaign_complete", False),
        "resume_succeeds_without_reprepare": resumed.returncode == 0 and packet_path.read_bytes() == packet_bytes,
        "accepted_clock_unchanged": clock_bytes == (root/"CAMPAIGN.json").read_bytes(),
        "saved_candidate_journals_and_units_unchanged": all((output/name).read_bytes() == payload for name,payload in saved.items()),
        "finite_search_and_followup_complete": completion["candidate_maker_records"] == 24 and completion["followup_maker_condition_records"] == 48,
        "archive_job_never_implies_campaign_completion": not completion["campaign_complete"]}
    result = {"source_identity": source_identity, "checks": checks,
        "instrument_state": "valid" if all(checks.values()) else "failed", "completed_at": now(),
        "files": {path.relative_to(directory).as_posix(): file_digest(path) for path in sorted(directory.rglob("*"))
                  if path.is_file() and path.suffix in {".json", ".tmp"}},
        "temporary_write_scope": "incomplete files left by actual termination are retained and checked separately from completed unit outputs"}
    write(receipt, result)
    return result
