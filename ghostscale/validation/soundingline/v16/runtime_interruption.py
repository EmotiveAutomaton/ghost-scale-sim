"""Real process termination/resumption with retained clocks and immutable units."""
import os
import subprocess
import sys
import time
import uuid
from .records import read, write, file_digest, now
from .runtime import REPO


def interruption_control(directory, source_packets):
    receipt = directory/"RECEIPT.json"
    if receipt.exists():
        saved = read(receipt)
        if saved["source_packets"] != source_packets:
            raise ValueError("interruption source binding changed")
        for relative, expected in saved["files"].items():
            if file_digest(directory/relative) != expected:
                raise ValueError("retained interruption evidence changed")
        return saved
    root = directory/"campaign"
    if (root/"CAMPAIGN.json").exists():
        raise ValueError("incomplete interruption calibration retained; explicitly diagnose before another attempt")
    campaign = read(REPO/"results/v16/CAMPAIGN.json")
    campaign["campaign_id"] = "fixture-"+uuid.uuid4().hex
    write(root/"CAMPAIGN.json", campaign)
    original_clock = (root/"CAMPAIGN.json").read_bytes()
    command = [sys.executable, "-B", "-m", "runners.v16_interrupt_fixture", "--root", str(root)]
    env = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    output = root/"interruption-native-fixture"
    with (directory/"interruption.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                saved_paths = list((output/"units").glob("*_points.json"))
                if saved_paths:
                    break
                if child.poll() is not None:
                    raise ValueError("fixture child stopped before a saved native unit")
                time.sleep(0.01)
            else:
                raise ValueError("bounded fixture produced no retained unit")
            child.terminate()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    saved_units = {path.name: path.read_bytes() for path in (output/"units").glob("*_points.json")}
    interrupted = read(root/"RUNNER_STATUS.json")
    write(directory/"INTERRUPTED_STATUS.json", interrupted)
    packet_path = root/"packets/interruption-native-fixture.json"
    original_packet = packet_path.read_bytes()
    with (directory/"resume.log").open("wb") as log:
        resumed = subprocess.run(command+["--resume"], cwd=REPO, env=env, stdout=log, stderr=log, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    final = read(root/"RUNNER_STATUS.json")
    aggregate = read(output/"AGGREGATE.json")
    checks = {"actual_owned_child_was_interrupted": child.returncode != 0 and 0 < len(saved_units) < 64,
        "interrupted_job_not_completed": interrupted["execution_state"] != "completed" and not interrupted.get("campaign_complete", False),
        "resumed_without_second_prepare": resumed.returncode == 0 and packet_path.read_bytes() == original_packet,
        "acceptance_clock_unchanged": (root/"CAMPAIGN.json").read_bytes() == original_clock,
        "completed_unit_bytes_unchanged": all((output/"units"/name).read_bytes() == payload for name, payload in saved_units.items()),
        "all_unique_units_reaggregated": aggregate["n"] == 64 and len(list((output/"units").glob("*_points.json"))) == 64,
        "job_complete_still_not_campaign_complete": final["execution_state"] == "completed" and not final["campaign_complete"]}
    files = {str(path.relative_to(directory)).replace("\\", "/"): file_digest(path)
             for path in sorted(directory.rglob("*.json"))}
    result = {"checks": checks, "instrument_state": "valid" if all(checks.values()) else "failed",
        "source_packets": source_packets, "scope": "actual shared runtime and native checkpoint control; not a resumption of scientific discovery",
        "interrupted_completed_units": len(saved_units), "files": files, "completed_at": now()}
    write(receipt, result)
    return result
