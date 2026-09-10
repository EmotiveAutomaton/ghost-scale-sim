"""Real child-process interruption; no mocked successful runtime receipts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from ghostscale.validation.soundingline.v16.runtime import REPO, supervisor
from ghostscale.validation.soundingline.v16.reaggregate import regenerate


def test_interrupted_packet_resumes_without_duplicates_or_changed_clocks(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    shutil.copyfile(REPO / "results/v16/CAMPAIGN.json", root / "CAMPAIGN.json")
    fixture_campaign=json.loads((root/"CAMPAIGN.json").read_bytes())
    fixture_campaign["campaign_id"]="fixture-"+uuid.uuid4().hex
    (root/"CAMPAIGN.json").write_text(json.dumps(fixture_campaign))
    accepted_bytes = (root / "CAMPAIGN.json").read_bytes()
    command = [sys.executable, "-B", "-m", "runners.run_v16",
               "--root", str(root), "--fixture-units", "64"]
    environment = dict(os.environ, PYTHONPATH=str(REPO), OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1")
    with (tmp_path / "interrupted.log").open("wb") as output:
        child = subprocess.Popen(command + ["--stage", "pilot"], cwd=REPO,
                                 env=environment, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                completed = list((root / "native-fixture-1/units").glob("*_points.json"))
                if completed:
                    break
                assert child.poll() is None, (tmp_path / "interrupted.log").read_text()
                time.sleep(0.01)
            else:
                raise AssertionError("child made no executable progress")
            child.terminate()  # Owned process handle; a real abrupt interruption.
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    saved = {path.name: path.read_bytes()
             for path in (root / "native-fixture-1/units").glob("*_points.json")}
    assert 0 < len(saved) < 64
    interrupted_status = json.loads((root / "RUNNER_STATUS.json").read_bytes())
    assert interrupted_status["execution_state"] != "completed"
    packet_bytes = (root / "packets/native-fixture-1.json").read_bytes()
    other=tmp_path/"other-checkout"
    other.mkdir()
    shutil.copyfile(root/"CAMPAIGN.json",other/"CAMPAIGN.json")
    with supervisor(other,"after-owned-process-death"):
        assert json.loads((other/"RUNNER_STATUS.json").read_bytes())["pid"]==os.getpid()
    resumed = subprocess.run(command + ["--stage", "resume"], cwd=REPO, env=environment,
                             capture_output=True, text=True, timeout=40)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (root / "CAMPAIGN.json").read_bytes() == accepted_bytes
    assert (root / "packets/native-fixture-1.json").read_bytes() == packet_bytes
    assert regenerate(root / "native-fixture-1")["n"] == 64
    for name, payload in saved.items():
        assert (root / "native-fixture-1/units" / name).read_bytes() == payload


def test_second_supervisor_cannot_take_status_ownership(tmp_path):
    import pytest
    with supervisor(tmp_path, "first"):
        before = (tmp_path / "RUNNER_STATUS.json").read_bytes()
        with pytest.raises(RuntimeError, match="another supervisor"):
            with supervisor(tmp_path, "second"):
                pass
        assert (tmp_path / "RUNNER_STATUS.json").read_bytes() == before
