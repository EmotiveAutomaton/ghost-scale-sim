"""Cross-checkout OS ownership, clock immutability and actual timed heartbeats."""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import pytest
from ghostscale.validation.soundingline.v16.runtime import REPO,supervisor,campaign

def fixture_campaign(root,identity=None):
    root.mkdir()
    value=json.loads((REPO/"results/v16/CAMPAIGN.json").read_bytes())
    value["campaign_id"]=identity or "fixture-"+uuid.uuid4().hex
    (root/"CAMPAIGN.json").write_text(json.dumps(value))
    return value["campaign_id"]

def test_campaign_lock_rejects_a_real_second_process_at_another_root(tmp_path):
    first=tmp_path/"first";second=tmp_path/"second"
    identity=fixture_campaign(first);fixture_campaign(second,identity)
    env=dict(os.environ,PYTHONPATH=str(REPO),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    with supervisor(first,"fixture"):
        before=(first/"RUNNER_STATUS.json").read_bytes()
        attempted=subprocess.run([sys.executable,"-B","-m","runners.run_v16","--stage","preflight","--root",str(second)],
            cwd=REPO,env=env,capture_output=True,text=True,timeout=20)
        assert attempted.returncode!=0
        assert "another supervisor owns this campaign" in attempted.stderr
        assert not (second/"RUNNER_STATUS.json").exists()
        assert (first/"RUNNER_STATUS.json").read_bytes()==before
        with pytest.raises(RuntimeError,match="another supervisor"):
            with supervisor(second,"same-process-second-root"):
                pass
    with supervisor(second,"owner-released"):
        assert (second/"RUNNER_STATUS.json").exists()

def test_heartbeat_advances_during_long_work_without_inventing_units(tmp_path):
    with supervisor(tmp_path,"fixture",heartbeat_interval=0.02) as heartbeat:
        heartbeat(completed_units=3,planned_units=7)
        initial=json.loads((tmp_path/"RUNNER_STATUS.json").read_bytes())
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            status=json.loads((tmp_path/"RUNNER_STATUS.json").read_bytes())
            if status["heartbeat"]!=initial["heartbeat"]:
                break
            time.sleep(0.01)
        assert status["heartbeat"]!=initial["heartbeat"]
        assert status["completed_units"]==3
        assert status["last_unit_report_at"]==initial["last_unit_report_at"]
        assert status["supervisor_elapsed_seconds"]>initial["supervisor_elapsed_seconds"]
        assert status["supervisor_cpu_seconds"]>=initial["supervisor_cpu_seconds"]
        assert not status["unit_loop_complete"]
        heartbeat(completed_units=7)
        assert json.loads((tmp_path/"RUNNER_STATUS.json").read_bytes())["unit_loop_complete"]

def test_acceptance_lock_rejects_clock_and_manifest_changes(tmp_path):
    root=tmp_path/"root";fixture_campaign(root)
    (root/"COMMISSION_MANIFEST.json").write_text('{"cards":[]}')
    campaign(root)
    accepted=(root/"CAMPAIGN.json").read_bytes()
    anchor=(root/"ACCEPTANCE_LOCK.json").read_bytes()
    changed=json.loads(accepted);changed["deadline"]="2099-01-01T00:00:00Z"
    (root/"CAMPAIGN.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="immutable acceptance"):
        campaign(root)
    (root/"CAMPAIGN.json").write_bytes(accepted)
    (root/"COMMISSION_MANIFEST.json").write_text('{"cards":["invented"]}')
    with pytest.raises(ValueError,match="immutable acceptance"):
        campaign(root)
    assert (root/"ACCEPTANCE_LOCK.json").read_bytes()==anchor

def test_existing_packet_clocks_anchor_acceptance_migration(tmp_path):
    root=tmp_path/"root";fixture_campaign(root)
    value=json.loads((root/"CAMPAIGN.json").read_bytes())
    (root/"packets").mkdir()
    (root/"packets/example.json").write_text(json.dumps({"accepted_at":value["accepted_at"],
        "deadline":"2000-01-01T00:00:00Z","identity":{"commission_hash":value["commission_sha256"]}}))
    with pytest.raises(ValueError,match="existing packet clock"):
        campaign(root)
    assert not (root/"ACCEPTANCE_LOCK.json").exists()
