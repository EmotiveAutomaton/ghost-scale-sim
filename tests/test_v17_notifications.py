"""Known event delivery, failure, acknowledgement and restart behavior."""
import json
import os
import sys
import time
from pathlib import Path
import pytest
from runners.watch_v17_notifications import Notifications
from ghostscale.validation.soundingline.v16.records import write

def setup(tmp_path, snippet="import sys; print(sys.stdin.read())"):
    root=tmp_path/"notifier"
    root.mkdir()
    (root/"ARMED").write_text("explicit handoff")
    config=dict(command=[sys.executable,"-c",snippet],cwd=str(tmp_path))
    return Notifications(tmp_path/"science",root,config)

def event(identity="complete",kind="run_complete"):
    return dict(id=identity,kind=kind,payload={},at="fixture")

def test_no_healthy_wake_and_completed_event_delivered_once(tmp_path):
    d=setup(tmp_path)
    d.tick([event(kind="heartbeat")])
    assert d.child is None
    d.tick([event()]);d.child.wait(timeout=10);d.tick([event()])
    assert d.state["delivered"]==["complete"] and len(d.state["attempts"])==1
    assert str(tmp_path/"science") in (d.root/"review-1.jsonl").read_text()
    restarted=Notifications(d.run_root,d.root,d.config)
    restarted.tick([event()])
    assert restarted.child is None

def test_prior_user_review_acknowledged_without_new_model_turn(tmp_path):
    d=setup(tmp_path)
    d.state["acknowledged"]=["old"];d.save()
    d.tick([event("old")])
    assert d.child is None and not d.state["attempts"]

def test_failures_are_bounded_and_preserved(tmp_path):
    d=setup(tmp_path,"import sys; sys.exit(7)")
    for _ in range(3):
        if d.state["attempts"]:d.state["attempts"][-1]["started_epoch"]=0
        d.tick([event()]);d.child.wait(timeout=10);d.tick([event()])
    assert d.state["failed"]==["complete"]
    d.tick([event()])
    assert len(d.state["attempts"])==3

def test_restart_preserves_a_live_delivery(tmp_path):
    d=setup(tmp_path)
    d.state["attempts"]=[dict(event_ids=["complete"],pid=os.getpid(),started_epoch=0)]
    d.save()
    restarted=Notifications(d.run_root,d.root,d.config)
    restarted.tick([event()])
    assert restarted.child is None and "exit_code" not in restarted.state["attempts"][0]

def test_missing_executable_records_failure_without_repeated_launch(tmp_path):
    d=setup(tmp_path);d.config["command"]=[str(tmp_path/"absent.exe")]
    d.tick([event()]);d.tick([event()])
    assert d.state["failed"]==["complete"] and len(d.state["attempts"])==1

def test_shared_session_resume_is_refused(tmp_path):
    d=setup(tmp_path);d.config["command"]=[sys.executable,"resume","a-thread","-"]
    with pytest.raises(ValueError,match="reopen"):d.tick([event()])

def test_changed_pinned_executable_is_never_run(tmp_path):
    d=setup(tmp_path);d.config["executable_sha256"]="wrong"
    d.tick([event()])
    assert d.child is None and d.state["failed"]==["complete"]
