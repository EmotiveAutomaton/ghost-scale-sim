from ghostscale.validation.soundingline.v27.common import write
from runners import queue_v27 as queue


def test_long_preparation_counts_until_terminal_even_if_pid_is_reused(tmp_path,monkeypatch):
    campaign=tmp_path/'v27'
    write(campaign/'COUNTED_PREPARATION.json',dict(batches=['particle-check']))
    status=campaign/'particle-check/STATUS.json'
    write(status,dict(state='RUNNING',pid=42))
    monkeypatch.setattr(queue,'alive',lambda pid:pid==42)
    assert queue.outside(campaign)==1
    write(status,dict(state='COMPLETE',pid=42),immutable=False)
    assert queue.outside(campaign)==0


def test_controller_restart_preserves_fresh_host_backoff(tmp_path):
    samples=[[1000.,.95],[1030.,.94]]
    write(tmp_path/'HOST_LOAD.json',dict(samples=samples,worker_limit=2))
    assert queue.resume_load(tmp_path,clock=1040.)==(2,samples)
    assert queue.resume_load(tmp_path,clock=1200.)==(4,[])
