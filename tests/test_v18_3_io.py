from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v18_3 import io


def test_atomic_sharing_retry_preserves_immutable_evidence(tmp_path,monkeypatch):
    path=tmp_path/'status.json';io.write(path,dict(step=0));original=io.os.replace;attempts=[0]
    def sharing(source,destination):
        attempts[0]+=1
        if attempts[0]<3:raise PermissionError('planted Windows sharing violation')
        return original(source,destination)
    monkeypatch.setattr(io.os,'replace',sharing)
    io.write(path,dict(step=1),immutable=False)
    assert attempts[0]==3 and io.read(path)==dict(step=1)
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ValueError,match='immutable'):io.write(path,dict(step=2))
    assert io.read(path)==dict(step=1)


def test_queue_drain_distinguishes_failed_packets(tmp_path,monkeypatch):
    from runners import watch_v18_3 as watcher
    root=tmp_path/'failed';root.mkdir();source=tmp_path/'source';source.mkdir()
    io.write(tmp_path/'ACCEPTANCE.json',dict(report_start=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
    io.write(root/'PLAN.json',dict(fixture=True));queue=tmp_path/'QUEUE.json'
    io.write(queue,dict(jobs=[dict(root=str(root),source=str(source),plan_sha256=io.file_digest(root/'PLAN.json'))]))
    class Failed:
        pid=999999;returncode=1
        def poll(self):return 1
    monkeypatch.setattr(watcher.subprocess,'Popen',lambda *a,**k:Failed())
    watcher.supervise(tmp_path,queue,Path('fixture-python'))
    result=io.read(tmp_path/'QUEUE-STATUS.json')
    assert result['state']=='drained_with_failures' and result['failed_packets']==['failed']
