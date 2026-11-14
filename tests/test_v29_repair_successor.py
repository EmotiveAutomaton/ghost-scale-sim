import pytest
from ghostscale.validation.soundingline.v29.common import read,write,digest
from runners import run_v29_repair_successor as s

def test_transfer_carries_actual_cost_once_and_rejects_reactivation(tmp_path,monkeypatch):
    previous=tmp_path/'previous';c=tmp_path/'next'
    write(previous/'ACCEPTANCE.json',dict(id='previous'))
    write(previous/'QUEUE_COMPLETE.json',dict(state='FINITE_END'))
    write(previous/'QUEUE_STATUS.json',dict(pid=99))
    write(previous/'accounting/science.json',dict(cpu_seconds=100))
    write(previous/'accounting/q2-h2-reservation.json',dict(reserved_cpu_seconds=2400,cpu_seconds=0))
    write(c/'ACCEPTANCE.json',dict(primary_campaign=str(previous),primary_acceptance_sha256=digest(previous/'ACCEPTANCE.json'),cpu_limits={'reserve':86400}))
    monkeypatch.setattr(s,'alive',lambda pid:False)
    seen=[];monkeypatch.setattr(s.queue,'run',lambda root:seen.append(root))
    s.activate(c)
    assert seen==[c]
    assert read(c/'accounting/prior-protected.json')['cpu_seconds']==100
    assert read(previous/'accounting/q2-h2-reservation.json')['reserved_cpu_seconds']==0
    with pytest.raises(ValueError,match='already activated'):s.activate(c)
