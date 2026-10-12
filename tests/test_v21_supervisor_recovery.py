from types import SimpleNamespace
import pytest
from runners import watch_v21 as S

def test_disappearing_atomic_file_is_not_supervisor_failure():
    class Gone:
        def is_file(self):return True
        def stat(self):raise FileNotFoundError('atomic rename')
    class Stable:
        def is_file(self):return True
        def stat(self):return SimpleNamespace(st_size=17)
    assert S.artifact_size(SimpleNamespace(rglob=lambda _:iter([Gone(),Stable()])))==17

def test_non_transient_error_is_not_hidden():
    class Denied:
        def is_file(self):raise PermissionError('not a disappearing temporary')
    with pytest.raises(PermissionError):S.artifact_size(SimpleNamespace(rglob=lambda _:iter([Denied()])))

def test_live_previous_owner_refuses_recovery(monkeypatch,tmp_path):
    monkeypatch.setattr(S,'native_identity',lambda _:dict(alive=True))
    with pytest.raises(ValueError,match='previous owner'):S.adopt_jobs(tmp_path,dict(previous_owner_pid=9),[])

def test_changed_queue_refuses_recovery(monkeypatch,tmp_path):
    monkeypatch.setattr(S,'native_identity',lambda _:None)
    monkeypatch.setattr(S,'file_digest',lambda _:'changed')
    with pytest.raises(ValueError,match='queue changed'):S.adopt_jobs(tmp_path,dict(previous_owner_pid=9,queue_sha256='bound'),[])

def test_exited_retained_owner_handle_is_not_live(monkeypatch,tmp_path):
    monkeypatch.setattr(S,'native_identity',lambda _:dict(alive=False))
    monkeypatch.setattr(S,'file_digest',lambda _:'bound')
    assert S.adopt_jobs(tmp_path,dict(previous_owner_pid=9,queue_sha256='bound',workers=[]),[])=={}

def test_idle_census_keeps_active_and_dispatch_storage_guards():
    assert S.artifact_check_due(float('inf'))
    assert not S.artifact_check_due(3599)
    assert S.artifact_check_due(3600)
    for key in ('active','dispatch'):
        assert not S.artifact_check_due(59,**{key:True})
        assert S.artifact_check_due(60,**{key:True})
