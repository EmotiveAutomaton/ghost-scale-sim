import os
import subprocess
import sys
import pytest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v23.common import write


def test_replay_compares_scientific_bytes_but_excludes_timing(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    for path in (a, b):
        write(path/'SUMMARY.json', {'same': True})
    write(a/'fit-cost.json', {'cpu': 1})
    write(b/'fit-cost.json', {'cpu': 2})
    assert runtime.compare(a, b)['state'] == 'passed'
    write(b/'SUMMARY.json', {'same': False}, immutable=False)
    with pytest.raises(ValueError, match='replay differs'): runtime.compare(a, b)


def test_atomic_record_write_retries_only_sharing_and_never_overwrites_immutable(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v23 import common
    actual=common.os.replace;attempts=[]
    def replace(a,b):
        attempts.append(1)
        if len(attempts)<3: raise PermissionError('transient sharing fixture')
        return actual(a,b)
    monkeypatch.setattr(common.os,'replace',replace)
    common.write(tmp_path/'status.json',{'ok':True},immutable=False)
    assert len(attempts)==3 and common.read(tmp_path/'status.json')['ok']
    common.write(tmp_path/'evidence.json',{'original':True})
    with pytest.raises(FileExistsError): common.write(tmp_path/'evidence.json',{'original':False})
    assert common.read(tmp_path/'evidence.json')['original']


@pytest.mark.skipif(os.name != 'nt', reason='native Windows CPU handle adapter')
def test_owned_child_cpu_remains_readable_after_exit(tmp_path):
    child = subprocess.Popen([sys.executable, '-B', '-m', 'runners.fixture_v23',
        '--out', str(tmp_path/'fixture.json'), '--accounting-handshake'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    handle = runtime.attach_child(child, tmp_path, 'fixture')
    child.wait(timeout=30)
    assert child.returncode == 0
    try:
        assert runtime.child_cpu(handle) > 0
    finally:
        runtime.ctypes.windll.kernel32.CloseHandle(runtime.ctypes.c_void_p(handle))
