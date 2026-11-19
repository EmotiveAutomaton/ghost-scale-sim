import json
from pathlib import Path
import pytest
from runners import archive_storage as a


def fixture(tmp_path):
    workspace=tmp_path/'workspace';dest=tmp_path/'archive';state=workspace/'.local/migration'
    src=workspace/'.local/v31/jobs/one/output';src.mkdir(parents=True);dest.mkdir()
    (src/'nested').mkdir();(src/'nested/evidence.bin').write_bytes(b'original evidence\0'*200)
    (src/'summary.json').write_text('{"passed":true}\n')
    return workspace,dest,state,src,src.relative_to(workspace)


def test_verified_copy_preserves_original_path_and_is_idempotent(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path);sha=a.digest(src/'nested/evidence.bin')
    row=a.relocate(workspace,dest,relative,state)
    assert row['stage']=='RECLAIMED' and src.is_symlink()
    assert a.digest(src/'nested/evidence.bin')==sha
    assert a.relocate(workspace,dest,relative,state)==row
    assert not list(src.parent.glob('*.archive-retired-*'))


def test_unowned_destination_cannot_be_overwritten(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path)
    target=dest/'retained'/relative;target.mkdir(parents=True);(target/'owner.txt').write_text('keep')
    with pytest.raises(ValueError,match='unowned'):a.relocate(workspace,dest,relative,state)
    assert not src.is_symlink() and (target/'owner.txt').read_text()=='keep'


def test_changed_source_keeps_local_evidence(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path)
    def change(**kw):
        if kw['file']=='summary.json':(src/'summary.json').write_text('changed after copy')
    with pytest.raises(ValueError,match='source changed'):a.relocate(workspace,dest,relative,state,change)
    assert src.is_dir() and not src.is_symlink()
    assert (src/'summary.json').read_text()=='changed after copy'


def test_link_failure_keeps_verified_and_retired_copies_for_resume(tmp_path,monkeypatch):
    workspace,dest,state,src,relative=fixture(tmp_path);real=a.os.symlink
    monkeypatch.setattr(a.os,'symlink',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('simulated link failure')))
    with pytest.raises(OSError,match='simulated'):a.relocate(workspace,dest,relative,state)
    assert list(src.parent.glob('*.archive-retired-*'))
    monkeypatch.setattr(a.os,'symlink',real)
    row=a.relocate(workspace,dest,relative,state)
    assert row['stage']=='RECLAIMED' and (src/'summary.json').read_text()=='{"passed":true}\n'


def test_tampered_archive_after_interrupted_cutover_prevents_deletion(tmp_path,monkeypatch):
    workspace,dest,state,src,relative=fixture(tmp_path);real=a.os.symlink
    monkeypatch.setattr(a.os,'symlink',lambda *args,**kwargs:(_ for _ in ()).throw(OSError('simulated')))
    with pytest.raises(OSError):a.relocate(workspace,dest,relative,state)
    (dest/'retained'/relative/'summary.json').write_text('corrupt')
    monkeypatch.setattr(a.os,'symlink',real)
    with pytest.raises(ValueError,match='archive content'):a.relocate(workspace,dest,relative,state)
    assert list(src.parent.glob('*.archive-retired-*'))


def test_paths_outside_approved_local_data_are_refused(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path)
    for invalid in ('../escape','.local/../../escape','repository/source'):
        with pytest.raises(ValueError):a.relocate(workspace,dest,invalid,state)


def test_paused_copy_preserves_source_and_resumes(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path)
    def pause(**kw):raise a.ArchivePaused('operator pause')
    with pytest.raises(a.ArchivePaused):a.relocate(workspace,dest,relative,state,pause)
    assert src.is_dir() and not src.is_symlink()
    assert a.relocate(workspace,dest,relative,state)['stage']=='RECLAIMED'


def test_transient_manifest_denial_retries_atomic_replace(tmp_path,monkeypatch):
    p=tmp_path/'receipt.json';p.write_text('{"old":true}');real=a.os.replace;calls=[]
    def retry(src,dst):
        calls.append(dst)
        if len(calls)<3:raise PermissionError('busy')
        real(src,dst)
    monkeypatch.setattr(a.os,'replace',retry);monkeypatch.setattr(a.time,'sleep',lambda _:None)
    a.save(p,{'new':True})
    assert len(calls)==3 and json.loads(p.read_text())=={'new':True}


def test_persistent_denial_preserves_both_manifest_versions(tmp_path,monkeypatch):
    p=tmp_path/'receipt.json';p.write_text('{"old":true}')
    def denied(*args):raise PermissionError('denied')
    monkeypatch.setattr(a.os,'replace',denied);monkeypatch.setattr(a.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):a.save(p,{'new':True})
    assert json.loads(p.read_text())=={'old':True}
    assert json.loads(p.with_suffix('.json.pending').read_text())=={'new':True}


def test_resume_reconciles_final_manifest_after_remote_denial(tmp_path):
    workspace,dest,state,src,relative=fixture(tmp_path)
    row=a.relocate(workspace,dest,relative,state)
    mirror=next((dest/'manifests').glob('*.json'))
    mirror.write_text(json.dumps(dict(row,stage='VERIFIED')))
    assert a.relocate(workspace,dest,relative,state)==row
    assert json.loads(mirror.read_text())==row


def test_empty_output_is_recorded_and_left_untouched(tmp_path):
    workspace=tmp_path/'workspace';dest=tmp_path/'archive';state=workspace/'.local/migration'
    src=workspace/'.local/v31/jobs/excluded/output';(src/'empty-child').mkdir(parents=True)
    dest.mkdir();relative=src.relative_to(workspace)
    row=a.relocate(workspace,dest,relative,state)
    assert row['stage']=='EMPTY_RETAINED' and row['bytes']==0
    assert src.is_dir() and not src.is_symlink() and (src/'empty-child').is_dir()
    assert not (dest/'retained'/relative).exists()
    mirror=next((dest/'manifests').glob('*.json'));mirror.write_text('{}')
    assert a.relocate(workspace,dest,relative,state)==row
    assert json.loads(mirror.read_text())==row
    (src/'later-evidence.bin').write_bytes(b'preserve this')
    with pytest.raises(ValueError,match='empty source changed'):a.relocate(workspace,dest,relative,state)
    assert (src/'later-evidence.bin').read_bytes()==b'preserve this'
