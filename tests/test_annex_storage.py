import hashlib
import os
from pathlib import Path
import pytest
from runners.annex_storage import entries,is_raw,link_object,safe_relative


@pytest.mark.parametrize('name',['../escape','C:/secret','folder\\file','/absolute','bad\nname'])
def test_bad_paths_refused(name):
    with pytest.raises(ValueError):safe_relative(name)


def test_raw_selection_keeps_operating_files_private():
    assert is_raw('study-original/world_points.npz')
    assert is_raw('raw/case.json')
    assert not is_raw('PLAN.json') and not is_raw('SOURCE.zip') and not is_raw('notifications/data.csv')


def test_only_final_bound_transactions_are_accepted(tmp_path):
    target=tmp_path/'retained/.local/v31/jobs/example/output'
    receipt=dict(stage='LINKED',original_path_preserved=True,relative='.local/v31/jobs/example/output',target=str(target),files={})
    with pytest.raises(ValueError,match='not complete'):entries(receipt,tmp_path)
    receipt['stage']='RECLAIMED';assert entries(receipt,tmp_path)==[]
    receipt['target']=str(tmp_path/'elsewhere')
    with pytest.raises(ValueError,match='binding'):entries(receipt,tmp_path)


def test_hardlink_does_not_duplicate_or_change_original(tmp_path):
    p=tmp_path/'raw.npz';p.write_bytes(b'raw scientific evidence')
    row=dict(source=p,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    target=link_object(row,tmp_path/'objects/key/key')
    assert os.path.samefile(p,target) and p.read_bytes()==b'raw scientific evidence'
    assert link_object(row,target)==target


def test_corrupt_existing_object_and_changed_source_are_refused(tmp_path):
    p=tmp_path/'raw.npz';p.write_bytes(b'raw');q=tmp_path/'object';q.write_bytes(b'bad')
    row=dict(source=p,size=3,mtime_ns=p.stat().st_mtime_ns,sha256=hashlib.sha256(b'raw').hexdigest())
    with pytest.raises(ValueError,match='differs'):link_object(row,q)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):link_object(row,tmp_path/'new')
    assert not (tmp_path/'new').exists()
