import json
from pathlib import Path
import sys,types
import numpy as np
import pytest
from ghostscale.validation.soundingline.v19 import runtime as R,tiny_reader as T
from ghostscale.validation.soundingline.v18_3.io import write,file_digest

def test_tiny_reader_capsule_excludes_privileged_previous_bank_and_development_truth(tmp_path):
    for split in ('train','development'):
        for condition in ('independent','copied'):
            p=tmp_path/'inputs/reader'/f'{split}-1-{condition}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            np.savez(p,codes=np.zeros((2,32),int),novel=np.ones((2,32),bool),target=np.ones((2,32,32))/8,before_bank_teacher=np.ones((2,32,32))/8)
    T.prepare(tmp_path,dict(training_draws=[1]))
    for p in (tmp_path/'data/reader').glob('*.npz'):
        with np.load(p) as z:assert set(z.files)==({'codes','novel','target'} if p.name.startswith('train-') else {'codes','novel'})

def fake_job(tmp_path,monkeypatch,handler,cap):
    root=tmp_path/'job';camp=tmp_path/'camp';root.mkdir();camp.mkdir();(root/'SOURCE.zip').write_bytes(b'fixture')
    write(root/'PLAN.json',dict(environment=R.fingerprint(),sources={},source_archive_sha256=file_digest(root/'SOURCE.zip'),design=dict(handler='tiny-reader',cpu_cap_seconds=cap,accounting_card='tiny-fixture')))
    write(camp/'ACCEPTANCE.json',dict(deadline='2099-01-01T00:00:00+00:00',cumulative_cpu_ceiling_seconds=cap,exploratory_cpu_ceiling_seconds=cap))
    write(camp/'INPUTS.json',dict(retained_root='unused-fixture'))
    monkeypatch.setitem(sys.modules,'ghostscale.validation.soundingline.v19.tiny_reader',types.SimpleNamespace(run=handler))
    return root,camp

def test_child_cpu_enforces_same_parent_cap_and_is_retained(tmp_path,monkeypatch):
    def handler(root,plan,pulse):pulse(child_cpu_seconds=10)
    root,camp=fake_job(tmp_path,monkeypatch,handler,5)
    with pytest.raises(TimeoutError):R.run(root,camp)
    receipt=json.loads(next((camp/'attempts').glob('*.json')).read_text())
    assert receipt['child_cpu_seconds']==10 and receipt['state']=='resource_cutoff'
    assert not (root/'COMPLETE.json').exists()

def test_checkpoint_measurements_separated_from_scientific_files(tmp_path,monkeypatch):
    def handler(root,plan,pulse):
        pulse(child_cpu_seconds=.01);(root/'model.pt').write_bytes(b'checkpoint')
        write(root/'CURRENT.json',dict(file='model.pt'));write(root/'CHILD_ACCOUNTING.json',dict(cpu_seconds=.01))
        return dict(controls=dict(fixture=True))
    root,camp=fake_job(tmp_path,monkeypatch,handler,100)
    assert R.run(root,camp)=='complete';r=json.loads((root/'COMPLETE.json').read_text())
    assert set(r['files'])=={'SUMMARY.json'} and set(r['execution_measurements'])=={'model.pt','CURRENT.json','CHILD_ACCOUNTING.json'}
