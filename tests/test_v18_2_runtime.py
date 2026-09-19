import json
import pytest
from ghostscale.validation.soundingline.v16.records import write,read,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v18_2 import runtime as r


def test_native_ownership_and_source_guard(tmp_path):
    campaign=tmp_path/'campaign';root=campaign/'case'
    write(campaign/'ACCEPTANCE.json',dict(report_start='2000-01-01T00:00:00+00:00',worker_cpu_ceiling_seconds=1))
    write(root/'PLAN.json',dict(sources={'runners/run_v18_2.py':'changed'},design=dict(branch='g0',namespace='guard')))
    with local_owner(campaign/'scientific-worker-owner'):
        with pytest.raises(RuntimeError,match='owns'):r.run(root,campaign)
    with pytest.raises(ValueError,match='source mismatch'):r.run(root,campaign)
    assert not (root/'STATUS.json').exists()


def test_absolute_cutoff_and_raw_corruption(tmp_path):
    campaign=tmp_path/'campaign';root=campaign/'case'
    write(campaign/'ACCEPTANCE.json',dict(report_start='2000-01-01T00:00:00+00:00',worker_cpu_ceiling_seconds=1))
    write(root/'PLAN.json',dict(sources={'runners/run_v18_2.py':file_digest(r.REPO/'runners/run_v18_2.py')},design=dict(branch='g0',namespace='cutoff')))
    r.run(root,campaign)
    assert read(root/'STATUS.json')['state']=='resource_cutoff'
    assert not (root/'COMPLETE.json').exists()
    r.keep(root,'fixture',[],0.,0.)
    assert r.load(root,'fixture')==[]
    (root/'raw/fixture_points.json.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='raw mismatch'):r.load(root,'fixture')
