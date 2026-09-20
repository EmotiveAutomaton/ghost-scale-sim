from pathlib import Path
import gzip
import pytest
from ghostscale.validation.soundingline.v18_4 import runtime as R
from ghostscale.validation.soundingline.v18_3.io import write


def test_source_closure_and_dispatch():
    assert all((R.REPO/p).is_file() for p in R.source_files())
    assert 'ghostscale/validation/soundingline/v18_3/torch_worker.py' in R.source_files()
    with pytest.raises(ValueError):R.dispatch(dict(family='unadmitted'))


def test_retained_block_identity_and_corruption(tmp_path):
    request=dict(family='P',index=0,cell=1,tilt=0.,rule=None)
    write(tmp_path/'PLAN.json',dict(design=dict(units=[request],block_size=1)))
    unit=R.dispatch(request);R.keep(tmp_path,'block-000000',[unit],0.,0.)
    assert len(R.load(tmp_path,'block-000000'))==1
    path=tmp_path/'raw/block-000000_points.json.gz'
    path.write_bytes(gzip.compress(b'[]'))
    with pytest.raises(ValueError):R.load(tmp_path,'block-000000')


def test_notifier_requires_exact_ack(tmp_path):
    from runners.watch_v18_4_events import pending
    from ghostscale.validation.soundingline.v18_3.io import file_digest
    state=tmp_path/'notifications';event=tmp_path/'events/x.json'
    write(event,dict(id='x',kind='fixture'))
    assert len(pending(tmp_path,state))==1
    ack=state/'acks/x.json';write(ack,dict(event_sha256=file_digest(event)))
    assert not pending(tmp_path,state)
    write(event,dict(id='x',kind='changed'),immutable=False)
    with pytest.raises(ValueError):pending(tmp_path,state)
