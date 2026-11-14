import pytest
from ghostscale.validation.soundingline.v29 import x2
from ghostscale.validation.soundingline.v29.common import read

def test_retained_audit_checks_full_paths_and_rejects_changed_raw(tmp_path):
    s=dict(part='X2',truth=[4,4,5],environment='scarce',regime='PROBE',
           buy_share=.1,start=0,steps=32,life=64,worlds=8,blocks=8)
    original=tmp_path/'original'
    x2.run(original,selected=[s],lane='excluded-date-audit-fixture')
    receipt=x2.audit_retained_dates(original,tmp_path/'audit')
    assert receipt['passed'] and receipt['no_new_worlds'] and receipt['retained_worlds']==8
    corrected=read(tmp_path/'audit/CORRECTED_SUMMARY.json')
    assert corrected['settings']==[s]
    raw=original/'cell-000_points.npz'
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='raw bytes changed'):
        x2.audit_retained_dates(original,tmp_path/'changed')
