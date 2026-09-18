import os
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v18_1 import g3_stitch,g3
from ghostscale.validation.soundingline.v16.records import read


def test_stitch_grounding_cost_storage_and_unsupported_world():
    learned={'accepted':[dict(name='pair',body=['seq',['place','c0'],['place','c1']],arity=0)]}
    training=[dict(program=[0,1])]*4
    assert g3_stitch.grounded(learned,training,'graphic',4)['fragments']==[]
    rep=g3_stitch.grounded(learned,training,'graphic',5)
    assert rep['fragments']==[[0,1]] and rep['storage']==5 and rep['grounding_operations']==10
    case=g3.make_cases('development-stitch-unsupported',per_stratum=1,histories=1,sizes=(5,),families=('chain',))[0]
    with pytest.raises(ValueError,match='codec'):g3_stitch.evaluate(case,Path('unused'))


def test_existing_stitch_binary_child_cpu_and_cache(tmp_path,monkeypatch):
    executable=os.environ.get('GS_V17_STITCH_EXE')
    if not executable:pytest.skip('optional existing binary not configured; not an admission pass')
    training=[dict(program=[0,1,2])]*8+[dict(program=[3,4,5])]*8
    learned=g3_stitch.learn(training,'graphic',tmp_path)
    assert learned['accepted']
    receipts=list((tmp_path/'child-attempts').glob('*.json'));assert len(receipts)==1
    receipt=read(receipts[0]);assert receipt['cpu_measured'] and receipt['returncode']==0 and receipt['child_cpu_seconds']>=0
    assert g3_stitch.learn(training,'graphic',tmp_path)==learned
    assert len(list((tmp_path/'child-attempts').glob('*.json')))==1
    assert g3_stitch.grounded(learned,training,'graphic',128)['fragments']
