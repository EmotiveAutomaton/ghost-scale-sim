import gzip
from itertools import product
import numpy as np
from ghostscale.validation.soundingline.v19 import routine_revision as R, local_world as L
from ghostscale.validation.soundingline.v18_3.io import canonical, file_digest, read


def test_revision_known_answers():
    assert all(R.controls().values())
    # All legal references obey native tool access and exactly retain the undo buffer.
    for m in L.MAKERS:
        for request in range(2):
            policy=R.task_policy((0,0,0),(1,1,1),m,request,False,.05)
            assert m[1] or all(op!='accept-tool' for _,op,_ in policy)
            values=R.evaluate(policy,(0,0,0),(1,1,1),m,request,'visible-task',.05,.01)[1]
            assert values['success']==1 and np.isclose(values['net_success'],.94)


def test_hidden_reference_cannot_consult_actual_hidden_bits():
    for m in L.MAKERS:
        for request in range(2):
            policies=[R.task_policy((1,0,a),(0,1,b),m,request,True,.05) for a,b in product(range(2),repeat=2)]
            assert all(p==policies[0] for p in policies)
            packets=[R.packet((1,0,a),(0,1,b),0,request,True) for a,b in product(range(2),repeat=2)]
            assert all(p==packets[0] for p in packets)


def test_group_retains_every_source_and_mass():
    records=[dict(maker=[0,0,0,0],final=[0,1,0],steps=[dict(before=[1,1,0])],requested_purpose=0,probability=p) for p in (.2,.3,.5)]
    groups=R.group(records)
    assert len(groups)==1 and groups[0][1]==dict(mass=1.,source_indices=[0,1,2])


def test_complete_synthetic_fixture(tmp_path):
    # Non-campaign fixture exercises all maker strata and both artifact presentations.
    records=[]
    for m,old,display in product(L.MAKERS,range(2),range(2)):
        records.append(dict(maker=list(m),final=[0,0,display],steps=[dict(before=[1,1,1-display])],requested_purpose=old,probability=1/64))
    inputs=tmp_path/'inputs';inputs.mkdir();p=inputs/'lineage-190968_points.json.gz'
    p.write_bytes(gzip.compress(canonical(records),mtime=0))
    cfg=dict(arms=list(R.ARMS),lineages=[190968],paths_per_lineage=64,operation_cost=.05,inspection_cost=.01,input_files={p.name:file_digest(p)})
    result=R.run(tmp_path,{'design':cfg},lambda **kw:None)
    assert all(result['controls'].values()) and result['paths']==64 and result['rows']==1280
    assert len(result['cells'])==160
    for r in result['cells']:
        if r['arm']=='keep':assert r['operation_cost']==0 and r['revision_rate']==0
        if r['arm']=='full-state-oracle':assert np.isclose(r['success'],1)
    for p in read(tmp_path/'PUBLIC_PACKET.json')['cases']:
        assert set(p['inputs'])=={'artifact','undo_artifact','original_request','current_request','observed_units'}
        assert not ({'maker','skill','purpose','routine','policy'} & set(p['inputs']))
