from copy import deepcopy
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v23 import population_campaign,extension_contract,registry
from ghostscale.validation.soundingline.v23.common import write,read
from runners import queue_v23_extensions as queue


def test_precision_is_budget_limited_not_silently_seven_looks():
    ids=[v['id'] for v in registry.variants('N1') if not registry.selection_eligible(v)]
    assert len(population_campaign.design(8,ids,precision=True))==204
    with pytest.raises(ValueError,match='rung'):population_campaign.design(8)


def test_measured_cost_includes_full_replay_and_maximum_fresh_training(tmp_path):
    from ghostscale.validation.soundingline.v23.conditional_values import variants
    row=dict(profile=[dict(variant=v['id'],cpu_seconds=1.,cases=1) for v in variants()],
        training_generation_cpu_seconds=2.,training_generation_labels=128,fit_cpu_seconds=1.,fit_epochs=2,fit_labels=128)
    for label in ('original','replay'):write(tmp_path/label/'pilot-cost.json',row)
    result=extension_contract.cost_estimate('conditional-fits',tmp_path)
    assert result['batch']>2*12*16*64
    assert result['fits']>2*2/128*(2*8192*120)
    row['profile'].pop();write(tmp_path/'replay/pilot-cost.json',row,immutable=False)
    with pytest.raises(ValueError,match='complete conditional'):extension_contract.cost_estimate('conditional-fits',tmp_path)


def test_controller_never_retries_a_failed_identity(tmp_path):
    packet=tmp_path/'packets/v23-n1-controls-1';write(packet/'PLAN.json',dict(task='population-controls'))
    write(packet/'STATUS.json',dict(state='failed',reason='known-answer'))
    original=read(packet/'STATUS.json')
    root,state=queue.ensure(tmp_path,packet.name,'population-controls','N1',300,{},'n1')
    assert root==packet and state=='INSTRUMENT_FAILED' and read(packet/'STATUS.json')==original


def test_oversized_extension_is_resource_disposition_without_exposure(tmp_path):
    root,state=queue.ensure(tmp_path,'v23-n1-batch-1','population-batch','N1',14401,{},'n1')
    assert state=='RESOURCE_DEFERRED' and not root.exists()


def test_complete_conditional_saved_score_reconstruction(tmp_path,monkeypatch):
    import os,numpy as np
    from ghostscale.validation.soundingline.v23 import conditional_values as cv,ledger
    from ghostscale.validation.soundingline.v23.common import Direct,digest
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    variant=cv.variants()[0];p,t=cv.sample(variant,'fit-fixture',0,0)
    x=np.array([cv.features(variant,p)]*4);y=np.array([t['y']]*4)
    model=Direct(x,y,x,y,2,'conditional-fixture',0,max_epochs=1);fits=tmp_path/'fits';fits.mkdir()
    model.save(fits/'fixture.npz');write(fits/'FITS.json',{'fixture':model.metadata})
    monkeypatch.setattr(cv,'variants',lambda:[variant])
    monkeypatch.setattr(cv,'controls',lambda:dict(passed=True))
    plan=dict(task='conditional-batch',worlds=16,cases=64,fit_directory=str(fits),fit_hashes={p.name:digest(p) for p in fits.iterdir()},
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),canary_key='fe'*32)
    cv.run(tmp_path/'original',plan,lambda **kw:None)
    assert cv.reconstruct(tmp_path/'original',plan)['state']=='passed'
