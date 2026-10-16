from copy import deepcopy
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import population_confirmation as confirmation,population_pipeline as pipe,registry,ledger
from ghostscale.validation.soundingline.v23.common import Direct,read,write,digest


def test_population_reserved_blocks_cover_all_cultures():
    result=confirmation.audit('N1-0014','a'*64,128)
    assert result['laws_per_block']==32 and result['independent_blocks']==4
    with pytest.raises(ValueError,match='32 cultures'):confirmation.audit('N1-0014','a'*64,16)


def test_reserved_population_consumer_keeps_makers_nested(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v23 import confirmation as generic,support_new,population_campaign
    v=deepcopy(registry.variants('N1')[0]);v['settings'].update(makers=2,records=2)
    monkeypatch.setattr(registry,'variants',lambda name:[v])
    monkeypatch.setattr(generic,'verify_definition',lambda plan:dict(variant=v['id']))
    x,y=pipe.training(tmp_path/'training',[v],'engineering-reserved-fit',4)
    model=Direct(x,y,x,y,3,'engineering-reserved-fit',0,max_epochs=1);fits=tmp_path/'fits';fits.mkdir()
    model.save(fits/'fixture.npz');write(fits/'FITS.json',{'fixture':model.metadata})
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    plan=dict(family='N1',task='confirmation',worlds=32,cases=2,variants=[v['id']],nomination_order_sha256='a'*64,
        fit_directory=str(fits),fit_hashes={p.name:digest(p) for p in fits.iterdir()},canary_key='be'*32,
        confirmation_support=support_new.bound('N1'),schema_directory=directory,
        schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()))
    confirmation.run(tmp_path/'original',plan,lambda **kw:None)
    evidence=read(tmp_path/'original/EVIDENCE.json')
    assert evidence['independent_blocks']==1 and evidence['worlds']==32 and evidence['makers_per_world']==2
    assert population_campaign.reconstruct(tmp_path/'original',plan)['state']=='passed'
    assert confirmation.reconstruct(tmp_path/'original',plan)==evidence
    folder=tmp_path/'original'/v['id']/'world-0000'
    identity=read(folder/'IDENTITY.json');identity['identity'][-1]=1;write(folder/'IDENTITY.json',identity,immutable=False)
    with pytest.raises(ValueError,match='binding changed'):confirmation.reconstruct(tmp_path/'original',plan)
