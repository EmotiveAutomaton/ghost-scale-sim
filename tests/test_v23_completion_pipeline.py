from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import conditional_values,population_campaign as population,registry,ledger,runtime
from ghostscale.validation.soundingline.v23.common import read,write,digest,Direct


def test_population_map_and_selection_keep_independent_worlds():
    first=population.design(1)
    assert len(first)==240
    assert {row['worlds'] for row in first}=={8,16}
    eligible=[row['variant']['id'] for row in first if registry.selection_eligible(row['variant'])]
    assert len(eligible)==36
    third=population.design(3,eligible)
    assert {row['worlds'] for row in third}=={24,48}
    with pytest.raises(ValueError,match='complete map'):population.design(1,eligible)
    with pytest.raises(ValueError,match='map cells'):population.design(3,[first[0]['variant']['id']])
    with pytest.raises(ValueError,match='separate'):population.design(3,eligible,precision=True)
    with pytest.raises(ValueError,match='two complete'):population.precision_candidates([])


def test_conditional_probe_preserves_alias_and_observed_label_timing():
    assert conditional_values.controls()['passed']
    for v in conditional_values.variants():
        p,t=conditional_values.sample(v,'engineering-only',15,0)
        pred,detail,diagnostic=conditional_values.evaluate(v,p,t)
        assert diagnostic['probe_label_used_for_forecast'] is False
        np.testing.assert_allclose(detail['mechanism_posterior'],1/3,atol=1e-12)
        changed=deepcopy(t);changed['mechanism']=(t['mechanism']+1)%3
        again=conditional_values.evaluate(v,p,changed)[0]
        for name in pred:np.testing.assert_array_equal(pred[name],again[name])


def test_complete_conditional_pilot_replays(tmp_path):
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    plan=dict(task='conditional-pilot',schema_directory=directory,
        schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),canary_key='bf'*32)
    for label in ('original','replay'):conditional_values.run(tmp_path/label,plan,lambda **kw:None)
    assert runtime.compare(tmp_path/'original',tmp_path/'replay')['state']=='passed'
    assert len(read(tmp_path/'original/SUMMARY.json')['variants'])==12


def test_population_full_consumer_reconstruction_and_replay(tmp_path,monkeypatch):
    from ghostscale.validation.soundingline.v23 import population_pipeline as pipe
    v=deepcopy(registry.variants('N1')[0]);v['settings'].update(makers=3,records=3)
    x,y=pipe.training(tmp_path/'training',[v],'engineering-only',6)
    fit=Direct(x,y,x,y,3,'population-consumer-fixture',0,max_epochs=1)
    fitroot=tmp_path/'fits';fitroot.mkdir();fit.save(fitroot/'fixture.npz');write(fitroot/'FITS.json',{'fixture':fit.metadata})
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    plan=dict(task='population-batch',batch=1,fit_directory=str(fitroot),fit_complete_sha256='fixture-only',
        fit_hashes={p.name:digest(p) for p in fitroot.iterdir()},canary_key='bc'*32,schema_directory=directory,
        schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()))
    monkeypatch.setattr(population,'validate',lambda _: [dict(variant=v,worlds=2)])
    for label in ('original','replay'):population.run(tmp_path/label,plan,lambda **kw:None)
    assert runtime.compare(tmp_path/'original',tmp_path/'replay')['state']=='passed'
    assert population.reconstruct(tmp_path/'original',plan)['state']=='passed'
    scorepath=tmp_path/'original'/v['id']/'world-0000/score_points.npz'
    with scorepath.open('ab') as output:output.write(b'corruption')
    with pytest.raises(ValueError,match='raw record'):population.reconstruct(tmp_path/'original',plan)
