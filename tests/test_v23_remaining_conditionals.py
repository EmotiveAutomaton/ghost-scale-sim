from copy import deepcopy
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import corrections_soft,precision_habits,goal_overlays,sacrifice,trust_attention,ledger,conditional_small
from ghostscale.validation.soundingline.v23.common import canonical,Direct,write,read,digest
MODELS=[corrections_soft,precision_habits,goal_overlays,sacrifice,trust_attention]

@pytest.mark.parametrize('model',MODELS)
def test_known_answers_and_complete_grid(model):
    assert len(model.variants())==12
    assert model.controls()['passed']

@pytest.mark.parametrize('model',MODELS)
def test_no_evaluator_or_future_target_in_forecasts_and_ledger(model):
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    shape=None
    for v in model.variants():
        p,t=model.sample(v,'engineering-fixture',7,1);a,_,metrics=model.evaluate(v,p,t)
        changed=dict(t,y=[(x+1)%model.CLASSES for x in t['y']],mechanism=(t['mechanism']+1)%2)
        b,_,_=model.evaluate(v,p,changed)
        for name in a:
            if name!='oracle':assert np.array_equal(a[name],b[name])
            assert np.all(a[name]>0) and np.allclose(a[name].sum(-1),1)
        assert np.isfinite(list(metrics.values())).all()
        features=model.features(v,p)
        if shape is None:shape=features.shape
        assert features.shape==shape and np.array_equal(features,model.features(model.variants()[0],p))
        bundle=model.emit(p,schema,['fixture',v['id']]);assert canonical(model.decode(bundle,schema))==canonical(p)
        row=next(x for x in bundle['rows']['Decision'] if x['context']['channel']=='query');row['taken']=row['options'][0]['option_id']
        with pytest.raises(ValueError,match='target'):model.decode(bundle,schema)

@pytest.mark.parametrize('model',MODELS)
def test_retained_complete_one_setting_and_tamper_rejection(model,tmp_path,monkeypatch):
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    v=model.variants()[-1];p,t=model.sample(v,'fixture-training',0,1)
    x=np.array([model.features(v,p)]*4);y=np.array([t['y']]*4)
    fitted=Direct(x,y,x,y,model.CLASSES,'remaining-fixture',0,max_epochs=1)
    fits=tmp_path/'fits';fits.mkdir();fitted.save(fits/'fixture.npz');write(fits/'FITS.json',dict(fixture=fitted.metadata))
    monkeypatch.setattr(model,'variants',lambda:[v]);monkeypatch.setattr(model,'controls',lambda:dict(passed=True))
    plan=dict(task='conditional-batch',worlds=16,cases=64,fit_directory=str(fits),fit_hashes={p.name:digest(p) for p in fits.iterdir()},
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),canary_key='ad'*32)
    conditional_small.run(tmp_path/'original',plan,lambda **kw:None,model)
    assert conditional_small.reconstruct(tmp_path/'original',plan,model)['state']=='passed'
    path=tmp_path/'original/SUMMARY.json';s=read(path);s['variants'][0]['metrics']['primary']['world_values'][0]+=.1;write(path,s,immutable=False)
    with pytest.raises(ValueError,match='world means differ'):conditional_small.reconstruct(tmp_path/'original',plan,model)

def test_auc_and_direction_are_conditional_not_zero_filled():
    v=corrections_soft.variants()[-1]
    metrics=dict(hidden_truth=[0,1],hidden_probability=[.2,.8],shuffled_hidden_probability=[.8,.2],direction_cosine=[0,.7])
    result=corrections_soft.block_readout(v,0,metrics)
    assert result['hidden_detection_auc']==1 and result['shuffled_detection_auc']==0
    assert result['direction_cosine_given_hidden']==.7
    metrics['hidden_truth']=[0,0];assert corrections_soft.block_readout(v,0,metrics)['hidden_detection_auc'] is None
