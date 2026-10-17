import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import conditional_small as pipeline,covariation,consideration,ledger
from ghostscale.validation.soundingline.v23.common import Direct,write,read,digest

def test_fresh_minibatches_cover_all_laws_and_fixed_training_stays_unchanged(monkeypatch):
    v=consideration.variants()[0];counts=np.zeros(16,int)
    # Integrate all equally likely offsets of the actual 256-label minibatch.
    for offset in range(16):
        monkeypatch.setattr(pipeline,'_law_offset',lambda family,phase,o=offset:o)
        for i in range(256):
            _,truth=pipeline.training_sample(consideration,v,['N9-training',['fresh',0,0,0]],i//12,i)
            counts[truth['law']]+=1
    assert np.array_equal(counts,np.full(16,256))
    phase=['N9-training',['fixed',2048]]
    from ghostscale.validation.soundingline.v23.common import canonical
    assert canonical(pipeline.training_sample(consideration,v,phase,7,0))==canonical(consideration.sample(v,phase,7,0))

@pytest.mark.parametrize('model',[covariation,consideration])
def test_complete_retained_pipeline_and_interval_reconstruction(tmp_path,monkeypatch,model):
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    v=model.variants()[0];p,t=model.sample(v,'fixture-training',0,0)
    x=np.array([model.features(v,p)]*4);y=np.array([t['y']]*4)
    fitted=Direct(x,y,x,y,model.CLASSES,'small-pipeline-fixture',0,max_epochs=1)
    fits=tmp_path/'fits';fits.mkdir();fitted.save(fits/'fixture.npz');write(fits/'FITS.json',dict(fixture=fitted.metadata))
    monkeypatch.setattr(model,'variants',lambda:[v])
    # Known-answer model controls have their own tests; this exercises full record plumbing.
    monkeypatch.setattr(model,'controls',lambda:dict(passed=True))
    plan=dict(task='conditional-batch',worlds=16,cases=64,fit_directory=str(fits),fit_hashes={p.name:digest(p) for p in fits.iterdir()},
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()),canary_key='fa'*32)
    pipeline.run(tmp_path/'original',plan,lambda **kw:None,model)
    assert pipeline.reconstruct(tmp_path/'original',plan,model)['intervals_reconstructed']>=5
    path=tmp_path/'original/SUMMARY.json';s=read(path)
    s['variants'][0]['metrics']['primary']['interval']['lower']+=.1
    write(path,s,immutable=False)
    with pytest.raises(ValueError,match='interval differs'):pipeline.reconstruct(tmp_path/'original',plan,model)
