from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import ledger,makers,population_pipeline as pipe,registry,runtime
from ghostscale.validation.soundingline.v23.common import Direct,read


def fixture_variant():
    v=deepcopy(registry.variants('N1')[3]);v['settings'].update(makers=3,records=3)
    return v


def test_population_retained_training_counts_and_truth_boundary(tmp_path):
    v=fixture_variant();x,y=pipe.training(tmp_path/'training',[v],'fixture',5)
    index=read(tmp_path/'training/INDEX.json')
    assert x.shape[0]==5 and y.shape==(5,)
    assert index['selected_labels']==5 and index['generated_labels']==6
    assert len(index['populations'])==2
    p,t=makers.n1_world(v,'feature-boundary',0)
    first=pipe.features(v,p);t['type'][:]=0;t['y'][:]=2;t['law']=31
    np.testing.assert_array_equal(first,pipe.features(v,p))
    with pytest.raises(ValueError,match='every design'):pipe.training(tmp_path/'undersized',[v,v],'fixture',1)


def test_retained_population_complete_engineering_replay(tmp_path):
    directory=os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory:pytest.skip('canonical schema required')
    schema,_=ledger.canonical_schema(directory);v=fixture_variant()
    x,y=pipe.training(tmp_path/'training',[v],'engineering-only',6)
    fit=Direct(x,y,x,y,3,'N1-engineering-only',0,max_epochs=2)
    a=pipe.retained_world(tmp_path/'original',v,'engineering-only',0,schema,'fa'*32,{'fixture-direct':fit})
    b=pipe.retained_world(tmp_path/'replay',v,'engineering-only',0,schema,'fa'*32,{'fixture-direct':fit})
    assert a==b and a['independent_max_error']<1e-10
    assert a['ledger']['partitions']==3 and a['ledger']['public_arrays_equal']
    assert runtime.compare(tmp_path/'original',tmp_path/'replay')['state']=='passed'
    with np.load(tmp_path/'original/score_points.npz') as scores:
        gains=scores['individual-only__outside__log_loss']-scores['law-marginal__outside__log_loss']
    assert a['primary_mean']==pytest.approx(gains.mean())
    assert a['raw_gain_min']==pytest.approx(gains.min())
    assert pipe.known_answers()['passed']


def test_population_ledger_corruption_is_rejected(tmp_path):
    directory=os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory:pytest.skip('canonical schema required')
    schema,_=ledger.canonical_schema(directory);v=fixture_variant();p,_=makers.n1_world(v,'boundary',0)
    bundle=ledger.emit(v,p,schema,['engineering','N1'])
    query=next(row for row in bundle['rows']['Decision'] if row['context']['channel']=='query')
    query['taken']=query['options'][0]['option_id']
    with pytest.raises(ValueError,match='query target leaked'):ledger.decode(v,bundle,schema)
