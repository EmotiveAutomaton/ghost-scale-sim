from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import twin_pipeline as pipe,twin_engineering,twin_graph,twin_ledger,ledger,registry,runtime
from ghostscale.validation.soundingline.v23.common import read,write,digest,canonical


def schema():return ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])[0]


def test_twin_reference_requires_actual_source_bytes(tmp_path):
    source=tmp_path/'aggregate-source';source.write_text('synthetic test fixture only')
    v=deepcopy(registry.variants('T4')[0]);v['settings'].update(makers=2,items=2)
    shape=dict(raters=2,items=2,ratings_per_rater=[2,2],raters_per_item=[2,2],rating_semantics='nominal_yes_no_unsure',rating_marginals=[.2,.3,.5])
    reference=tmp_path/'REFERENCE.json';write(reference,dict(source=dict(revision='a'*40,sha256=digest(source),citation='fixture, no empirical claim'),aggregates=shape))
    plan=dict(family='T4',empirical_reference=str(reference),empirical_reference_sha256=digest(reference),empirical_source=str(source))
    assert pipe.reference(plan)['aggregates']==shape
    source.write_text('changed')
    with pytest.raises(ValueError,match='authenticate'):pipe.reference(plan)


def test_full_small_twin_world_and_engine_consumer_replay(tmp_path):
    v=deepcopy(registry.variants('T4')[0]);v['settings'].update(makers=3,items=3)
    shape=dict(raters=3,items=3,ratings_per_rater=[3]*3,raters_per_item=[3]*3,rating_semantics='nominal_yes_no_unsure',rating_marginals=[.2,.3,.5])
    reference=dict(source=dict(revision='a'*40,sha256='b'*64,citation='engineering fixture only'),aggregates=shape)
    for label in ('original','replay'):
        pipe.evaluate_world(tmp_path/label,v,'engineering-world-only',0,reference,schema(),'de'*32,lambda **kw:None)
    compared=runtime.compare(tmp_path/'original',tmp_path/'replay')
    assert 'ENGINE_BASELINE.json' in compared['files']
    graph=read(tmp_path/'original/WORLD_GRAPH.json');assert graph['makers']==3 and graph['queries']==3 and not graph['evaluator_loaded']
    result=read(tmp_path/'original/ENGINE_BASELINE.json');assert result['state']=='ENGINE_BASELINE_EXECUTED' and len(result['rows'])==3
    assert not result['engine_module_admission']


def test_global_revision_graph_preserves_shared_articles_and_true_chronology(tmp_path):
    from ghostscale.validation.soundingline.v23 import twin_wiki,engine_bridge
    v=deepcopy(registry.variants('T3')[0]);v['settings'].update(makers=3,max_records=100)
    generated=list(twin_wiki.world(v,'graph-fixture',0,makers=3,records=3));identity=['T3',v['id'],'graph-fixture',0]
    for maker in range(3):
        selected=[(p,t) for p,t in generated if p['maker']==maker]
        p=dict(maker=maker,history=[p for p,_ in selected]);unit=tmp_path/f'maker-{maker:05d}';unit.mkdir()
        write(unit/'PUBLIC.json',p);write(unit/'TRUTH.json',dict(y=[selected[0][1]['purpose']]))
        bundle=twin_ledger.emit(v,p,schema(),[*identity,maker],world_identity=identity)
        engine_bridge.verify_export(unit/'engine',v,bundle,schema())
    result=twin_graph.evaluate(tmp_path,v,identity,schema(),lambda **kw:None)
    assert result['state']=='TARGET_UNAVAILABLE'
    reader=read(tmp_path/'WHOLE_WORLD_READER.json');records=reader['records']
    assert sum(row['type']=='Maker' for row in records)==3
    articles=[row['record'] for row in records if row['type']=='Artifact' and row['record']['revisions']]
    assert len(articles)==2 and sum(len(a['revisions']) for a in articles)==9
    times=[row['record']['t'] for row in records if row['type']=='Revision']
    assert len(set(times))==9


def test_twin_training_keeps_entire_population_and_selected_labels(tmp_path):
    v=deepcopy(registry.variants('T4')[0]);v['settings'].update(makers=3,items=2)
    x,y=pipe.training(tmp_path/'draw',[v],'training-fixture',2,dict(rating_semantics='nominal_yes_no_unsure'),lambda **kw:None)
    result=read(tmp_path/'draw/INDEX.json')
    assert x.shape==(2,128) and y.shape==(2,1)
    assert result['generated_labels']==3 and result['selected_labels']==2


def test_writing_option_histogram_changes_only_public_menu_sizes():
    from ghostscale.validation.soundingline.v23 import twin_writing
    v,p,_=twin_engineering.sample('T1',False)
    shape=dict(records_per_maker=[16],sessions_per_maker=[2],options_per_decision={'2':1,'5':1},
        latency_quantiles_s=dict(probabilities=[0,1],values=[1,4]),keystroke_quantiles=dict(probabilities=[0,1],values=[0,12]),handling_marginals=[.5,.22,.23,.05])
    p,t=twin_writing.maker(v,'histogram-fixture',0,0,shape)
    assert set(len(row['options']) for row in p['history'])=={2,5}
    assert len(t['y'])==1


@pytest.mark.parametrize('name',['T1','T2'])
def test_supported_engine_targets_consume_public_history_and_future_options(tmp_path,name):
    from ghostscale.validation.soundingline.v23 import engine_bridge
    v,p,t=twin_engineering.sample(name,False);identity=[name,v['id'],'engine-fixture-only',0]
    folder=tmp_path/'maker-00000';folder.mkdir();write(folder/'PUBLIC.json',p);write(folder/'TRUTH.json',t)
    bundle=twin_ledger.emit(v,p,schema(),[*identity,0],world_identity=identity)
    engine_bridge.verify_export(folder/'engine',v,bundle,schema())
    outcome=twin_graph.evaluate(tmp_path,v,identity,schema(),lambda **kw:None)
    assert outcome['state']=='ENGINE_BASELINE_EXECUTED'
    result=read(tmp_path/'ENGINE_BASELINE.json');assert len(result['rows'])==1
    assert np.isclose(sum(result['rows'][0]['probabilities']),1)
    assert result['train_ids'] and not result['engine_module_admission']
