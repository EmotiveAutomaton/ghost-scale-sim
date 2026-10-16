from copy import deepcopy
import os
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v23 import ledger,registry,dispatch,engine_bridge
from ghostscale.validation.soundingline.v23.common import write


@pytest.mark.parametrize('family',['R1','N2','N3','N4','N5','N6','N7','N8'])
def test_actual_pinned_engine_consumes_public_projection(tmp_path,family):
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    v=deepcopy(registry.variants(family)[0])
    if 'records' in v['settings']:v['settings']['records']=2
    public,_=dispatch.sample(v,'engineering-engine-transport',0,0)
    bundle=ledger.emit(v,public,schema,[family,'engine-transport'])
    result=engine_bridge.verify_export(tmp_path/'export',v,bundle,schema,canaries=['evaluator-only-random-token'])
    assert result['native_observations_recovered'] and result['resolved_baseline_choices']>0
    assert result['scientific_engine_admission'] is False


def test_engine_bridge_rejects_truth_and_future_leaks(tmp_path):
    from tompathy_engine.ghost.bridge import load_reader
    from tompathy_engine.ledger.ids import digest
    schema,_=ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    v=registry.variants('N5')[0];public,_=dispatch.sample(v,'bridge-mutation',0,0)
    bundle=ledger.emit(v,public,schema,['bridge-mutation'])
    bad=deepcopy(bundle);query=bad['rows']['Decision'][-1];query['taken']=query['options'][0]['option_id']
    with pytest.raises(ValueError,match='target leaked'):engine_bridge.export(bad,schema)
    result=engine_bridge.export(bundle,schema);result['records'][0]['record']['truth_label']='secret'
    result['reader_hash']=digest({k:v for k,v in result.items() if k!='reader_hash'})
    write(tmp_path/'bad.json',result)
    with pytest.raises(ValueError,match='truth field'):load_reader(tmp_path/'bad.json')
