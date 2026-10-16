"""Actual frozen ToMpathy research boundary; constructed fixtures, no real-data claim."""
from datetime import timedelta
import os
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import ledger, twin_ledger, twin_engineering, engine_bridge
from ghostscale.validation.soundingline.v23.common import write


def test_interface_matches_live_imported_schema_and_implementation():
    from tompathy_engine.interface import contract
    from tompathy_engine.admission import implementation_hash
    value = contract()
    schema, _ = ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    assert value['format'] == 'tompathy-research-interface-1'
    assert value['ledger_schema_hash'] == schema.schema_hash()
    assert value['implementation_hash'] == implementation_hash()
    assert set(schema.RECORDS) <= set(value['record_schemas'])


@pytest.mark.parametrize('name', ['T1','T2','T3','T4'])
@pytest.mark.parametrize('dirty', [False, True])
def test_each_twin_native_ingestion_preserves_baseline_fits(tmp_path, name, dirty):
    from tompathy_engine.ghost.bridge import load_reader, reader_choices, ingest_reader
    from tompathy_engine.ledger.store import Ledger
    from tompathy_engine.models.backbone import Backbone
    schema, _ = ledger.canonical_schema(os.environ['V23_SCHEMA_DIRECTORY'])
    v, public, truth = twin_engineering.sample(name, dirty)
    bundle = twin_ledger.emit(v, public, schema, ['interface-roundtrip',name,dirty])
    projected = engine_bridge.export(bundle, schema, canaries=['evaluator-only-canary'])
    path = tmp_path/'reader.json'; write(path, projected)
    native, _ = load_reader(path)
    stored = Ledger(tmp_path/'ledger')
    receipt = ingest_reader(path, stored)
    assert receipt['truth_loaded'] is False
    a, b = reader_choices(native), reader_choices(stored.all())
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert {k:v for k,v in vars(left).items() if k!='features'} == {k:v for k,v in vars(right).items() if k!='features'}
        assert np.array_equal(left.features, right.features)
    if name == 'T3':
        # Edit events have no option-choice target; do not manufacture one.
        assert not a
        return
    assert a
    cutoff = max(row.resolved for row in a)+timedelta(seconds=1)
    direct, transported = Backbone.fit(a, until=cutoff), Backbone.fit(b, until=cutoff)
    for row in a:
        assert np.allclose(direct.predict(row.maker, row.features),
                           transported.predict(row.maker, row.features), atol=1e-12, rtol=0)
