"""Canonical dependency controls; a missing dependency is a visible skip."""
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import ledger, registry, repairs
from ghostscale.validation.soundingline.v23 import dispatch


@pytest.fixture(scope='module')
def schema():
    directory = os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory:
        pytest.skip('canonical schema directory not supplied; scientific admission remains blocked')
    return ledger.canonical_schema(Path(directory))[0]


@pytest.mark.parametrize('family', ['R1', 'R2', 'R3', 'R4'])
def test_canonical_roundtrip_and_order_free_rows(schema, family):
    v = registry.variants(family)[0]
    p, truth = repairs.sample(v, 'ledger-control', 0, 1)
    bundle = ledger.emit(v, p, schema, [family, 0, 1])
    ledger.validate(bundle, schema, canaries=['independent-unobserved-canary'])
    assert ledger.decode(v, bundle, schema) == p
    native = repairs.evaluate(v, p)[0]['law-marginal']
    module = ledger.ReferenceModule(v, schema)
    for rows in bundle['rows'].values(): rows.reverse()
    np.testing.assert_allclose(module.predict(module.fit(bundle)), native, atol=1e-12)
    # Opaque identifiers can all be permuted without changing a forecast.
    mapping = {}
    for kind, rows in bundle['rows'].items():
        for row in rows:
            value = row[schema.ID_FIELDS[kind]]
            mapping[value] = 'renamed-'+value
            for option in row.get('options',[]):mapping[option['option_id']]='renamed-'+option['option_id']
    def rename(x):
        if isinstance(x, str): return mapping.get(x, x)
        if isinstance(x, list): return [rename(a) for a in x]
        if isinstance(x, dict): return {k: rename(a) for k, a in x.items()}
        return x
    changed = rename(bundle)
    np.testing.assert_allclose(module.predict(module.fit(changed)), native, atol=1e-12)


def test_reference_api_query_binding_and_incidental_time_mask(schema):
    from datetime import datetime,timedelta
    v=registry.variants('R1')[0];p,_=repairs.sample(v,'api-conformance',0,0)
    bundle=ledger.emit(v,p,schema,['api-conformance']);module=ledger.ReferenceModule(v,schema)
    fitted=module.fit(bundle);query=bundle['rows']['Decision'][-1]
    np.testing.assert_allclose(module.predict(fitted,query),module.predict(fitted)[0],atol=1e-12)
    explanation=module.explain(fitted,bundle['rows']['Maker'][0]['maker_id'])
    assert isinstance(explanation['hypotheses'],dict)
    bad=deepcopy(query);bad['options'].reverse()
    with pytest.raises(ValueError,match='boundary'):module.predict(fitted,bad)
    shifted=deepcopy(bundle)
    for rows in shifted['rows'].values():
        for row in rows:
            for key in ('t','t_start','t_end','t_open','t_resolve','created_at','captured_at'):
                if row.get(key):row[key]=(datetime.fromisoformat(row[key].replace('Z','+00:00'))+timedelta(days=9)).isoformat()
    np.testing.assert_allclose(module.predict(module.fit(shifted)),module.predict(fitted),atol=1e-12)


def test_truth_future_and_schema_mismatch_fail(schema):
    v = registry.variants('R1')[0]; p, _ = repairs.sample(v, 'ledger-control', 2, 0)
    b = ledger.emit(v, p, schema, ['R1', 2, 0])
    bad = deepcopy(b); bad['rows']['Decision'][0]['outcome']['future'] = 1
    with pytest.raises(ValueError, match='forbidden'): ledger.validate(bad, schema)
    bad = deepcopy(b); bad['schema_hash'] = '0'*64
    with pytest.raises(ValueError, match='mismatch'): ledger.validate(bad, schema)
    bad = deepcopy(b); bad['rows']['Maker'][0]['display_label'] = 'CANARY-991'
    with pytest.raises(ValueError, match='canary'): ledger.validate(bad, schema, canaries=['CANARY-991'])
    bad = deepcopy(b); q = bad['rows']['Decision'][-1]; q['taken'] = q['options'][0]['option_id']
    with pytest.raises(ValueError, match='target leaked'): ledger.validate(bad, schema)


@pytest.mark.parametrize('family', ['N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'])
def test_new_mechanism_observation_roundtrip(schema, family):
    v = registry.variants(family)[0]
    public, _ = dispatch.sample(v, 'ledger-control', 0, 0)
    bundle = ledger.emit(v, public, schema, [family, 0, 0])
    assert ledger.decode(v, bundle, schema) == public
    for rows in bundle['rows'].values(): rows.reverse()
    assert ledger.decode(v, bundle, schema) == public


def test_population_roundtrip_keeps_individual_ownership(schema):
    from ghostscale.validation.soundingline.v23 import makers
    v = next(v for v in registry.variants('N1') if v['settings']['makers'] == 10 and
             v['settings']['records'] == 2 and v['settings']['structure'] == 'alias')
    public, _ = makers.n1_world(v, 'ledger-control', 0)
    bundle = ledger.emit(v, public, schema, ['population-control', 0])
    for rows in bundle['rows'].values(): rows.reverse()
    recovered = ledger.decode(v, bundle, schema)
    assert len(bundle['rows']['Maker']) == 10
    for key in public: np.testing.assert_array_equal(public[key], recovered[key])


def test_revision_and_supersession_integrity(schema):
    v = registry.variants('N3')[0]; public, _ = dispatch.sample(v, 'ledger-integrity', 0, 0)
    bundle = ledger.emit(v, public, schema, ['references'])
    bad = deepcopy(bundle); bad['rows']['Maker'][0]['supersedes'] = 'missing'
    with pytest.raises(ValueError, match='unresolved'): ledger.validate(bad, schema)
    bad = deepcopy(bundle); bad['rows']['Revision'][0]['parent_revision_ids'] = ['missing']
    with pytest.raises(ValueError, match='unresolved'): ledger.validate(bad, schema)
