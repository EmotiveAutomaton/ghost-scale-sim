from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import checker, repairs, registry, ledger, pipeline, runtime
from ghostscale.validation.soundingline.v23.common import Direct, digest, write


def test_independent_full_posteriors_across_laws():
    for v in registry.variants('R1')[::5]:
        for law in (0, 7, 15):
            public, truth = repairs.sample(v, 'independent-check-control', law, 3)
            pred, details = repairs.evaluate(v, public, truth)
            assert checker.verify_r1(v, public, truth, pred, details) < 1e-12
            bad = deepcopy(pred); bad['masked'] = np.array([[.1, .9]])
            with pytest.raises(ValueError, match='mismatch'):
                checker.verify_r1(v, public, truth, bad, details)


def test_top_down_does_not_use_leaf_evidence():
    v = deepcopy(registry.variants('R4')[-1]); p, _ = repairs.sample(v, 'control', 0, 0)
    q = deepcopy(p); q['evidence'] = [1]*len(q['evidence'])
    np.testing.assert_allclose(repairs.r4_infer(v, p, mode='top-down'),
                               repairs.r4_infer(v, q, mode='top-down'), atol=1e-12)


@pytest.mark.parametrize('family', ['R2', 'R3', 'R4'])
def test_independent_repair_posteriors(family):
    for v in registry.variants(family)[::max(1, len(registry.variants(family))//4)]:
        p, truth = repairs.sample(v, 'independent-control', 8, 3)
        predictions, details = repairs.evaluate(v, p, truth)
        assert getattr(checker, 'verify_'+family.lower())(v, p, truth, predictions, details) < 1e-12


def test_complete_small_engineering_replay(tmp_path):
    directory = os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory: pytest.skip('canonical schema required for engineering replay')
    schema, hashes = ledger.canonical_schema(directory)
    fitroot = tmp_path/'fixture-fits'; fitroot.mkdir()
    x, y = pipeline.training('R1', 'engineering-fixture', 36)
    model = Direct(x, y, x, y, 2, 'engineering-only', 0, max_epochs=2)
    model.save(fitroot/'fixture-direct.npz'); write(fitroot/'FITS.json', {'fixture-direct': model.metadata})
    plan = dict(family='R1', batch=1, variants=['R1-0011'], worlds=2, cases=2,
        schema_directory=directory, schema_pin=dict(files=hashes, schema_version=schema.SCHEMA_VERSION,
        schema_hash=schema.schema_hash()), fit_directory=str(fitroot),
        fit_hashes={p.name: digest(p) for p in fitroot.iterdir()}, canary_key='f0'*32)
    packet=tmp_path/'packet';packet.mkdir()
    pipeline.run_batch(packet/'original', plan, lambda **kw: None)
    pipeline.run_batch(packet/'replay', plan, lambda **kw: None)
    replay=runtime.compare(packet/'original', packet/'replay')
    assert replay['state'] == 'passed'
    write(packet/'PLAN.json',plan);write(packet/'REPLAY.json',replay)
    write(packet/'COMPLETE.json',dict(state='verified',plan_sha256=digest(packet/'PLAN.json'),
        replay_sha256=digest(packet/'REPLAY.json'),summary_sha256=digest(packet/'original/SUMMARY.json')))
    from ghostscale.validation.soundingline.v23.family import reconstruct
    audited=reconstruct(packet)
    assert audited['state']=='passed' and audited['variants']['R1-0011']['cases']==4
