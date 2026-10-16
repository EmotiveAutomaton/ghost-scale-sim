from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import registry, dispatch, ledger, runtime
from ghostscale.validation.soundingline.v23 import pipeline_new as pipeline
from ghostscale.validation.soundingline.v23.common import Direct, write, digest, read
from ghostscale.validation.soundingline.v23.family import rows


def test_projection_independent_arithmetic_and_mutation():
    for v in registry.variants('N5')[::17]:
        p,t = dispatch.sample(v,'engineering-projection',9,0)
        prediction,detail = dispatch.evaluate(v,p,t)
        assert pipeline.verify_case(v,p,t,prediction,detail)<1e-11
        broken=deepcopy(prediction); broken['calibrated-projection']=np.array([[.25]*4])
        with pytest.raises(ValueError,match='mismatch'):
            pipeline.verify_case(v,p,t,broken,detail)
        assert not detail['headroom_eligible']


def test_projection_sufficient_likelihood_matches_scalar_nodes():
    from ghostscale.validation.soundingline.v23 import projection
    for v in registry.variants('N5')[::13]:
        for world in (0,15):
            p,_=projection.sample(v,'engineering-projection-vector',world,3)
            for power in (8,10):
                np.testing.assert_allclose(projection.quadrature(v,p,power),
                    projection.quadrature_scalar(v,p,power),atol=1e-12,rtol=0)


def test_empty_support_never_becomes_a_finite_selected_case_average():
    assert pipeline.score(None,{'y':[0]})['available'] is False
    result=pipeline.aggregate([.3,None,.4],['engineering','missing'])
    assert result['mean'] is None and not result['finite'] and result['unavailable']==1
    assert pipeline.aggregate([0.,0.],['engineering','structural'])['mean']==0


def test_prefit_stage_reconstructs_validation_and_refuses_corrupt_metadata(tmp_path,monkeypatch):
    def small_fit(root,family,pulse):
        root.mkdir();x=np.array([[0.,1.],[1.,0.],[1.,1.],[0.,0.]])*4;y=np.array([[0],[1],[2],[3]])
        model=Direct(x,y,x,y,4,'engineering-prefit',0,max_epochs=2)
        np.savez_compressed(root/'validation.npz',x=x,y=y);model.save(root/'fixture.npz')
        return {'fixture':model},{'fixture':dict(model.metadata)}
    monkeypatch.setattr(pipeline,'fit_family',small_fit)
    pipeline.run_fits(tmp_path/'good',dict(family='N5'),lambda **kw:None)
    assert read(tmp_path/'good/SUMMARY.json')['discovery_exposure'] is False
    def corrupt(*args):
        models,meta=small_fit(*args);meta['fixture']['validation_log_loss']+=.1;return models,meta
    monkeypatch.setattr(pipeline,'fit_family',corrupt)
    with pytest.raises(ValueError,match='validation score mismatch'):
        pipeline.run_fits(tmp_path/'bad',dict(family='N5'),lambda **kw:None)


def test_retained_diagnosticity_fit_rebuilds_without_test_makers(tmp_path):
    v=registry.variants('N2')[0]
    pipeline.retain_auxiliary(tmp_path,v)
    fit=read(tmp_path/'TIMING_FIT.json')
    assert fit['makers']==4096 and fit['independent_reconstruction']
    assert fit['test_labels_used'] is False
    assert (tmp_path/'timing_training_points.jsonl.gz').stat().st_size>1000


def test_optimized_diagnosticity_matches_scalar_reference_and_independent_masses():
    from ghostscale.validation.soundingline.v23 import diagnosticity
    for v in registry.variants('N2')[::11]:
        p,t=dispatch.sample(v,'engineering-vector-equivalence',15,2)
        actual,detail=diagnosticity.evaluate(v,p,t)
        expected,_=diagnosticity.evaluate_scalar(v,p,t)
        for name in actual:np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=0)
        assert pipeline.verify_case(v,p,t,actual,detail)<1e-11


def test_training_table_cache_preserves_original_random_draws_exactly():
    from ghostscale.validation.soundingline.v23 import diagnosticity,diagnosticity_training
    from ghostscale.validation.soundingline.v23.common import canonical
    for v in registry.variants('N2'):
        for law in (0,15):
            args=(v,'training-cache-equivalence',law,3)
            assert canonical(diagnosticity_training.sample(*args))==canonical(diagnosticity.sample(*args))


def test_optimized_correction_checker_matches_scalar_including_empty_support():
    from ghostscale.validation.soundingline.v23 import checker_new
    for v in (registry.variants('N3')[0],registry.variants('N3')[-1]):
        p,t=dispatch.sample(v,'engineering-correction-equivalence',15,0)
        for name in ('law-marginal','corrections-only','half-space','soft-constraint-supplement','final-choice-only'):
            a,ap=checker_new.n3(v,p,name);b,bp=checker_new.n3_scalar(v,p,name)
            if a is None:assert b is None
            else:
                np.testing.assert_allclose(a,b,atol=1e-11,rtol=0)
                np.testing.assert_allclose(ap,bp,atol=1e-11,rtol=0)


@pytest.mark.parametrize('empty_first', [False, True])
def test_n3_mixed_empty_support_retains_unavailable_diagnostic(tmp_path, monkeypatch, empty_first):
    directory = os.environ['V23_SCHEMA_DIRECTORY']
    schema, hashes = ledger.canonical_schema(directory)
    v = registry.variants('N3')[0]
    public, truth = dispatch.sample(v, 'engineering-empty-support', 0, 0)
    valid = deepcopy(public)
    for row in valid['history']:
        row['picked'] = 0
    empty = deepcopy(valid)
    direction = np.zeros(v['settings']['dimensions']); direction[0] = 1
    for row, sign in zip(empty['history'][:2], (1, -1), strict=True):
        row['picked'] = 1
        row['edits'][1] = (sign*direction).tolist()
    assert dispatch.evaluate(v, valid, truth)[0]['half-space'] is not None
    assert dispatch.evaluate(v, empty, truth)[0]['half-space'] is None
    monkeypatch.setattr(dispatch, 'sample', lambda variant, phase, world, case:
        (deepcopy(empty if (case == 0) == empty_first else valid), deepcopy(truth)))
    fitroot = tmp_path/'fits'; fitroot.mkdir()
    x = np.array([pipeline.features(v, valid)]*4); y = np.array([truth['y']]*4)
    model = Direct(x, y, x, y, 4, 'engineering-empty-support', 0, max_epochs=1)
    model.save(fitroot/'fixture.npz'); write(fitroot/'FITS.json', {'fixture':model.metadata})
    plan = dict(family='N3', batch=1, variants=[v['id']], worlds=2, cases=2,
        schema_directory=directory, schema_pin=dict(files=hashes,
        schema_version=schema.SCHEMA_VERSION, schema_hash=schema.schema_hash()),
        fit_directory=str(fitroot), fit_hashes={p.name:digest(p) for p in fitroot.iterdir()},
        canary_key='f0'*32)
    pipeline.run_batch(tmp_path/'original', plan, lambda **kw:None)
    summary = read(tmp_path/'original/SUMMARY.json')['variants'][0]
    assert summary['unavailable_predictions']['half-space'] == 2
    assert summary['world_gains'] == [None, None]
    assert summary['primary_status'] == 'UNAVAILABLE_NO_CASE_DELETION'
    assert summary['secondary']['half-space:direction_cosine']['finite'] is False
    assert summary['secondary']['half-space:direction_cosine']['unavailable'] == 2
    assert summary['secondary']['half-space:available']['mean'] == .5
    assert len(list(rows(tmp_path/'original'/v['id']/'prediction_points.jsonl.gz'))) == 4


@pytest.mark.parametrize('family',['N2','N3','N4','N5','N6','N7','N8'])
def test_new_family_complete_retention_and_replay(tmp_path,family):
    directory=os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory: pytest.fail('canonical schema is required; no silent replay skip')
    schema,hashes=ledger.canonical_schema(directory)
    v=registry.variants(family)[0]
    p,t=dispatch.sample(v,'engineering-direct',0,0)
    x=np.array([pipeline.features(v,p)]*4); y=np.array([t['y']]*4)
    fitroot=tmp_path/'fits'; fitroot.mkdir()
    model=Direct(x,y,x,y,pipeline.CLASSES[family],'engineering-only',0,max_epochs=1)
    model.save(fitroot/'fixture.npz');write(fitroot/'FITS.json',{'fixture':model.metadata})
    plan=dict(family=family,batch=1,variants=[v['id']],worlds=2,cases=2,
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,
        schema_hash=schema.schema_hash()),fit_directory=str(fitroot),
        fit_hashes={p.name:digest(p) for p in fitroot.iterdir()},canary_key='f0'*32)
    pipeline.run_batch(tmp_path/'original',plan,lambda **kw:None)
    pipeline.run_batch(tmp_path/'replay',plan,lambda **kw:None)
    replay=runtime.compare(tmp_path/'original',tmp_path/'replay')
    assert replay['state']=='passed'
    write(tmp_path/'PLAN.json',plan);write(tmp_path/'REPLAY.json',replay)
    write(tmp_path/'COMPLETE.json',dict(state='verified',plan_sha256=digest(tmp_path/'PLAN.json'),
        replay_sha256=digest(tmp_path/'REPLAY.json'),summary_sha256=digest(tmp_path/'original/SUMMARY.json')))
    from ghostscale.validation.soundingline.v23.family import reconstruct
    assert reconstruct(tmp_path)['state']=='passed'
    summary=read(tmp_path/'original/SUMMARY.json')['variants'][0]
    assert summary['validity']['ledger_forecast_max_error']<1e-12
    if family=='N5': assert not summary['headroom_eligible'] and not summary['matched_headroom_groups']
    if family=='N6': assert sum(summary['primary_counts'])<=4
