from copy import deepcopy
import math
import numpy as np
import pytest
from ghostscale.validation.soundingline.v23 import confirmation, pipeline, registry, support
from ghostscale.validation.soundingline.v23.common import write, digest, records


def frozen_fixture(tmp_path):
    v=registry.variants('R1')[-1]
    card=tmp_path/'CARD.json'
    write(card,dict(state='NOMINATED',family='R1',best_id=v['id'],curve=[dict(variant=v,shrunk_mean=.03)]))
    order=dict(state='DISCOVERY_FROZEN',dispositions={f:'UNIMPLEMENTED_AT_FREEZE' for f in registry.ORDER},
        nominations=[dict(family='R1',variant=v['id'],shrunk_mean=.03,machine_card_sha256=digest(card))])
    order['dispositions']['R1']='NOMINATED';path=tmp_path/'ORDER.json';write(path,order)
    review=tmp_path/'REVIEW.json';write(review,dict(state='verified_discovery',decision='NOMINATED',family='R1',
        nominated=v['id'],full_fresh_process_replay=True,receipts=[dict(machine_card_sha256=digest(card))]))
    plan=dict(task='confirmation',family='R1',batch='confirmation',variants=[v['id']],worlds=256,cases=256,
        nomination_order=str(path),nomination_order_sha256=digest(path),nomination_card=str(card),confirmation_index=1,
        nomination_review=str(review),nomination_review_sha256=digest(review),
        confirmation_support=support.repair_bound('R1'))
    return plan,order


def test_confirmation_refuses_open_discovery_and_changed_primary(tmp_path):
    plan,order=frozen_fixture(tmp_path)
    assert confirmation.verify_definition(plan)['variant']=='R1-0035'
    bad=deepcopy(plan);bad.pop('nomination_review')
    with pytest.raises(ValueError,match='review required'):confirmation.verify_definition(bad)
    bad=deepcopy(order);bad['state']='DISCOVERY_ACTIVE'
    with pytest.raises(ValueError,match='not frozen'):confirmation.validate_order(bad)
    bad=deepcopy(order);bad['dispositions'].pop('N1')
    with pytest.raises(ValueError,match='every family'):confirmation.validate_order(bad)
    bad=deepcopy(plan);bad['variants']=['R1-0034']
    with pytest.raises(ValueError,match='replace'):confirmation.verify_definition(bad)
    bad=deepcopy(plan);bad['worlds']=128
    with pytest.raises(ValueError,match='allocation'):confirmation.verify_definition(bad)
    bad=deepcopy(plan);bad['fit_directory']='discovery-fits'
    with pytest.raises(ValueError,match='refitting'):confirmation.verify_definition(bad)


def test_frozen_order_requires_strongest_first(tmp_path):
    _,order=frozen_fixture(tmp_path)
    order['dispositions']['R2']='NOMINATED'
    order['nominations'].append(dict(family='R2',variant='R2-0000',shrunk_mean=.04,machine_card_sha256='a'*64))
    with pytest.raises(ValueError,match='strongest first'):confirmation.validate_order(order)


def test_reserved_world_seeds_and_variant_only_training():
    result=confirmation.seed_audit('R1-0035','a'*64,128)
    assert result['independent_blocks']==8 and result['reserved_case_identities']==32768
    v=registry.variants('R1')[-1]
    x,y=pipeline.training('R1',['confirmation-fixture','training'],32,variant=v)
    mx,my=pipeline.training('R1',['confirmation-fixture','training'],32,masked=True,variant=v)
    np.testing.assert_array_equal(y,my)
    np.testing.assert_array_equal(x[:,34:],mx[:,34:])
    assert np.all(mx[:,2:34]==0)
    np.testing.assert_array_equal(x[:,-len(registry.variant_features(v)):],np.tile(registry.variant_features(v),(32,1)))


def raw_fixture(root, bad_law=False):
    pp=[];tt=[]
    for world in range(32):
        # First complete block loses, second wins. Direct is intentionally
        # excellent; it must never replace the nominated masked comparison.
        mechanism=.25 if world<16 else .75
        pp.append(dict(world=world,case=0,probabilities={'masked':[[.5,.5]],
            'law-marginal':[[mechanism,1-mechanism]],'direct':[[.99,.01]]}))
        tt.append(dict(identity=['R1','R1-0035','confirmation',world,0],truth=dict(y=[0],law=(0 if bad_law else world%16))))
    folder=root/'R1-0035';folder.mkdir(parents=True)
    records(folder/'prediction_points.jsonl.gz',pp);records(folder/'truth_points.jsonl.gz',tt)


def test_independent_raw_evidence_preserves_negative_gain_and_law_blocks(tmp_path):
    raw_fixture(tmp_path)
    plan=dict(family='R1',variants=['R1-0035'],worlds=32,cases=1,confirmation_support=support.repair_bound('R1'))
    result=confirmation.reconstruct_evidence(tmp_path,plan)
    expected=[math.log(.5),math.log(1.5)]
    np.testing.assert_allclose(result['block_means'],expected,rtol=0,atol=1e-15)
    fraction=.5/(.02-support.repair_bound('R1')['lower_bound'])
    assert result['log_e']==pytest.approx(sum(math.log1p(fraction*(v-.02)) for v in expected))
    assert result['truncated']==0 and result['independent_blocks']==2
    assert result['primary_readers']==['masked','law-marginal']
    bad=tmp_path/'bad';raw_fixture(bad,bad_law=True)
    with pytest.raises(ValueError,match='law block'):confirmation.reconstruct_evidence(bad,plan)


def test_confirmation_admission_preserves_group_and_global_reserves(tmp_path,monkeypatch):
    from runners.admit_v23_confirmation import capacity
    from ghostscale.validation.soundingline.v23 import runtime
    write(tmp_path/'ACCEPTANCE.json',dict(max_scientific_workers=3,cpu_seconds=324000,protected_cpu_seconds=64800))
    monkeypatch.setattr(runtime,'cost',lambda campaign:250000)
    with pytest.raises(ValueError,match='protected reserve'):capacity(tmp_path,[],10000)
    monkeypatch.setattr(runtime,'cost',lambda campaign:100)
    completed=[(tmp_path,dict(budget_group='confirmation'),dict(state='complete',cpu_seconds=54000))]
    with pytest.raises(ValueError,match='group cap'):capacity(tmp_path,completed,1)
    active=[(tmp_path,dict(budget_group='repairs',cpu_cap_seconds=100),dict(state='running'))]*3
    with pytest.raises(ValueError,match='slot'):capacity(tmp_path,active,100)
    assert capacity(tmp_path,[],100)==0


def test_full_confirmation_engineering_products_replay(tmp_path,monkeypatch):
    """Tiny known-answer fixture; deliberately not a scientific admission."""
    import os
    from ghostscale.validation.soundingline.v23 import ledger, runtime
    from ghostscale.validation.soundingline.v23.common import Direct
    directory=os.environ.get('V23_SCHEMA_DIRECTORY')
    if not directory:pytest.skip('canonical schema required')
    schema,hashes=ledger.canonical_schema(directory)
    v=registry.variants('R1')[-1]
    x,y=pipeline.training('R1','confirmation-engineering-only',32,variant=v)
    model=Direct(x,y,x,y,2,'confirmation-engineering-only',0,max_epochs=2)
    def fixture_fit(root,family,pulse,variant=None,namespace=None):
        assert variant==v and namespace==['confirmation','a'*64]
        root.mkdir();model.save(root/'fixture.npz');write(root/'FITS.json',{'fixture':model.metadata})
        return {'fixture':model},{'fixture':model.metadata}
    monkeypatch.setattr(pipeline,'fit_family',fixture_fit)
    monkeypatch.setattr(confirmation,'verify_definition',lambda plan:dict(variant=v['id']))
    plan=dict(task='confirmation',family='R1',batch='confirmation',variants=[v['id']],worlds=16,cases=1,
        nomination_order_sha256='a'*64,confirmation_support=support.repair_bound('R1'),canary_key='c0'*32,
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()))
    for label in ('original','replay'):confirmation.run(tmp_path/label,plan,lambda **kw:None)
    compared=runtime.compare(tmp_path/'original',tmp_path/'replay')
    assert 'EVIDENCE.json' in compared['files'] and 'SEED_AUDIT.json' in compared['files']


def test_n_family_confirmation_replays_with_reserved_variant_only_fits(tmp_path,monkeypatch):
    import os
    from ghostscale.validation.soundingline.v23 import ledger,runtime,pipeline_new,support_new
    from ghostscale.validation.soundingline.v23.common import Direct
    directory=os.environ['V23_SCHEMA_DIRECTORY'];schema,hashes=ledger.canonical_schema(directory)
    v=registry.variants('N5')[0];x,y=pipeline_new.training('N5','engineering-confirmation-only',32,variant=v)
    model=Direct(x,y,x,y,4,'engineering-confirmation-only',0,max_epochs=2)
    def fixture_fit(root,family,pulse,variant=None,namespace=None):
        assert variant==v and namespace==['confirmation','a'*64]
        root.mkdir();model.save(root/'fixture.npz');write(root/'FITS.json',{'fixture':model.metadata})
        return {'fixture':model},{'fixture':model.metadata}
    monkeypatch.setattr(pipeline_new,'fit_family',fixture_fit)
    monkeypatch.setattr(confirmation,'verify_definition',lambda plan:dict(variant=v['id']))
    plan=dict(task='confirmation',family='N5',batch='confirmation',variants=[v['id']],worlds=16,cases=1,
        nomination_order_sha256='a'*64,confirmation_support=support_new.bound('N5'),canary_key='b1'*32,
        schema_directory=directory,schema_pin=dict(files=hashes,schema_version=schema.SCHEMA_VERSION,schema_hash=schema.schema_hash()))
    for label in ('original','replay'):confirmation.run(tmp_path/label,plan,lambda **kw:None)
    assert 'EVIDENCE.json' in runtime.compare(tmp_path/'original',tmp_path/'replay')['files']
    with pytest.raises(ValueError,match='finite raw-gain'):support_new.verify(dict(task='confirmation',family='N3',confirmation_support=support_new.bound('N3')))


def test_n6_confirmation_excludes_other_strata_without_case_deletion(tmp_path):
    from ghostscale.validation.soundingline.v23 import support_new
    pp=[];tt=[]
    for world in range(16):
        for case in range(2):
            pp.append(dict(world=world,case=case,probabilities={'withdraw':[[.5,.5,0.,0.]],'model-and-gate':[[.75,.25,0.,0.]]}))
            tt.append(dict(identity=['N6','N6-0000','confirmation',world,case],truth=dict(y=[0],law=world,warmth=case,competence=case)))
    folder=tmp_path/'N6-0000';folder.mkdir();records(folder/'prediction_points.jsonl.gz',pp);records(folder/'truth_points.jsonl.gz',tt)
    plan=dict(family='N6',variants=['N6-0000'],worlds=16,cases=2,confirmation_support=support_new.bound('N6'))
    # The registered contrast must be maintained; use its exact reader names.
    from ghostscale.validation.soundingline.v23.common import read
    actual=confirmation.reconstruct_evidence(tmp_path,plan)
    assert actual['independent_blocks']==1
