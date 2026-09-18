from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v18_1 import g2,g2_mask
from ghostscale.validation.soundingline.v18_1.verify import independent_assembly


def fixture():
    world=dict(kind='assembly',parents=[-1,0,0],defaults=[0,0,0],forbidden=[])
    queries=[dict(kind='action',initial=[0,0,0],program=[6]),dict(kind='action',initial=[0,0,0],program=[7])]
    public=dict(schema=g2.SCHEMA,models=[world],observations=[dict(query=q,outcome=g2.observed(world,q),source_context='fixture') for q in queries],
                menu=[],initial=[0,0,0],target=[1,0,0],max_steps=8,query_seed=0,action_order=list(range(10)))
    probes=[dict(kind='action',initial=[1,1,1],program=[a]) for a in (6,7,8)]
    return world,public,probes


def test_mask_positive_generalizes_orientation_and_retains_unknown():
    world,public,probes=fixture();result=g2_mask.predict(canonical(public),probes)
    assert result['probabilities']==[0.,1.,0.5]
    assert [independent_assembly(world,q['initial'],q['program'],8)['legal'] for q in probes[:2]]==[False,True]
    old=g2.predictions(public['models'],public['observations'],probes,'episodes')[0]
    assert old==[0.5,0.5,0.5] and result['covered_probes']==2


def test_mask_placebo_empty_evidence_and_tiny_budget():
    _,public,probes=fixture();public['observations']=[]
    assert g2_mask.predict(canonical(public),probes)['probabilities']==[0.5]*3
    answer=g2_mask.predict(canonical(public),probes,budget=1)
    assert answer['missing_output'] and answer['costs']['total_online']<=1


def test_mask_rejects_hidden_field_and_never_uses_candidate_truth():
    _,public,probes=fixture();first=g2_mask.predict(canonical(public),probes)
    swapped=deepcopy(public);swapped['models'][0]['parents']=[-1,-1,-1]
    assert g2_mask.predict(canonical(swapped),probes)==first
    public['private']={'true_world':None}
    with pytest.raises(ValueError):g2_mask.predict(canonical(public),probes)


def test_mask_attachment_and_stopping_are_not_discarded():
    world,public,probes=fixture()
    q=dict(kind='routine',initial=[0,0,0],program=[9,7])
    public['observations']=[dict(query=q,outcome=g2.observed(world,q),source_context='fixture')]
    # An illegal action after STOP does not imply it is illegal in a fresh state.
    assert g2_mask.predict(canonical(public),probes)['probabilities']==[0.5]*3
    public['observations']=fixture()[1]['observations']
    probe=dict(kind='action',initial=[1,-1,-1],program=[6])
    assert g2_mask.predict(canonical(public),[probe])['probabilities']==[0.5]


def test_mask_reference_resume_corruption_and_independent_report(tmp_path):
    from datetime import datetime,timedelta,timezone
    from ghostscale.validation.soundingline.v16.records import read,file_digest,digest
    from ghostscale.validation.soundingline.v18_1 import runtime,mask_verify
    world,public,_=fixture()
    original=dict(case_id='fixture',structural_unit='fixture-unit',public=public,private=dict(true_world=world))
    rows=g2.evaluate(original,budgets=(8192,),query_counts=(0,))
    origin=tmp_path/'origin';runtime.retain(origin,'first',dict(units=[dict(case=original,rows=rows)]),0,0)
    receipt=read(origin/'blocks/first.json')
    case=dict(case_id='mask-fixture',structural_unit='fixture-unit',origin_case_id='fixture',origin_run='origin',origin_block='first',origin_index=0,
              origin_raw_sha256=receipt['raw_sha256'],n=3,family='fork',selection='fixture',donor='same-law',truth_excluded=False)
    acceptance=dict(started_at=datetime.now(timezone.utc).isoformat(),report_start=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                    delivery_deadline=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),worker_cpu_ceiling_seconds=144000,new_law_context_unit_ceiling=200000)
    admission=dict(passed=True,sources={p:file_digest(runtime.REPO/p) for p in runtime.source_files()})
    root=tmp_path/'masked';runtime.freeze_cases(root,acceptance,admission,[case],dict(block_size=1),'g2-mask','fixture')
    runtime.run(root,None,stop_after_blocks=0)
    assert not (root/'COMPLETE.json').exists()
    runtime.run(root,None);raw=next((root/'raw').glob('*.gz'));before=raw.read_bytes()
    runtime.run(root,None);assert raw.read_bytes()==before
    assert mask_verify.report(root,tmp_path/'report')['rows_checked']==3
    case['origin_raw_sha256']='wrong'
    with pytest.raises(AssertionError):g2_mask.evaluate(case,tmp_path)
