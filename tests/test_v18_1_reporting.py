"""Branch acceptance and matched-evidence regression tests."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import gzip,json
import pytest
from ghostscale.validation.soundingline.v18_1 import runtime,g0,g1,g2,common
from ghostscale.validation.soundingline.v18_1 import g3,verify
from ghostscale.validation.soundingline.v16.records import file_digest,write,read


def test_generic_frozen_case_resume_and_costs(tmp_path):
    context=next(c for c in g1.contexts() if c['stratum']=='required-detour')
    cases=[g1.make_case(context,0,'correct','complete')]
    acceptance=dict(started_at=datetime.now(timezone.utc).isoformat(),
        report_start=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        delivery_deadline=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
        worker_cpu_ceiling_seconds=144000,new_law_context_unit_ceiling=200000)
    admission=dict(passed=True,sources={p:file_digest(runtime.REPO/p) for p in runtime.source_files()},scope='engineering fixture')
    root=tmp_path/'run'
    runtime.freeze_cases(root,acceptance,admission,cases,{'budgets':[128],'block_size':1},'g1-native','test')
    runtime.run(root,None)
    before=(root/'raw/g1-native-000_points.json.gz').read_bytes()
    runtime.run(root,None)
    assert before==(root/'raw/g1-native-000_points.json.gz').read_bytes()
    data=json.loads(gzip.decompress(before))
    assert all(row['costs']['total_online']<=128 for row in data['units'][0]['rows'])
    report=verify.branch_report(root,tmp_path/'report')
    assert report['rows_checked']==6 and report['distinct_context_units']==1
    assert sum(c['mean_success'] for c in report['cells'])==sum(r['success'] for r in data['units'][0]['rows'])
    (root/'INPUTS.json.gz').write_bytes((root/'INPUTS.json.gz').read_bytes()+b'bad')
    with pytest.raises(ValueError,match='cases mismatch'):runtime.run(root,None)


def test_frozen_case_plan_can_name_precommitted_untouched_sampling(tmp_path):
    context=next(c for c in g1.contexts() if c['stratum']=='required-detour')
    case=g1.make_case(context,0,'correct','complete')
    acceptance=dict(started_at=datetime.now(timezone.utc).isoformat(),
        report_start=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        delivery_deadline=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
        worker_cpu_ceiling_seconds=144000,new_law_context_unit_ceiling=200000)
    admission=dict(passed=True,sources={p:file_digest(runtime.REPO/p) for p in runtime.source_files()},scope='engineering fixture')
    sampling='frozen untouched structural sampling; primary direction, margin, allocation and analysis precommitted'
    plan=runtime.freeze_cases(tmp_path/'run',acceptance,admission,[case],
        {'budgets':[128],'block_size':1,'sampling':sampling},'g1-native','test-confirmation')
    assert plan['sampling']==sampling


def test_g2_methods_receive_identical_paid_evidence_and_fixed_forecasts():
    case=g2.make_cases('v18.1-g2-development',histories=1)[0]
    rows=g2.evaluate(case,budgets=(512,),query_counts=(1,))
    for policy in ('uniform','fixed','decision'):
        selected=[r for r in rows if r['query_policy']==policy]
        assert selected[0]['observation_record']==selected[1]['observation_record']==selected[2]['observation_record']
        assert selected[0]['query_indices']==selected[1]['query_indices']==selected[2]['query_indices']
    assert all(r['forecast_probes']==rows[0]['forecast_probes'] for r in rows)


def test_balanced_g0_support_pairing_and_crossed_orders():
    cases=g0.balanced_cases('v18.1-g0-development',histories=1)
    assert len(cases)==72 and len({c['structural_unit'] for c in cases})==72
    for case in cases:
        a,b=[{r['offer']:r for r in case['acquisitions'][name]['request']['processed']} for name in ('broad','focused')]
        assert all(a[i]==b[i] for i in a.keys()&b.keys())
    rows=g0.common_matrix(cases[0],budgets=(32,),boundary_budgets=(),balanced=True)
    assert {r['action_order_stratum'] for r in rows if r['planner']=='state'}=={*[f'rotation-{i}' for i in range(8)],'goal-relevance'}
    assert all(r['costs']['total_online']<=r['budget'] for r in rows if r['planner']=='state')


def test_g3_longer_targets_have_independent_legal_witnesses_and_real_dependency_changes():
    cases=g3.make_cases('v18.1-g3-development',per_stratum=3,histories=1)
    assert len({c['case_id'] for c in cases})==len(cases)
    for case in cases:
        p=case['public'];truth=case['private']['true_world'];witness=case['private']['target_witness']
        reference=verify.independent_assembly(truth,p['initial'],witness,p['max_steps'])
        assert reference['legal'] and reference['stopped'] and reference['state']==p['target']
        if case['condition']=='changed-dependency':assert truth['parents']!=case['private']['donor_world']['parents']
        assert len(p['training'])==16
    assert any(len(c['private']['target_witness'])>8 for c in cases if c['n']==7)


def test_g3_storage_and_computation_budgets_with_failed_and_censored_evidence():
    case=next(c for c in g3.make_cases('v18.1-g3-development',per_stratum=1,histories=1,sizes=(3,5)) if c['condition']=='missing-observation')
    rows=g3.evaluate(case,budgets=(128,512),storage_caps=(16,64))
    for row in rows:
        assert row['costs']['total_online']<=row['budget']
        assert row['acquisition']['storage_tokens']<=row['storage_cap']
        assert row['acquisition']['processed_trials']==16 and row['acquisition']['censored_trials']==8
    original=case['public']['world']['parents'].copy()
    case['private']['true_world']['parents'][1]=-1
    assert case['public']['world']['parents']==original


def test_g3_graphic_has_true_state_search_and_reachable_changed_constraint():
    cases=g3.graphic_cases('v18.1-g3-development',per_stratum=2,histories=1)
    assert {c['condition'] for c in cases}=={'familiar','new-combination','changed-constraint','missing-observation'}
    for case in cases:
        p=case['public'];actual=common.execute(p['world'],p['initial'],case['private']['target_witness'],p['max_steps'])
        assert actual['legal'] and actual['state']==p['target']
        row=g3.evaluate(case,budgets=(512,),storage_caps=(32,))[0]
        assert row['costs']['total_online']<=512
def test_independent_target_checker_distinguishes_detour_depth_and_impossibility():
    from runners.verify_v18_1_targets import reachable
    world = dict(kind='assembly', parents=[-1, 0, 0], defaults=[0, 0, 0], forbidden=[])
    assert reachable(world, [0, 0, 0], [1, 0, 0], 8)
    assert not reachable(world, [0, 0, 0], [1, 0, 0], 8, monotone=True)
    assert not reachable(world, [0, 0, 0], [1, 0, 0], 5)
    assert not reachable(world, [0, 0, 0], [-1, 0, 0], 12)
