"""Known-answer admission for V18.1; branch tests added before each freeze."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timedelta,timezone

import pytest

from ghostscale.validation.soundingline.v16 import assembly, craft, world
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v18.study import make_case,use_library
from ghostscale.validation.soundingline.v18_1 import common,g0
from ghostscale.validation.soundingline.v18_1 import runtime
from ghostscale.validation.soundingline.v18_1 import g1
from ghostscale.validation.soundingline.v18_1 import g2
from ghostscale.validation.soundingline.v16.records import file_digest,read,write


def test_g0_original_enumeration_exact_known_ruler():
    for target in range(16):
        for library in ([],[[0,1]],[[0,1],[2,3]]):
            for budget in (0,1,8,32,128):
                native=craft.construct(target,library,primitive_budget=budget)
                result=g0.enumeration(target,library,budget,list(range(8)))
                assert all(result[k]==native[k] for k in result)


def test_g0_original_capacity_one_matches_and_failed_records_stay():
    case=make_case('v18.1-development-1',0,0,8)
    for row in g0.original_matrix(case):
        if row['capacity']!=1:continue
        old=use_library(canonical(dict(schema='v18.selective-acquisition.1',library=row['request']['library'],
            target=row['request']['target'],budget=row['budget'],checking=row['checking'],extra_search=False)))
        assert row['success']==old['success']
        assert row['costs']['total_online']==old['costs']['online_primitives']
    records=deepcopy(case['acquisitions']['broad']['request']['processed'])
    for r in records:r['feedback']=False
    result=g0.acquire(canonical(dict(schema=g0.SCHEMA,processed=records,capacity=2,representation='fragments')))
    assert result['library']==[] and result['costs']['processed_trials']==16


def test_g0_private_fields_cannot_enter_reader():
    request=dict(schema=g0.SCHEMA,library=[[0,1]],target=7,budget=128,checking=True,
                 planner='state',action_order=list(range(8)),ordering='fixed',representation='fragments')
    result=g0.predict(canonical(request))
    private={'hidden_goal':9,'generator':'changed'}
    assert g0.predict(canonical(request))==result
    with pytest.raises(ValueError,match='schema'):
        g0.predict(canonical(dict(request,private=private)))


def test_common_budget_and_empty_library_placebo():
    for target in range(16):
        for budget in (0,1,32,128,512):
            request=dict(schema=g0.SCHEMA,library=[],target=target,budget=budget,checking=True,
                         planner='state',action_order=list(range(8)),ordering='fixed',representation='primitive')
            result=g0.predict(canonical(request))
            assert result['costs']['total_online']<=budget
            assert result['costs']['checking']==0
            if result['success']:
                assert world.execute(result['program']).artifact==target
    request['target']=7;request['budget']=512
    assert g0.predict(canonical(request))['success']


def test_parameterized_native_three_part_complete_state_action_agreement():
    for parents in product(*[range(-1,i) for i in range(3)]):
        for defaults in product((0,1),repeat=3):
            native=assembly.World(parents,defaults)
            model=dict(kind='assembly',parents=list(parents),defaults=list(defaults),forbidden=[])
            for state in product((-1,0,1),repeat=3):
                if any(state[i]>=0 and p>=0 and state[p]<0 for i,p in enumerate(parents)):continue
                for stopped in (False,True):
                    for action in range(11):
                        assert common.step(model,state,stopped,action)==assembly.step(native,state,stopped,action)


def test_detour_witness_and_unreachable_goal():
    model=dict(kind='assembly',parents=[-1,0,0],defaults=[0,0,0],forbidden=[])
    witness=[4,5,6,1,2,9]
    result=common.execute(model,[0,0,0],witness,6)
    assert result['legal'] and result['stopped'] and result['state']==(1,0,0)
    assert [common.distance([0,0,0],[1,0,0])]+[common.distance(t['after'],[1,0,0]) for t in result['trace']]==[1,2,3,2,1,0,0]
    assert common.exhaustive(model,[0,0,0],[1,0,0],6)['reachable']
    assert not common.exhaustive(model,[0,0,0],[1,0,0],6,monotone=True)['reachable']
    assert not common.exhaustive(model,[0,0,0],[-1,0,0],10)['reachable']


def test_common_state_search_all_small_reachable_targets():
    # Complete native finite state support, including explicit stopping.
    model=dict(kind='assembly',parents=[-1,0,0],defaults=[0,0,0],forbidden=[])
    for target in product((-1,0,1),repeat=3):
        reference=common.exhaustive(model,[-1,-1,-1],target,8)
        result=common.solve(model,[-1,-1,-1],target,{},100000,8,list(range(10)))
        scored=common.score_submission(model,[-1,-1,-1],target,result,8)
        assert scored['success']==reference['reachable']
        assert scored['costs']['total_online']<=100000


def test_shallower_arrival_reopens_state_after_long_macro():
    world=dict(kind='graphic',cells=4,forbidden=[])
    rep=dict(fragments=[[0,0,0],[1,1,1],[2,2,2]])
    result=common.solve(world,0,7,rep,10000,3,list(range(8)))
    assert common.score_submission(world,0,7,result,3)['success']
    assert len(result['program'])==3
    with pytest.raises(ValueError,match='initial'):
        common.execute(dict(kind='assembly',parents=[-1,0,0],defaults=[0,0,0]),[-1,0,0],[],8)


def test_literal_block_resume_source_refusal_and_corruption(tmp_path):
    archive=Path(__file__).resolve().parents[1]/'results/v18/replay.zip'
    acceptance=dict(started_at=datetime.now(timezone.utc).isoformat(),
                    report_start=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                    delivery_deadline=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
                    worker_cpu_ceiling_seconds=144000)
    admission=dict(passed=True,sources={p:file_digest(runtime.REPO/p) for p in runtime.source_files()},scope='engineering fixture only')
    runtime.freeze(tmp_path,acceptance,archive,admission,constructors=8)
    cmd=[sys.executable,'-B','-m','runners.run_v18_1','run','--root',str(tmp_path),'--archive',str(archive)]
    result=subprocess.run(cmd+['--stop-after-blocks','0'],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'COMPLETE.json').exists()
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    retained={p.name:p.read_bytes() for p in (tmp_path/'raw').glob('*.gz')}
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert retained=={p.name:p.read_bytes() for p in (tmp_path/'raw').glob('*.gz')}
    raw=next((tmp_path/'raw').glob('*.gz'));raw.write_bytes(raw.read_bytes()+b'corruption')
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode!=0 and 'checksum' in result.stderr
    plan=read(tmp_path/'PLAN.json');plan['sources']['runners/run_v18_1.py']='wrong'
    write(tmp_path/'PLAN.json',plan,immutable=False)
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode!=0 and 'source mismatch' in result.stderr


def test_g1_gate_known_detour_and_harmful_program():
    model=dict(kind='assembly',parents=[-1,0,0],defaults=[0,0,0],forbidden=[])
    target=[1,0,0];start=[0,0,0];witness=[4,5,6,1,2,9]
    local=common.Work(10000);whole=common.Work(10000)
    assert not g1.gate_for('local')(model,start,target,witness,8,local)
    assert g1.gate_for('endpoint')(model,start,target,witness,8,whole)
    assert local.spent==whole.spent==6
    assert not g1.gate_for('endpoint')(model,start,target,[8,9],8,common.Work(10000))
    # Checks at the actual current state: rotating the support is illegal here.
    assert not g1.gate_for('endpoint')(model,start,target,[6,9],8,common.Work(10000))


def test_g1_complete_declared_native_context_support_is_valid():
    contexts=g1.contexts()
    assert len({c['unit_id'] for c in contexts})==len(contexts)
    assert {c['stratum'] for c in contexts}=={'monotone','required-detour','harmful','unreachable'}
    for context in contexts:
        if context['stratum']=='unreachable':assert not context['reference']['reachable']
        elif context['stratum']=='required-detour':
            assert context['reference']['reachable'] and not context['monotone_reference']['reachable']
        result=common.execute(context['true_world'],context['initial'],context['proposed_routine'],context['max_steps'])
        if context['stratum']!='unreachable':
            assert result['legal'] and result['stopped']
        case=g1.make_case(context,0,'correct','complete')
        assert len(case['public']['training'])==4
        assert all(t['feedback']==(result['legal'] and result['stopped']) for t in case['public']['training'])
        if not result['legal']:
            assert case['public']['representation']['fragments']==[]


def test_g1_incorrect_model_and_hidden_fields_are_separate():
    context=next(c for c in g1.contexts() if c['stratum']=='required-detour')
    case=g1.make_case(context,0,'incomplete','complete')
    payload=canonical(case['public']);original=g1.predict(payload,'local',512)
    case['private']['true_world']={'hidden':'swapped'}
    assert g1.predict(payload,'local',512)==original
    with pytest.raises(ValueError,match='schema'):
        g1.predict(canonical(dict(case['public'],truth=case['private'])),'local',512)
    assert all(row['costs']['total_online']<=row['budget'] for row in g1.evaluate(g1.make_case(context,0,'incomplete','complete')))


def test_g2_observation_equivalence_query_split_and_unrelated_dependency():
    models=[dict(kind='assembly',parents=list(p),defaults=[0,0,0],forbidden=[])
            for p in product(*[range(-1,i) for i in range(3)])]
    query=dict(kind='action',initial=[0,0,0],program=[8])
    observations=[dict(query=query,outcome=g2.observed(models[0],query),source_context='own-law')]
    assert len(g2.compatible(models,observations))==6  # Leaf action does not reveal missing supports.
    context=dict(kind='context',part=2)
    added=dict(query=context,outcome={'parent':0,'primitive_cost':0},source_context='paid-observation')
    retained=g2.compatible(models,observations+[added])
    assert len(retained)==2 and all(m['parents'][2]==0 for m in retained)
    assert g2.parent_marginals(models)[1]==g2.parent_marginals(retained)[1]


def test_g2_failed_donor_is_retained_and_public_schema_is_strict():
    cases=g2.make_cases('v18.1-g2-development',histories=1)
    failed=next(c for c in cases if c['donor']=='cross-law' and not c['public']['observations'][-1]['outcome']['legal'])
    assert failed['private']['donor_source_success']
    assert len(failed['public']['observations'])==3
    assert failed['private']['true_world'] in g2.compatible(failed['public']['models'],failed['public']['observations'])
    payload=canonical(failed['public'])
    with pytest.raises(ValueError,match='schema'):
        g2.contract(canonical(dict(failed['public'],hidden=failed['private'])))
    assert g2.contract(payload)['observations']==failed['public']['observations']


def test_g2_costs_forecast_support_and_truth_exclusion_disposition():
    cases=g2.make_cases('v18.1-g2-development',histories=1)
    chosen=[next(c for c in cases if c['truth_excluded']==excluded and c['donor']=='cross-law' and c['selection']=='task-relevant')
            for excluded in (False,True)]
    for case in chosen:
        rows=g2.evaluate(case,budgets=(512,),query_counts=(0,1,2))
        assert all(r['costs']['total_online']<=512 for r in rows)
        assert {r['forecast_probe_count'] for r in rows}=={20}
        assert all(0<=r['forecast_brier']<=1 for r in rows)
        assert all(r['forecast_brier']==0 for r in rows if r['method']=='known-law')
        assert all(not r['dependency_recovered'] for r in rows if r['method']=='dependencies' and not r['inference_completed'])
