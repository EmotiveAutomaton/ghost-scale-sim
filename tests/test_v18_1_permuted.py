from copy import deepcopy

import pytest

from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v18_1 import common,cyclic_union,direct,g2,g3,permuted
from ghostscale.validation.soundingline.v18_1.verify import independent_assembly


def test_arbitrary_dag_labels_execute_and_cycles_fail():
    world=dict(kind='assembly',parents=[2,2,-1],defaults=[0,0,0],forbidden=[])
    common.validate_world(world)
    result=common.execute(world,[0,0,0],[4,3,5,2,0,1,9],8)
    assert result['legal'] and result['stopped'] and result['state']==(0,0,0)
    bad=deepcopy(world);bad['parents']=[1,0,-1]
    with pytest.raises(ValueError,match='cyclic'):
        common.validate_world(bad)


def test_relabel_preserves_physics_and_hides_order():
    world=dict(kind='assembly',parents=[-1,0,0,2,2],defaults=[0,1,0,1,0],forbidden=[])
    permutation=[3,1,4,0,2]
    transformed=permuted.relabel_world(world,permutation)
    initial=[0,1,0,1,0];program=[8,7,6,5,9,0,1,2,3,4,15]
    original=independent_assembly(world,initial,program,20)
    relabeled=independent_assembly(transformed,permuted.relabel_state(initial,permutation),
        [permuted.relabel_action(action,permutation) for action in program],20)
    assert relabeled['legal']==original['legal']
    assert relabeled['state']==permuted.relabel_state(original['state'],permutation)
    assert any(parent>child for child,parent in enumerate(transformed['parents']) if parent>=0)


def test_permuted_transfer_keeps_public_evidence_and_direct_rival_equal():
    original=g2.transfer_cases('development-permuted-base',per_stratum=1,histories=1,sizes=(5,),families=('fork',))
    cases=permuted.make_cases(original,'development-permuted')
    assert len(cases)==6
    for case in cases:
        public=case['public'];truth=case['private']['true_world']
        assert any(parent>child for child,parent in enumerate(truth['parents']) if parent>=0)
        assert all(retained['outcome']==g2.observed(truth,retained['query']) for retained in public['observations'])
        request=dict(schema=direct.SCHEMA,models=public['models'],initial=public['initial'],target=public['target'],max_steps=public['max_steps'])
        result=direct.predict(canonical(request),32768)
        assert result['program'] is not None
        for model in public['models']:
            actual=independent_assembly(model,public['initial'],result['program'],public['max_steps'])
            assert actual['legal'] and actual['stopped'] and actual['state']==public['target']
        rows=permuted.evaluate(case,budgets=(32768,),query_counts=(0,2))
        assert len(rows)==19 and any(row['method']=='structural-direct' for row in rows)
        assert all(row['costs']['total_online']<=row['budget'] for row in rows)


def test_direct_refuses_candidate_union_cycle():
    models=[dict(kind='assembly',parents=[-1,0,-1],defaults=[0,0,0],forbidden=[]),
            dict(kind='assembly',parents=[1,-1,-1],defaults=[0,0,0],forbidden=[])]
    request=dict(schema=direct.SCHEMA,models=models,initial=[0,0,0],target=[1,0,0],max_steps=8)
    result=direct.predict(canonical(request),512)
    assert result['program'] is None and 'no shared dependency order' in result['unsupported_reason']


def test_cyclic_union_requires_evidence_and_keeps_action_rivals_equal():
    cases=cyclic_union.make_cases('development-cyclic-union',per_stratum=1,histories=1,
                                  sizes=(5,),families=('chain',))
    assert len(cases)==1
    case=cases[0];public=case['public'];truth=case['private']['true_world']
    assert truth in public['models'] and len(public['models'])==12
    assert len(g2.compatible(public['models'],public['observations']))==12
    assert direct.shared_order(public['models'],common.Work(10000)) is None
    assert all(model['parents'][case['target_part']]!=truth['parents'][case['target_part']]
               for model in public['models'] if model!=truth)
    assert 'true_world' not in public and 'label_permutation' not in public

    rows=cyclic_union.evaluate(case,budgets=(32768,),query_counts=(0,1))
    assert len(rows)==30
    by_key={(row['method'],row['query_policy'],row['requested_queries']):row for row in rows}
    assert by_key['known-law','fixed',0]['success']
    for method in ('dependencies','conditioned-direct','candidate-set-primitive'):
        assert not by_key[method,'fixed',0]['success']
        assert by_key[method,'fixed',1]['success']
        assert by_key[method,'fixed',1]['acquired_queries']==1
    for policy in ('uniform','fixed','decision'):
        for count in (0,1):
            records={canonical(row['observation_record']) for row in rows
                     if row['query_policy']==policy and row['requested_queries']==count}
            assert len(records)==1
    assert all(row['costs']['total_online']<=row['budget'] for row in rows)


def test_cached_decision_separates_selector_work_without_changing_its_evidence():
    case=cyclic_union.make_cases('development-cyclic-cost',per_stratum=1,histories=1,
                                 sizes=(5,),families=('chain',))[0]
    full=cyclic_union.evaluate(case,budgets=(32768,),query_counts=(1,))
    full={(row['method'],row['query_policy']):row for row in full}
    cached=cyclic_union.evaluate_cached_decision(case,32768,32768)
    assert {row['method'] for row in cached}=={
        'dependencies','known-law','conditioned-direct','candidate-set-primitive'}
    assert all(row['query_indices']==full[row['method'],'decision']['query_indices'] for row in cached)
    assert all(canonical(row['observation_record'])==canonical(full[row['method'],'decision']['observation_record'])
               for row in cached)
    assert all(row['selector_operations']>0 and row['costs']['total_online']<=32768 for row in cached)
    assert all(row['combined_operations']==row['selector_operations']+row['costs']['total_online'] for row in cached)
    known=next(row for row in cached if row['method']=='known-law')
    assert known['success'] and known['combined_operations']==full['known-law','decision']['costs']['total_online']


def test_target_aware_queries_resolve_two_public_target_cycles_cheaply():
    case=cyclic_union.make_multi_target_cases(
        'development-cyclic-target',per_stratum=1,histories=1,families=('chain',))[0]
    public=case['public'];truth=case['private']['true_world']
    assert case['target_parts']==[1,2] and len(public['models'])==4
    fixed,_,fixed_exhausted,fixed_work=cyclic_union.acquire(public,truth,'fixed',2,32768)
    one,one_indices,one_exhausted,one_work=cyclic_union.acquire(
        public,truth,'target-aware',1,32768)
    two,two_indices,two_exhausted,two_work=cyclic_union.acquire(
        public,truth,'target-aware',2,32768)
    assert not fixed_exhausted and not one_exhausted and not two_exhausted
    assert len(g2.compatible(public['models'],fixed))==4
    assert len(g2.compatible(public['models'],one))==2
    assert g2.compatible(public['models'],two)==[truth]
    assert {public['menu'][index]['part'] for index in two_indices}=={1,2}
    assert one_indices[0] in two_indices and two_work.spent<256<fixed_work.cap
    assert one_work.spent<two_work.spent
    rows=cyclic_union.evaluate_target_aware(case,32768,(1,2))
    selected={(row['method'],row['query_policy'],row['requested_queries']):row for row in rows}
    assert len(rows)==24
    assert selected['dependencies','target-aware',2]['success']
    assert selected['conditioned-direct','target-aware',2]['success']
    assert selected['known-law','target-aware',2]['success']
    assert not selected['dependencies','target-aware',1]['query_isolates_truth']
    assert not selected['dependencies','fixed',2]['query_isolates_truth']


def test_target_action_evidence_resolves_cycles_without_parent_cues():
    case=cyclic_union.make_multi_target_cases(
        'development-cyclic-action',per_stratum=1,histories=1,
        families=('chain',),action_probes=True)[0]
    public=case['public'];truth=case['private']['true_world'];n=len(public['initial'])
    fixed,_,fixed_exhausted,_=cyclic_union.acquire(public,truth,'fixed',2,32768)
    parent,parent_indices,parent_exhausted,_=cyclic_union.acquire(
        public,truth,'target-aware',2,32768)
    one,one_indices,one_exhausted,one_work=cyclic_union.acquire(
        public,truth,'target-action',1,32768)
    two,two_indices,two_exhausted,two_work=cyclic_union.acquire(
        public,truth,'target-action',2,32768)
    assert not any((fixed_exhausted,parent_exhausted,one_exhausted,two_exhausted))
    assert len(g2.compatible(public['models'],fixed))==4
    assert g2.compatible(public['models'],parent)==[truth]
    assert len(g2.compatible(public['models'],one))==2
    assert g2.compatible(public['models'],two)==[truth]
    assert one_indices[0] in two_indices and one_work.spent<two_work.spent<512
    for index in two_indices:
        query=public['menu'][index]
        assert query['kind']=='action' and len(query['program'])==1
        assert query['program'][0]%n in case['target_parts']
        assert cyclic_union._common_valid_state(public['models'],query['initial'])
    assert all(o['query']['kind']!='context' for o in two[len(public['observations']):])
    rows=cyclic_union.evaluate_target_action(case,32768,(1,2))
    selected={(row['method'],row['query_policy'],row['requested_queries']):row for row in rows}
    assert len(rows)==24
    assert selected['dependencies','target-action',2]['success']
    assert selected['conditioned-direct','target-action',2]['success']
    assert not selected['dependencies','target-action',2]['direct_parent_cues_used']
    assert selected['dependencies','target-action',2]['target_action_parts']==[1,2]


def test_target_action_selector_uses_public_evidence_under_every_candidate_truth():
    from copy import deepcopy
    from ghostscale.validation.soundingline.v18_1.verify import independent_assembly
    case=cyclic_union.make_multi_target_cases('development-action-counterfactual',
        per_stratum=1,histories=1,families=('groups',),action_probes=True)[0]
    public=case['public'];first=[]
    for truth in public['models']:
        observations,indices,exhausted,work=cyclic_union.acquire(public,truth,'target-action',2,32768)
        first.append(indices[0])
        assert not exhausted and g2.compatible(public['models'],observations)==[truth]
        for observation in observations[len(public['observations']):]:
            query=observation['query'];state=query['initial'];outcome=observation['outcome']
            for model in public['models']:
                assert all(value==-1 or parent==-1 or state[parent]!=-1
                           for value,parent in zip(state,model['parents']))
            actual=independent_assembly(truth,state,query['program'],public['max_steps'])
            assert actual['legal']==outcome['legal'] and actual['state']==outcome['state']
        reordered=deepcopy(public);reordered['models'].reverse()
        repeated,selected,_,_=cyclic_union.acquire(reordered,truth,'target-action',2,32768)
        assert selected==indices and repeated==observations
    assert len(set(first))==1


def test_target_action_acquisition_exhaustion_never_yields_free_observation():
    case=cyclic_union.make_multi_target_cases('development-action-budget',
        per_stratum=1,histories=1,families=('chain',),action_probes=True)[0]
    public=case['public'];truth=case['private']['true_world']
    for budget in (0,1,8):
        observations,indices,exhausted,work=cyclic_union.acquire(public,truth,'target-action',2,budget)
        assert exhausted and work.spent<=budget
        assert observations==public['observations'] and indices==[]


def test_complete_action_menu_is_exhaustive_and_never_inspects_outcomes(monkeypatch):
    from itertools import product
    case=cyclic_union.make_multi_target_cases('development-complete-action-menu',
        per_stratum=1,histories=1,families=('chain',))[0]
    public=case['public'];models=public['models'];n=len(public['initial'])
    with monkeypatch.context() as patch:
        def forbidden(*args,**kwargs):raise AssertionError('menu construction inspected outcomes')
        patch.setattr(g2,'observed',forbidden)
        menu=cyclic_union.complete_action_menu(models)
    expected=set()
    for present in product((False,True),repeat=n):
        state=tuple(d if on else -1 for d,on in zip(models[0]['defaults'],present))
        if all(all(state[i]==-1 or p==-1 or state[p]!=-1 for i,p in enumerate(m['parents'])) for m in models):
            expected.update((state,action) for action in range(3*n))
    assert {(tuple(q['initial']),q['program'][0]) for q in menu if q['kind']=='action'}==expected
    public['menu']=menu
    observations,_,exhausted,work=cyclic_union.acquire(public,case['private']['true_world'],'target-action',2,32768)
    assert not exhausted and work.spent<32768
    assert g2.compatible(models,observations)==[case['private']['true_world']]


def test_physical_action_selector_builds_queries_without_truth_or_oracle_setup():
    case=cyclic_union.make_multi_target_cases('development-physical-action',
        per_stratum=1,histories=1,families=('groups',))[0]
    public=case['public'];public['menu']=cyclic_union.complete_action_menu(public['models'])
    first=[]
    for truth in public['models']:
        observations,indices,setups,reachable,exhausted,work=cyclic_union.acquire_physical(
            public,truth,2,32768)
        first.append(indices[0])
        assert not exhausted and len(indices)==len(setups)==len(reachable)==2
        assert g2.compatible(public['models'],observations)==[truth]
        prior=deepcopy(public['observations'])
        for index,setup,observation in zip(indices,setups,observations[len(prior):]):
            hypotheses=g2.compatible(public['models'],prior)
            terminal=[]
            for model in hypotheses:
                result=independent_assembly(model,public['initial'],setup,public['max_steps'])
                assert result['legal'] and not result['stopped']
                terminal.append(result['state'])
            assert len({tuple(state) for state in terminal})==1
            assert terminal[0]==public['menu'][index]['initial']==observation['query']['initial']
            prior.append(observation)
        assert work.counts['checking']>=sum(map(len,setups))+len(setups)
    assert len(set(first))==1


def test_physical_action_evaluation_charges_setup_and_keeps_one_query_unresolved():
    case=cyclic_union.make_multi_target_cases('development-physical-evaluation',
        per_stratum=1,histories=1,families=('fork',))[0]
    case['public']['menu']=cyclic_union.complete_action_menu(case['public']['models'])
    rows=cyclic_union.evaluate_physical_action(case,32768,(1,2))
    selected={(row['method'],row['requested_queries']):row for row in rows}
    assert len(rows)==8
    assert selected['dependencies',1]['query_compatible_laws']==2
    assert not selected['dependencies',1]['success']
    assert selected['dependencies',2]['query_isolates_truth']
    assert selected['conditioned-direct',2]['success']
    if not selected['dependencies',2]['success']:
        assert selected['dependencies',2]['costs']['total_online']==32768
    row=selected['dependencies',2]
    assert row['query_policy']=='target-action-physical'
    assert row['physical_setup_operations']==sum(map(len,row['physical_setup_paths']))
    assert row['candidate_family_supplied'] and not row['setup_uses_evaluator_truth']


def test_physical_action_acquisition_exhaustion_never_yields_free_setup_or_evidence():
    case=cyclic_union.make_multi_target_cases('development-physical-budget',
        per_stratum=1,histories=1,families=('chain',))[0]
    public=case['public'];public['menu']=cyclic_union.complete_action_menu(public['models'])
    observations,indices,setups,reachable,exhausted,work=cyclic_union.acquire_physical(
        public,case['private']['true_world'],2,8)
    assert exhausted and work.spent<=8
    assert observations==public['observations']
    assert indices==setups==reachable==[]


def test_misspecified_action_evidence_detects_excluded_truth_and_abstains():
    case=cyclic_union.make_misspecified_cases('development-cyclic-misspecified',
        per_stratum=1,histories=1,families=('chain',))[0]
    public=case['public'];truth=case['private']['true_world'];n=len(public['initial'])
    assert case['truth_excluded'] and truth not in public['models'] and len(public['models'])==3
    assert public['menu']==cyclic_union.complete_action_menu(public['models'])
    one,one_indices,one_exhausted,_=cyclic_union.acquire(
        public,truth,'misspecification-action',1,32768)
    two,two_indices,two_exhausted,_=cyclic_union.acquire(
        public,truth,'misspecification-action',2,32768)
    assert not one_exhausted and not two_exhausted
    assert len(g2.compatible(public['models'],one))==1
    assert g2.compatible(public['models'],two)==[]
    assert one_indices==two_indices[:1]
    assert len(two_indices)==2
    assert len({public['menu'][index]['program'][0]%n for index in two_indices})==2
    assert all(public['menu'][index]['kind']=='action' for index in two_indices)

    rows=cyclic_union.evaluate_misspecified(case,32768,(1,2))
    assert len(rows)==12
    selected={(row['method'],row['requested_queries']):row for row in rows}
    methods={'dependencies','known-law','conditioned-direct','candidate-set-primitive',
             'episodes','forced-candidate-direct'}
    assert {row['method'] for row in rows}==methods
    for method in ('dependencies','conditioned-direct','candidate-set-primitive'):
        row=selected[method,2]
        assert row['candidate_aware'] and row['misspecification_detected']
        assert row['abstained_on_inconsistency'] and row['program'] is None
        assert not row['unsafe_action_attempted_after_inconsistency']
    forced=selected['forced-candidate-direct',2]
    assert forced['program'] is not None and forced['unsafe_action_attempted_after_inconsistency']
    assert selected['known-law',2]['success'] and selected['known-law',2]['method_receives_evaluator_truth']
    assert all(row['costs']['total_online']<=row['budget'] for row in rows)


def test_misspecification_selector_never_uses_evaluator_truth():
    case=cyclic_union.make_misspecified_cases('development-misspecification-no-oracle',
        per_stratum=1,histories=1,families=('groups',))[0]
    public=case['public'];truth=case['private']['true_world']
    observations,indices,exhausted,_=cyclic_union.acquire(
        public,truth,'misspecification-action',2,32768)
    assert not exhausted and not g2.compatible(public['models'],observations)
    alternatives=[]
    for alternative in public['models']:
        _,chosen,failed,_=cyclic_union.acquire(
            public,alternative,'misspecification-action',2,32768)
        alternatives.append(chosen[0])
        assert not failed
    assert len(set(alternatives+[indices[0]]))==1
