from copy import deepcopy

import pytest

from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v18_1 import common,cyclic_union,direct,g2,permuted
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
