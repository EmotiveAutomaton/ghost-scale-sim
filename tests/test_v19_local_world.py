import pytest
from ghostscale.validation.soundingline.v19 import local_world as L


@pytest.fixture(scope='module')
def records():return L.enumerate_world(L.law(999))


def test_local_rulers():assert all(L.controls().values())


def test_complete_mass_and_markov_transitions(records):
    assert len(records)==13824
    assert sum(r['probability'] for r in records)==pytest.approx(1.)
    for r in records[::37]:
        for event in r['steps']:
            assert L.execute(event['before'],event['undo_buffer'],event['operation'],r['maker'])==tuple(event['after'])


def test_projected_reader_never_needs_actual_goal_or_maker(records):
    a=records[0]
    for tier in L.TIERS:
        packet=L.project(a,tier)
        assert L.validate_public(packet)
        result=L.infer(packet,records)
        assert result['candidate_count']>0
        forged={**packet,'maker_index':0}
        with pytest.raises(ValueError):L.infer(forged,records)


def test_endpoint_alias_and_operation_goal_alias(records):
    a=next(r for r in records if all(e['operation']=='inspect' for e in r['steps']))
    artifact=L.infer(L.project(a,'E0'),records)
    witnessed=L.infer(L.project(a,'E2-full'),records)
    assert len(artifact['process_support'])>1
    assert len(witnessed['process_support'])==1
    assert sum(g['step']==0 for g in witnessed['goal_support'])==3


def test_no_outside_family_false_certainty(records):
    p=dict(schema='v19.local.public.1',tier='E0',inputs=dict(artifact=[2,0,0]))
    assert L.infer(p,records)['unknown']
