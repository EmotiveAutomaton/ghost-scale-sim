from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v23 import registry,family,pipeline,repairs


def test_structural_zero_stays_on_curve_without_invented_se(monkeypatch):
    variants=registry.variants('R1')
    def packet(batch):
        data=[]
        for v in variants:
            zero=v['settings']['strength']==.5
            data.append(dict(variant=v,structural_zero=zero,world_gains=[0.]*16 if zero else [-.08]*16,
                primary=dict(mean=0. if zero else -.08,se=0. if zero else .01)))
        return dict(family='R1',batch=batch,worlds=16,variants=[v['id'] for v in variants]),dict(variants=data)
    monkeypatch.setattr(family,'verify_packet',packet)
    decision=family.decide('R1',[1,2],dict(kill_threshold=.1,prior=[.05,.03]))
    assert decision['state']=='STOPPED_BELOW_MARGIN'
    assert len(decision['curve'])==36
    assert sum(row['structural_zero'] for row in decision['curve'])==9
    with pytest.raises(ValueError,match='terminal'):family.decide('R1',[1,2,3],dict(kill_threshold=.1,prior=[.05,.03]))


def test_unavailable_primary_stops_without_deleting_failed_cells(monkeypatch):
    variants=registry.variants('N3')
    plan=dict(family='N3',batch=1,worlds=16,variants=[v['id'] for v in variants])
    data=[dict(variant=v,primary=dict(finite=False,mean=None),primary_status='UNAVAILABLE_NO_CASE_DELETION') for v in variants]
    monkeypatch.setattr(family,'verify_packet',lambda root:(plan,dict(variants=data)))
    decision=family.decide('N3',['fixture'],{})
    assert decision['state']=='PRIMARY_UNAVAILABLE' and len(decision['curve'])==96
    assert not decision['active_ids'] and decision['best_id'] is None


@pytest.mark.parametrize('identity',['R1','R2','R3','R4'])
def test_secondary_readouts_are_available_on_known_world(identity):
    v=deepcopy(registry.variants(identity)[0]);p,t=repairs.sample(v,'secondary-control',0,0)
    _,detail=repairs.evaluate(v,p,t)
    output=pipeline.secondary(v,t,detail)
    assert output and all(isinstance(x,float) for x in output.values())
