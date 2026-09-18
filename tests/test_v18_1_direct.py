from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v18_1 import direct,g1,g3
from ghostscale.validation.soundingline.v18_1.verify import independent_assembly


def test_structural_direct_does_not_need_true_native_graph():
    models=list(g1.laws(3));models=[m for m in models if m['defaults']==[0,0,0]]
    p=dict(schema=direct.SCHEMA,models=models,initial=[0,0,0],target=[1,0,0],max_steps=8)
    result=direct.predict(canonical(p),128)
    assert result['program'] is not None and result['costs']['total_online']<=128
    for truth in models:
        actual=independent_assembly(truth,p['initial'],result['program'],8)
        assert actual['legal'] and actual['stopped'] and actual['state']==p['target']
    assert direct.predict(canonical(p),2)['program'] is None
    bad=deepcopy(p);bad['models'][0]['private_truth']=True
    with pytest.raises(ValueError):direct.predict(canonical(bad),128)


def test_direct_larger_and_graphic_targets_independently_execute():
    cases=g3.make_cases('development-structural-direct',per_stratum=2,histories=1,sizes=(5,7))
    for case in cases:
        p=case['public'];truth=case['private']['true_world']
        request=dict(schema=direct.SCHEMA,models=[p['world']],initial=p['initial'],target=p['target'],max_steps=p['max_steps'])
        result=direct.predict(canonical(request),512)
        actual=independent_assembly(truth,p['initial'],result['program'],p['max_steps'])
        assert actual['legal'] and actual['stopped'] and actual['state']==p['target']
    p=dict(schema=direct.SCHEMA,models=[dict(kind='graphic',cells=4,forbidden=[])],initial=0,target=7,max_steps=3)
    assert direct.predict(canonical(p),32)['program']==[0,1,2]
