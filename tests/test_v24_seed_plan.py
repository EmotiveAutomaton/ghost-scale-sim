import pytest
from ghostscale.validation.soundingline.v24 import seed_plan
from ghostscale.validation.soundingline.v23.common import read,write


@pytest.mark.parametrize("value,expanded",[(.03,True),(.01,False),(.02,True)])
def test_complete_prospective_look_consumed_once_across_restarts(tmp_path,value,expanded):
    root=tmp_path/"plan"
    seed_plan.freeze(root,[2,7,10],threshold=.02,protocol_sha256="a"*64)
    result=seed_plan.decide(root,{2:value,7:value,10:value},source_sha256="b"*64,protocol_sha256="a"*64)
    assert result["extension"]==([11,12,13] if expanded else [])
    assert not result["fixed_sample_inference_on_pooled_outcomes_authorized"]
    assert read(root/"DECISION.json")==result
    with pytest.raises(FileExistsError):
        seed_plan.decide(root,{2:value,7:value,10:value},source_sha256="b"*64,protocol_sha256="a"*64)


def test_missing_nonfinite_or_rebound_look_never_consumes_decision(tmp_path):
    root=tmp_path/"plan"
    seed_plan.freeze(root,[2,7],threshold=.02,protocol_sha256="a"*64)
    for values in ({2:.03},{2:.03,7:float("nan")},{2:.03,7:.04,8:.05}):
        with pytest.raises(ValueError):
            seed_plan.decide(root,values,source_sha256="b"*64,protocol_sha256="a"*64)
    assert not (root/"DECISION.json").exists()
    plan=read(root/"PLAN.json");plan["trigger"]["threshold"]=-10.
    write(root/"PLAN.json",plan,immutable=False)
    with pytest.raises(ValueError,match="changed"):
        seed_plan.decide(root,{2:.03,7:.04},source_sha256="b"*64,protocol_sha256="a"*64)
