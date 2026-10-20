"""Independent checker exercises on fully known cold-start records."""
import numpy as np
import pytest

from ghostscale.validation.soundingline.v23.common import read, write, digest
from ghostscale.validation.soundingline.v23.pipeline import Stream
from ghostscale.validation.soundingline.v24 import fit_fixture


def fixture(tmp_path):
    source=tmp_path/"inputs";output=tmp_path/"original";output.mkdir()
    record=dict(key="known",writer="fixture",prompt="zero",
                open_at=1000,resolved_at=2000,
                features={arm:[[0.,0.]]*4 for arm in fit_fixture.ARMS})
    for condition in ("planted","uninformative","constant"):
        write(source/condition/"READER.json",[record])
        write(source/condition/"LABELS.json",[dict(key="known",action=0)])
    plan=dict(identity="v24-retained-fit-preparation-1",new_scientific_worlds=0,
              input_root=str(source),future_prefix=1,
              input_files={p.relative_to(source).as_posix():digest(p) for p in source.rglob("*.json")})
    for condition in fit_fixture.CONDITIONS:
        stream=Stream(output/(condition+"_fits_points.jsonl.gz"))
        for i in range(3):
            stream.write(dict(call=i,row_ids=[],center=[0.,0.],penalty=.005,offset=[0.,0.],receipt={}))
        stream.close()
        prediction=dict(key="known",writer="fixture",prompt="zero",
                        arms={arm:dict(probabilities=[.25]*4,training_keys=[],cold_start=True)
                              for arm in fit_fixture.ARMS})
        write(output/(condition+"_predictions_points.json"),[prediction])
        write(output/(condition+"_SUMMARY.json"),dict(targets=1,fit_calls=3))
    write(output/"RAW_INDEX.json",{p.name:dict(sha256=digest(p),bytes=p.stat().st_size)
                                   for p in output.iterdir()})
    return plan,output


def test_complete_known_null_reconstructs_and_does_not_claim_science(tmp_path):
    plan,output=fixture(tmp_path)
    result=fit_fixture.reconstruct(output,plan,lambda **_:None)
    assert result["new_scientific_worlds"]==0
    assert not result["complete_fresh_pilot"]
    assert result["controls"]==dict(future=0.,option=0.,constant=0.)
    assert all(x["fit_calls"]==3 for x in result["conditions"].values())


def test_valid_hash_does_not_hide_corrupted_training_membership(tmp_path):
    plan,output=fixture(tmp_path)
    file=output/"planted_predictions_points.json"
    data=read(file);data[0]["arms"]["own"]["training_keys"]=["future-row"]
    write(file,data,immutable=False)
    index=read(output/"RAW_INDEX.json");index[file.name]=dict(sha256=digest(file),bytes=file.stat().st_size)
    write(output/"RAW_INDEX.json",index,immutable=False)
    with pytest.raises(ValueError,match="chronological"):
        fit_fixture.reconstruct(output,plan,lambda **_:None)


def test_independent_scalar_derivatives_have_known_answer():
    records=[dict(key="a",features=dict(own=[[0.],[1.]])),
             dict(key="b",features=dict(own=[[0.],[1.]]))]
    loss,gradient=fit_fixture.objective_gradient(records,dict(a=0,b=1),"own",[0.],np.zeros(1),.5)
    assert loss==pytest.approx(2*np.log(2),abs=1e-14)
    assert gradient.tolist()==[0.]
