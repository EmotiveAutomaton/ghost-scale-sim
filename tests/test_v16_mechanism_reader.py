import random
import pytest
from ghostscale.validation.soundingline.v16.mechanism_models import draw,board_prior
from ghostscale.validation.soundingline.v16.mechanism_reader import public_reader
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def fixture():
    world={"family":"W1","option_training":[[0,1],[2,3]]*8}
    return {"schema_version":"v16.mechanism-reader.1","task_id":"opaque","world":world,"training":{"attempts":8},
            "current":{"goal":3,"artifact":3},"prior_works":[{"goal":3,"artifact":3}]*4,"future_goal":12,
            "production_budget":128,"max_steps":3,"beta":1.0,"length_cost":0.3,"reader_budget":128}


def test_same_information_joint_mixture_matches_and_uses_evidence():
    public=fixture()
    predicted=public_reader(canonical(public),"mixture")
    direct=public_reader(canonical(public),"direct-mixture")
    assert abs(sum(predicted["future_probabilities"])-1)<1e-10
    assert max(abs(a-b) for a,b in zip(predicted["future_probabilities"],direct["future_probabilities"]))<1e-10
    public["prior_works"]=[{"goal":3,"artifact":0}]*4
    changed=public_reader(canonical(public),"mixture")
    assert max(abs(predicted["family_posterior"][name]-changed["family_posterior"][name]) for name in predicted["family_posterior"])>1e-6


def test_off_model_reverse_direction_is_an_explicit_failure_not_uniform():
    public=fixture()
    public["current"]={"goal":3,"artifact":4}
    public["prior_works"]=[]
    bounded=public_reader(canonical(public),"bounded-model")
    assert bounded["model_mismatch"]
    assert "future_probabilities" not in bounded
    assert not public_reader(canonical(public),"mixture")["model_mismatch"]


def test_mechanism_reader_public_boundary_and_unknown_truth_fields(tmp_path):
    public=fixture()
    extension="ghostscale.validation.soundingline.v16.mechanism_reader:public_reader"
    with ReaderProcess(tmp_path/"reader",extensions=[extension]) as reader:
        assert reader.request(extension,public)==public_reader(canonical(public))
    public["true_family"]="habit"
    with pytest.raises(ValueError,match="schema"):
        public_reader(canonical(public))
