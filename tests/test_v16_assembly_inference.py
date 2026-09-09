from collections import defaultdict
from itertools import product
import math
import pytest
from ghostscale.validation.soundingline.v16.assembly import World,execute,fragments
from ghostscale.validation.soundingline.v16.assembly_inference import prior,kernel,training_example,public_reader,decode
from ghostscale.validation.soundingline.v16.assembly_reference import stopped_histories
from ghostscale.validation.soundingline.v16.reference import token_cost
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def test_assembly_acquisition_prior_matches_independent_ordered_training_enumeration():
    world=World()
    libraries,masses=prior(world,attempts=4,reliability=0.9,topic_probability=0.8)
    exact=defaultdict(float)
    for preferred in range(2):
        probabilities=[0.9*(0.8 if kind==preferred else 0.2) for kind in range(2)]+[0.1]
        for sequence in product(range(3),repeat=4):
            examples=[training_example(world,kind) for kind in sequence]
            library=tuple(tuple(fragment) for fragment in fragments(examples,world)["library"])
            exact[library]+=0.5*math.prod(probabilities[kind] for kind in sequence)
    assert abs(sum(masses)-1)<1e-10
    for library,mass in zip(libraries,masses):
        assert abs(mass-exact[library])<1e-10


def test_assembly_artifact_likelihood_matches_independent_scalar_programs():
    for world in [World(),World((-1,0,1),(1,0,1))]:
        libraries,_=prior(world,attempts=4)
        for library in libraries:
            for target in [(0,0,0),(1,0,-1),(-1,-1,-1)]:
                distribution=kernel(world,library,target)
                scalar=defaultdict(float)
                for program,result in stopped_histories(world.public()):
                    distance=sum(a!=b for a,b in zip(result["state"],target))
                    scalar[tuple(result["state"])]+=math.exp(-distance-0.2*token_cost(program,library))
                total=sum(scalar.values())
                for state,probability in zip(distribution["support"],distribution["artifact_probabilities"]):
                    assert abs(probability-scalar[state]/total)<1e-10


def test_assembly_reader_uses_history_without_claiming_unique_route(tmp_path):
    world=World()
    visible=execute(world,[0,6,1,9])["artifact"]
    public={"schema_version":"v16.assembly-reader.1","task_id":"opaque","world":world.public(),
            "training_contract":{"attempts":8,"reliability":0.9,"topic_probability":0.8},
            "current":{"goal":[1,0,-1],"artifact":visible},
            "prior_works":[{"goal":[1,0,-1],"artifact":visible}]*4,"future_goal":[0,0,0],
            "max_steps":4,"beta":1.0,"length_cost":0.2,"search_budget":1024}
    maker=public_reader(canonical(public))
    generic=public_reader(canonical(public),"generic")
    direct=public_reader(canonical(public),"direct-table")
    assert max(abs(a-b) for a,b in zip(maker["library_posterior"],generic["library_posterior"]))>1e-6
    assert max(abs(a-b) for a,b in zip(maker["future_probabilities"],direct["future_probabilities"]))<1e-10
    with ReaderProcess(tmp_path/"reader",extensions=["ghostscale.validation.soundingline.v16.assembly_inference:public_reader"]) as reader:
        assert reader.request("ghostscale.validation.soundingline.v16.assembly_inference:public_reader",public)==maker
    empty={"parts":[],"relations":[]}
    public["current"]["artifact"]=empty
    assert public_reader(canonical(public))["history_collision_count"]>1
    public["current"]["artifact"]={"parts":[[0,1],[1,1],[2,1]],"relations":[[0,1],[0,2]]}
    assert public_reader(canonical(public))["model_mismatch"]
    with pytest.raises(ValueError,match="initial"):
        decode(world,{"parts":[[1,0]],"relations":[[0,1]]})
