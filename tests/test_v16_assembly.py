from ghostscale.validation.soundingline.v16.assembly import World,execute,plan,public_construct
from ghostscale.validation.soundingline.v16.assembly_gates import run
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.records import canonical


def test_independent_assembly_physics_and_actual_revision_controls():
    assert run()["instrument_state"]=="valid"


def test_bounded_assembly_planner_reports_timeout_and_unreachable_separately():
    world=World()
    assert plan(world,(0,0,0),budget=0)["search_timeout"]
    assert plan(world,(-1,0,-1),budget=10000)["unreachable"]
    result=plan(world,(1,1,1),budget=10000)
    assert execute(world,result["program"])["state"]==[1,1,1]


def test_assembly_extension_runs_through_public_only_worker(tmp_path):
    extension="ghostscale.validation.soundingline.v16.assembly:public_construct"
    public={"schema_version":"v16.assembly.1","task_id":"opaque","world":World().public(),
            "training":[{"program":[0,1,9],"target":[0,0,-1]}]*4,"targets":[[0,0,0]],
            "budget":4096,"initial":[-1,-1,-1],"max_steps":8}
    with ReaderProcess(tmp_path/"reader",extensions=[extension]) as reader:
        assert reader.request(extension,public)==public_construct(canonical(public))
