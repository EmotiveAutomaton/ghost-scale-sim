from datetime import datetime
import pytest
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v24 import contract,d5
from runners import run_v24_queue


def test_finite_queue_requires_exact_window_roster_and_plan_bindings(tmp_path):
    write(tmp_path/"ACCEPTANCE.json",contract.acceptance())
    manifest=dict(acceptance_sha256=digest(tmp_path/"ACCEPTANCE.json"),packets=[])
    for family in ("N16","D5"):
        name=f"v24-{family.lower()}-pilot-1"+("-r1" if family=="D5" else "")
        path=tmp_path/"packets"/name/"PLAN.json"
        write(path,dict(kind="pilot",family=family))
        manifest["packets"].append(dict(packet=name,family=family,plan_sha256=digest(path)))
    opened=datetime.fromisoformat(contract.OPEN)
    assert run_v24_queue.validate_queue(tmp_path,manifest,opened)
    for time in ("2026-10-20T12:00:00-07:00",contract.CLOSE,contract.PRODUCTION_OPEN):
        with pytest.raises(ValueError,match="outside"):
            run_v24_queue.validate_queue(tmp_path,manifest,datetime.fromisoformat(time))
    manifest["packets"].reverse()
    with pytest.raises(ValueError,match="two prepared"):
        run_v24_queue.validate_queue(tmp_path,manifest,opened)
    manifest["packets"].reverse()
    write(path,dict(kind="production",family=family),immutable=False)
    with pytest.raises(ValueError,match="plan changed"):
        run_v24_queue.validate_queue(tmp_path,manifest,opened)


def test_d5_cannot_open_before_complete_retained_equivalence(tmp_path):
    proof=tmp_path/"proof.json";write(proof,dict(state="RETAINED_FULL_FIXTURE_VERIFIED"))
    plan=dict(kind="pilot",family="D5",identity="v24-d5-pilot-1",configuration=d5.configuration(),
        world_indices=[0],confirmation=False,cpu_cap_seconds=5400,wall_cap_seconds=14400,
        backbone_sha256="5563cc4b7e46638640d2455bf4aaae69c7ffb32f1c434f381b1b7bedc251e244",
        profile_sha256=d5.base.PROFILE_HASH,task_sha256=d5.base.TASK_HASH,
        retained_equivalence=str(proof),retained_equivalence_sha256=digest(proof))
    d5.validate_design(plan)
    plan["configuration"]["pilot_seed_namespace"]="ghost-v23-s15-profile-successor-cost-v1"
    with pytest.raises(ValueError,match="fresh bounded"):
        d5.validate_design(plan)
    plan["configuration"]=d5.configuration()
    write(proof,dict(state="PARTIAL"),immutable=False)
    plan["retained_equivalence_sha256"]=digest(proof)
    with pytest.raises(ValueError,match="complete retained"):
        d5.validate_design(plan)


def test_heartbeat_throttling_never_throttles_resource_checks(monkeypatch,tmp_path):
    from runners import run_v24_pilot as runner
    clock=[0.];cpu=[0.];writes=[]
    monkeypatch.setattr(runner.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(runner.time,"process_time",lambda:cpu[0])
    monkeypatch.setattr(runner,"now",lambda:datetime.fromisoformat(contract.OPEN))
    monkeypatch.setattr(runner,"write",lambda *args,**kw:writes.append(args[1]))
    pulse=runner.progress_callback(tmp_path,"original",dict(cpu_cap_seconds=30),0)
    pulse(phase="starting")
    for i in range(100):pulse(phase="fit")
    assert len(writes)==1
    clock[0]=5;pulse(phase="fit")
    assert len(writes)==2
    cpu[0]=11
    with pytest.raises(TimeoutError,match="CPU"):pulse(phase="fit")
    assert len(writes)==2
    cpu[0]=0
    monkeypatch.setattr(runner,"now",lambda:datetime.fromisoformat(contract.CLOSE))
    with pytest.raises(TimeoutError,match="closed"):pulse(phase="fit")
