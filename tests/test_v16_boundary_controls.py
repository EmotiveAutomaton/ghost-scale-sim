import pytest
from ghostscale.validation.soundingline.v16.records import read,write
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.boundary_runner import execute as generate_known
from ghostscale.validation.soundingline.v16.boundary_control_runner import source_specification,execute
from ghostscale.validation.soundingline.v16.boundary_control_interruption import control
from ghostscale.validation.soundingline.v16.aggregate_science import registered_designs


def test_final_source_control_actual_interruption_and_unique_evidence_scope(tmp_path):
    source=tmp_path/"source"
    fixture_campaign(source)
    generate_known(source,lambda **kw:None,fixture="K01")
    specification,entries=source_specification(source,"boundary-fixture-K01")
    assert specification["planned_controls"]==3
    assert entries[0]["n_per_condition"]==64
    assert registered_designs(specification)=={}, "a control join must not register duplicate scientific samples"
    receipt=control(tmp_path/"runtime",source,"boundary-fixture-K01")
    assert receipt["instrument_state"]=="valid",receipt["checks"]
    assert receipt["condition_controls"]==3
    done=tmp_path/"runtime/campaign/boundary-controls-1/COMPLETION.json"
    assert read(done)["source_condition_records"]==192
    write(source/"boundary-fixture-K01/COMPLETION.json",{"execution_state":"running","instrument_state":"valid"},immutable=False)
    with pytest.raises(ValueError,match="completed valid allocation"):
        source_specification(source,"boundary-fixture-K01")


def test_final_source_control_missing_resume_never_prepares(tmp_path):
    fixture_campaign(tmp_path/"campaign")
    with pytest.raises(ValueError,match="existing packet"):
        execute(tmp_path/"campaign",lambda **kw:None,resume=True)
