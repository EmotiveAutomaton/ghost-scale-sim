import pytest
from ghostscale.validation.soundingline.v16.boundary_plan import choose
from ghostscale.validation.soundingline.v16.boundary_runner import execute
from ghostscale.validation.soundingline.v16.boundary_interruption import control
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.expansion_adapter import CARDS


def test_final_boundary_selection_keeps_nulls_harms_conditions_and_finite_limit():
    entries=[]
    summaries={}
    for index,card in enumerate(CARDS[:26]):
        entries.append({"card_id":card,"design":{"card_id":card,"conditions":[{"id":"known"}],
                        "primary":[["a","b","success","success_fraction",.05]]}})
        low,high=[(-.01,.01),(.04,.06),(-.08,-.03),(-.03,.03)][index] if index<4 else (.2,.3)
        estimand={"id":"primary","arm":"a","rival":"b","target":"success","units":"success_fraction","practical_bar":.05}
        primary={"estimand":estimand,"mean":(low+high)/2,"interval_95":[low,high],"criterion_state":"inconclusive"}
        secondary={"estimand":dict(estimand,id="secondary",target="unregistered"),"mean":0,"interval_95":[-.5,.5],"criterion_state":"inconclusive"}
        summaries[card]={"conditions":{"known":{"contrasts":[primary,secondary]}}}
    closed=[{"card_id":card,"reason":"known closed"} for card in CARDS[26:]]
    result=choose(entries,summaries,closed)
    assert [row["card_id"] for row in result["cards"]]==CARDS[1:4]
    assert len(result["dispositions"])==30
    assert result["planned_maker_condition_records"]==3*1024
    assert result["maximum_further_expansions_after_this"]==0
    assert all(row["constructors"]==128 and row["n_per_condition"]==1024 for row in result["cards"])
    assert all(row["design"]["conditions"]==[{"id":"known"}] for row in result["cards"])


@pytest.mark.parametrize("card",["K01","R02"])
def test_final_boundary_actual_interruption_resume(tmp_path,card):
    receipt=control(tmp_path/card,card)
    assert receipt["instrument_state"]=="valid",receipt["checks"]


def test_final_boundary_missing_resume_is_rejected(tmp_path):
    fixture_campaign(tmp_path/"campaign")
    with pytest.raises(ValueError,match="never prepares"):
        execute(tmp_path/"campaign",lambda **kw:None,resume=True,fixture="K01")
