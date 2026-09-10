import pytest
from ghostscale.validation.soundingline.v16.expansion_plan import plan, REASONS, CLOSED
from ghostscale.validation.soundingline.v16.expansion_adapter import CARDS
from ghostscale.validation.soundingline.v16.expansion_runner import execute
from ghostscale.validation.soundingline.v16.expansion_interruption import control, fixture_campaign


def test_expansion_plan_is_finite_balanced_and_explicit():
    result = plan()
    assert set(REASONS)|set(CLOSED) == set(CARDS) and not set(REASONS)&set(CLOSED)
    assert result["planned_maker_condition_records"] == 76800
    assert len(result["cards"]) == 26 and len(result["closed_at_scout"]) == 4
    for row in result["cards"]:
        assert row["n_per_condition"] == 256 and row["constructors"] == 32
        assert row["reason"] and row["design"]["conditions"]
    assert len({row["namespace"] for row in result["cards"]}) == 26


@pytest.mark.parametrize("card", ["K01", "R02"])
def test_expansion_actual_interruption_and_resume(tmp_path, card):
    result = control(tmp_path/card, card)
    assert result["instrument_state"] == "valid"
    assert all(result["checks"].values())


def test_expansion_resume_refuses_missing_packet(tmp_path):
    fixture_campaign(tmp_path/"campaign")
    with pytest.raises(ValueError, match="never prepares"):
        execute(tmp_path/"campaign", lambda **kw: None, resume=True, fixture="K01")
