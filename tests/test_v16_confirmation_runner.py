from ghostscale.validation.soundingline.v16.confirmation_interruption import control


def test_actual_confirmation_cli_preserves_partial_data_and_checks_its_source(tmp_path):
    result = control(tmp_path)
    assert result["instrument_state"] == "valid"
    assert all(result["checks"].values())
