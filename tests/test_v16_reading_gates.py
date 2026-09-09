from ghostscale.validation.soundingline.v16.reading_gates import run


def test_reader_gate_family_before_discovery():
    receipt = run()
    assert receipt["instrument_state"] == "valid", receipt
