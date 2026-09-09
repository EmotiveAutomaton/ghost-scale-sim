from ghostscale.validation.soundingline.v16.self_trajectory import gates


def test_self_generated_evidence_is_conditioned_on_known_resets():
    result = gates()
    assert result["instrument_state"] == "valid", result
