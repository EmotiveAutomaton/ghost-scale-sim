from ghostscale.validation.soundingline.v16.self_gates import run


def test_self_repair_and_goal_estimand_gates():
    receipt = run()
    assert receipt["instrument_state"] == "valid", receipt
