from ghostscale.validation.soundingline.v16.confirmation_candidates import allocate


def test_actual_estimand_power_and_history_cap_precede_frozen_selection():
    null = {"card_id": "S02", "kind": "equivalence", "rank": [0],
        "histories_per_constructor_packet": 2, "grouping": "two histories in each independent constructor packet"}
    capability = {"card_id": "K01", "kind": "capability", "rank": [2],
        "histories_per_constructor_packet": 1, "discovery_variance": .25,
        "estimands": [{"practical_bar": .05}]}
    costly = {**capability, "card_id": "M01", "rank": [1],
        "histories_per_constructor_packet": 2, "discovery_variance": .95}
    duplicate = {**capability, "rank": [3]}
    result = allocate([duplicate, capability, costly, null], maximum_claims=2)
    assert [row["card_id"] for row in result["selected"]] == ["S02", "K01"]
    assert all(row["total_fresh_acquisition_histories"] <= 4096 for row in result["selected"])
    assert all(row["power"]["planning_state"] == "adequate" for row in result["selected"])
    assert result["selected"][0]["n_constructor_packets"] == 320
    failed = next(row for row in result["candidate_dispositions"] if row["candidate"]["card_id"] == "M01")
    assert failed["state"] == "exploratory"
    assert failed["power"]["n_independent_makers"] == 2048
    assert failed["power"]["planning_state"] == "inadequate"
    assert result["confirmation_data_accessed"] is False


def test_empty_candidate_set_is_an_explicit_empty_selection():
    result = allocate([])
    assert not result["selected"] and not result["confirmation_data_accessed"]
