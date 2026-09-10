from copy import deepcopy
import pytest

from ghostscale.validation.soundingline.v16.study_products import project_card, confirmation_join, run
from ghostscale.validation.soundingline.v16.records import write


def known():
    definition = {"card_id": "K01", "conditions": [{"id": "low", "budget": 8}, {"id": "high", "budget": 128}],
                  "access_arms": {"personal": "own training", "generic": "same count generic training"}}
    estimand = {"id": "personal-generic", "arm": "personal", "rival": "generic", "target": "success",
                "units": "fraction", "practical_bar": .05, "numerator": "paired success difference",
                "denominator": "makers including failures", "aggregation": "equal maker mean",
                "interval_target": "constructor-clustered maker mean"}
    summary = {"card_id": "K01", "n_maker_packets": 512, "conditions": {}}
    for condition, mean, state in [("low", .25, "held"), ("high", .01, "failed")]:
        summary["conditions"][condition] = {"arms": {"personal": {"success_mean": .9, "search_primitives_mean": 8},
            "generic": {"success_mean": .9-mean, "search_primitives_mean": 16}},
            "contrasts": [{"estimand": estimand, "mean": mean, "n_makers": 256, "n_constructors": 32,
                "interval_95": [mean-.01, mean+.01], "bootstrap": {"method": "constructors then makers"}, "criterion_state": state}]}
    disposition = {"card_id": "K01", "execution_state": "completed", "instrument_state": "valid",
                   "n_per_condition": 256, "final_source": "constructor/K01",
                   "criterion_counts": {"held": 1, "failed": 1, "inconclusive": 0}}
    return disposition, definition, summary


def test_report_preserves_opposite_regimes_and_access_without_pooling():
    disposition, definition, summary = known()
    card = project_card(disposition, definition, summary)
    assert [row["mean"] for row in card["comparisons"]] == [.25, .01]
    assert [row["criterion_state"] for row in card["comparisons"]] == ["held", "failed"]
    assert all(row["n_makers"] == 256 for row in card["comparisons"])
    assert card["performance_cost"][0]["access_contract"]["access_arms"] == definition["access_arms"]
    assert card["confirmed_claims"] == []
    assert "not an equivalence" in card["criterion_scope"]


def test_report_refuses_omitted_conditions_changed_n_and_untyped_statistics():
    disposition, definition, summary = known()
    for mutation in ["condition", "denominator", "units", "criterion"]:
        bad = deepcopy(summary)
        if mutation == "condition":
            del bad["conditions"]["high"]
        elif mutation == "denominator":
            bad["conditions"]["low"]["contrasts"][0]["n_makers"] = 512
        elif mutation == "units":
            del bad["conditions"]["low"]["contrasts"][0]["estimand"]["units"]
        else:
            bad["conditions"]["low"]["contrasts"][0]["criterion_state"] = "failed"
        with pytest.raises(ValueError):
            project_card(disposition, definition, bad)


def test_confirmation_failure_stays_joined_to_only_its_frozen_regime():
    card = project_card(*known())
    selected = [{"claim_id": "C01-K01", "card_id": "K01", "condition_ids": ["low"],
        "n_constructor_packets": 320, "total_fresh_acquisition_histories": 320}]
    outcome = {"claim_id": "C01-K01", "card_id": "K01", "p": .8, "rejected": False}
    ledger = {"claims": [outcome], "frozen_primary_count": 1, "familywise_alpha": .05, "no_failed_claim_replaced": True}
    primary = {"C01-K01": {"card_id": "K01", "primary_p": .8,
        "n_independent_constructor_packets": 320, "total_fresh_acquisition_histories": 320}}
    confirmation_join([card], selected, ledger, primary)
    assert card["confirmed_claims"][0]["multiplicity_outcome"]["rejected"] is False
    assert card["confirmed_claims"][0]["claim"]["condition_ids"] == ["low"]
    assert card["comparisons"][1]["condition"] == "high"
    bad = deepcopy(primary)
    bad["C01-K01"]["card_id"] = "S02"
    with pytest.raises(ValueError, match="identity"):
        confirmation_join([card], selected, ledger, bad)
    with pytest.raises(ValueError, match="frozen claim family"):
        confirmation_join([card], selected, {**ledger, "claims": []}, primary)


def test_report_cannot_close_from_a_running_ladder_or_overwrite_a_previous_attempt(tmp_path):
    root = tmp_path/"campaign"
    write(root/"expansion-closure/COMPLETION.json", {"execution_state": "running", "instrument_state": "valid"})
    with pytest.raises(ValueError, match="completed finite"):
        run(root, tmp_path/"new")
    assert not (tmp_path/"new").exists()
    (tmp_path/"old").mkdir()
    with pytest.raises(ValueError, match="new retained"):
        run(root, tmp_path/"old")
