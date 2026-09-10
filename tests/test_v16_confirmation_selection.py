import pytest
from ghostscale.validation.soundingline.v16 import confirmation_plan as planner
from ghostscale.validation.soundingline.v16.confirmation_runner import sources
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze
from ghostscale.validation.soundingline.v16.records import read, write, file_digest


def test_actual_selector_records_empty_reserve_and_rejects_changed_dependencies(tmp_path):
    # The upstream thirty-card disposition is tested separately. This known
    # ledger exercises selection persistence with a deliberately ineligible n.
    root = tmp_path/"campaign"
    fixture_campaign(root)
    freeze(root, "known-predecessor", [PACKAGE/"confirmation_plan.py"], {"scope": "known bookkeeping fixture"})
    summary = root/"known-source/K01/AGGREGATE.json"
    write(summary, {"n_maker_packets": 2, "conditions": {"known": {"contrasts": [{"n_constructors": 2}]}}})
    write(root/"expansion-closure/COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid",
        "all_thirty_dispositions_explicit": True, "source_hashes": {summary.relative_to(root).as_posix(): file_digest(summary)},
        "cards": [{"card_id": "K01", "final_source": "known-source/K01"}], "scope": "known upstream-ledger fixture"})
    write(root/"case-catalogue-2/COMPLETION.json", {"instrument_state": "valid"})
    write(root/"confirmation-setup/SELECTION_ADMISSION.json", {"instrument_state": "valid",
        "source_hashes": {path.relative_to(REPO).as_posix(): file_digest(path) for path in sources(root)}})
    result = planner.execute(root)
    assert result["empty_selection"] and result["selected_claim_count"] == 0
    assert not result["confirmation_data_opened"] and not result["campaign_complete"]
    assert planner.execute(root, resume=True) == result
    assert not (root/"confirmation").exists()
    assert not read(root/"confirmation-selection/CLAIMS.json")["selected"]
    write(summary, {"changed": True}, immutable=False)
    with pytest.raises(ValueError, match="evidence changed"):
        planner.execute(root, resume=True)
