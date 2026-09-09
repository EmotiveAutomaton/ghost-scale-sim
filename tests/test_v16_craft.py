from ghostscale.validation.soundingline.v16.craft_gates import run
from ghostscale.validation.soundingline.v16.craft import DESIGN
from ghostscale.validation.soundingline.v16.study import unit, summarize
from ghostscale.validation.soundingline.v16.audit_craft import audit


def test_duplicate_maker_cannot_inflate_the_analysis(tmp_path):
    import pytest
    row = unit(tmp_path, DESIGN["conditions"][1], 0, packet_hash="fixture",
               namespace="duplicate-fixture", constructors=8, evidence_scope="fixture")
    with pytest.raises(ValueError, match="duplicate maker"):
        summarize([row, row])


def test_acquisition_cost_access_and_estimand_controls():
    receipt = run()
    assert receipt["instrument_state"] == "valid", receipt


def test_independent_regeneration_of_paired_hierarchical_interval(tmp_path):
    rows = [unit(tmp_path, DESIGN["conditions"][1], index, packet_hash="fixture",
                 namespace="craft-test-fixture", constructors=8, evidence_scope="fixture")
            for index in range(16)]
    summary = summarize(rows)
    receipt = audit(tmp_path, summary)
    assert receipt["all_reported_aggregates_reproduced"]
    summary["conditions"]["budget-32"]["contrasts"][0]["interval_95"][0] -= 0.1
    import pytest
    with pytest.raises(ValueError, match="hierarchical interval"):
        audit(tmp_path, summary)
