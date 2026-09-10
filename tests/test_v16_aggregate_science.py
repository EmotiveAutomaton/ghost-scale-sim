import copy
import json
from pathlib import Path
import pytest
from ghostscale.validation.soundingline.v16.aggregate_science import validate_layout, registered_designs, scientific, native_fixture
from ghostscale.validation.soundingline.v16.craft import DESIGN
from ghostscale.validation.soundingline.v16.study import unit, summarize
from ghostscale.validation.soundingline.v16.records import write, read

def known_packet(tmp_path):
    design = copy.deepcopy(DESIGN)
    design["conditions"] = [design["conditions"][1]]
    rows = [unit(tmp_path, design["conditions"][0], i, packet_hash="closeout-fixture",
                 namespace="closeout-known-fixture", constructors=4, evidence_scope="fixture") for i in range(8)]
    summary = summarize(rows)
    return rows, summary, design


def test_independent_closeout_rebuilds_actual_physics_and_intervals(tmp_path):
    rows, summary, design = known_packet(tmp_path)
    write(tmp_path/"SUMMARY.json", summary)
    actual = scientific(tmp_path, tmp_path/"SUMMARY.json", design, 8)
    assert actual["calculation"]["n_raw_units"] == 8
    broken = copy.deepcopy(summary)
    broken["conditions"]["budget-32"]["contrasts"][0]["interval_95"][0] -= .1
    (tmp_path/"BROKEN.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="interval"):
        scientific(tmp_path, tmp_path/"BROKEN.json", design, 8)


def test_closeout_rejects_omitted_conditions_contrasts_and_denominators(tmp_path):
    rows, summary, design = known_packet(tmp_path)
    with pytest.raises(ValueError, match="condition set"):
        validate_layout(rows, summary, DESIGN, 8)
    broken = copy.deepcopy(summary)
    broken["conditions"]["budget-32"]["contrasts"].pop()
    with pytest.raises(ValueError, match="contrast omitted"):
        validate_layout(rows, broken, design, 8)
    with pytest.raises(ValueError, match="denominator"):
        validate_layout(rows[:-1], summary, design, 8)
    broken = copy.deepcopy(summary)
    broken["conditions"]["budget-32"]["contrasts"][0]["estimand"]["practical_bar"] = 0
    with pytest.raises(ValueError, match="contrast omitted"):
        validate_layout(rows, broken, design, 8)


def test_registration_reads_both_frozen_design_shapes():
    compact = {"card_id": "S02", "conditions": [{"id":"a"}],
               "primary": [["self-model","bayes-error","net_repair","net_repair_fraction",.05]]}
    assert set(registered_designs({"cards":{"S02":compact,"K01":DESIGN}})) == {"S02","K01"}
    assert set(registered_designs({"cards":[{"design":compact}]})) == {"S02"}


def test_native_recount_uses_saved_predictions_and_rejects_truth_tampering(tmp_path):
    from ghostscale.validation.soundingline.v16.vertical import run_case
    row = run_case(tmp_path, packet_hash="closeout-native-fixture")
    expected = {"n":1, "legal_count":int(row["outcomes"]["legal"]), "success_count":int(row["outcomes"]["success"]),
                "ambiguous_history_count":int(row["outcomes"]["history_class_size"] > 1),
                "future_log_score_sum":row["outcomes"]["future_log_score"]}
    write(tmp_path/"AGGREGATE.json", expected)
    assert native_fixture(tmp_path)["reproduced"] == expected
    path = tmp_path/"units"/(row["unit_id"]+"_points.json")
    row["truth"]["future_artifact"] = (row["truth"]["future_artifact"]+1)%16
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="embedded source"):
        native_fixture(tmp_path)

