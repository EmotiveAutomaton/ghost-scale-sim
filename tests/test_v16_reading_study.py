import pytest
from ghostscale.validation.soundingline.v16.reading_study import execute_unit,summarize,DESIGNS
from ghostscale.validation.soundingline.v16.audit_reading import audit


def test_paired_future_scores_are_independently_regenerated(tmp_path):
    rows = [execute_unit(tmp_path,"K02",DESIGNS["K02"]["conditions"][0],index,
                         namespace="prospective-fixture",packet_hash="fixture",
                         constructors=8,evidence_scope="fixture") for index in range(16)]
    summary = summarize("K02",rows)
    assert audit(tmp_path,summary)["all_reported_aggregates_reproduced"]
    assert abs(summary["conditions"]["prior-4"]["contrasts"][1]["mean"]) < 1e-10
    summary["conditions"]["prior-4"]["contrasts"][0]["interval_95"][1] += 0.1
    with pytest.raises(ValueError,match="hierarchical"):
        audit(tmp_path,summary)


def test_process_tool_and_prior_work_observations_are_realized(tmp_path):
    for condition in DESIGNS["P02"]["conditions"]:
        row = execute_unit(tmp_path,"P02",condition,0,namespace="query-fixture",
                           packet_hash="fixture",constructors=8,evidence_scope="fixture")
        if condition["query"] == "tool":
            assert row["public"]["permitted_prior_artifacts"][0]["feasible"] == list(range(1,8))
        if condition["query"] == "process":
            assert row["public"]["declared_context"]["current_observation"]["prefix"] == row["private"]["true_production_record"]["program"][:1]
