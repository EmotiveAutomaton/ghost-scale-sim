import pytest
from ghostscale.validation.soundingline.v16.behavior_designs import DESIGNS
from ghostscale.validation.soundingline.v16.behavior_study import execute_unit,summarize
from ghostscale.validation.soundingline.v16.audit_behavior import audit

PACKET={"packet_hash":"fixture","identity":{"commission_hash":"fixture","environment":{"python":"fixture"}}}


@pytest.mark.parametrize("card",list(DESIGNS))
def test_behavioral_scores_and_intervals_reaggregate_from_real_units(tmp_path,card):
    rows=[]
    for condition in DESIGNS[card]["conditions"]:
        for index in range(2):
            rows.append(execute_unit(tmp_path,card,condition,index,namespace="behavior-fixture-"+card,
                                     packet=PACKET,constructors=2,evidence_scope="fixture"))
    summary=summarize(card,rows)
    receipt=audit(tmp_path,summary)
    assert receipt["all_reported_aggregates_reproduced"]
    first=next(iter(summary["conditions"].values()))["contrasts"][0]
    first["interval_95"][0]-=0.1
    with pytest.raises(ValueError,match="interval"):
        audit(tmp_path,summary)


def test_trajectory_resume_preserves_round_predictions_and_submitted_observations(tmp_path):
    condition=DESIGNS["S05"]["conditions"][1]
    row=execute_unit(tmp_path,"S05",condition,0,namespace="trajectory-resume-fixture",
                     packet=PACKET,constructors=2,evidence_scope="fixture")
    prior={path.name:path.read_bytes() for path in (tmp_path/"predictions").glob("*.json")}
    # Simulate only loss of the final unit checkpoint, retaining actual submissions.
    # A real process interruption test separately covers the OS/runtime boundary.
    (tmp_path/"units"/(row["unit_id"]+"_points.json")).unlink()
    resumed=execute_unit(tmp_path,"S05",condition,0,namespace="trajectory-resume-fixture",
                         packet=PACKET,constructors=2,evidence_scope="fixture")
    assert resumed["arms"]==row["arms"]
    assert {path.name:path.read_bytes() for path in (tmp_path/"predictions").glob("*.json")}==prior
