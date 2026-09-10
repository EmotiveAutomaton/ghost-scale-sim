import copy
import json
import pytest
from ghostscale.validation.soundingline.v16.aggregate_controls import recoding, hierarchy
from ghostscale.validation.soundingline.v16.records import write
from ghostscale.validation.soundingline.v16.dependence import hierarchical_control

def test_control_recount_does_not_confuse_frames_calls_and_units(tmp_path):
    base=tmp_path/"recoding-attack-fixture-1"
    checks=[{"encodings":[{},{}],"physical":{"passed":True}},
            {"encodings":[{},{}],"physical":{}}]
    write(base/"units/a_points.json",{"card_id":"K01","checks":checks})
    write(base/"private/TRANSFER_points.json",[{"representation":[{},{}],"physical":{"passed":True}}])
    report={"source_units":1,"consumer_requests_checked":{"K01":2,"B01":1},"actual_reader_requests":8}
    write(base/"COMPLETION.json",report)
    assert recoding(tmp_path)==report
    report["actual_reader_requests"]=3
    (base/"COMPLETION.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="actual_reader_requests"):
        recoding(tmp_path)

def test_independent_known_hierarchy_rebuilds_interval_and_rejects_false_precision(tmp_path):
    record=hierarchical_control()
    write(tmp_path/"private/HIERARCHICAL_CONTROL_points.json",record)
    checked=hierarchy(tmp_path)
    assert checked["n_makers"]==64
    assert checked["n_constructors"]==8
    record["valid_summary"]["interval_95"]=[-.01,.01]
    (tmp_path/"private/HIERARCHICAL_CONTROL_points.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="hierarchy interval"):
        hierarchy(tmp_path)

