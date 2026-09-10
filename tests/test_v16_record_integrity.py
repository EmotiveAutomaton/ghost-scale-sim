import copy
import pytest
from ghostscale.validation.soundingline.v16.record_integrity import independent_units,raw_integrity
from ghostscale.validation.soundingline.v16.records import write

def row(index=0):
    return {"unit_id":f"unit-{index}","lineage":"fixture","card_id":"fixture","condition":"same-condition",
        "constructor_id":"same-constructor","maker_history_id":f"history-{index}","seed_components":{"index":index},"outcome":1}

def test_sampling_audit_rejects_renamed_duplicate_histories_and_seeds():
    original=row()
    renamed=copy.deepcopy(original);renamed["unit_id"]="renamed"
    with pytest.raises(ValueError,match="duplicate maker"):
        independent_units([original,renamed])
    renamed["maker_history_id"]="renamed-too"
    with pytest.raises(ValueError,match="duplicate maker"):
        independent_units([original,renamed])
    renamed["seed_components"]["index"]=1;renamed["maker_history_id"]=original["maker_history_id"]
    with pytest.raises(ValueError,match="duplicate maker"):
        independent_units([original,renamed])
    result=independent_units([original,row(1)])
    assert result["n_retained_unit_records"]==2
    assert result["constructor_groups"][0]["makers"]==2

def test_sampling_audit_allows_coincident_artifacts_and_retains_context_groups():
    first=row();second=row(1)
    third=copy.deepcopy(first);third["unit_id"]="other-condition";third["condition"]="paired-condition"
    result=independent_units([first,second,third])
    assert len(result["constructor_groups"])==2
    assert sum(group["makers"] for group in result["constructor_groups"])==3
    with pytest.raises(ValueError,match="empty"):
        independent_units([])

def test_raw_audit_rejects_missing_changed_and_extra_files(tmp_path):
    packet=tmp_path/"fixture-packet"
    record={**row(),"packet_hash":"packet"}
    expected=write(packet/"units/unit-0_points.json",record)
    manifest={"files":{"units/unit-0_points.json":expected}}
    write(packet/"RAW_MANIFEST.json",manifest)
    assert raw_integrity(tmp_path,{"fixture-packet":"packet"})["raw_files_checked"]==1
    write(packet/"private/extra.json",{"new":True})
    with pytest.raises(ValueError,match="unmanifested"):
        raw_integrity(tmp_path,{"fixture-packet":"packet"})
    manifest["files"]["private/extra.json"]="0"*64
    write(packet/"RAW_MANIFEST.json",manifest,immutable=False)
    with pytest.raises(ValueError,match="bytes differ"):
        raw_integrity(tmp_path,{"fixture-packet":"packet"})
