import pytest
from ghostscale.validation.soundingline.v16.expansion_adapter import design,execute_unit
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.packet_controls import condition_control,METER

@pytest.mark.parametrize("card",["K01","K02","P04","R02"])
def test_fresh_source_controls_use_actual_guarded_and_metered_endpoints(tmp_path,card):
    source=tmp_path/"source"
    condition=design(card)["conditions"][0]
    packet={"packet_hash":"known-source-"+card,"identity":{"commission_hash":"fixture"}}
    required=set(design(card)["adversaries"])
    with ReaderProcess(tmp_path/"generation-reader",extensions=EXTENSIONS) as reader:
        row=execute_unit(card,source,condition,0,namespace="known-expansion-controls-"+card,
                         packet=packet,reader=reader,constructors=8,scope="fixture")
    with ReaderProcess(tmp_path/"control-reader",extensions=EXTENSIONS) as reader:
        result=condition_control(source,row,tmp_path/"control",reader,required)
    assert result["instrument_state"]=="valid"
    assert result["evidence"]["runtime"]=="pending whole-packet driver join"
    assert result["completed_local_checks"]["X01"]
    assert result["evidence"]["private_reads_denied"]==[True,True]
    if card=="K02":
        assert result["evidence"]["reading_meter_join"]
        assert all(request["actual_endpoint"]==METER for request in result["requests"] if request["requested_kind"]=="reading")
    class ChangedReader:
        child=reader.child
        def request(self,kind,public,**options):
            if kind==METER:
                return {"result":{"deliberately_changed_prediction":True},"invocations":{},"scope":"fixture"}
            return {"deliberately_changed_prediction":True}
    with pytest.raises(ValueError,match="committed prediction"):
        condition_control(source,row,tmp_path/"broken-control",ChangedReader(),required)
