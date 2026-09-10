import pytest
from ghostscale.validation.soundingline.v16.expansion_adapter import CARDS, design, module, execute_unit, Calls, guarded_function
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.recoded_interface import OPERATIONS
from ghostscale.validation.soundingline.v16.consumer_frames import requests
from ghostscale.validation.soundingline.v16.fairness import audit_unit
from ghostscale.validation.soundingline.v16.records import read, file_digest, canonical

EXTENSIONS = [name for name in OPERATIONS if ":" in name]+["ghostscale.validation.soundingline.v16.reading_cost_meter:public_reader"]
PACKET = {"packet_hash": "fixture", "identity": {"commission_hash": "fixture", "environment": {"scope": "known fixture"}}}


@pytest.mark.parametrize("card", CARDS)
def test_all_expansion_families_keep_physics_scores_and_guarded_reader_outputs(tmp_path, card):
    base = tmp_path/card
    scientific = module(card)
    before = file_digest(__import__("pathlib").Path(scientific.__file__))
    with ReaderProcess(tmp_path/"reader", extensions=EXTENSIONS) as reader:
        row = execute_unit(card, base, design(card)["conditions"][0], 0,
            namespace="expansion-dispatch-known-fixture-"+card, packet=PACKET,
            reader=reader, constructors=32, scope="fixture", crosscheck=True)
        frames = list(requests(base, row))
        assert frames
        for frame in frames:
            assert reader.request(frame["kind"], frame["public"], **frame["options"]) == frame["result"]
        if card.startswith("R"):
            from ghostscale.validation.soundingline.v16.audit_inquiry_stable import audit_unit as inquiry_audit
            inquiry_audit(base, row)
        else:
            audit_unit(base, row, frames)
        saved = (base/"units"/(row["unit_id"]+"_points.json")).read_bytes()
        resumed = execute_unit(card, base, design(card)["conditions"][0], 0,
            namespace="expansion-dispatch-known-fixture-"+card, packet=PACKET,
            reader=reader, constructors=32, scope="fixture", crosscheck=True)
        assert canonical(resumed) == canonical(row)
        assert saved == (base/"units"/(row["unit_id"]+"_points.json")).read_bytes()
    assert file_digest(__import__("pathlib").Path(scientific.__file__)) == before
    resource = read(base/"private"/(row["unit_id"]+"-expansion-resources.json"))
    assert resource["requests"] and all(item["resource_state"] == "measured" for item in resource["requests"])
    if card in {"K02", "P01", "P02", "P03"}:
        assert len(resource["reading_invocation_corrections"]) == 5


def test_public_dispatch_uses_copied_bindings_and_rejects_hidden_fields(tmp_path):
    original = module("K01").construct_public
    with ReaderProcess(tmp_path/"reader", extensions=EXTENSIONS) as reader:
        calls = Calls(reader)
        copied = guarded_function("K01", calls)
        assert copied.__code__ is module("K01").unit.__code__
        assert copied.__globals__ is not module("K01").unit.__globals__
        assert module("K01").construct_public is original
        with pytest.raises(RuntimeError):
            copied.__globals__["construct_public"](b'{"private_answer": 1}')
