from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.records import read, write, file_digest
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from runners.reaggregate_v16_incremental import completed_card, cached, run


def complete(root, name, card):
    base = root/name/card
    write(base/"AGGREGATE.json", {"card_id": card, "known": 1})
    write(base/"RAW_MANIFEST.json", {"files": {"known_points.json": "fixture"}})
    write(base/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid", "packet_hash": "known-packet",
        "aggregate_sha256": file_digest(base/"AGGREGATE.json"), "raw_manifest_sha256": file_digest(base/"RAW_MANIFEST.json")})


def test_partial_card_is_never_opened_even_when_an_unfinished_summary_exists(tmp_path):
    name, entry, packet = "boundary-expansion-1", {"card_id": "K01"}, {"packet_hash": "known-packet"}
    base = tmp_path/name/"K01"
    base.mkdir(parents=True)
    (base/"AGGREGATE.json").write_text("unfinished invalid JSON", encoding="utf-8")
    assert completed_card(tmp_path, name, packet, entry) is None
    write(base/"COMPLETION.json", {"execution_state": "running"})
    assert completed_card(tmp_path, name, packet, entry) is None


def test_completed_card_rejects_wrong_identity_and_changed_manifest(tmp_path):
    name, entry = "boundary-expansion-1", {"card_id": "K01"}
    complete(tmp_path, name, "K01")
    with pytest.raises(ValueError, match="another packet"):
        completed_card(tmp_path, name, {"packet_hash": "wrong"}, entry)
    assert len(completed_card(tmp_path, name, {"packet_hash": "known-packet"}, entry)) == 3
    write(tmp_path/name/"K01/RAW_MANIFEST.json", {"changed": True}, immutable=False)
    with pytest.raises(ValueError, match="manifest changed"):
        completed_card(tmp_path, name, {"packet_hash": "known-packet"}, entry)


def test_incremental_analysis_resumes_completed_cards_without_recalculating_or_claiming_partial_completion(tmp_path, monkeypatch):
    import runners.reaggregate_v16_incremental as module
    root, output, name = tmp_path/"campaign", tmp_path/"analysis", "boundary-expansion-1"
    fixture_campaign(root)
    entries = [{"card_id": card, "n_per_condition": 1, "design": {"card_id": card,
        "conditions": [{"id": "known"}], "primary": []}} for card in ["K01", "K03"]]
    write(root/"packets"/(name+".json"), {"packet_hash": "known-packet", "identity": {"design": {"cards": entries}}})
    monkeypatch.setattr(module, "source_locks", lambda *args: {"working_source": "valid known fixture"})
    calls = []
    def calculate(base, summary, design, n):
        calls.append(design["card_id"])
        return {"card_id": design["card_id"], "calculation": {"n_raw_units": n}}
    monkeypatch.setattr(module, "scientific", calculate)
    monkeypatch.setattr(module, "inventory", lambda *args: (root/"packets"/(name+".json"), root/name/"COMPLETION.json",
        [(root/name/entry["card_id"]/"AGGREGATE.json", entry["design"], 1) for entry in entries]))
    complete(root, name, "K01")
    first = run(root, output, name)
    assert first["execution_state"] == "checkpointed" and first["waiting_card"] == "K03"
    assert not (output/"RECEIPT.json").exists() and calls == ["K01"]
    saved = (output/"items/K01/CALCULATION.json").read_bytes()
    complete(root, name, "K03")
    write(root/name/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid"})
    final = run(root, output, name, resume=True)
    assert final["execution_state"] == "completed" and final["raw_records"] == 2
    assert calls == ["K01", "K03"] and (output/"items/K01/CALCULATION.json").read_bytes() == saved
    before = {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}
    assert run(root, output, name, resume=True) == final
    assert before == {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob("*") if p.is_file()}
    (output/"items/K01/CALCULATION.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="retained independent"):
        run(root, output, name, resume=True)
