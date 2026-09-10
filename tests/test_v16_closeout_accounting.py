from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.closeout_accounting import native_state, build, ORIGINAL_CONTROLS
from ghostscale.validation.soundingline.v16.expansion_adapter import CARDS
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign
from ghostscale.validation.soundingline.v16.operational_queue import PROOF_FIELDS
from ghostscale.validation.soundingline.v16.records import write, file_digest, read


def native(card="K01"):
    return {"card_id": card, "criterion_counts": {"held": 1, "failed": 1, "inconclusive": 0},
        "confirmed_claims": [], "comparisons": [{"n_constructors": 32}], "definition": {"access_arms": "same public evidence"},
        "final_source": "constructor-expansion-1/"+card, "n_per_condition": 256,
        "source_qualifications": {"historical_inference": "not established by execution alone"},
        "expansion_disposition": {"additional_expansions_allowed": 0}}


def fixture(root):
    fixture_campaign(root)
    commissioned = [*CARDS, *ORIGINAL_CONTROLS, "B01", "B02", "B03", "B04"]
    write(root/"COMMISSION_MANIFEST.json", {"cards": [{"card_id": cid, "question_and_comparison": "Known accounting fixture",
        "dependency_ids": ["X01"] if cid in CARDS or cid == "B01" else []} for cid in commissioned]})
    valid = {"execution_state": "completed", "instrument_state": "valid"}
    for cid in CARDS:
        write(root/"constructor-expansion-1"/cid/"COMPLETION.json", valid)
        write(root/"constructor-controls-1"/cid/"COMPLETION.json", {"instrument_state": "valid", "required_adversaries": ["X01"]})
    write(root/"study/NATIVE_CARDS.json", {"cards": [native(cid) for cid in CARDS]})
    write(root/"study/RECEIPT.json", {**valid, "output_hashes": {"NATIVE_CARDS.json": file_digest(root/"study/NATIVE_CARDS.json")}, "input_hashes": {}})
    for attack, entries in ORIGINAL_CONTROLS.items():
        for packet, key in entries:
            write(root/packet/"COMPLETION.json", {**valid, key: [*CARDS, "B01"]})
    for name in ["constructor-controls-1", "boundary-controls-1", "confirmation", "case-catalogue-2", "confirmation-selection"]:
        write(root/name/"COMPLETION.json", valid)
    write(root/"transfer-handoff-1/RECEIPT.json", {**valid,
        "all_payload_bytes_verified": True, "extracted_consumer_executed": True,
        "extracted_known_prediction_matched": True, "extracted_private_read_denied": True,
        "actual_consumer_controls": {"access-attack-fixture-2": {"sha256": file_digest(root/"access-attack-fixture-2/COMPLETION.json")}}})
    write(root/"archive-search-1/CONSUMER_ATTACKS.json", {"instrument_state": "valid", "attack_checks": {"X07": True}})
    write(root/"archive-search-1/COMPLETION.json", {**valid, "consumer_attack_sha256": file_digest(root/"archive-search-1/CONSUMER_ATTACKS.json")})
    write(root/"archive-search-1/private/runtime-control/RECEIPT.json", {"instrument_state": "valid", "checks": {"interrupted_then_resumed": True}})
    proofs = {}
    for name, field in PROOF_FIELDS.items():
        write(root/"proofs"/(name+".json"), {**valid, field: True, "scope": "known accounting fixture, not scientific campaign evidence"})
        proofs[name] = {"path": "proofs/"+name+".json", "sha256": file_digest(root/"proofs"/(name+".json"))}
    return proofs


def test_final_accounting_covers_all_42_without_whole_card_confirmation(tmp_path):
    proofs = fixture(tmp_path)
    inputs, result = build(tmp_path, "study", proofs)
    assert result["campaign_closed"] and result["card_count"] == 42
    assert len(inputs["expansions"]) == 42
    card = next(row for row in inputs["cards"] if row["card_id"] == "K01")
    assert card["criterion_state"] == "inconclusive" and card["warrant_status"] == "MECHANISM CANDIDATE"
    assert card["confirmation_claims"] == []
    assert "not established" in card["source_qualifications"]["historical_inference"]


def test_accounting_refuses_missing_proof_and_real_source_dependency(tmp_path):
    proofs = fixture(tmp_path)
    bad = dict(proofs)
    del bad["scientific_replay"]
    with pytest.raises(ValueError, match="four distinct"):
        build(tmp_path, "study", bad)
    write(tmp_path/"constructor-controls-1/K01/COMPLETION.json", {"instrument_state": "valid", "required_adversaries": []}, immutable=False)
    with pytest.raises(ValueError, match="commissioned adversary"):
        build(tmp_path, "study", proofs)


def test_accounting_refuses_changed_report_and_failed_runtime_control(tmp_path):
    proofs = fixture(tmp_path)
    saved = (tmp_path/"study/NATIVE_CARDS.json").read_bytes()
    (tmp_path/"study/NATIVE_CARDS.json").write_bytes(saved+b"\n")
    with pytest.raises(ValueError, match="report changed"):
        build(tmp_path, "study", proofs)
    (tmp_path/"study/NATIVE_CARDS.json").write_bytes(saved)
    write(tmp_path/"archive-search-1/private/runtime-control/RECEIPT.json", {"instrument_state": "valid", "checks": {"interrupted_then_resumed": False}}, immutable=False)
    with pytest.raises(ValueError, match="interruption/resume"):
        build(tmp_path, "study", proofs)


def test_failed_confirmation_and_supplied_state_do_not_become_support():
    card = native()
    card["definition"]["family"] = "critic"
    card["confirmed_claims"] = [{"claim": {"kind": "capability", "condition_ids": ["only-one"]},
        "multiplicity_outcome": {"familywise_rejected": False}}]
    result = native_state(card)
    assert result["warrant_status"] == "DESCRIPTIVE ONLY" and result["pursuit_status"] == "EXHAUSTED"
    card["confirmed_claims"][0]["claim"]["kind"] = "equivalence"
    card["confirmed_claims"][0]["multiplicity_outcome"]["familywise_rejected"] = True
    assert native_state(card)["warrant_status"] == "BOUNDARY ESTABLISHED"
