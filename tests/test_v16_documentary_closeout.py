from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.documentary_closeout import documents, run, DOCUMENTS, REVIEWS
from ghostscale.validation.soundingline.v16.operational_queue import PROOF_FIELDS
from ghostscale.validation.soundingline.v16.records import write, file_digest


def document_fixture(repo):
    paths = {**DOCUMENTS, "exchange": "docs/exchange/2026-09-10-codex-v16-fixture.md"}
    headline = "V16 known fixture: execution does not identify a unique history."
    texts = "# V16 known document fixture\n\n"+headline+"\n\n## Inverse inference\nKnown evidence row.\nRevised qualified afterword.\n"
    refs = {}
    for role, name in paths.items():
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(texts, encoding="utf-8")
        refs[role] = {"path": name, "sha256": file_digest(path)}
    headlines = [{"kind": "method result", "text": headline}]
    owners = [{"section": "## Inverse inference", "evidence_row": "Known evidence row.", "afterword": "Revised qualified afterword."}]
    return refs, headlines, owners


def test_documentary_requires_identical_headlines_and_revised_theory_afterwords(tmp_path):
    refs, headlines, owners = document_fixture(tmp_path)
    assert len(documents(tmp_path, refs, headlines, owners)) == 9
    with pytest.raises(ValueError, match="same headline"):
        documents(tmp_path, refs, [{"kind": "method result", "text": "changed result"}], owners)
    with pytest.raises(ValueError, match="afterword"):
        documents(tmp_path, refs, headlines, [{**owners[0], "afterword": "not revised"}])
    with pytest.raises(ValueError, match="result/measurement class"):
        documents(tmp_path, refs, [{"kind": "human evidence", "text": headlines[0]["text"]}], owners)


def test_documentary_rejects_changed_bytes_missing_owner_and_path_escape(tmp_path):
    refs, headlines, owners = document_fixture(tmp_path)
    bad = deepcopy(refs)
    del bad["findings"]
    with pytest.raises(ValueError, match="both public channels"):
        documents(tmp_path, bad, headlines, owners)
    bad = deepcopy(refs)
    bad["exchange"]["path"] = "../outside.md"
    with pytest.raises(ValueError, match="escapes"):
        documents(tmp_path, bad, headlines, owners)
    (tmp_path/refs["theory"]["path"]).write_text("V16 old theory", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        documents(tmp_path, refs, headlines, owners)


def test_documentary_full_fixture_cannot_replace_missing_scientific_proof(tmp_path):
    repo, root = tmp_path, tmp_path/"results/v16"
    refs, headlines, owners = document_fixture(repo)
    valid = {"execution_state": "completed", "instrument_state": "valid"}
    proofs = {}
    for name in ["raw_archive", "independent_aggregates", "scientific_replay"]:
        path = root/"proofs"/(name+".json")
        write(path, {**valid, PROOF_FIELDS[name]: True, "scope": "known fixture only"})
        proofs[name] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    write(root/"study/NATIVE_CARDS.json", {"fixture": True})
    write(root/"study/RECEIPT.json", {**valid, "native_cards": 30,
        "output_hashes": {"NATIVE_CARDS.json": file_digest(root/"study/NATIVE_CARDS.json")}, "input_hashes": {}})
    inputs = {"proofs": proofs, "documents": refs, "headlines": headlines, "theory_owner_updates": owners,
        "interpretive_review": {name: "Explicit known-fixture review; no actual scientific claim" for name in REVIEWS},
        "study_projection": {"path": "study/RECEIPT.json", "sha256": file_digest(root/"study/RECEIPT.json")}}
    input_path = root/"closeout/document-inputs.json"
    write(input_path, inputs)
    result = run(repo, root, root/"closeout/known-documents-1", input_path)
    assert result["write_through_verified"] and not result["campaign_complete"]
    path = root/proofs["scientific_replay"]["path"]
    write(path, {**valid, "whole_unit_replay": False}, immutable=False)
    inputs["proofs"]["scientific_replay"]["sha256"] = file_digest(path)
    write(root/"closeout/bad-inputs.json", inputs)
    with pytest.raises(ValueError, match="incomplete scientific proof"):
        run(repo, root, root/"closeout/known-documents-2", root/"closeout/bad-inputs.json")
    assert not (root/"closeout/known-documents-2").exists()
