"""Bind reviewed final prose to its unchanged documents and scientific proofs.

Mechanical text/provenance checks complement the retained interpretive review;
they do not independently establish whether a scientific interpretation is true.
"""
from pathlib import Path
from .records import read, write, file_digest, now
from .completion_guard import bound_receipt
from .operational_queue import PROOF_FIELDS

DOCUMENTS = {"findings": "FINDINGS.md", "theory": "docs/theory/READING_INTENT.md",
    "theory_format": "docs/theory/README.md", "methods": "docs/METHODS.md",
    "study": "docs/versions/v16-acquired-craft/RESULTS.md",
    "transfer": "docs/versions/v16-acquired-craft/TRANSFER.md",
    "version_index": "docs/versions/v16-acquired-craft/README.md", "public_index": "README.md"}
KINDS = {"method result", "constructed mechanism result", "missing measurement"}
REVIEWS = {"scope_and_measurement", "criteria_and_confirmation", "theory_owners_and_afterwords",
           "transfer_interface_limits", "preserved_failures", "next_discriminator"}


def documents(repo, references, headlines, owner_updates):
    if set(references) != set(DOCUMENTS)|{"exchange"}:
        raise ValueError("documentary write-through requires both public channels, methods, study, transfer and exchange")
    texts = {}
    for role, item in references.items():
        relative = Path(item["path"])
        path = (repo/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(repo.resolve()):
            raise ValueError("documentary evidence escapes the repository")
        if role in DOCUMENTS and relative.as_posix() != DOCUMENTS[role]:
            raise ValueError("documentary proof points at a substitute owner")
        if role == "exchange" and (relative.parent.as_posix() != "docs/exchange" or relative.suffix != ".md"):
            raise ValueError("final exchange must use the existing exchange convention")
        if not path.is_file() or file_digest(path) != item["sha256"]:
            raise ValueError("reviewed documentary bytes changed or are missing")
        texts[role] = path.read_text(encoding="utf-8")
    if not headlines or any(row.get("kind") not in KINDS or not row.get("text") for row in headlines):
        raise ValueError("each final headline needs an explicit result/measurement class")
    for headline in headlines:
        for role in ["study", "exchange"]:
            if headline["text"] not in texts[role]:
                raise ValueError("study and exchange do not retain the same headline wording")
    if not owner_updates:
        raise ValueError("theory write-through lacks its changed owners and afterwords")
    for owner in owner_updates:
        if not all(owner.get(key) and owner[key] in texts["theory"] for key in ["section", "evidence_row", "afterword"]):
            raise ValueError("theory owner, evidence row or revised afterword is absent")
    for role in ["findings", "theory", "methods", "study", "transfer", "version_index", "public_index", "exchange"]:
        if "V16" not in texts[role]:
            raise ValueError("required owner does not identify the current commission")
    return {role: {**references[role], "bytes": (repo/references[role]["path"]).stat().st_size}
            for role in references}


def run(repo, root, output, input_path):
    if output.exists() or not output.resolve().is_relative_to((root/"closeout").resolve()):
        raise ValueError("documentary proof requires a new administrative closeout attempt")
    inputs = read(input_path)
    proofs = inputs["proofs"]
    expected = {"raw_archive", "independent_aggregates", "scientific_replay"}
    if set(proofs) != expected:
        raise ValueError("documentary closeout awaits all three scientific retention/reproduction proofs")
    for name, item in proofs.items():
        receipt = bound_receipt(root, item["path"], item["sha256"])
        if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt.get(PROOF_FIELDS[name]) is not True:
            raise ValueError("documentary closeout cannot relabel an incomplete scientific proof")
    if set(inputs["interpretive_review"]) != REVIEWS or any(not value.strip() for value in inputs["interpretive_review"].values()):
        raise ValueError("documentary closeout lacks the retained interpretive review")
    study_ref = inputs["study_projection"]
    study = bound_receipt(root, study_ref["path"], study_ref["sha256"])
    if study.get("execution_state") != "completed" or study.get("instrument_state") != "valid" or study["native_cards"] != 30:
        raise ValueError("documentary closeout needs the completed native numerical index")
    study_base = (root/study_ref["path"]).parent
    for name, expected_sha in study["output_hashes"].items():
        path = (study_base/name).resolve()
        if not path.is_relative_to(study_base.resolve()) or file_digest(path) != expected_sha:
            raise ValueError("documentary numerical table changed after generation")
    for name, expected_sha in study["input_hashes"].items():
        bound_receipt(root, name, expected_sha)
    checked = documents(repo, inputs["documents"], inputs["headlines"], inputs["theory_owner_updates"])
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "write_through_verified": True, "documents": checked, "proofs": proofs,
        "study_projection": study_ref, "headlines": inputs["headlines"],
        "theory_owner_updates": inputs["theory_owner_updates"], "interpretive_review": inputs["interpretive_review"],
        "mechanical_scope": "Reviewed byte identities, required owners, exact study/exchange headlines, updated theory rows/afterwords, completed scientific proof and numerical-table joins",
        "semantic_scope": "Interpretation is the recorded agent review, not an independent automated truth judgment",
        "inputs_sha256": file_digest(input_path), "producer_sha256": file_digest(Path(__file__)),
        "campaign_complete": False, "remaining_step": "All-card accounting and the ordinary supervisor's completion guard"}
    write(output/"RECEIPT.json", result)
    return result
