"""Final 42-card accounting from actual completed evidence and four proofs."""
from copy import deepcopy
from pathlib import Path
from .records import read, write, file_digest, now
from .completion_guard import bound_receipt, PROOFS
from .operational_queue import PROOF_FIELDS, verify_closeout

ORIGINAL_CONTROLS = {
    "X01": [("access-attack-fixture-1", "consumer_cards"), ("access-attack-fixture-2", "consumer_cards")],
    "X02": [("recoding-attack-fixture-1", "consumer_requests_checked")],
    "X03": [("collision-attack-fixture-1", "source_condition_calibrations")],
    "X04": [("fairness-attack-fixture-1", "native_consumer_condition_profiles")],
    "X05": [("context-attack-fixture-1", "source_condition_calibrations")],
    "X06": [("misspecification-attack-fixture-1", "source_condition_calibrations")],
    "X07": [("dependence-attack-fixture-1", "native_consumer_condition_calibrations")],
    "X08": [("noise-runtime-attack-fixture-1", "native_consumer_condition_calibrations")],
}


def native_state(card):
    """A card is an inventory of claims, never one pooled criterion."""
    counts = card["criterion_counts"]
    states = [name for name, count in counts.items() if count]
    if not states or set(states)-{"held", "failed", "inconclusive"}:
        raise ValueError("native accounting lacks explicit criterion dispositions")
    confirmed = [item for item in card["confirmed_claims"] if item["multiplicity_outcome"]["familywise_rejected"]]
    criterion = states[0] if len(states) == 1 else "inconclusive"
    constructors = min(item["n_constructors"] for item in card["comparisons"])
    if confirmed:
        warrant = "BOUNDARY ESTABLISHED" if all(item["claim"]["kind"] == "equivalence" for item in confirmed) else "CONFIRMATORY SUPPORT"
    elif counts["held"] and constructors >= 20 and card["definition"].get("family") != "critic":
        warrant = "MECHANISM CANDIDATE"
    else:
        warrant = "DESCRIPTIVE ONLY"
    return {"criterion_state": criterion,
        "criterion_state_scope": "Unanimous condition/comparator state when possible; mixed states are represented as inconclusive and retained individually",
        "criterion_counts": deepcopy(counts),
        "scientific_criterion": "supported" if counts["held"] or confirmed else "boundary",
        "scientific_criterion_scope": "Accounting category for registered comparisons; failed positive criteria are never relabeled as equivalence or a population null",
        "warrant_status": warrant, "warrant_scope": "Only the explicit held discovery comparisons or joined frozen confirmation claims",
        "pursuit_status": "PROMOTE" if confirmed else "EXHAUSTED",
        "pursuit_scope": "Finite V16 sampling exhausted; PROMOTE recommends the validated comparison for a future eligible interface test, without launching one",
        "confirmation_claims": deepcopy(card["confirmed_claims"]),
        "evidence_scope": "confirmation" if card["confirmed_claims"] else "discovery"}


def build(root, study_relative, proof_references, supplemental_controls=None):
    if set(proof_references) != set(PROOFS):
        raise ValueError("final accounting requires all four distinct proof references")
    def reference(name):
        return {"path": name, "sha256": file_digest(root/name)}
    def checked(ref, *, completed=True):
        value = bound_receipt(root, ref["path"], ref["sha256"])
        if value.get("instrument_state") != "valid" or (completed and value.get("execution_state") != "completed"):
            raise ValueError("final accounting encountered incomplete or invalid evidence: "+ref["path"])
        return value
    for name, ref in proof_references.items():
        if checked(ref).get(PROOF_FIELDS[name]) is not True:
            raise ValueError("final accounting encountered an incomplete "+name+" proof")
    supplements = {}
    if supplemental_controls is not None:
        from .commission_control_audit_v2 import verify
        supplement = verify(root, supplemental_controls)
        supplements = {row["card_id"]: row for row in supplement["specification"]["gaps"]}
    study_ref = reference(study_relative+"/RECEIPT.json")
    study = checked(study_ref)
    for name, expected in study["output_hashes"].items():
        path = (root/study_relative/name).resolve()
        if not path.is_relative_to((root/study_relative).resolve()) or file_digest(path) != expected:
            raise ValueError("final numerical report changed after generation")
    for name, expected in study["input_hashes"].items():
        bound_receipt(root, name, expected)
    native = read(root/study_relative/"NATIVE_CARDS.json")["cards"]
    manifest = read(root/"COMMISSION_MANIFEST.json")["cards"]
    native_ids = {row["card_id"] for row in manifest if not row["card_id"].startswith(("X", "B"))}
    if len(native) != len(native_ids) or {row["card_id"] for row in native} != native_ids:
        raise ValueError("final accounting omitted or duplicated a native card")
    definitions = {row["card_id"]: row for row in manifest}
    original = {}
    for attack, entries in ORIGINAL_CONTROLS.items():
        original[attack] = [(reference(packet+"/COMPLETION.json"), key) for packet, key in entries]
        for ref, key in original[attack]:
            if key not in checked(ref):
                raise ValueError("original control omitted its consumer inventory")
    expansion_refs = [reference(name+"/COMPLETION.json") for name in ["constructor-controls-1", "boundary-controls-1"]]
    for ref in expansion_refs:
        checked(ref)
    confirmation_ref = reference("confirmation/COMPLETION.json")
    checked(confirmation_ref)
    cards, expansions = [], []
    for card in native:
        cid = card["card_id"]
        dependencies = definitions[cid]["dependency_ids"]
        evidence = [study_ref, reference(card["final_source"]+"/COMPLETION.json"), proof_references["independent_aggregates"]]
        checked(evidence[1])
        phase = card["final_source"].split("/")[0]
        if phase in {"constructor-expansion-1", "boundary-expansion-1"}:
            control = phase.replace("expansion", "controls")+"/"+cid+"/COMPLETION.json"
            ref = reference(control)
            record = checked(ref, completed=False)
            if not set(dependencies) <= set(record["required_adversaries"]):
                missing = set(dependencies)-set(record["required_adversaries"])
                extra = supplements.get(cid)
                if extra is None or extra["original_control"] != ref or set(extra["missing_adversaries"]) != missing:
                    raise ValueError("final source control omitted a commissioned adversary")
                evidence.append(supplemental_controls)
            evidence.append(ref)
        else:
            for attack in dependencies:
                matches = [ref for ref,key in original[attack] if cid in checked(ref)[key]]
                if not matches:
                    raise ValueError("unexpanded native source lacks its actual commissioned control")
                evidence.extend(matches)
        definition = card["definition"]
        access = {key: deepcopy(value) for key,value in definition.items() if "access" in key or "information" in key}
        if definition.get("family") == "critic":
            access["qualification"] = "Supplied-state resource benchmark; not learned-reader admission"
        cards.append({"card_id": cid, "question": definitions[cid]["question_and_comparison"],
            "execution_state": "completed", "instrument_state": "valid", **native_state(card),
            "access_tier": access, "dependency_ids": dependencies, "unresolved_dependencies": [],
            "evidence": evidence, "n_per_condition": card["n_per_condition"],
            "comparisons": deepcopy(card["comparisons"]), "source_qualifications": card["source_qualifications"]})
        expansions.append({"card_id": cid, "state": "exhausted",
            "reason": "The finite discovery ladder is exhausted; unresolved intervals remain unresolved",
            "detail": card["expansion_disposition"]})
    transfer_ref = reference("transfer-handoff-1/RECEIPT.json")
    transfer = checked(transfer_ref)
    if not all(transfer[name] is True for name in ["all_payload_bytes_verified", "extracted_consumer_executed", "extracted_known_prediction_matched", "extracted_private_read_denied"]):
        raise ValueError("B01 lacks its executed standalone handoff")
    for packet, item in transfer["actual_consumer_controls"].items():
        checked({"path": packet+"/COMPLETION.json", "sha256": item["sha256"]})
    archive_ref = reference("archive-search-1/COMPLETION.json")
    archive = checked(archive_ref)
    attack_ref = reference("archive-search-1/CONSUMER_ATTACKS.json")
    attacks = checked(attack_ref, completed=False)
    if archive["consumer_attack_sha256"] != attack_ref["sha256"] or not all(attacks["attack_checks"].values()):
        raise ValueError("B02 source controls changed or failed")
    runtime_ref = reference("archive-search-1/private/runtime-control/RECEIPT.json")
    runtime = checked(runtime_ref, completed=False)
    if not runtime.get("checks") or not all(runtime["checks"].values()):
        raise ValueError("B02 actual interruption/resume proof is incomplete")
    catalogue_ref = reference("case-catalogue-2/COMPLETION.json")
    checked(catalogue_ref)
    selection_ref = reference("confirmation-selection/COMPLETION.json")
    checked(selection_ref)
    for cid in sorted(set(definitions)-native_ids):
        if cid.startswith("X"):
            evidence = [ref for ref,_ in original[cid]]+expansion_refs+[confirmation_ref, proof_references["independent_aggregates"]]
            if cid in {"X04", "X07", "X08"}:
                evidence += [attack_ref, runtime_ref]
            if cid in definitions["B01"]["dependency_ids"]:
                evidence.append(transfer_ref)
            scope = "Known adversarial instruments plus actual source joins; earlier pending-consumer snapshots remain historical"
        elif cid == "B01":
            evidence, scope = [transfer_ref, proof_references["independent_aggregates"]], "Actual standalone synthetic task transfer; no real-text reader claim"
        elif cid == "B02":
            evidence, scope = [archive_ref, attack_ref, runtime_ref, catalogue_ref, proof_references["independent_aggregates"]], "Fixed/adaptive search and separate descriptive catalogue; no additional pooled sample"
        elif cid == "B03":
            evidence, scope = [selection_ref, confirmation_ref, proof_references["independent_aggregates"]], "Frozen finite claim selection, power and fresh-lineage identity; failures are not replaced"
        elif cid == "B04":
            evidence, scope = [confirmation_ref, *proof_references.values()], "Actual confirmations and all four distinct final proof producers"
        else:
            raise ValueError("unknown commissioned administrative/control card")
        cards.append({"card_id": cid, "question": definitions[cid]["question_and_comparison"],
            "execution_state": "completed", "instrument_state": "valid", "criterion_state": "not_applicable",
            "scientific_criterion": "not_applicable", "evidence_scope": "transfer" if cid == "B01" else "fixture",
            "access_tier": scope, "dependency_ids": definitions[cid]["dependency_ids"], "unresolved_dependencies": [],
            "pursuit_status": "EXHAUSTED", "warrant_status": "DESCRIPTIVE ONLY", "warrant_scope": scope,
            "evidence": evidence, "criterion_scope": "Instrument/commission completion; no new population-mechanism criterion"})
        expansions.append({"card_id": cid, "state": "not_applicable", "reason": "Finite instrument or bridge task; no native discovery expansion ladder"})
    inputs = {"cards": sorted(cards, key=lambda row: row["card_id"]),
        "expansions": sorted(expansions, key=lambda row: row["card_id"]), "proofs": proof_references}
    result = verify_closeout(root, inputs)
    if not result["campaign_closed"]:
        raise ValueError("final accounting remains blocked: "+repr(result["blocking_reasons"]))
    return inputs, result


def run(root, output, study_relative, proof_references, supplemental_controls=None):
    if output.exists() or not output.resolve().is_relative_to((root/"closeout").resolve()):
        raise ValueError("final accounting requires a new administrative closeout directory")
    inputs, result = build(root, study_relative, proof_references, supplemental_controls)
    write(output/"INPUTS.json", inputs)
    write(output/"PROGRESS.json", {"recorded_at": now(), "cards": inputs["cards"],
        "expansions": inputs["expansions"], "campaign_complete": result["campaign_closed"],
        "scope": "Final file-bound accounting, with per-comparison and per-claim qualifications"})
    write(output/"RECEIPT.json", {**result, "inputs_sha256": file_digest(output/"INPUTS.json"),
        "producer_sha256": file_digest(Path(__file__)),
        "scope": "Read-only completion-guard validation; the ordinary supervisor still records campaign closure"})
    return result
