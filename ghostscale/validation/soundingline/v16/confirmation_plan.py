"""B03 freeze after the finite discovery ladder and actual controls are closed."""
from .runtime import REPO, freeze
from .records import read, write, file_digest, now, digest
from .record_integrity import source_locks
from .expansion_adapter import design
from .confirmation_candidates import enumerate_candidates, allocate

PACKET = "confirmation-selection-1"


def execute(root, heartbeat=None, *, resume=False):
    output = root/"confirmation-selection"
    source_locks(root, REPO)
    closed_path = root/"expansion-closure/COMPLETION.json"
    closed = read(closed_path)
    if closed.get("execution_state") != "completed" or closed.get("instrument_state") != "valid" or not closed["all_thirty_dispositions_explicit"]:
        raise ValueError("confirmation selection requires the whole finite discovery ladder")
    for name, expected in closed["source_hashes"].items():
        if file_digest(root/name) != expected:
            raise ValueError("final discovery evidence changed before confirmation selection")
    catalogue_path = root/"case-catalogue-2/COMPLETION.json"
    if read(catalogue_path).get("instrument_state") != "valid":
        raise ValueError("confirmation selection requires the completed explanatory catalogue")
    from .confirmation_runner import sources
    files = sources(root)
    admission_path = root/"confirmation-setup/SELECTION_ADMISSION.json"
    admission = read(admission_path)
    if admission.get("instrument_state") != "valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix()) != file_digest(path) for path in files):
        raise ValueError("confirmation selection requires the admitted complete runner and planning sources")
    if (output/"COMPLETION.json").exists():
        result = read(output/"COMPLETION.json")
        if file_digest(output/"CLAIMS.json") != result["claims_sha256"]:
            raise ValueError("completed confirmation selection changed")
        return result
    entries, paths = [], [closed_path, catalogue_path, admission_path]
    for disposition in closed["cards"]:
        base = root/disposition["final_source"]
        summary_path = base/("AGGREGATE.json" if (base/"AGGREGATE.json").exists() else "SUMMARY.json")
        summary = read(summary_path)
        contrasts = [row for value in summary["conditions"].values() for row in value["contrasts"]]
        constructors = min(row["n_constructors"] for row in contrasts)
        timing_path = base/"TIMING.json"
        seconds = read(timing_path)["wall_seconds"]/summary["n_maker_packets"] if timing_path.exists() else 0.
        if constructors >= 20 and seconds <= 0:
            raise ValueError("a mature confirmation candidate needs its measured cost source")
        entries.append({"card_id": disposition["card_id"], "constructors": constructors,
            "design": design(disposition["card_id"]), "summary": summary,
            "source_base": disposition["final_source"], "source_sha256": file_digest(summary_path),
            "seconds_per_condition_record": seconds})
        paths.append(summary_path)
        if timing_path.exists():
            paths.append(timing_path)
    candidates, exclusions = enumerate_candidates(entries)
    chosen = allocate(candidates)
    for index, claim in enumerate(chosen["selected"]):
        claim["claim_id"] = f"C{index+1:02d}-"+claim["card_id"]
        claim["namespace"] = "v16-confirmation-1-"+claim["claim_id"]+"-"+digest([claim["condition_ids"], claim["estimands"]])[:12]
    specification = {"source_hashes": {path.relative_to(root).as_posix(): file_digest(path) for path in paths},
        "claims": chosen["selected"], "selection_rule": chosen["selection_rule"], "exclusions": exclusions,
        "familywise_alpha": .05, "all_candidate_count": len(candidates), "discovery_selected_regimes": True,
        "confirmation_data_opened": False, "replacement_after_confirmation_forbidden": True}
    packet = freeze(root, PACKET, files, specification)
    write(output/"candidate_ledger_points.json", {"candidates": candidates, "exclusions": exclusions, "dispositions": chosen["candidate_dispositions"]})
    claims_record = {"selected": chosen["selected"], "packet_hash": packet["packet_hash"],
        "familywise_alpha": .05, "planning_allocation": .05/3, "final_adjustment": "Holm",
        "empty_selection": not chosen["selected"], "no_failed_claim_replacement": True}
    write(output/"CLAIMS.json", claims_record)
    result = {"execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
        "packet_hash": packet["packet_hash"], "claims_sha256": file_digest(output/"CLAIMS.json"),
        "selected_claim_count": len(chosen["selected"]), "empty_selection": not chosen["selected"],
        "candidate_ledger_sha256": file_digest(output/"candidate_ledger_points.json"),
        "confirmation_data_opened": False, "campaign_complete": False}
    write(output/"COMPLETION.json", result)
    return result
