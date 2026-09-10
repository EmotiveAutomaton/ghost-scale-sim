"""Close the finite discovery ladder only after actual final-source controls."""
from .records import read, write, file_digest, now
from .runtime import REPO
from .record_integrity import source_locks
from .catalogue_runner_v2 import inventory


def final_control_join(science, control, packet):
    spec = packet["identity"]["design"]
    entries = spec["cards"]
    if any(row.get("execution_state") != "completed" or row.get("instrument_state") != "valid" for row in [science, control]):
        raise ValueError("final ladder requires completed science and actual valid source controls")
    if science["packet_hash"] != packet["packet_hash"] or control["source_packet_hash"] != packet["packet_hash"]:
        raise ValueError("final controls refer to a different source packet")
    expected_records = sum(entry["n_per_condition"]*len(entry["design"]["conditions"]) for entry in entries)
    expected_conditions = sum(len(entry["design"]["conditions"]) for entry in entries)
    if science["completed_condition_records"] != expected_records or control["source_condition_records"] != expected_records or control["condition_controls"] != expected_conditions:
        raise ValueError("final source-control denominators do not cover the whole allocation")
    for label, rows in [("science", science["cards"]), ("controls", control["cards"])]:
        if len(rows) != len(entries) or {row["card_id"] for row in rows} != {entry["card_id"] for entry in entries}:
            raise ValueError("final "+label+" omitted or duplicated a card")
    return {"packet_hash": packet["packet_hash"], "source_condition_records": expected_records,
        "controlled_conditions": expected_conditions, "controlled_cards": len(entries)}


def checked_summary(base, summary_name, audit_name, *, expected_n, expected_conditions):
    completion, summary, audit = [read(base/name) for name in ["COMPLETION.json", summary_name, audit_name]]
    if completion.get("execution_state") != "completed" or completion.get("instrument_state") != "valid" or audit.get("instrument_state") != "valid":
        raise ValueError("final disposition includes an incomplete or invalid source")
    if "aggregate_sha256" in completion and completion["aggregate_sha256"] != file_digest(base/summary_name):
        raise ValueError("final summary changed after completed-source controls")
    if set(summary["conditions"]) != set(expected_conditions) or summary["n_maker_packets"] != expected_n*len(expected_conditions):
        raise ValueError("final summary changed the registered condition or maker denominator")
    for condition in summary["conditions"].values():
        if any(contrast["n_makers"] != expected_n for contrast in condition["contrasts"]):
            raise ValueError("final contrast denominator differs from the original allocation")
    return summary


def execute(root, heartbeat=None, *, resume=False):
    output = root/"expansion-closure/COMPLETION.json"
    source_locks(root, REPO)
    selection_path = root/"expansion-closure/SELECTION.json"
    selection = read(selection_path)
    for name, expected in selection["source_hashes"].items():
        if file_digest(root/name) != expected:
            raise ValueError("final disposition changed the original expansion selection")
    science_path = root/"boundary-expansion-1/COMPLETION.json"
    control_path = root/"boundary-controls-1/COMPLETION.json"
    packet_path = root/"packets/boundary-expansion-1.json"
    science, control, packet = [read(path) for path in [science_path, control_path, packet_path]]
    joined = final_control_join(science, control, packet)
    selected = {entry["card_id"]: entry for entry in packet["identity"]["design"]["cards"]}
    original = {entry["card_id"]: entry for entry in inventory(root)}
    paths = [selection_path, science_path, control_path, packet_path]
    results = []
    for disposition in selection["dispositions"]:
        card = disposition["card_id"]
        entry = original[card]
        if card in selected:
            entry = {**entry, **selected[card], "base": "boundary-expansion-1/"+card,
                "summary_name": "AGGREGATE.json", "audit_name": "INDEPENDENT_AUDIT.json"}
            n = 1024
        else:
            n = 256 if entry["packet_id"] == "constructor-expansion-1" else 64
        base = root/entry["base"]
        summary = checked_summary(base, entry["summary_name"], entry["audit_name"], expected_n=n,
            expected_conditions=[condition["id"] for condition in entry["design"]["conditions"]])
        paths.extend(base/name for name in ["COMPLETION.json", entry["summary_name"], entry["audit_name"]])
        contrasts = [{"condition": condition, **contrast} for condition, values in summary["conditions"].items() for contrast in values["contrasts"]]
        results.append({"card_id": card, "execution_state": "completed", "instrument_state": "valid",
            "expansion_state": "finite ladder exhausted", "n_per_condition": n,
            "original_disposition": disposition, "final_source": entry["base"],
            "criterion_counts": {state: sum(row["criterion_state"] == state for row in contrasts) for state in ["held", "failed", "inconclusive"]},
            "remaining_uncertain_contrasts": [{"condition": row["condition"], "estimand": row["estimand"], "interval_95": row["interval_95"]}
                for row in contrasts if row["criterion_state"] == "inconclusive"],
            "confirmation_state": "not confirmed by discovery", "additional_expansions_allowed": 0,
            "qualification": "Unresolved intervals remain unresolved at the finite allocation limit"})
    if len(results) != 30 or len({row["card_id"] for row in results}) != 30:
        raise ValueError("the final ladder must account for every native card exactly once")
    hashes = {path.relative_to(root).as_posix(): file_digest(path) for path in paths}
    if output.exists():
        saved = read(output)
        if saved["source_hashes"] != hashes:
            raise ValueError("completed ladder source evidence changed")
        return saved
    result = {"execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
        "source_hashes": hashes, "actual_final_control_join": joined, "cards": results,
        "maximum_further_expansions_after_this": 0, "all_thirty_dispositions_explicit": True,
        "campaign_complete": False, "remaining_work": "frozen confirmations and all scientific/documentary closeout proofs"}
    write(output, result)
    return result
