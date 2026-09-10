"""Frozen fresh-constructor confirmations, actual source controls and one ledger."""
from copy import deepcopy
import json
import time
from .runtime import REPO, PACKAGE, campaign, freeze, remaining_seconds
from .records import read, write, file_digest, now, canonical, digest
from .record_integrity import source_locks
from .expansion_adapter import execute_unit, design, auditor
from .expansion_runner import EXTENSIONS, verify_completed
from .study import raw_manifest
from .reader_process import ReaderProcess
from .confirmation_summary import summarize_subset
from .confirmation_protocol import analyze_primary
from .confirmation_bounded import holm
from .packet_control_runner import source_inventory, next_attempt
from .packet_controls import condition_control, ActualReader
from .consumer_frames import requests

PACKET = "confirmation-1"
FILES = ["confirmation_candidates.py", "confirmation_bounded.py", "confirmation_math.py",
    "confirmation_summary.py", "confirmation_protocol.py", "confirmation_runner.py",
    "confirmation_plan.py", "confirmation_interruption.py", "runtime_status_retry.py"]


def sources(root):
    if (root/"packets"/(PACKET+".json")).exists():
        return sorted(REPO/name for name in read(root/"packets"/(PACKET+".json"))["identity"]["files"])
    paths = {PACKAGE/name for name in FILES}
    for path in (REPO/"results/v16/packets").glob("*.json"):
        paths.update(REPO/name for name in read(path)["identity"]["files"])
    paths.update([REPO/"runners/confirm_v16.py", REPO/"tests/test_v16_confirmation_runner.py"])
    return sorted(paths)


def fixture_claims():
    claims = []
    for card in ["K01", "S02"]:
        native = design(card)
        if card == "K01":
            from .study import ESTIMANDS
            from dataclasses import asdict
            specs = [asdict(ESTIMANDS[0])]
            conditions = [native["conditions"][1]["id"]]
        else:
            from .behavior_study import estimands
            from dataclasses import asdict
            specs = [asdict(item) for item in estimands(card)]
            conditions = [condition["id"] for condition in native["conditions"]]
        n = 64
        claims.append({"claim_id": "fixture-"+card, "card_id": card,
            "kind": "capability" if card == "K01" else "equivalence", "condition_ids": conditions,
            "estimands": specs, "n_constructor_packets": n, "histories_per_constructor_packet": len(conditions),
            "total_fresh_acquisition_histories": n*len(conditions), "alpha_planning": .05/3,
            "external_difference_bound": 2., "margin": .05,
            "grouping": "One independent constructor with a distinct history in each original condition",
            "namespace": "fixture-v16-confirmation-"+card, "power": {"planning_state": "fixture only"}})
    return claims


def specification(root, fixture):
    if fixture:
        if not campaign(root)["campaign_id"].startswith("fixture-"):
            raise ValueError("known confirmation fixture cannot use the scientific campaign")
        claims = fixture_claims()
        selection_hash = "known fixture; no scientific claim selected"
    else:
        selection = read(root/"confirmation-selection/COMPLETION.json")
        if selection["execution_state"] != "completed" or selection["instrument_state"] != "valid":
            raise ValueError("confirmation selection is not valid and complete")
        path = root/"confirmation-selection/CLAIMS.json"
        if file_digest(path) != selection["claims_sha256"]:
            raise ValueError("frozen confirmation claim selection changed")
        claims = read(path)["selected"]
        selection_hash = file_digest(path)
    cards = []
    for claim in claims:
        native = deepcopy(design(claim["card_id"]))
        native["conditions"] = [condition for condition in native["conditions"] if condition["id"] in claim["condition_ids"]]
        if {condition["id"] for condition in native["conditions"]} != set(claim["condition_ids"]):
            raise ValueError("confirmation changed an original native regime")
        cards.append({"card_id": claim["card_id"], "design": native, "n_per_condition": claim["n_constructor_packets"],
            "constructors": claim["n_constructor_packets"], "namespace": claim["namespace"], "claim": claim})
    result = {"cards": cards, "selection_sha256": selection_hash, "maximum_claims": 3,
        "evidence_scope": "fixture" if fixture else "confirmation",
        "primary_rule": "Exactly one bounded mean or joint equivalence primary per claim; all native contrasts are descriptive",
        "planned_condition_records": sum(entry["n_per_condition"]*len(entry["design"]["conditions"]) for entry in cards)}
    return json.loads(canonical(result))


def actual_controls(root, base, entry, packet, destination, runtime_join):
    inventory, selected = source_inventory(base, entry, packet["packet_hash"])
    write(destination/"source_inventory_points.json", inventory)
    checks = []
    with ReaderProcess(next_attempt(destination/"warm-readers"), extensions=EXTENSIONS) as raw:
        reader = ActualReader(raw)
        warm_pid = raw.child.pid
        for condition in entry["design"]["conditions"]:
            row = selected[condition["id"]]
            folder = destination/"conditions"/digest(condition["id"])[:16]
            path = folder/"RECEIPT.json"
            if path.exists():
                receipt = read(path)
                if receipt["source_unit_sha256"] != file_digest(base/"units"/(row["unit_id"]+"_points.json")):
                    raise ValueError("confirmation control resumed with different scientific source")
                for name, expected in receipt["retained_control_files"].items():
                    if file_digest(folder/name) != expected:
                        raise ValueError("preserved confirmation control evidence changed")
            else:
                attempt = next_attempt(folder)
                receipt = condition_control(base, row, attempt, reader, set(entry["design"]["adversaries"]))
                receipt["retained_control_files"] = {path.relative_to(folder).as_posix(): file_digest(path) for path in attempt.rglob("*.json")}
                write(path, receipt)
            required = set(entry["design"]["adversaries"])-{"X08"}
            if not required <= {name for name, passed in receipt["completed_local_checks"].items() if passed}:
                raise ValueError("confirmation omitted a required actual source adversary")
            checks.append({"condition": condition["id"], "receipt_sha256": file_digest(path)})
    cold_path = destination/"COLD.json"
    if not cold_path.exists():
        with ReaderProcess(next_attempt(destination/"cold-readers"), extensions=EXTENSIONS) as raw:
            reader = ActualReader(raw)
            for condition in entry["design"]["conditions"]:
                for frame in requests(base, selected[condition["id"]]):
                    if reader.request(frame["kind"], frame["public"], **frame["options"]) != frame["result"]:
                        raise ValueError("confirmation cold reader changed its prediction")
            write(cold_path, {"instrument_state": "valid", "fresh_process": raw.child.pid != warm_pid,
                "requests": reader.calls, "warm_pid": warm_pid, "cold_pid": raw.child.pid})
    if not read(cold_path)["fresh_process"]:
        raise ValueError("confirmation cold-reader process was not fresh")
    result = {"execution_state": "completed", "instrument_state": "valid", "card_id": entry["card_id"],
        "source_packet_hash": packet["packet_hash"], "source_condition_records": len(inventory["unit_hashes"]),
        "conditions": checks, "cold_sha256": file_digest(cold_path), "runtime_join": runtime_join,
        "X08_scope": "Actual confirmation CLI interruption plus source-bound known-noise checks; no continuous-occupancy assertion"}
    write(destination/"COMPLETION.json", result)
    return result


def execute(root, heartbeat, *, resume=False, fixture=False):
    files = sources(root)
    if resume and not (root/"packets"/(PACKET+".json")).exists():
        raise ValueError("confirmation resume requires the existing frozen packet")
    runtime_join = "known fixture, evaluated by its external interruption control"
    if not fixture:
        source_locks(root, REPO)
        admission = read(root/"confirmation-setup/ADMISSION.json")
        if admission.get("instrument_state") != "valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix()) != file_digest(path) for path in files):
            raise ValueError("confirmation runner lacks its current source-bound admission")
        runtime_join = admission["runtime_join"]
        if file_digest(root/runtime_join["path"]) != runtime_join["sha256"]:
            raise ValueError("confirmation interruption proof changed")
        forecast = read(root/"confirmation-setup/FORECAST.json")
        if forecast["conservative_seconds"]+forecast["closeout_reserve_seconds"] >= remaining_seconds(root):
            raise ValueError("confirmation no longer fits the immutable remaining horizon")
    spec = specification(root, fixture)
    packet = freeze(root, PACKET, files, spec)
    output = root/"confirmation"
    completed = 0
    primaries = []
    for entry in spec["cards"]:
        card = entry["card_id"]
        base = output/card
        began = time.perf_counter()
        if (base/"COMPLETION.json").exists():
            verify_completed(base, read(base/"COMPLETION.json"))
            rows = [read(path) for path in (base/"units").glob("*_points.json")]
            completed += len(rows)
        else:
            rows = []
            with ReaderProcess(next_attempt(base/"readers"), extensions=EXTENSIONS) as reader:
                for condition in entry["design"]["conditions"]:
                    for index in range(entry["n_per_condition"]):
                        row = execute_unit(card, base, condition, index, namespace=entry["namespace"], packet=packet,
                            reader=reader, constructors=entry["constructors"], scope=spec["evidence_scope"])
                        rows.append(row)
                        completed += 1
                        heartbeat(card_id=card, completed_units=completed, planned_units=spec["planned_condition_records"], operation="frozen confirmation units")
            summary = summarize_subset(card, rows, entry["claim"]["condition_ids"])
            audit = auditor(card)(base, summary)
            if audit["instrument_state"] != "valid":
                raise ValueError("confirmation native physics or statistical audit failed")
            write(base/"AGGREGATE.json", summary)
            write(base/"INDEPENDENT_AUDIT.json", audit)
            write(base/"RAW_MANIFEST.json", raw_manifest(base))
            write(base/"TIMING.json", {"wall_seconds": time.perf_counter()-began, "n_condition_records": len(rows)})
            write(base/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid", "card_id": card,
                "packet_hash": packet["packet_hash"], "aggregate_sha256": file_digest(base/"AGGREGATE.json"),
                "raw_manifest_sha256": file_digest(base/"RAW_MANIFEST.json"), "completed_at": now(),
                "scope": "Native execution complete; actual source controls and primary confirmation follow"})
        controls = actual_controls(root, base, entry, packet, root/"confirmation-controls"/card, runtime_join)
        claim = {**entry["claim"], "execution_packet_hash": packet["packet_hash"]}
        primary = analyze_primary(claim, rows)
        primary["actual_source_control_sha256"] = file_digest(root/"confirmation-controls"/card/"COMPLETION.json")
        write(base/"PRIMARY.json", primary)
        primaries.append({"claim_id": claim["claim_id"], "card_id": card, "p": primary["primary_p"],
            "primary_sha256": file_digest(base/"PRIMARY.json"), "source_controls": controls["instrument_state"]})
    ledger = {"familywise_alpha": .05, "frozen_primary_count": len(primaries),
        "claims": holm(primaries) if primaries else [], "empty_selection": not primaries,
        "no_failed_claim_replaced": True, "native_secondary_contrasts": "Descriptive; not additional confirmations"}
    write(output/"MULTIPLICITY.json", ledger)
    result = {"execution_state": "completed", "instrument_state": "valid", "packet_hash": packet["packet_hash"],
        "completed_at": now(), "claims": ledger["claims"], "condition_records": completed,
        "empty_selection": ledger["empty_selection"], "source_controls_completed": True,
        "campaign_complete": False, "remaining_work": "Complete raw archive, independent final calculations/replay and documentary write-through"}
    if (output/"COMPLETION.json").exists():
        return read(output/"COMPLETION.json")
    write(output/"COMPLETION.json", result)
    return result
