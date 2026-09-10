"""Source-bound controls for a completed finite expansion, with immutable resume."""
from collections import Counter
import time
from .records import read, write, digest, file_digest, now
from .runtime import REPO, PACKAGE, campaign, freeze, remaining_seconds
from .expansion_runner import sources as expansion_sources, EXTENSIONS
from .reader_process import ReaderProcess
from .consumer_frames import requests
from .record_integrity import independent_units
from .packet_controls import condition_control, ActualReader

PACKET = "constructor-controls-1"
FILES = ["packet_controls.py", "packet_control_runner.py", "packet_control_interruption.py"]


def sources(root):
    return sorted(set(expansion_sources(root)) | {PACKAGE/name for name in FILES} |
                  {REPO/"runners/control_v16_expansion.py", REPO/"tests/test_v16_packet_controls.py",
                   REPO/"tests/test_v16_packet_control_runner.py"})


def specification(source_root, source_name):
    completion = read(source_root/source_name/"COMPLETION.json")
    if completion["execution_state"] != "completed" or completion["instrument_state"] != "valid":
        raise ValueError("source expansion is not complete and valid")
    lock = read(source_root/"packets"/(source_name+".json"))
    return {"source_name": source_name, "source_packet_hash": lock["packet_hash"],
            "source_completion_sha256": file_digest(source_root/source_name/"COMPLETION.json"),
            "cards": lock["identity"]["design"]["cards"],
            "selection": "Every registered condition, immutable index zero; no outcome-based selection",
            "grouping": "All retained source units; conditions sharing constructor draws remain separate",
            "planned_condition_controls": sum(len(row["design"]["conditions"]) for row in lock["identity"]["design"]["cards"])}


def source_inventory(base, entry, packet_hash):
    completion = read(base/"COMPLETION.json")
    manifest = read(base/"RAW_MANIFEST.json")
    if file_digest(base/"RAW_MANIFEST.json") != completion["raw_manifest_sha256"]:
        raise ValueError("new source raw manifest changed")
    rows = []
    hashes = {}
    selected = {}
    for path in sorted((base/"units").glob("*_points.json")):
        relative = path.relative_to(base).as_posix()
        actual = file_digest(path)
        if manifest["files"].get(relative) != actual:
            raise ValueError("new source unit differs from its saved manifest")
        row = read(path)
        if row["packet_hash"] != packet_hash or row["card_id"] != entry["card_id"] or row["lineage"] != entry["namespace"]:
            raise ValueError("new source scientific identity changed")
        rows.append(row)
        hashes[relative] = actual
        if row["seed_components"]["index"] == 0:
            selected[row["condition"]] = row
    grouping = independent_units(rows)
    expected = {condition["id"] for condition in entry["design"]["conditions"]}
    count = Counter(row["condition"] for row in rows)
    if set(count) != expected or set(selected) != expected or any(n != entry["n_per_condition"] for n in count.values()):
        raise ValueError("new source omitted or duplicated a registered condition")
    for condition in expected:
        current = [row for row in rows if row["condition"] == condition]
        if {row["seed_components"]["index"] for row in current} != set(range(entry["n_per_condition"])):
            raise ValueError("new source has incomplete immutable seed coverage")
        if len({row["constructor_id"] for row in current}) != min(entry["constructors"], entry["n_per_condition"]):
            raise ValueError("new source constructor coverage differs")
    return {"unit_hashes": hashes, "grouping": grouping,
            "source_raw_manifest_sha256": file_digest(base/"RAW_MANIFEST.json")}, selected


def next_attempt(directory):
    attempts = list(directory.glob("attempt-*"))
    number = max([int(path.name.split("-")[-1]) for path in attempts]+[0])+1
    path = directory/("attempt-"+str(number))
    path.mkdir(parents=True)
    return path


def execute(root, heartbeat, *, resume=False, source_root=None, source_name="constructor-expansion-1"):
    source_root = root if source_root is None else source_root
    fixture = campaign(root)["campaign_id"].startswith("fixture-")
    if not fixture and (source_root.resolve() != root.resolve() or source_name != "constructor-expansion-1"):
        raise ValueError("scientific controls must bind the admitted local source")
    if resume and not (root/"packets"/(PACKET+".json")).exists():
        raise ValueError("control resume requires an existing frozen packet")
    files = sources(root)
    runtime = None
    if not fixture:
        admission = read(root/"packet-control-setup/ADMISSION.json")
        if admission["instrument_state"] != "valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix()) != file_digest(path) for path in files):
            raise ValueError("source controls lack current admission")
        runtime = read(root/"packet-control-setup/runtime/RECEIPT.json")
        if runtime["instrument_state"] != "valid" or file_digest(root/"packet-control-setup/runtime/RECEIPT.json") != admission["runtime_sha256"]:
            raise ValueError("source controls lack actual interruption proof")
        forecast = read(root/"packet-control-setup/FORECAST.json")
        if forecast["conservative_seconds"]+forecast["closeout_reserve_seconds"] >= remaining_seconds(root):
            raise ValueError("source controls cannot fit immutable horizon and reserve")
    spec = specification(source_root, source_name)
    packet = freeze(root, PACKET, files, spec)
    output = root/PACKET
    completed = 0
    cards = []
    started = time.perf_counter()
    for entry in spec["cards"]:
        card = entry["card_id"]
        source = source_root/source_name/card
        destination = output/card
        inventory, selected = source_inventory(source, entry, spec["source_packet_hash"])
        write(destination/"source_inventory_points.json", inventory)
        receipts = []
        with ReaderProcess(destination/"reader-work", extensions=EXTENSIONS) as reader:
            warm_pid = reader.child.pid
            for condition in entry["design"]["conditions"]:
                row = selected[condition["id"]]
                directory = destination/"conditions"/digest(condition["id"])[:16]
                receipt_path = directory/"RECEIPT.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    if receipt["source_unit_sha256"] != inventory["unit_hashes"]["units/"+row["unit_id"]+"_points.json"]:
                        raise ValueError("resumed control source changed")
                    for name, expected in receipt["retained_control_files"].items():
                        if file_digest(directory/name) != expected:
                            raise ValueError("retained control evidence changed")
                else:
                    attempt = next_attempt(directory)
                    receipt = condition_control(source, row, attempt, reader, set(entry["design"]["adversaries"]))
                    receipt["control_packet_hash"] = packet["packet_hash"]
                    receipt["retained_control_files"] = {path.relative_to(directory).as_posix(): file_digest(path) for path in attempt.rglob("*.json")}
                    write(receipt_path, receipt)
                receipts.append({"condition": condition["id"], "receipt": receipt_path.relative_to(root).as_posix(),
                                 "sha256": file_digest(receipt_path), "checks": receipt["completed_local_checks"]})
                completed += 1
                heartbeat(card_id=card, completed_units=completed, planned_units=spec["planned_condition_controls"],
                          operation="actual source-specific controls", last_unit=row["unit_id"])
        cold_path = destination/"COLD.json"
        if not cold_path.exists():
            with ReaderProcess(next_attempt(destination/"cold-readers"), extensions=EXTENSIONS) as raw:
                cold = ActualReader(raw)
                for condition in entry["design"]["conditions"]:
                    row = selected[condition["id"]]
                    for frame in requests(source, row):
                        if cold.request(frame["kind"], frame["public"], **frame["options"]) != frame["result"]:
                            raise ValueError("fresh-process source prediction differs")
                write(cold_path, {"instrument_state": "valid", "fresh_process": raw.child.pid != warm_pid,
                    "warm_pid": warm_pid, "cold_pid": raw.child.pid, "requests": cold.calls, "resources": cold.reader.samples,
                    "scope": "A new guarded process starts the fixed card-wide request sequence with an empty process cache"})
        cold = read(cold_path)
        if cold["instrument_state"] != "valid" or not cold["fresh_process"]:
            raise ValueError("cold-process control did not pass")
        receipt = {"card_id": card, "instrument_state": "valid", "conditions": receipts,
                   "source_inventory_sha256": file_digest(destination/"source_inventory_points.json"),
                   "cold_sha256": file_digest(cold_path), "required_adversaries": entry["design"]["adversaries"],
                   "source_condition_records": len(inventory["unit_hashes"])}
        write(destination/"COMPLETION.json", receipt)
        cards.append(receipt)
    result = {"execution_state": "completed", "instrument_state": "valid", "campaign_complete": False,
              "packet_hash": packet["packet_hash"], "source_packet_hash": spec["source_packet_hash"],
              "condition_controls": completed, "source_condition_records": sum(row["source_condition_records"] for row in cards),
              "cards": cards, "runtime": runtime if runtime else "known fixture: actual interruption proof is evaluated externally",
              "runtime_scope": "Actual control CLI interruption/resume plus admitted expansion interruption fixtures; no claim of continuous occupancy",
              "wall_seconds": time.perf_counter()-started, "completed_at": now()}
    if (output/"COMPLETION.json").exists():
        return read(output/"COMPLETION.json")
    write(output/"COMPLETION.json", result)
    return result
