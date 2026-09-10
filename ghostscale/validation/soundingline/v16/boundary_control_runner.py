"""Independent source joins for the final allocation; no inherited green labels."""
import time
from .runtime import REPO, PACKAGE, campaign, freeze, remaining_seconds
from .records import read, write, digest, file_digest, now
from .record_integrity import source_locks
from .expansion_runner import EXTENSIONS
from .reader_process import ReaderProcess
from .consumer_frames import requests
from .packet_controls import condition_control, ActualReader
from .packet_control_runner import source_inventory, next_attempt
from .completion_guard import bound_receipt

PACKET = "boundary-controls-1"
FILES = ["boundary_control_runner.py", "boundary_control_interruption.py", "runtime_status_read.py"]


def sources(root):
    frozen = root/"packets"/(PACKET+".json")
    if frozen.exists():
        return sorted(REPO/name for name in read(frozen)["identity"]["files"])
    paths = {PACKAGE/name for name in FILES}|{REPO/"runners/control_v16_boundary.py", REPO/"tests/test_v16_boundary_controls.py"}
    source = read(REPO/"results/v16/packets/boundary-expansion-1.json")
    paths.update(REPO/name for name in source["identity"]["files"])
    return sorted(paths)


def source_specification(root, name):
    completion = read(root/name/"COMPLETION.json")
    if completion.get("execution_state")!="completed" or completion.get("instrument_state")!="valid":
        raise ValueError("final source controls require a completed valid allocation")
    packet = read(root/"packets"/(name+".json"))
    entries = packet["identity"]["design"]["cards"]
    # Reference scientific designs without registering a second scientific study.
    spec = {"source_packet_id":name,"source_packet_hash":packet["packet_hash"],
        "source_completion_sha256":file_digest(root/name/"COMPLETION.json"),
        "source_design_sha256":digest(entries),
        "selection":"Immutable index zero of every original condition; every source unit enters grouping/hash checks",
        "source_cards":[{"card_id":entry["card_id"],"registered_condition_count":len(entry["design"]["conditions"]),
            "design_sha256":digest(entry["design"]),"n_per_condition":entry["n_per_condition"],"constructors":entry["constructors"]} for entry in entries],
        "planned_controls":sum(len(entry["design"]["conditions"]) for entry in entries)}
    return spec, entries


def execute(root, heartbeat, *, resume=False, fixture_source=None, fixture_name=None):
    fixture = campaign(root)["campaign_id"].startswith("fixture-")
    if (fixture_source is not None or fixture_name is not None) and not fixture:
        raise ValueError("scientific final controls cannot substitute a fixture source")
    source_root = root if fixture_source is None else fixture_source
    source_name = "boundary-expansion-1" if fixture_name is None else fixture_name
    if resume and not (root/"packets"/(PACKET+".json")).exists():
        raise ValueError("final control resume requires its existing packet")
    files = sources(root)
    runtime_join = None
    if not fixture:
        source_locks(root,REPO)
        admission = read(root/"boundary-control-setup/ADMISSION.json")
        if admission.get("instrument_state")!="valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix())!=file_digest(path) for path in files):
            raise ValueError("final controls need current actual admission")
        runtime_join = admission.get("runtime_join", {"path":"boundary-control-setup/runtime/RECEIPT.json", "sha256":admission["runtime_sha256"]})
        runtime_receipt = bound_receipt(root, runtime_join["path"], runtime_join["sha256"])
        if runtime_receipt["instrument_state"]!="valid" or runtime_join["sha256"]!=admission["runtime_sha256"]:
            raise ValueError("final controls lack their actual interruption proof")
        forecast = read(root/"boundary-control-setup/FORECAST.json")
        if forecast["conservative_seconds"]+forecast["closeout_reserve_seconds"]>=remaining_seconds(root):
            raise ValueError("final control forecast exceeds immutable remaining horizon")
    spec, entries = source_specification(source_root,source_name)
    packet = freeze(root,PACKET,files,spec)
    output = root/PACKET
    count = 0
    cards = []
    started = time.perf_counter()
    for entry in entries:
        card = entry["card_id"]
        source, destination = source_root/source_name/card, output/card
        inventory, selected = source_inventory(source,entry,spec["source_packet_hash"])
        write(destination/"source_inventory_points.json",inventory)
        conditions = []
        with ReaderProcess(destination/"reader-work",extensions=EXTENSIONS) as reader:
            warm_pid = reader.child.pid
            for condition in entry["design"]["conditions"]:
                row = selected[condition["id"]]
                folder = destination/"conditions"/digest(condition["id"])[:16]
                path = folder/"RECEIPT.json"
                if path.exists():
                    receipt = read(path)
                    if receipt["source_unit_sha256"]!=inventory["unit_hashes"]["units/"+row["unit_id"]+"_points.json"] or receipt["control_packet_hash"]!=packet["packet_hash"]:
                        raise ValueError("final control resumed with a different immutable source")
                    for name, expected in receipt["retained_control_files"].items():
                        if file_digest(folder/name)!=expected:
                            raise ValueError("final control partial evidence changed")
                else:
                    attempt = next_attempt(folder)
                    receipt = condition_control(source,row,attempt,reader,set(entry["design"]["adversaries"]))
                    receipt["control_packet_hash"] = packet["packet_hash"]
                    receipt["retained_control_files"] = {path.relative_to(folder).as_posix():file_digest(path) for path in attempt.rglob("*.json")}
                    write(path,receipt)
                conditions.append({"condition":condition["id"],"receipt":path.relative_to(root).as_posix(),
                                   "sha256":file_digest(path),"checks":receipt["completed_local_checks"]})
                count += 1
                heartbeat(card_id=card,completed_units=count,planned_units=spec["planned_controls"],last_unit=row["unit_id"],operation="final-source controls")
        cold_path = destination/"COLD.json"
        if not cold_path.exists():
            with ReaderProcess(next_attempt(destination/"cold-readers"),extensions=EXTENSIONS) as raw:
                cold = ActualReader(raw)
                for condition in entry["design"]["conditions"]:
                    for frame in requests(source,selected[condition["id"]]):
                        if cold.request(frame["kind"],frame["public"],**frame["options"])!=frame["result"]:
                            raise ValueError("final-source cold prediction differs")
                write(cold_path,{"instrument_state":"valid","fresh_process":raw.child.pid!=warm_pid,
                    "warm_pid":warm_pid,"cold_pid":raw.child.pid,"requests":cold.calls,"resources":cold.reader.samples})
        if read(cold_path).get("instrument_state")!="valid" or not read(cold_path)["fresh_process"]:
            raise ValueError("final-source fresh process control failed")
        required = set(entry["design"]["adversaries"])
        for condition in conditions:
            if not required-{"X08"} <= {name for name,passed in condition["checks"].items() if passed}:
                raise ValueError("final source omitted a required actual adversary")
        receipt = {"card_id":card,"instrument_state":"valid","conditions":conditions,"required_adversaries":sorted(required),
            "source_condition_records":len(inventory["unit_hashes"]),"source_inventory_sha256":file_digest(destination/"source_inventory_points.json"),
            "cold_sha256":file_digest(cold_path),"runtime_join":runtime_join if runtime_join else "fixture runtime evaluated externally",
            "X08_scope":"Actual final-control interruption and source-bound noise fixtures; no continuous-occupancy claim"}
        write(destination/"COMPLETION.json",receipt)
        cards.append(receipt)
    result = {"execution_state":"completed","instrument_state":"valid","campaign_complete":False,"completed_at":now(),
        "packet_hash":packet["packet_hash"],"source_packet_hash":spec["source_packet_hash"],"cards":cards,
        "condition_controls":count,"source_condition_records":sum(row["source_condition_records"] for row in cards),
        "runtime_join":runtime_join,"wall_seconds":time.perf_counter()-started}
    if (output/"COMPLETION.json").exists():
        return read(output/"COMPLETION.json")
    write(output/"COMPLETION.json",result)
    return result
