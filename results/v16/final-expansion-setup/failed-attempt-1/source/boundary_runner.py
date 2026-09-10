"""The last unchanged scientific allocation, using new independent namespaces."""
import time
from .records import read, write, file_digest, now
from .runtime import REPO, PACKAGE, campaign, freeze, remaining_seconds
from .expansion_runner import EXTENSIONS, verify_completed
from .expansion_adapter import execute_unit, summarizer, auditor, design
from .reader_process import ReaderProcess
from .record_integrity import independent_units, source_locks
from .study import raw_manifest
from .boundary_plan import freeze_selection

PACKET = "boundary-expansion-1"
FILES = ["boundary_plan.py", "boundary_runner.py", "boundary_interruption.py"]


def sources(root, name=PACKET):
    saved = root/"packets"/(name+".json")
    if saved.exists():
        return sorted(REPO/path for path in read(saved)["identity"]["files"])
    paths = {PACKAGE/name for name in FILES}|{REPO/"runners/run_v16_boundary.py", REPO/"tests/test_v16_boundary.py"}
    for packet in (REPO/"results/v16/packets").glob("*.json"):
        paths.update(REPO/name for name in read(packet)["identity"]["files"])
    return sorted(paths)


def fixture_spec(card):
    return {"cards":[{"card_id":card,"design":design(card),"n_per_condition":64 if card=="K01" else 8,
        "constructors":32,"namespace":"known-final-boundary-"+card}],
        "evidence_scope":"fixture", "dispositions":[]}


def execute(root, heartbeat, *, resume=False, fixture=None):
    accepted = campaign(root)
    if fixture and not accepted["campaign_id"].startswith("fixture-"):
        raise ValueError("boundary fixture cannot own science")
    name = PACKET if fixture is None else "boundary-fixture-"+fixture
    lock_path = root/"packets"/(name+".json")
    if resume and not lock_path.exists():
        raise ValueError("boundary resume requires an existing packet; never prepares")
    files = sources(root,name)
    if fixture is None:
        source_locks(root,REPO)
        admission = read(root/"final-expansion-setup/ADMISSION.json")
        if admission["instrument_state"]!="valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix())!=file_digest(path) for path in files):
            raise ValueError("final boundary source lacks current admission")
        forecast = read(root/"final-expansion-setup/FORECAST.json")
        if forecast["conservative_seconds"]+forecast["closeout_reserve_seconds"] >= remaining_seconds(root):
            raise ValueError("final boundary work exceeds the immutable horizon with reserve")
        spec = freeze_selection(root)
    else:
        spec = fixture_spec(fixture)
    packet = freeze(root,name,files,spec)
    output = root/name
    completed = 0
    receipts = []
    planned = sum(entry["n_per_condition"]*len(entry["design"]["conditions"]) for entry in spec["cards"])
    for entry in spec["cards"]:
        card = entry["card_id"]
        base = output/card
        count = entry["n_per_condition"]*len(entry["design"]["conditions"])
        if (base/"COMPLETION.json").exists():
            receipt = read(base/"COMPLETION.json")
            verify_completed(base,receipt)
            receipts.append(receipt)
            completed += count
            continue
        started, cpu = time.perf_counter(), time.process_time()
        rows = []
        with ReaderProcess(base/"reader-work",extensions=EXTENSIONS) as reader:
            for condition in entry["design"]["conditions"]:
                for index in range(entry["n_per_condition"]):
                    if remaining_seconds(root)<60:
                        return {"execution_state":"checkpointed","reason":"immutable deadline reserve","campaign_complete":False}
                    row = execute_unit(card,base,condition,index,namespace=entry["namespace"],packet=packet,
                        reader=reader,constructors=entry["constructors"],scope=spec["evidence_scope"])
                    rows.append(row)
                    completed += 1
                    heartbeat(card_id=card,completed_units=completed,planned_units=planned,last_unit=row["unit_id"],operation="last finite discovery expansion")
        summary = summarizer(card,rows)
        write(base/"AGGREGATE.json",summary)
        audit = auditor(card)(base,summary)
        if audit["instrument_state"]!="valid":
            raise ValueError("final boundary independent physics/statistics failed")
        write(base/"INDEPENDENT_AUDIT.json",audit)
        write(base/"SAMPLING.json",independent_units(rows))
        write(base/"RAW_MANIFEST.json",raw_manifest(base))
        write(base/"TIMING.json",{"wall_seconds":time.perf_counter()-started,"parent_cpu_seconds":time.process_time()-cpu,
              "includes":"generation, guarded readers, scoring, independent audit and raw hashing"})
        receipt = {"execution_state":"completed","instrument_state":"valid","card_id":card,"packet_hash":packet["packet_hash"],
            "n_makers_per_condition":entry["n_per_condition"],"n_condition_records":len(rows),
            "constructors_per_condition":min(entry["constructors"],entry["n_per_condition"]),
            "raw_manifest_sha256":file_digest(base/"RAW_MANIFEST.json"),"aggregate_sha256":file_digest(base/"AGGREGATE.json"),
            "completed_at":now(),"evidence_scope":spec["evidence_scope"],"confirmation_state":"not a confirmation","campaign_complete":False}
        write(base/"COMPLETION.json",receipt)
        receipts.append(receipt)
    result = {"execution_state":"completed","instrument_state":"valid","completed_at":now(),"cards":receipts,
        "completed_condition_records":completed,"packet_hash":packet["packet_hash"],"campaign_complete":False,
        "remaining_work":"actual final-source controls, final ladder disposition, confirmations and closeout"}
    if (output/"COMPLETION.json").exists():
        return read(output/"COMPLETION.json")
    write(output/"COMPLETION.json",result)
    return result
