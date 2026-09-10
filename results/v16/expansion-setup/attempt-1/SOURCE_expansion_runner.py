"""Finite, source-frozen constructor expansions with independent per-card audits."""
import time
from pathlib import Path
from .records import read, write, file_digest, now
from .runtime import REPO, PACKAGE, campaign, freeze, remaining_seconds
from .expansion_adapter import execute_unit, summarizer, auditor
from .expansion_plan import plan
from .reader_process import ReaderProcess
from .recoded_interface import OPERATIONS
from .record_integrity import independent_units
from .study import raw_manifest

PACKET = "constructor-expansion-1"
EXTENSIONS = sorted({name for name in OPERATIONS if ":" in name} |
                    {"ghostscale.validation.soundingline.v16.reading_cost_meter:public_reader"})
FILES = ["expansion_adapter.py", "expansion_plan.py", "expansion_runner.py", "expansion_interruption.py"]


def sources(root, name=PACKET):
    saved = root/"packets"/(name+".json")
    if saved.exists():
        return sorted(REPO/path for path in read(saved)["identity"]["files"])
    files = {PACKAGE/path for path in FILES} | {
        REPO/"runners/run_v16_expansion.py", REPO/"tests/test_v16_expansion_adapter.py",
        REPO/"tests/test_v16_expansion_runner.py"}
    for path in (REPO/"results/v16/packets").glob("*.json"):
        files.update(REPO/name for name in read(path)["identity"]["files"])
    return sorted(files)


def specification(fixture=None):
    result = plan()
    if fixture:
        if fixture == "profile":
            for row in result["cards"]:
                row["n_per_condition"] = 1
        else:
            result["cards"] = [row for row in result["cards"] if row["card_id"] == fixture]
            for row in result["cards"]:
                row["n_per_condition"] = 64 if fixture == "K01" else 8
        for row in result["cards"]:
            row["namespace"] = "v16-expansion-known-"+fixture+"-"+row["card_id"]
        result["evidence_scope"] = "fixture"
        result["planned_maker_condition_records"] = sum(row["n_per_condition"]*len(row["design"]["conditions"]) for row in result["cards"])
    else:
        result["evidence_scope"] = "discovery expansion"
    return result


def verify_completed(base, result):
    manifest = base/"RAW_MANIFEST.json"
    if file_digest(manifest) != result["raw_manifest_sha256"]:
        raise ValueError("completed expansion raw manifest changed")
    for name, expected in read(manifest)["files"].items():
        if file_digest(base/name) != expected:
            raise ValueError("completed expansion raw bytes changed")
    if file_digest(base/"AGGREGATE.json") != result["aggregate_sha256"]:
        raise ValueError("completed expansion aggregate changed")


def execute(root, heartbeat, *, resume=False, fixture=None):
    name = PACKET if fixture is None else "expansion-fixture-"+fixture
    if resume and not (root/"packets"/(name+".json")).exists():
        raise ValueError("expansion resume requires an existing packet; never prepares")
    accepted = campaign(root)
    if fixture and not accepted["campaign_id"].startswith("fixture-"):
        raise ValueError("fixture cannot own the scientific campaign")
    files = sources(root, name)
    if fixture is None:
        admission = read(root/"expansion-setup/ADMISSION.json")
        if admission["instrument_state"] != "valid" or any(admission["source_hashes"].get(path.relative_to(REPO).as_posix()) != file_digest(path) for path in files):
            raise ValueError("expansion source has not passed its actual admission tests")
        forecast = read(root/"expansion-setup/FORECAST.json")
        if forecast["instrument_state"] != "valid" or forecast["conservative_seconds"]+6*3600 > remaining_seconds(root):
            raise ValueError("expansion cannot fit the remaining horizon plus closeout reserve")
    spec = specification(fixture)
    packet = freeze(root, name, files, spec)
    output = root/name
    completed = 0
    receipts = []
    for entry in spec["cards"]:
        card = entry["card_id"]
        base = output/card
        completion_path = base/"COMPLETION.json"
        count = entry["n_per_condition"]*len(entry["design"]["conditions"])
        if completion_path.exists():
            receipt = read(completion_path)
            verify_completed(base, receipt)
            receipts.append(receipt)
            completed += count
            continue
        started, cpu = time.perf_counter(), time.process_time()
        rows = []
        timings = []
        with ReaderProcess(base/"reader-work", extensions=EXTENSIONS) as reader:
            for condition in entry["design"]["conditions"]:
                for index in range(entry["n_per_condition"]):
                    if remaining_seconds(root) < 60:
                        return {"execution_state": "checkpointed", "reason": "immutable deadline reserve", "campaign_complete": False}
                    unit_start = time.perf_counter()
                    row = execute_unit(card, base, condition, index, namespace=entry["namespace"], packet=packet,
                        reader=reader, constructors=entry["constructors"], scope=spec["evidence_scope"])
                    rows.append(row)
                    timings.append({"condition": condition["id"], "index": index, "elapsed_seconds": time.perf_counter()-unit_start})
                    completed += 1
                    heartbeat(card_id=card, completed_units=completed, planned_units=spec["planned_maker_condition_records"],
                        last_unit=row["unit_id"], operation="constructor expansion" if fixture is None else "known fixture")
        sampling = independent_units(rows)
        summary = summarizer(card, rows)
        write(base/"AGGREGATE.json", summary)
        audit = auditor(card)(base, summary)
        if audit["instrument_state"] != "valid":
            raise ValueError("independent expansion physics/statistics audit failed")
        write(base/"INDEPENDENT_AUDIT.json", audit)
        write(base/"SAMPLING.json", sampling)
        write(base/"RAW_MANIFEST.json", raw_manifest(base))
        write(base/"TIMING.json", {"unit_calls": timings, "wall_seconds": time.perf_counter()-started,
            "parent_cpu_seconds": time.process_time()-cpu, "includes": "reader startup, units, scoring, independent audit and raw hashing",
            "resume_scope": "unit-call timings include verification-only calls for already saved units after a restart"})
        receipt = {"card_id": card, "execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
            "n_makers_per_condition": entry["n_per_condition"], "n_condition_records": len(rows),
            "constructors_per_condition": min(entry["constructors"], entry["n_per_condition"]),
            "evidence_scope": spec["evidence_scope"], "confirmation_state": "not a confirmation",
            "aggregate_sha256": file_digest(base/"AGGREGATE.json"), "raw_manifest_sha256": file_digest(base/"RAW_MANIFEST.json"),
            "packet_hash": packet["packet_hash"], "campaign_complete": False}
        write(completion_path, receipt)
        receipts.append(receipt)
    result = {"execution_state": "completed", "instrument_state": "valid", "cards": receipts,
        "completed_condition_records": completed, "completed_at": now(), "campaign_complete": False,
        "remaining_work": "source-specific adversary joins, remaining expansion disposition, confirmation and four closeout proofs"}
    path = output/"COMPLETION.json"
    if path.exists():
        return read(path)
    write(path, result)
    return result
