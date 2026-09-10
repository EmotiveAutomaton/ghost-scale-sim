"""Independent arithmetic for an explicit completed native scientific packet."""
import argparse
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.aggregate_science import registered_designs, scientific


def inventory(root, name):
    if name not in {"constructor-expansion-1", "boundary-expansion-1", "confirmation-1"}:
        raise ValueError("this final calculation entry requires an explicit completed expansion or confirmation")
    packet_path = root/"packets"/(name+".json")
    packet = read(packet_path)
    base = root/("confirmation" if name == "confirmation-1" else name)
    completion = read(base/"COMPLETION.json")
    if completion.get("execution_state") != "completed" or completion.get("instrument_state") != "valid":
        raise ValueError("independent final calculation requires completed valid science with its original packet")
    if name != "constructor-expansion-1" and completion["packet_hash"] != packet["packet_hash"]:
        raise ValueError("final calculation top-level packet identity differs")
    entries = packet["identity"]["design"]["cards"]
    designs = registered_designs(packet["identity"]["design"])
    if len(entries) != len(designs) or {entry["card_id"] for entry in entries} != set(designs):
        raise ValueError("final calculation packet omitted or duplicated a scientific card")
    work = []
    for entry in entries:
        folder = base/entry["card_id"]
        done = read(folder/"COMPLETION.json")
        summary = folder/"AGGREGATE.json"
        if done.get("execution_state") != "completed" or done.get("instrument_state") != "valid" or done["aggregate_sha256"] != file_digest(summary) or done["packet_hash"] != packet["packet_hash"]:
            raise ValueError("final calculation encountered incomplete or changed native evidence")
        work.append((summary, designs[entry["card_id"]], entry["n_per_condition"]))
    if name == "constructor-expansion-1":
        if {row["card_id"] for row in completion["cards"]} != set(designs) or len(completion["cards"]) != len(designs):
            raise ValueError("expansion completion omitted or duplicated a registered native card")
        if any(row != read(base/row["card_id"]/"COMPLETION.json") for row in completion["cards"]):
            raise ValueError("expansion top-level card identity differs from its native completion")
    return packet_path, base/"COMPLETION.json", work


def run(root, output, name):
    began, cpu = time.perf_counter(), time.process_time()
    locks = source_locks(root, REPO)
    packet_path, completion_path, work = inventory(root, name)
    reports = {}
    for path, design, n in work:
        label = path.relative_to(root).as_posix()
        report = scientific(path.parent, path, design, n)
        write(output/"items"/(label.replace("/", "__")), report)
        reports[label] = report
        print({"independently_recalculated": label, "raw_records": report["calculation"]["n_raw_units"]}, flush=True)
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "process_id": os.getpid(), "clean_analysis_entry": "runners.reaggregate_v16_packet",
        "packet_id": name, "source_checks": locks, "packet_sha256": file_digest(packet_path),
        "completion_sha256": file_digest(completion_path), "scientific_summaries": reports,
        "raw_records": sum(report["calculation"]["n_raw_units"] for report in reports.values()),
        "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in
            [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/aggregate_science.py"]},
        "wall_seconds": time.perf_counter()-began, "cpu_seconds": time.process_time()-cpu,
        "full_aggregate_regeneration": False, "campaign_complete": False,
        "scope": "All registered native contrasts of this completed packet; final control, catalogue and confirmation-primary accounting remain separate joins"}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", required=True)
    args = parser.parse_args()
    try:
        result = run(args.root, args.output, args.packet)
    except Exception as error:
        write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise
    print({"execution_state": result["execution_state"], "raw_records": result["raw_records"]}, flush=True)


if __name__ == "__main__":
    main()
