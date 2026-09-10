"""Frozen endpoint whole-unit replay for one completed expansion or confirmation."""
import argparse
from pathlib import Path
import os
import time
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.replay_plan import lookup, item
from ghostscale.validation.soundingline.v16.whole_replay import replay_item
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
from runners.reaggregate_v16_packet import inventory

CAPS = {"constructor-expansion-1": 52, "boundary-expansion-1": 34, "confirmation-1": 6}


def run(root, output, name):
    if output.exists() or name not in CAPS:
        raise ValueError("whole-packet replay requires a new retained attempt and a declared finite packet")
    began = time.perf_counter()
    checks = source_locks(root, REPO)
    packet_path, completion_path, work = inventory(root, name)
    selected = []
    for summary, design, n in work:
        for condition, index in [(design["conditions"][0], 0), (design["conditions"][-1], n-1)]:
            path, row = lookup(summary.parent, condition["id"], index)
            selected.append(item(root, path, name, "native-card", condition, row["seed_components"]["constructors"]))
    if len(selected) > CAPS[name] or len({row["source_unit"] for row in selected}) != len(selected):
        raise ValueError("whole-packet replay exceeds its finite bound or duplicates a unit")
    paths = [Path(__file__).resolve(), REPO/"runners/reaggregate_v16_packet.py",
        REPO/"ghostscale/validation/soundingline/v16/replay_plan.py",
        REPO/"ghostscale/validation/soundingline/v16/whole_replay.py"]
    plan = {"packet_id": name, "packet_sha256": file_digest(packet_path), "completion_sha256": file_digest(completion_path),
        "selected": selected, "n_units": len(selected), "hard_cap": CAPS[name], "planned_at": now(),
        "source_checks": checks, "replay_sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in paths},
        "outcomes_used_for_selection": False, "selection": "First condition/index zero and last condition/final index of every native card in the completed frozen packet",
        "overall_native_and_search_bound": 160, "separate_descriptive_catalogue_bound": 36,
        "budget_accounting": "68 prior scout/native/search endpoints, 52 constructor endpoints, 34 final boundary endpoints, at most 6 confirmation endpoints; the 36-case catalogue is separate",
        "scientific_rule": "Regenerate complete worlds, acquisition histories, guarded predictions, continuations, programs and scores with the original frozen replay comparator",
        "not_recomputed": "Original OS resource samples remain observed measurements; opaque aliases and event timestamps may differ"}
    write(output/"PLAN.json", plan)
    receipts = []
    with ReaderProcess(output/"reader", extensions=EXTENSIONS) as reader:
        for index, selected_item in enumerate(selected):
            receipt = replay_item(root, output/"private"/f"case-{index:03d}", selected_item, reader)
            receipt["selected_source"] = selected_item["source_unit"]
            write(output/"checks"/f"case-{index:03d}.json", receipt)
            receipts.append(receipt)
            print({"wholly_replayed": index+1, "planned": len(selected), "source": selected_item["source_unit"]}, flush=True)
    result = {"execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
        "process_id": os.getpid(), "packet_id": name, "plan_sha256": file_digest(output/"PLAN.json"),
        "n_units": len(receipts), "checks": receipts, "selected_units_wholly_replayed": True,
        "empty_confirmation_selection": not selected and name == "confirmation-1",
        "whole_unit_replay": False, "campaign_complete": False, "wall_seconds": time.perf_counter()-began,
        "scope": "Every frozen endpoint for this packet; final proof must join all declared replay phases and the descriptive catalogue"}
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
    print({"execution_state": "completed", "whole_units": result["n_units"]})


if __name__ == "__main__":
    main()
