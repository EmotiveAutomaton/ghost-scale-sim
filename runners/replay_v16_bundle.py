"""Regenerate the fixed whole-unit cases in an extracted portable replay bundle."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.replay_coverage import checked_file
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.whole_replay import replay_item
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS


def run(root, plan_path, output):
    if output.exists():
        raise ValueError("portable replay requires a new retained attempt")
    plan = read(plan_path)
    selected = plan["selected"]
    if not selected or len(selected) > 196 or len({row["source_unit"] for row in selected}) != len(selected):
        raise ValueError("portable replay allocation is empty, duplicated or exceeds its finite bound")
    path = checked_file(REPO, plan["dependencies"], plan["dependencies_sha256"])
    dependencies = read(path)["files"]
    if not dependencies:
        raise ValueError("portable replay lacks its dependency inventory")
    for name, expected in dependencies.items():
        checked_file(REPO, name, expected)
    source_locks(root, REPO)
    results = []
    with ReaderProcess(output/"reader", extensions=EXTENSIONS) as reader:
        for index, item in enumerate(selected):
            checked_file(root, item["source_unit"], item["source_unit_sha256"])
            result = replay_item(root, output/"private"/f"case-{index:03d}", item, reader)
            write(output/"checks"/f"case-{index:03d}.json", result)
            results.append(result)
            print({"wholly_replayed": index+1, "planned": len(selected)}, flush=True)
    receipt = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "plan_sha256": file_digest(plan_path), "distinct_units": len(selected), "selected_units_wholly_replayed": True,
        "checks": results, "scientific_observations_added": 0,
        "scope": "Reproduction of the fixed bounded bundle; original measured OS resource values are not regenerated"}
    write(output/"RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing portable replay attempt is retained")
    try:
        run(args.root.resolve(), args.plan.resolve(), args.output.resolve())
    except Exception as error:
        write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
