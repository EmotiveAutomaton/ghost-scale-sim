"""Read-only clean-process calculation inventory; never owns live run status."""
import argparse
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.aggregate_science import registered_designs, scientific, native_fixture, transfer, archive

REPO = Path(__file__).resolve().parents[1]


def inventory(root, excluded=()):
    work = []
    for packet_path in sorted((root/"packets").glob("*.json")):
        name = packet_path.stem
        packet = read(packet_path)
        design = packet["identity"]["design"]
        if name in excluded:
            if (root/name/"COMPLETION.json").exists():
                raise ValueError("cannot exclude a completed scientific packet")
            continue
        designs = registered_designs(design)
        if not designs:
            continue
        base = root/name
        candidates = [*base.glob("SUMMARY.json"), *base.glob("*/SUMMARY.json"),
                      *base.glob("AGGREGATE.json"), *base.glob("*/AGGREGATE.json")]
        candidates = [path for path in candidates if "conditions" in read(path)]
        if not candidates:
            raise ValueError("registered science has no completed summary: "+name)
        if name.startswith("constructor-") and not (base/"COMPLETION.json").exists():
            raise ValueError("unfinished expansion requires explicit partial-check exclusion")
        found = set()
        for path in candidates:
            card = read(path)["card_id"]
            if card in found or card not in designs:
                raise ValueError("duplicate or unregistered card summary")
            found.add(card)
            n = design.get("n_per_condition")
            if isinstance(design.get("cards"), list):
                entry = next(item for item in design["cards"] if item["card_id"] == card)
                n = entry["n_per_condition"]
            work.append((path, designs[card], n, name == "inquiry-scout-1"))
        if found != set(designs):
            raise ValueError("registered card summary missing: "+name)
    return work


def run(root, output, excluded=(), *, include_profile=True):
    began, cpu = time.perf_counter(), time.process_time()
    source = source_locks(root, REPO)
    reports = {}
    for path, design, n, invalid in inventory(root, excluded):
        label = path.relative_to(root).as_posix()
        reports[label] = scientific(path.parent, path, design, n, invalid_original=invalid)
        print({"reproduced": label, "raw_records": reports[label]["calculation"]["n_raw_units"]}, flush=True)
        write(output/"items"/(label.replace("/", "__")), reports[label])
    fixture_reports = {}
    if include_profile:
        profile_root = root/"expansion-setup/profile/campaign"
        for path, design, n, invalid in inventory(profile_root):
            label = path.relative_to(root).as_posix()
            fixture_reports[label] = scientific(path.parent, path, design, n, invalid_original=invalid)
            print({"reproduced_fixture": label}, flush=True)
    native = native_fixture(root/"native-fixture-1")
    transfers = {name: transfer(root/name) for name in ["transfer-fixture-1", "transfer-fixture-2"]}
    b02 = archive(root/"archive-search-1")
    # The broader final scope requires controls, subsequent expansions,
    # catalogue and confirmation reports. This development pass cannot close it.
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
              "process_id": os.getpid(), "clean_analysis_entry": "runners.reaggregate_v16",
              "source_checks": source, "scientific_summaries": reports, "profile_fixtures": fixture_reports,
              "native_fixture": native, "transfer_recounts": transfers, "archive_comparison": b02,
              "excluded_unfinished_packets": list(excluded),
              "full_aggregate_regeneration": False,
              "remaining_scope": ["control and platform measurement aggregates", "final boundary and catalogue reports", "confirmation results"],
              "scientific_replay": False, "raw_archive_handoff": False,
              "wall_seconds": time.perf_counter()-began, "cpu_seconds": time.process_time()-cpu,
              "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in
                          [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/aggregate_science.py"]}}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-incomplete-packet", action="append", default=[])
    args = parser.parse_args()
    try:
        result = run(args.root, args.output, args.exclude_incomplete_packet)
    except Exception as error:
        write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved",
            "recorded_at": now(), "error": repr(error), "full_aggregate_regeneration": False})
        raise
    print({"execution_state": result["execution_state"], "full_aggregate_regeneration": result["full_aggregate_regeneration"]})


if __name__ == "__main__":
    main()
