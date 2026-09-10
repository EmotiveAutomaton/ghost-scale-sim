"""Verify scientific samples and non-sampling fixture archives separately."""
import argparse
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.record_integrity import source_locks, raw_integrity, independent_units

ARCHIVES = {"archive-search-1"}

FIXTURES = {"transfer-fixture-1": "B01 standalone transfer",
            "access-attack-fixture-1": "X01 access/cache fixture",
            "transfer-fixture-2": "B01 corrected inquiry transfer",
            "access-attack-fixture-2": "X01 corrected inquiry and transfer access/cache fixture",
            "recoding-attack-fixture-1": "X02 macro and physical recoding fixture",
            "collision-attack-fixture-1": "X03 actual consumer collision controls",
            "fairness-attack-fixture-1": "X04 native consumer information/cost profiles; B02 still pending",
            "context-attack-fixture-1": "X05 actual false-context and correction controls",
            "misspecification-attack-fixture-1": "X06 alternative-generator and omitted-procedure controls",
            "dependence-attack-fixture-1": "X07 source identities, nesting and rejected-work inventory; bridge consumers pending",
            "noise-runtime-attack-fixture-1": "X08 physical noise and actual interruption controls; bridge consumers pending"}


def fixture_integrity(root, packets):
    records = []
    for packet in packets:
        base = root/packet
        mapping = read(base/"RAW_MANIFEST.json")["files"]
        actual = {str(path.relative_to(base)).replace("\\", "/")
                  for directory in ("public", "private", "predictions", "units", "runtime-fixture")
                  for path in (base/directory).rglob("*") if path.is_file()
                  and path.suffix not in {".log", ".lock", ".tmp"}}
        if set(mapping) != actual or not mapping:
            raise ValueError("fixture raw archive has missing or unmanifested files")
        total = 0
        for relative, expected in mapping.items():
            path = (base/relative).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("fixture path escapes its archive")
            if file_digest(path) != expected:
                raise ValueError("fixture raw bytes changed")
            total += path.stat().st_size
        report = read(base/("PUBLIC_REPORT.json" if packet.startswith("transfer-") else "COMPLETION.json"))
        if report["execution_state"] != "completed" or report["instrument_state"] != "valid":
            raise ValueError("fixture has not completed validly")
        records.append({"packet": packet, "kind": FIXTURES[packet], "files": len(mapping),
                        "bytes": total, "scientific_maker_sample_size": "not applicable",
                        "interrupted_temporary_writes": {
                            str(path.relative_to(base)).replace("\\", "/"): file_digest(path)
                            for folder in ["runtime-fixture", "private"]
                            for path in sorted((base/folder).rglob("*.tmp"))},
                        "manifest_sha256": file_digest(base/"RAW_MANIFEST.json")})
    return records


def archive_integrity(root, packets):
    records = []
    for packet in packets:
        base = root/packet
        manifest = read(base/"RAW_MANIFEST.json")
        mapping = manifest["files"]
        actual = {path.relative_to(base).as_posix() for folder in ["public", "private", "predictions", "units"]
                  for path in (base/folder).rglob("*.json")}
        if not mapping or set(mapping) != actual:
            raise ValueError("archive has missing or unmanifested raw files")
        for name, expected in mapping.items():
            path = (base/name).resolve()
            if not path.is_relative_to(base.resolve()) or file_digest(path) != expected:
                raise ValueError("archive raw source differs")
        completion = read(base/"COMPLETION.json")
        if completion["execution_state"] != "completed" or completion["instrument_state"] != "valid":
            raise ValueError("archive has not completed validly")
        groups = {}
        for group in ["search", "followup"]:
            rows = [read(path) for path in sorted((base/"private"/group).glob("**/units/*_points.json"))]
            groups[group] = independent_units(rows)
        if groups["search"]["n_retained_unit_records"] != completion["candidate_maker_records"] or groups["followup"]["n_retained_unit_records"] != completion["followup_maker_condition_records"]:
            raise ValueError("archive search or frozen follow-up denominator differs")
        records.append({"packet": packet, "files": len(mapping), "manifest_sha256": file_digest(base/"RAW_MANIFEST.json"),
            "bytes": sum((base/name).stat().st_size for name in mapping), "sampling": groups,
            "interrupted_temporary_writes": {path.relative_to(base).as_posix(): file_digest(path) for path in (base/"private").rglob("*.tmp")},
            "scope": "search candidate histories and shared-condition follow-up histories kept separate; nested operational fixtures excluded from these denominators"})
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--source-repo", type=Path, default=REPO)
    parser.add_argument("--git-ref", choices=["HEAD", "origin/main"])
    parser.add_argument("--exclude-incomplete-packet", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    sources = source_locks(args.root, args.source_repo, args.git_ref)
    packets = sources["packets"]
    excluded = {}
    for name in args.exclude_incomplete_packet:
        if name not in packets or (args.root/name/"COMPLETION.json").exists():
            raise ValueError("only an explicitly named unfinished packet can be excluded from raw completion checks")
        excluded[name] = {"packet_hash": packets[name], "raw_completion": "not checked; packet still executing"}
    science = raw_integrity(args.root, {key: value for key, value in packets.items() if key not in set(FIXTURES)|ARCHIVES|set(excluded)})
    fixtures = fixture_integrity(args.root, [key for key in packets if key in FIXTURES])
    archives = archive_integrity(args.root, [key for key in packets if key in ARCHIVES])
    report = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
              "source_checks": sources, "scientific_retention": science, "fixture_retention": fixtures, "archive_search_retention": archives,
              "unfinished_packets_excluded_from_raw_checks": excluded,
              "elapsed_seconds": time.perf_counter()-started, "parent_cpu_seconds": time.process_time()-cpu,
              "aggregate_regeneration": False, "whole_unit_replay": False, "final_archive_handoff": False,
              "verifier_sources": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path)
                   for path in [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/record_integrity.py"]}}
    write(args.output, report)
    print({"scientific_files": science["raw_files_checked"],
           "scientific_unit_records": science["retained_unit_records"],
           "fixture_files": sum(row["files"] for row in fixtures), "archive_files": sum(row["files"] for row in archives), "packet_sources": len(packets)})


if __name__ == "__main__":
    main()
