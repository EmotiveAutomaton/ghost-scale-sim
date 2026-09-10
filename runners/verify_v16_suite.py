"""Verify scientific samples and non-sampling fixture archives separately."""
import argparse
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.record_integrity import source_locks, raw_integrity

FIXTURES = {"transfer-fixture-1": "B01 standalone transfer",
            "access-attack-fixture-1": "X01 access/cache fixture",
            "transfer-fixture-2": "B01 corrected inquiry transfer",
            "access-attack-fixture-2": "X01 corrected inquiry and transfer access/cache fixture",
            "recoding-attack-fixture-1": "X02 macro and physical recoding fixture",
            "collision-attack-fixture-1": "X03 actual consumer collision controls",
            "fairness-attack-fixture-1": "X04 native consumer information/cost profiles; B02 still pending",
            "context-attack-fixture-1": "X05 actual false-context and correction controls",
            "misspecification-attack-fixture-1": "X06 alternative-generator and omitted-procedure controls"}


def fixture_integrity(root, packets):
    records = []
    for packet in packets:
        base = root/packet
        mapping = read(base/"RAW_MANIFEST.json")["files"]
        actual = {str(path.relative_to(base)).replace("\\", "/")
                  for directory in ("public", "private", "predictions", "units")
                  for path in (base/directory).rglob("*") if path.is_file()
                  and path.name not in {"reader-stderr.log", "consumer-stderr.log"}}
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
                        "manifest_sha256": file_digest(base/"RAW_MANIFEST.json")})
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--source-repo", type=Path, default=REPO)
    parser.add_argument("--git-ref", choices=["HEAD", "origin/main"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    sources = source_locks(args.root, args.source_repo, args.git_ref)
    packets = sources["packets"]
    science = raw_integrity(args.root, {key: value for key, value in packets.items() if key not in FIXTURES})
    fixtures = fixture_integrity(args.root, [key for key in packets if key in FIXTURES])
    report = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
              "source_checks": sources, "scientific_retention": science, "fixture_retention": fixtures,
              "elapsed_seconds": time.perf_counter()-started, "parent_cpu_seconds": time.process_time()-cpu,
              "aggregate_regeneration": False, "whole_unit_replay": False, "final_archive_handoff": False,
              "verifier_sources": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path)
                   for path in [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/record_integrity.py"]}}
    write(args.output, report)
    print({"scientific_files": science["raw_files_checked"],
           "scientific_unit_records": science["retained_unit_records"],
           "fixture_files": sum(row["files"] for row in fixtures), "packet_sources": len(packets)})


if __name__ == "__main__":
    main()
