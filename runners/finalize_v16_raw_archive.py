"""Complete raw coverage only after scientific execution and independent proofs."""
import argparse
from collections import defaultdict
from pathlib import Path
import os
from ghostscale.validation.soundingline.v16.runtime import REPO, remaining_seconds
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.completion_guard import bound_receipt
from ghostscale.validation.soundingline.v16.operational_queue import dispatchable
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create
from ghostscale.validation.soundingline.v16.archive_coverage import snapshot, coverage


def baseline_matches(required, expected):
    if not expected or any(required.get(name, {}).get("sha256") != sha for name,sha in expected.items()):
        raise ValueError("current raw files differ from independently regenerated input baseline")


def prerequisites(root, path):
    inputs = read(path)
    raw_baseline = None
    for name, flag in [("independent_aggregates", "full_aggregate_regeneration"), ("scientific_replay", "whole_unit_replay")]:
        item = inputs["proofs"][name]
        receipt = bound_receipt(root, item["path"], item["sha256"])
        if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt.get(flag) is not True:
            raise ValueError("raw closeout requires the completed independent calculation and replay proofs")
        if name == "independent_aggregates":
            reference = receipt["raw_input_baseline"]
            raw_baseline = bound_receipt(root, reference["path"], reference["sha256"])
    for name in ["boundary-expansion-1", "boundary-controls-1", "expansion-closure", "confirmation"]:
        receipt = read(root/name/"COMPLETION.json")
        if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid":
            raise ValueError("raw closeout cannot precede the finite scientific queue")
    forecast = bound_receipt(root, inputs["forecast"]["path"], inputs["forecast"]["sha256"])
    if not dispatchable(root, {"job_id": "final raw archive"}, forecast)["eligible_within_deadline"]:
        raise ValueError("raw closeout forecast exceeds the immutable remaining horizon")
    return inputs, raw_baseline


def validate_output(root, output):
    if output.exists() or not output.resolve().is_relative_to((root/"closeout").resolve()):
        raise ValueError("raw closeout requires a new retained attempt under its administrative directory")


def run(root, archive_root, output, input_path):
    validate_output(root, output)
    inputs, raw_baseline = prerequisites(root, input_path)
    checks = source_locks(root, REPO)
    required = snapshot(REPO, root)
    baseline_matches(required, raw_baseline["files"])
    write(output/"required_files_points.json", {"files": required,
        "scope": "All retained V16 scientific/control/setup/replay evidence and implementation, commission/method/environment inputs",
        "excluded": "Live status, owner locks, the actual root operations/ACTIVE_JOB.json cursor, temporary/cache files and final closeout administration; frozen queue plans and failures remain included; final documents/admin receipts remain in ordinary Git"})
    receipts = sorted((archive_root/"RECEIPTS").glob("*.json"))
    existing = coverage(archive_root, receipts, required, verify_bytes=False)
    groups = defaultdict(list)
    for name in existing["missing_or_different_files"]:
        parts = Path(name).parts
        label = parts[2] if len(parts) > 3 and parts[:2] == ("results", "v16") else "implementation-and-environment"
        groups[label].append(REPO/name)
    batch_name = "final-residual-"+output.name
    new_chunks = []
    for label, files in sorted(groups.items()):
        for number, start in enumerate(range(0, len(files), 25000)):
            if remaining_seconds(root) < 60:
                raise ValueError("immutable campaign ceiling reached during raw archival closeout")
            plan = manifest(REPO, files[start:start+25000], scope={"section": label, "batch": batch_name,
                "evidence_state": "Original scientific, fixture and failure labels retained; archival bytes add no independent observations"})
            directory = archive_root/"final-residuals"/batch_name/label/f"chunk-{number:04d}"
            receipt = create(REPO, directory, plan)
            new_chunks.append({"relative_archive": directory.relative_to(archive_root).as_posix()+"/raw.zip",
                "member_count": receipt["verified_members"], "uncompressed_bytes": receipt["uncompressed_bytes"],
                "archive_identity": receipt["archive_identity"], "plan_sha256": receipt["plan_sha256"]})
            print({"new_verified_raw_chunk": label, "chunk": number, "members": receipt["verified_members"]}, flush=True)
    if new_chunks:
        receipt_path = archive_root/"RECEIPTS"/(batch_name+".json")
        write(receipt_path, {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
            "chunks": new_chunks, "complete_campaign_archive": False})
        receipts.append(receipt_path)
    def progress(name, covered, expected):
        print({"archive_fully_reread": name, "current_files_covered": covered, "required_files": expected}, flush=True)
    result = coverage(archive_root, receipts, required, report=progress)
    if not result["verified_complete_accessible"] or snapshot(REPO, root) != required:
        raise ValueError("raw archive lacks current complete coverage or its source changed during verification")
    write(output/"coverage_points.json", result)
    final = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "process_id": os.getpid(), "source_checks": checks, "verified_complete_accessible": True,
        "required_files": result["required_files"], "required_bytes": result["required_bytes"], "verified_chunks": len(result["chunks"]),
        "all_archive_member_bytes_reread": True, "current_source_snapshot_unchanged": True,
        "independent_calculation_input_baseline_matched": True,
        "required_inventory_sha256": file_digest(output/"required_files_points.json"),
        "coverage_sha256": file_digest(output/"coverage_points.json"),
        "batch_receipts": {path.relative_to(archive_root).as_posix(): file_digest(path) for path in receipts},
        "prerequisites_sha256": file_digest(input_path),
        "retention": "Keep the complete raw archive directory, member manifests, batch receipts and this coverage record; no automatic deletion",
        "location": "The retained raw-archive directory beside the active execution checkout; member paths are portable within that directory",
        "administrative_scope": "Final closure receipts and final document updates are tracked separately; no raw scientific or failed unit is excluded on that basis",
        "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/archive_coverage.py"]}}
    write(output/"RECEIPT.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prerequisites", type=Path, required=True)
    args = parser.parse_args()
    # Refusing an old/outside attempt must not add a failure into that directory.
    validate_output(args.root, args.output)
    try:
        result = run(args.root, args.archive_root, args.output, args.prerequisites)
    except Exception as error:
        write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error), "partial_archives_preserved": True})
        raise
    print({"execution_state": "completed", "verified_complete_accessible": result["verified_complete_accessible"]})


if __name__ == "__main__":
    main()
