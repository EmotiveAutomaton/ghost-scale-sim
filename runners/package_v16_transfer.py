"""Package the completed B01 public whitelist and verify an extracted consumer."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, now, file_digest, canonical, digest
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create, identity
from runners.review_v16_transfer_stable import review


def run(root, output, archive_root):
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("public transfer packaging requires a new retained campaign directory")
    base = root/"transfer-fixture-2"
    original = {name: file_digest(base/name) for name in ["PUBLIC_DISTRIBUTION.json", "PUBLIC_REPORT.json", "COMPLETION.json", "PREDICTIONS_COMMITTED.json"]}
    # The frozen reviewer only rewrites identical immutable receipts. It does
    # not own status here, execute science, or alter the retained export.
    reviewed = review(root, lambda **updates: None)
    if original != {name: file_digest(base/name) for name in original}:
        raise ValueError("public packaging changed an existing scientific export")
    dependencies = {}
    for name, field in [("access-attack-fixture-2", "consumer_cards"), ("recoding-attack-fixture-1", "consumer_requests_checked"),
                        ("collision-attack-fixture-1", "source_condition_calibrations"), ("context-attack-fixture-1", "source_condition_calibrations"),
                        ("misspecification-attack-fixture-1", "source_condition_calibrations")]:
        path = root/name/"COMPLETION.json"
        receipt = read(path)
        if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt[field].get("B01", 0) < 1:
            raise ValueError("corrected transfer lacks an actual consumer-specific adversary")
        dependencies[name] = {"sha256": file_digest(path), "B01_checks_or_requests": receipt[field]["B01"]}
    distribution = read(base/"PUBLIC_DISTRIBUTION.json")
    paths = []
    for name, sha in distribution["files"].items():
        path = (base/name).resolve()
        if name != "PUBLIC_MANIFEST.json" and Path(name).parts[0] != "public":
            raise ValueError("private/evaluator data cannot enter the reader bundle")
        if not path.is_relative_to(base.resolve()) or file_digest(path) != sha:
            raise ValueError("reader whitelist path is escaping or changed")
        paths.append(path)
    plan = manifest(base, paths, scope={"kind": "B01 reader-facing payload", "source_packet": "transfer-fixture-2",
        "distribution_sha256": original["PUBLIC_DISTRIBUTION.json"], "payload_rule": "Exactly the original reviewed whitelist; the additional ZIP inventory is packaging metadata only"})
    directory = archive_root/"public-transfer"/output.name
    archived = create(base, directory, plan)
    output.mkdir(parents=True)
    shutil.copy2(directory/"raw.zip", output/"reader.zip")
    if identity(output/"reader.zip") != archived["archive_identity"]:
        raise ValueError("versioned reader bundle differs from the verified archive")
    extracted = output/"private/extracted"
    with zipfile.ZipFile(output/"reader.zip") as bundle:
        if set(bundle.namelist()) != set(distribution["files"])|{"ARCHIVE_MEMBER_MANIFEST.json"}:
            raise ValueError("reader ZIP contains an unapproved payload")
        bundle.extractall(extracted)
    observations = read(base/"PUBLIC_MANIFEST.json")["observations"]
    selected = sorted(observations)[0]
    observation = read(extracted/selected)
    sentinel = output/"private/evaluator-sentinel.txt"
    sentinel.write_text("known private-access test witness", encoding="utf-8")
    request = b"\n".join([canonical(observation), canonical({"operation": "probe-private", "path": str(sentinel.resolve())}), canonical({"operation": "shutdown"})])+b"\n"
    environment = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, "-s", "-B", "-u", "-m", "consumer"], cwd=extracted/"public/consumer",
        env=environment, input=request, capture_output=True, timeout=60)
    (output/"private/consumer.stdout.log").write_bytes(completed.stdout)
    (output/"private/consumer.stderr.log").write_bytes(completed.stderr)
    if completed.returncode:
        raise ValueError("extracted standalone reader command failed")
    import json
    response = [json.loads(line) for line in completed.stdout.splitlines()]
    expected = read(base/"predictions"/(observation["task_id"]+".json"))
    if len(response) != 3 or response[0].get("ready") is not True or response[0]["ghostscale_imports"] or response[1].get("ok") is not True or response[1]["result"] != expected["result"] or response[1]["observation_sha256"] != digest(observation):
        raise ValueError("extracted standalone reader differs from its frozen public prediction")
    if response[2].get("ok") is not False or "PermissionError" not in response[2].get("error", ""):
        raise ValueError("extracted standalone reader did not deny the evaluator file")
    expected_sources = read(base/"PUBLIC_MANIFEST.json")["consumer_sources"]
    if response[0]["sources"] != expected_sources:
        raise ValueError("extracted standalone reader loaded different source bytes")
    (output/"README.md").write_text("# V16 public task bundle\n\nExtract reader.zip into a new empty directory. "
        "Its payload is the corrected transfer-fixture-2 public whitelist: 44 recorded synthetic cases and 1,574 public reader tasks. "
        "Evaluator mappings and committed predictions are retained separately.\n\n"
        "From public/consumer, run `python -s -B -u -m consumer` in a Python environment with NumPy. "
        "Send one public observation JSON object per input line. The first output is the ready/source receipt; subsequent outputs are predictions. "
        "Send `{\"operation\":\"shutdown\"}` to stop. Neither Ghost Scale nor Sounding Line needs to be installed.\n\n"
        "See docs/versions/v16-acquired-craft/TRANSFER.md in the repository for the evidence interface and real-record limits. "
        "This is an executed synthetic task interface, not a deployed Sounding Line reader or evidence of human capability.\n", encoding="utf-8", newline="\n")
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "source_packet": "transfer-fixture-2", **reviewed, "source_receipts": original, "actual_consumer_controls": dependencies,
        "reader_archive_identity": archived["archive_identity"], "payload_members": archived["verified_members"],
        "all_payload_bytes_verified": True, "reader_payload_matches_original_whitelist": True,
        "extracted_consumer_executed": True, "extracted_known_prediction_matched": True, "extracted_private_read_denied": True,
        "test_task_sha256": file_digest(base/selected), "standalone_source_hashes": response[0]["sources"],
        "selection_rule": "One lexically first opaque public task for packaging verification; the original complete 1,574-task prediction comparison remains separately retained",
        "scientific_observations_added": 0, "real_text_reader_launched": False, "campaign_complete": False,
        "source_sha256": file_digest(Path(__file__).resolve())}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing public transfer handoff is retained")
    try:
        result = run(args.root.resolve(), args.output.resolve(), args.archive_root.resolve())
    except Exception as error:
        if args.output.resolve().is_relative_to(args.root.resolve()):
            write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise
    print({key:result[key] for key in ["execution_state", "n_cases", "n_tasks", "extracted_consumer_executed", "extracted_private_read_denied"]})


if __name__ == "__main__":
    main()
