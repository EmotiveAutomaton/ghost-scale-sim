"""Separate distribution review; never changes the frozen transfer or predictions."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, canonical, now
from ghostscale.validation.soundingline.v16.runtime import REPO, supervisor

FORBIDDEN = {"seed", "seed_components", "constructor_id", "condition", "condition_spec",
             "true_production_record", "expected_predictions", "scoring_contract"}


def public_keys(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN:
            raise ValueError("evaluator field in public transfer data")
        for item in value.values():
            public_keys(item)
    elif isinstance(value, list):
        for item in value:
            public_keys(item)


def review(root, heartbeat):
    output = root/"transfer-fixture-2"
    manifest = read(output/"PUBLIC_MANIFEST.json")
    original = read(output/"COMPLETION.json")
    committed = read(output/"PREDICTIONS_COMMITTED.json")
    if original["instrument_state"] != "valid" or len(committed["prediction_files"]) != manifest["n_tasks"]:
        raise ValueError("incomplete transfer cannot receive a distribution receipt")
    files = {"PUBLIC_MANIFEST.json": file_digest(output/"PUBLIC_MANIFEST.json")}
    for path, expected in manifest["observations"].items():
        public = read(output/path)
        public_keys(public)
        for key in ("task_id", "lineage_id"):
            if len(public[key]) != 32 or any(char not in "0123456789abcdef" for char in public[key]):
                raise ValueError("malformed opaque public alias")
        if file_digest(output/path) != expected:
            raise ValueError("public input differs")
        files[path] = expected
        stage9 = f"public/stage9/{public['task_id']}.json"
        evidence = read(output/stage9)
        if set(evidence) != {"prefix", "options"} or canonical(public).decode() not in evidence["prefix"]:
            raise ValueError("Stage9 evidence differs from its public input")
        if file_digest(output/stage9) != manifest["stage9_evidence"][stage9]:
            raise ValueError("Stage9 public bytes changed")
        files[stage9] = file_digest(output/stage9)
    for path, expected in manifest["consumer_sources"].items():
        relative = "public/consumer/"+path
        if file_digest(output/relative) != expected:
            raise ValueError("public reference source changed")
        files[relative] = expected
    for relative, expected in committed["prediction_files"].items():
        if file_digest(output/relative) != expected:
            raise ValueError("retained standalone prediction changed")
    receipt = {"schema_version": "v16.transfer-distribution.1", "files": files,
               "rule": "only this explicit whitelist belongs in the reader-facing distribution",
               "evaluation_mapping": "private; original COMPLETION.json contains evaluator source-to-alias mapping",
               "standalone_predictions": "private until evaluation; excluded from reader distribution",
               "n_cases": manifest["n_cases"], "n_tasks": manifest["n_tasks"],
               "source_completion_sha256": file_digest(output/"COMPLETION.json")}
    write(output/"PUBLIC_DISTRIBUTION.json", receipt)
    public_report = {key: value for key, value in original.items() if key not in {"cases"}}
    public_report.update(distribution_review="valid", distribution_manifest_sha256=file_digest(output/"PUBLIC_DISTRIBUTION.json"),
        public_alias_mapping="retained only with evaluator data",
        status_reconciliation="original driver reported each fiftieth task; final count reconciled from all committed predictions")
    write(output/"PUBLIC_REPORT.json", public_report)
    heartbeat(card_id="B01", packet_id="transfer-fixture-2", completed_units=manifest["n_tasks"],
              planned_units=manifest["n_tasks"], execution_state="completed", result=public_report)
    return {key: public_report[key] for key in ("n_cases", "n_tasks", "distribution_review")}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    args = parser.parse_args()
    # Save the actual prior status before the new owner creates its own status.
    output = args.root/"transfer-fixture-2"
    target = args.root/"transfer-review-2/private/ORIGINAL_DRIVER_STATUS.json"
    if not target.exists():
        write(target, read(args.root/"RUNNER_STATUS.json"))
    with supervisor(args.root, "transfer-distribution-review") as heartbeat:
        print(review(args.root, heartbeat))


if __name__ == "__main__":
    main()
