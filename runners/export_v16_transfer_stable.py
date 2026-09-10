"""B01 standalone transfer packet; run with the scientific interpreter in module form."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze, supervisor
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.transfer_stable import DESIGN, MODULES, export, consume, evaluate

PACKET_ID = "transfer-fixture-2"
REQUIRED_TESTS = [
    "test_amended_standalone_retains_recoding_and_zero_value_repairs",
    "test_amended_standalone_transfer_known_answer_and_private_read_denial",
    "test_amended_standalone_transfer_learning_changes_executed_competence_and_noise_does_not",
    "test_amended_stage9_transfer_has_complete_opaque_options_and_exact_evidence_shape",
    "test_amended_transfer_evaluator_requires_committed_predictions_and_rejects_changed_results"]


def sources():
    return [PACKAGE/f"{name}.py" for name in [*MODULES, "transfer_stable", "transfer_consumer", "inquiry_stable"]] + [
        REPO/"runners/export_v16_transfer_stable.py", REPO/"runners/replay_v16_readers.py",
        REPO/"tests/test_v16_transfer_stable.py", REPO/"runners/review_v16_transfer_stable.py"]


def admit(root, junit):
    tests = {}
    for case in ET.parse(junit).findall(".//testcase"):
        name = case.attrib["name"]
        if name in tests:
            raise ValueError("duplicate test identity")
        tests[name] = "failed" if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) else "passed"
    if not tests or any(value != "passed" for value in tests.values()) or any(name not in tests for name in REQUIRED_TESTS):
        raise ValueError("transfer admission suite is incomplete or failed")
    record = {"recorded_at": now(), "tests": tests, "junit_sha256": file_digest(junit),
              "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in sources()}}
    write(root/"transfer-stable-setup/ADMISSION.json", record)
    return {"recorded_passed_tests": len(tests)}


def execute(root, heartbeat):
    admission = read(root/"transfer-stable-setup/ADMISSION.json")
    if any(admission["tests"].get(name) != "passed" for name in REQUIRED_TESTS):
        raise ValueError("standalone transfer has not passed its actual admission tests")
    for path in sources():
        if admission["source_hashes"].get(str(path.relative_to(REPO)).replace("\\", "/")) != file_digest(path):
            raise ValueError("transfer source changed after admission")
    packet = freeze(root, PACKET_ID, sources(), DESIGN)
    destination = root/PACKET_ID
    manifest = export(root, destination, packet)
    heartbeat(card_id="B01", packet_id=PACKET_ID, completed_units=0, planned_units=manifest["n_tasks"])
    consume(destination, heartbeat)
    report = evaluate(destination)
    from runners.review_v16_transfer_stable import review
    review(root, heartbeat)
    files = {str(path.relative_to(destination)).replace("\\", "/"): file_digest(path)
             for directory in ["public", "private", "predictions"] for path in sorted((destination/directory).rglob("*"))
             if path.is_file() and not path.name.endswith(".log")}
    write(destination/"RAW_MANIFEST.json", {"files": files, "retained": True,
          "scope": "exported public, private and committed prediction files including reference source",
          "retention": "through verified archival handoff"})
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--admit-from-junit", type=Path)
    args = parser.parse_args()
    if args.admit_from_junit:
        print(admit(args.root, args.admit_from_junit))
        return
    with supervisor(args.root, "transfer") as heartbeat:
        report = execute(args.root, heartbeat)
        heartbeat(execution_state=report["execution_state"], result={key: value for key, value in report.items() if key != "cases"})
    print({key: value for key, value in report.items() if key != "cases"})


if __name__ == "__main__":
    main()
