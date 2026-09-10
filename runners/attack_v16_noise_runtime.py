"""Admit source-bound X08 controls and run the finite native consumer profile."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze, supervisor
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.attack_noise_runtime import DESIGN, SNAPSHOT, execute
from ghostscale.validation.soundingline.v16.active_consumers import attack_consumers

PACKET = "noise-runtime-attack-fixture-1"
REQUIRED = ['test_executed_noise_fools_progress_but_has_no_expected_learning_value[decline]', 'test_executed_noise_fools_progress_but_has_no_expected_learning_value[rise]', 'test_noise_control_rejects_progress_mislabeled_as_value_learning', 'test_completion_guard_separates_valid_null_missing_work_and_missing_proofs', 'test_completion_guard_keeps_invalid_or_blocked_work_qualified', 'test_closeout_proofs_bind_actual_saved_bytes_and_archive_location', 'test_actual_interruption_control_preserves_clock_units_and_scope']


def sources(root):
    previous = root/"packets"/f"{PACKET}.json"
    if previous.exists():
        return sorted(REPO/path for path in read(previous)["identity"]["files"])
    files = {REPO/"runners/attack_v16_noise_runtime.py", REPO/"tests/test_v16_noise_runtime.py"}
    files.update(PACKAGE/name for name in ["attack_noise_runtime.py", "noise_cases.py", "completion_guard.py", "runtime_interruption.py"])
    files.add(REPO/SNAPSHOT)
    files.add(REPO/"runners/v16_interrupt_fixture.py")
    for path in (root/"packets").glob("*.json"):
        files.update(REPO/relative for relative in read(path)["identity"]["files"])
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--admit-from-junit", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not (args.root/"packets"/f"{PACKET}.json").exists():
        raise ValueError("resume never prepares noise_runtime controls")
    files = sources(args.root)
    admission_path = args.root/"noise-runtime-setup/ADMISSION.json"
    if args.admit_from_junit:
        tests = {}
        for case in ET.parse(args.admit_from_junit).findall(".//testcase"):
            name = case.attrib["name"]
            if name in tests:
                raise ValueError("duplicate executed test identity")
            tests[name] = "failed" if any(case.find(tag) is not None for tag in ("error", "failure", "skipped")) else "passed"
        if not tests or any(value != "passed" for value in tests.values()) or any(name not in tests for name in REQUIRED):
            raise ValueError("noise_runtime admission tests are missing or failed")
        write(admission_path, {"tests": tests, "recorded_at": now(), "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in files}})
        print({"passed_tests": len(tests), "bound_sources": len(files)})
        return
    admission = read(admission_path)
    if any(admission["tests"].get(name) != "passed" for name in REQUIRED):
        raise ValueError("noise_runtime required executed admission missing")
    for path in files:
        if admission["source_hashes"].get(str(path.relative_to(REPO)).replace("\\", "/")) != file_digest(path):
            raise ValueError("noise_runtime source differs from executed test receipt")
    with supervisor(args.root, "noise_runtime-attack") as heartbeat:
        packet = freeze(args.root, PACKET, files, {**DESIGN, "consumer_cards": attack_consumers(args.root)["X08"]})
        output = args.root/PACKET
        report = read(output/"COMPLETION.json") if (output/"COMPLETION.json").exists() else execute(args.root, output, heartbeat, packet)
        heartbeat(execution_state=report["execution_state"], result=report)
    print(report)


if __name__ == "__main__":
    main()
