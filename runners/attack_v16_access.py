"""Freeze and execute X01 against the actual V16 consumers."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze, supervisor
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.attack_access import DESIGN, execute

REQUIRED = ["test_consumer_frames_preserve_both_paid_phases_without_private_data",
            "test_access_aliases_preserve_scientific_information_and_broken_output_is_detected",
            "test_standalone_transfer_known_answer_and_private_read_denial"]
PACKET = "access-attack-fixture-1"


def sources(root):
    existing = root/"packets"/f"{PACKET}.json"
    if existing.exists():
        return sorted(REPO/path for path in read(existing)["identity"]["files"])
    paths = {REPO/"runners/attack_v16_access.py", PACKAGE/"consumer_frames.py", PACKAGE/"attack_access.py",
             PACKAGE/"reader_process.py", PACKAGE/"transfer.py", REPO/"runners/v16_reader_worker.py",
             REPO/"tests/test_v16_consumer_frames.py"}
    for packet_path in (root/"packets").glob("*.json"):
        if packet_path.stem != PACKET:
            paths.update(REPO/path for path in read(packet_path)["identity"]["files"])
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--admit-from-junit", type=Path)
    args = parser.parse_args()
    files = sources(args.root)
    admission_path = args.root/"access-attack-setup/ADMISSION.json"
    if args.admit_from_junit:
        tests = {}
        for case in ET.parse(args.admit_from_junit).findall(".//testcase"):
            name = case.attrib["name"]
            if name in tests:
                raise ValueError("duplicate test identity")
            tests[name] = "failed" if any(case.find(tag) is not None for tag in ("error", "failure", "skipped")) else "passed"
        if not tests or any(value != "passed" for value in tests.values()) or any(name not in tests for name in REQUIRED):
            raise ValueError("incomplete access admission tests")
        write(admission_path, {"tests": tests, "recorded_at": now(),
            "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in files}})
        print({"passed_tests": len(tests)})
        return
    admission = read(admission_path)
    for path in files:
        if admission["source_hashes"].get(str(path.relative_to(REPO)).replace("\\", "/")) != file_digest(path):
            raise ValueError("source changed after access admission")
    with supervisor(args.root, "consumer-access-attack") as heartbeat:
        packet = freeze(args.root, PACKET, files, DESIGN)
        output = args.root/PACKET
        if (output/"COMPLETION.json").exists():
            report = read(output/"COMPLETION.json")
            heartbeat(execution_state="completed", result=report)
        else:
            report = execute(args.root, output, heartbeat)
    print(report)


if __name__ == "__main__":
    main()
