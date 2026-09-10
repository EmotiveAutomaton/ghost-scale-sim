"""Admit and execute the bounded consumer-specific X05 controls."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze, supervisor
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.attack_context import DESIGN, TRANSFER_FAMILIES, execute
from ghostscale.validation.soundingline.v16.context_cases import CASES
from ghostscale.validation.soundingline.v16.active_consumers import attack_consumers

PACKET = "context-attack-fixture-1"
REQUIRED = ["test_false_context_controls_reject_ignored_evidence_and_rewritten_goal"] + [
    f"test_actual_false_context_correction_and_trust_boundary[{family}]" for family in sorted(CASES)] + [
    f"test_actual_standalone_false_context_controls[{family}]" for family in TRANSFER_FAMILIES]


def sources(root):
    previous = root/"packets"/f"{PACKET}.json"
    if previous.exists():
        return sorted(REPO/path for path in read(previous)["identity"]["files"])
    files = {REPO/"runners/attack_v16_context.py", PACKAGE/"attack_context.py",
             PACKAGE/"context_cases.py", REPO/"tests/test_v16_context.py"}
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
        raise ValueError("resume never prepares context controls")
    files = sources(args.root)
    admission_path = args.root/"context-setup/ADMISSION.json"
    if args.admit_from_junit:
        tests = {}
        for case in ET.parse(args.admit_from_junit).findall(".//testcase"):
            name = case.attrib["name"]
            if name in tests:
                raise ValueError("duplicate executed test identity")
            tests[name] = "failed" if any(case.find(tag) is not None for tag in ("error", "failure", "skipped")) else "passed"
        if not tests or any(value != "passed" for value in tests.values()) or any(name not in tests for name in REQUIRED):
            raise ValueError("context admission tests are missing or failed")
        write(admission_path, {"tests": tests, "recorded_at": now(), "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in files}})
        print({"passed_tests": len(tests), "bound_sources": len(files)})
        return
    admission = read(admission_path)
    if any(admission["tests"].get(name) != "passed" for name in REQUIRED):
        raise ValueError("context required executed admission missing")
    for path in files:
        if admission["source_hashes"].get(str(path.relative_to(REPO)).replace("\\", "/")) != file_digest(path):
            raise ValueError("context source differs from executed test receipt")
    with supervisor(args.root, "context-attack") as heartbeat:
        design = {**DESIGN, "consumer_cards": attack_consumers(args.root)["X05"]}
        packet = freeze(args.root, PACKET, files, design)
        output = args.root/PACKET
        report = read(output/"COMPLETION.json") if (output/"COMPLETION.json").exists() else execute(args.root, output, heartbeat, packet)
        heartbeat(execution_state=report["execution_state"], result=report)
    print(report)


if __name__ == "__main__":
    main()
