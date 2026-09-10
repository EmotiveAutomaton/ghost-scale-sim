"""Admit and execute the bounded B02 search with a separately frozen follow-up."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, supervisor, freeze, campaign
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.archive_design import DESIGN
from ghostscale.validation.soundingline.v16.archive_study import execute
from ghostscale.validation.soundingline.v16.archive_interruption import control

PACKET = "archive-search-1"
FILES = ["archive_design.py", "archive_cases.py", "archive_reference.py", "archive_search.py", "archive_study.py", "archive_interruption.py", "archive_controls.py"]
REQUIRED = ["test_archive_known_selection_ambiguity_and_corrected_error_cases",
    "test_archive_semantics_reject_uuid_novelty_and_false_equivalence",
    "test_fixed_and_adaptive_search_have_frozen_bounded_edits",
    "test_archive_candidate_independent_physics_and_exact_budget",
    "test_archive_followup_cannot_replace_failed_first_witness",
    "test_actual_archive_interrupt_resume_preserves_all_saved_work"]


def sources(root, name=PACKET):
    existing = root/"packets"/(name+".json")
    if existing.exists():
        return sorted(REPO/path for path in read(existing)["identity"]["files"])
    files = {PACKAGE/name for name in FILES}
    files.update([REPO/"runners/run_v16_archive.py", REPO/"tests/test_v16_archive.py"])
    for path in (REPO/"results/v16/packets").glob("*.json"):
        files.update(REPO/name for name in read(path)["identity"]["files"])
    return sorted(files)


def execute_campaign(root, heartbeat, *, resume=False, fixture=False):
    name = "archive-known-fixture-1" if fixture else PACKET
    if resume and not (root/"packets"/(name+".json")).exists():
        raise ValueError("archive resume never prepares")
    files = sources(root, name)
    if fixture:
        if not campaign(root)["campaign_id"].startswith("fixture-"):
            raise ValueError("fixture cannot use the live campaign identity")
    else:
        admission = read(root/"archive-setup/ADMISSION.json")
        if any(admission["tests"].get(name) != "passed" for name in REQUIRED):
            raise ValueError("missing archive admission test")
        if any(admission["source_hashes"].get(path.relative_to(REPO).as_posix()) != file_digest(path) for path in files):
            raise ValueError("archive differs from executed admission source")
    packet = freeze(root, name, files, {**DESIGN, "evidence_scope": "fixture" if fixture else "archive discovery"})
    output = root/name
    if (output/"COMPLETION.json").exists():
        result = read(output/"COMPLETION.json")
        for relative, expected in read(output/"RAW_MANIFEST.json")["files"].items():
            if file_digest(output/relative) != expected:
                raise ValueError("completed archive changed before resume")
        return result
    if not fixture:
        runtime = control(output/"private/runtime-control", packet["packet_hash"])
        if runtime["instrument_state"] != "valid":
            raise ValueError("actual archive runtime failed its interruption control")
    return execute(root, output, heartbeat, packet, fixture=fixture)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--admit-from-junit", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.admit_from_junit:
        tests = {case.attrib["name"]: "failed" if any(case.find(tag) is not None for tag in ["failure", "error", "skipped"]) else "passed"
                 for case in ET.parse(args.admit_from_junit).findall(".//testcase")}
        if not tests or any(value != "passed" for value in tests.values()) or any(name not in tests for name in REQUIRED):
            raise ValueError("archive tests missing or failed")
        files = sources(args.root)
        write(args.root/"archive-setup/ADMISSION.json", {"tests": tests, "recorded_at": now(),
            "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {path.relative_to(REPO).as_posix(): file_digest(path) for path in files}})
        print({"tests": len(tests), "source_files": len(files)})
        return
    with supervisor(args.root, "archive") as heartbeat:
        report = execute_campaign(args.root, heartbeat, resume=args.resume, fixture=args.fixture)
        heartbeat(execution_state=report["execution_state"], result=report)
    print(report)


if __name__ == "__main__":
    main()
