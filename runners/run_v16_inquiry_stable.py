"""Source-bound admission and finite R-family numerical repair discovery."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, freeze, supervisor, remaining_seconds
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.inquiry_designs import DESIGNS
from ghostscale.validation.soundingline.v16.inquiry_stable_study import execute_unit, summarize, STABLE_OPERATIONS
from ghostscale.validation.soundingline.v16.inquiry_stable_gates import run
from ghostscale.validation.soundingline.v16.audit_inquiry_stable import audit
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.resource_accounting import MeasuredReader

PACKET = "inquiry-scout-2"
REQUIRED = ["test_amended_learning_noise_and_broken_controls_are_source_bound",
    "test_stable_action_ties_and_resolved_differences",
    "test_corrected_policy_matches_independent_scalar_known_answers",
    "test_known_mapping_noise_mixture_has_no_construction_value_from_queries",
    "test_amended_episodes_independent_math_physics_and_resources",
    "test_inquiry_recoding_retains_old_failure_and_validates_corrected_reader"]


def sources(root):
    existing = root/"packets"/f"{PACKET}.json"
    if existing.exists():
        return sorted(REPO/path for path in read(existing)["identity"]["files"])
    paths = {REPO/"runners/run_v16_inquiry_stable.py", REPO/"runners/replay_v16_readers.py",
             REPO/"tests/test_v16_inquiry_stable.py", REPO/"tests/test_v16_recoding.py"}
    paths.update(PACKAGE/name for name in ["inquiry_stable.py", "inquiry_stable_study.py",
        "inquiry_stable_gates.py", "inquiry_stable_reference.py", "audit_inquiry_stable.py",
        "resource_accounting.py", "recoding.py"])
    for packet in (root/"packets").glob("*.json"):
        paths.update(REPO/path for path in read(packet)["identity"]["files"])
    return sorted(paths)


def execute(root, heartbeat, packet):
    output = root/PACKET
    gate_path = output/"GATES.json"
    gates = read(gate_path) if gate_path.exists() else run()
    write(gate_path, gates)
    if gates["instrument_state"] != "valid":
        raise ValueError("bounded repair admission failed; retain failure")
    total = sum(64*len(design["conditions"]) for design in DESIGNS.values())
    completed = 0
    with ReaderProcess(output/"public/reader-workspace", extensions=list(STABLE_OPERATIONS.values())) as reader:
        for card, design in DESIGNS.items():
            destination, rows = output/card, []
            for condition in design["conditions"]:
                for index in range(64):
                    if remaining_seconds(root) <= 0:
                        return {"execution_state": "checkpointed", "completed_units": completed,
                                "planned_units": total, "campaign_complete": False, "reason": "immutable ceiling"}
                    row = execute_unit(destination, card, condition, index,
                        namespace=f"v16-inquiry-discovery-2-{card}", packet=packet, reader=MeasuredReader(reader))
                    rows.append(row)
                    completed += 1
                    heartbeat(packet_id=PACKET, card_id=card, condition=condition["id"],
                        completed_units=completed, planned_units=total, last_unit=row["unit_id"], reader_pid=reader.identity["pid"])
            summary = summarize(card, rows)
            receipt = audit(destination, summary)
            summary["independent_reaggregation"] = "valid"
            write(destination/"REAGGREGATION.json", receipt)
            write(destination/"SUMMARY.json", summary)
            manifest = {str(path.relative_to(destination)).replace("\\", "/"): file_digest(path)
                        for directory in ("public", "private", "predictions", "units")
                        for path in sorted((destination/directory).glob("*.json"))}
            write(destination/"RAW_MANIFEST.json", {"files": manifest, "archive_location": "this packet directory",
                "retention": "through final verified archival handoff"})
            write(destination/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid",
                "criterion_state": "see typed per-condition contrasts", "evidence_scope": "amended discovery",
                "n_maker_packets": len(rows), "warrant": "DESCRIPTIVE ONLY", "pursuit": "OPENED",
                "dependencies": {name: "valid" for name in design["dependencies"]},
                "adversaries": {name: "pending for amended reader" for name in design["adversaries"]},
                "expansion_state": "pending", "confirmation_state": "untested", "repair_id": "inquiry-ties-1"})
    return {"execution_state": "completed", "packet_id": PACKET, "n_maker_packets": completed,
            "cards": list(DESIGNS), "independent_reaggregation": "valid", "campaign_complete": False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--admit-from-junit", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not (args.root/"packets"/f"{PACKET}.json").exists():
        raise ValueError("resume never prepares the repair packet")
    files = sources(args.root)
    admission_path = args.root/"inquiry-stable-setup/ADMISSION.json"
    if args.admit_from_junit:
        tests = {}
        for case in ET.parse(args.admit_from_junit).findall(".//testcase"):
            name = case.attrib["name"]
            if name in tests:
                raise ValueError("duplicate admission test identity")
            tests[name] = "failed" if any(case.find(tag) is not None for tag in ("error", "failure", "skipped")) else "passed"
        if not tests or any(state != "passed" for state in tests.values()) or any(name not in tests for name in REQUIRED):
            raise ValueError("amended inquiry admission tests are incomplete")
        write(admission_path, {"tests": tests, "recorded_at": now(), "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {str(path.relative_to(REPO)).replace("\\", "/"): file_digest(path) for path in files}})
        print({"passed_tests": len(tests), "bound_source_files": len(files)})
        return
    admission = read(admission_path)
    if any(admission["tests"].get(name) != "passed" for name in REQUIRED):
        raise ValueError("missing amended inquiry admission")
    for path in files:
        if admission["source_hashes"].get(str(path.relative_to(REPO)).replace("\\", "/")) != file_digest(path):
            raise ValueError("amended inquiry source changed after tests")
    with supervisor(args.root, "amended-inquiry-discovery") as heartbeat:
        packet = freeze(args.root, PACKET, files, {"cards": DESIGNS, "n_per_condition": 64, "constructors": 8,
            "seed_namespace": "v16-inquiry-discovery-2", "scope": "amended discovery", "repair_id": "inquiry-ties-1",
            "original_packet": "inquiry-scout-1", "numerical_decision_resolution": 2.0**-46})
        report = execute(args.root, heartbeat, packet)
        heartbeat(execution_state=report["execution_state"], result=report)
    print(report)


if __name__ == "__main__":
    main()
