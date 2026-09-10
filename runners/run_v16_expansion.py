"""Admit, profile and execute the finite constructor expansion packet."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO, supervisor
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.expansion_runner import execute, sources, specification

REQUIRED = {"test_expansion_plan_is_finite_balanced_and_explicit", "test_expansion_actual_interruption_and_resume",
            "test_expansion_resume_refuses_missing_packet", "test_all_expansion_families_keep_physics_scores_and_guarded_reader_outputs",
            "test_public_dispatch_uses_copied_bindings_and_rejects_hidden_fields"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", choices=["profile", "K01", "R02"])
    parser.add_argument("--admit-from-junit", type=Path)
    parser.add_argument("--forecast-from", type=Path)
    args = parser.parse_args()
    if args.admit_from_junit:
        tests = {case.attrib["name"]: "failed" if any(case.find(tag) is not None for tag in ["failure", "error", "skipped"]) else "passed"
                 for case in ET.parse(args.admit_from_junit).findall(".//testcase")}
        names = {name.split("[")[0] for name in tests}
        if not REQUIRED <= names or any(value != "passed" for value in tests.values()) or len(tests) < 35:
            raise ValueError("expansion admission tests missing, failed or skipped")
        write(args.root/"expansion-setup/ADMISSION.json", {"instrument_state": "valid", "tests": tests,
            "recorded_at": now(), "junit_sha256": file_digest(args.admit_from_junit),
            "source_hashes": {path.relative_to(REPO).as_posix(): file_digest(path) for path in sources(args.root)}})
        print({"admission_tests": len(tests)})
        return
    if args.forecast_from:
        spec = specification()
        records = []
        for entry in spec["cards"]:
            card = entry["card_id"]
            base = args.forecast_from/"expansion-fixture-profile"/card
            completion, timing = read(base/"COMPLETION.json"), read(base/"TIMING.json")
            if completion["instrument_state"] != "valid":
                raise ValueError("profile did not pass independent unit/summary checks")
            # Linear scaling of observed complete-card wall includes audits and bootstrap work;
            # the separately stated factor protects against load and file-count growth.
            estimate = timing["wall_seconds"]*entry["n_per_condition"]
            records.append({"card_id": card, "measured_profile_seconds": timing["wall_seconds"],
                "profile_condition_records": completion["n_condition_records"], "scaled_seconds": estimate,
                "timing_sha256": file_digest(base/"TIMING.json"), "profile_packet_hash": completion["packet_hash"]})
        estimate = sum(row["scaled_seconds"] for row in records)
        write(args.root/"expansion-setup/FORECAST.json", {"instrument_state": "valid", "recorded_at": now(), "cards": records,
            "estimated_seconds": estimate, "conservative_seconds": estimate*2, "conservative_factor": 2,
            "worker_cap": 1, "forecast_is_not_measurement": True, "planned_condition_records": spec["planned_maker_condition_records"],
            "closeout_reserve_seconds": 21600, "scope": "all original conditions profiled with one fresh fixture maker each; linear scaling plus explicit safety factor"})
        print({"estimated_hours": estimate/3600, "conservative_hours": estimate*2/3600})
        return
    with supervisor(args.root, "discovery") as heartbeat:
        result = execute(args.root, heartbeat, resume=args.resume, fixture=args.fixture)
        heartbeat(execution_state=result["execution_state"], result=result)
    print({key: value for key, value in result.items() if key != "cards"})


if __name__ == "__main__":
    main()
