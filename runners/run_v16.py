"""V16 finite campaign supervisor. Launch only as python -m runners.run_v16."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.runtime import REPO, PACKAGE, supervisor, freeze, campaign
from ghostscale.validation.soundingline.v16.gates import run_gates
from ghostscale.validation.soundingline.v16.vertical import run_case
from ghostscale.validation.soundingline.v16.reaggregate import regenerate


def pilot(root, heartbeat, units):
    packet = freeze(root, "native-fixture-1",
                    [PACKAGE / name for name in
                     ["world.py", "reference.py", "learning.py", "inference.py", "gates.py",
                      "vertical.py", "records.py", "reaggregate.py"]],
                    {"scope": "fixture", "units": units, "max_steps": 3, "canvas": [2, 2],
                     "tolerance": 1e-10, "seed_namespace": "v16-native-fixture"})
    output = root / "native-fixture-1"
    gate_path = output / "GATES.json"
    gates = read(gate_path) if gate_path.exists() else run_gates()
    write(gate_path, gates)
    if gates["instrument_state"] != "valid":
        raise ValueError("native admission failed")
    for index in range(units):
        record = run_case(output, packet_hash=packet["packet_hash"], index=index)
        heartbeat(completed_units=index + 1, last_unit=record["unit_id"])
    aggregate = regenerate(output)
    write(output / "AGGREGATE.json", aggregate)
    manifest = {str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
                for directory in ["public", "private", "predictions", "units"]
                for path in sorted((output / directory).glob("*.json"))}
    write(output / "RAW_MANIFEST.json", {"files": manifest, "retained": True,
                                        "location": "relative to this packet directory",
                                        "retention": "through campaign closeout and verification"})
    write(output / "CAPABILITY.json", {
        "evidence_scope": "fixture", "execution_state": "completed",
        "generator": "valid", "serialized_access": "valid", "exact_inference": "valid",
        "legal_construction": "valid", "successful_construction": "valid",
        "prediction_submission": "valid", "independent_reaggregation": "valid",
        "discovery": "untested", "historical_uniqueness": "not_identifiable",
        "completed_at": read(output / "units" / (record["unit_id"] + "_points.json"))["scored_at"]})
    return {"evidence_scope": "fixture", "aggregate": aggregate, "packet_hash": packet["packet_hash"]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["preflight", "pilot", "discovery", "transfer",
                                 "confirmation", "close", "resume"])
    parser.add_argument("--root", type=Path, default=REPO / "results/v16")
    parser.add_argument("--fixture-units", type=int, default=2)
    parser.add_argument("--packet", choices=["acquisition","reading","behavior","inquiry","options","purpose","attention","mechanism","dependency"], default="acquisition")
    args = parser.parse_args()
    if args.fixture_units < 1:
        parser.error("fixture units must be positive")
    accepted = campaign(args.root)
    with supervisor(args.root, args.stage) as heartbeat:
        if args.stage == "preflight":
            report = {"commission": accepted["commission_sha256"], "code_root": str(REPO),
                      "execution_state": "completed", "capability": "acceptance identity checked"}
        elif args.packet == "dependency" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.dependency_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "mechanism" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.mechanism_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "attention" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.attention_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "purpose" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.purpose_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "options" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.options_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "inquiry" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.inquiry_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "behavior" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.behavior_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.packet == "reading" and args.stage in {"discovery","resume"}:
            from ghostscale.validation.soundingline.v16.reading_runner import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.stage == "discovery" or (
                args.stage == "resume" and (args.root / "packets/k01-scout-1.json").exists()):
            from ghostscale.validation.soundingline.v16.acquisition_checkpoint import execute
            report = execute(args.root, heartbeat, resume=args.stage == "resume")
        elif args.stage in {"pilot", "resume"}:
            if args.stage == "resume" and not (args.root / "packets/native-fixture-1.json").exists():
                raise ValueError("resume requires an existing packet; it never prepares")
            report = pilot(args.root, heartbeat, args.fixture_units)
        else:
            raise ValueError(f"{args.stage} instrument has not yet been admitted")
        heartbeat(execution_state=report.get("execution_state", "completed"), result=report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
