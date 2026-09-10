"""Owned, bounded interruption fixture using the actual native unit/checkpoint path."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import read, write, file_digest
from ghostscale.validation.soundingline.v16.runtime import supervisor, freeze, REPO, PACKAGE
from ghostscale.validation.soundingline.v16.vertical import run_case
from ghostscale.validation.soundingline.v16.reaggregate import regenerate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    name = "interruption-native-fixture"
    if args.resume and not (args.root/"packets"/(name+".json")).exists():
        raise ValueError("resume must not prepare")
    with supervisor(args.root, "pilot") as heartbeat:
        packet = freeze(args.root, name, [REPO/"runners/v16_interrupt_fixture.py"]+
            [PACKAGE/filename for filename in ["world.py", "reference.py", "learning.py", "inference.py", "gates.py", "vertical.py", "records.py", "reaggregate.py", "runtime.py", "campaign_ownership.py"]],
            {"scope": "fixture", "units": 64, "seed_namespace": "v16-interruption-fixture"})
        output = args.root/name
        for index in range(64):
            row = run_case(output, packet_hash=packet["packet_hash"], index=index)
            heartbeat(completed_units=index+1, planned_units=64, last_unit=row["unit_id"])
        aggregate = regenerate(output)
        write(output/"AGGREGATE.json", aggregate)
        files = {str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
                 for folder in ["public", "private", "predictions", "units"] for path in sorted((output/folder).glob("*.json"))}
        write(output/"RAW_MANIFEST.json", {"files": files})
        heartbeat(execution_state="completed", campaign_complete=False, evidence_scope="fixture")


if __name__ == "__main__":
    main()
