"""Independent power arithmetic for the actual frozen confirmation selection."""
import argparse
from pathlib import Path
import os
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.confirmation_power_audit import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = source_locks(args.root, REPO)
    try:
        result = run(args.root)
    except Exception as error:
        write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise
    write(args.output/"RECEIPT.json", {"execution_state": "completed", **result, "recorded_at": now(),
        "process_id": os.getpid(), "source_checks": checks, "full_aggregate_regeneration": False,
        "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in
            [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/confirmation_power_audit.py"]}})
    print({"execution_state": "completed", "power_calculations": result["claim_count"]})


if __name__ == "__main__":
    main()
