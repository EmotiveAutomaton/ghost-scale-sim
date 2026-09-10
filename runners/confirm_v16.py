"""Execute frozen V16 confirmations through the single resilient supervisor."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.runtime_status_retry import supervisor
from ghostscale.validation.soundingline.v16.confirmation_runner import execute


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    with supervisor(args.root, "confirmation") as heartbeat:
        result = execute(args.root, heartbeat, resume=args.resume, fixture=args.fixture)
        heartbeat(execution_state=result["execution_state"], result=result)
    print({"completed_claims": len(result["claims"]), "execution_state": result["execution_state"]}, flush=True)


if __name__ == "__main__":
    main()
