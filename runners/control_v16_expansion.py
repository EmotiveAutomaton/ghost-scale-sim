"""Run finite actual-source controls under the same campaign ownership guard."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO, supervisor
from ghostscale.validation.soundingline.v16.packet_control_runner import execute


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-name", default="constructor-expansion-1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with supervisor(args.root, "discovery") as heartbeat:
        result = execute(args.root, heartbeat, resume=args.resume, source_root=args.source_root, source_name=args.source_name)
        heartbeat(execution_state=result["execution_state"], result={key:value for key,value in result.items() if key not in {"cards", "runtime"}})
    print({key:value for key,value in result.items() if key not in {"cards", "runtime"}})


if __name__ == "__main__":
    main()
