"""Generate the final numerical index only after discovery and confirmation."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.study_products import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print({key: result[key] for key in ["execution_state", "native_cards", "registered_conditions", "registered_comparisons", "frozen_confirmation_claims"]}, flush=True)


if __name__ == "__main__":
    main()
