"""Prepare final file-bound card accounting for the ordinary campaign supervisor."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read
from ghostscale.validation.soundingline.v16.closeout_accounting import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    inputs = read(args.inputs)
    result = run(args.root, args.output, inputs["study_projection_directory"], inputs["proofs"])
    print({"card_count": result["card_count"], "campaign_closed": result["campaign_closed"]}, flush=True)


if __name__ == "__main__":
    main()
