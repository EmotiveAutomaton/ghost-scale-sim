"""Verify final documentary write-through against retained scientific evidence."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.documentary_closeout import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    result = run(REPO, args.root, args.output, args.inputs)
    print({"write_through_verified": result["write_through_verified"], "documents": len(result["documents"])}, flush=True)


if __name__ == "__main__":
    main()
