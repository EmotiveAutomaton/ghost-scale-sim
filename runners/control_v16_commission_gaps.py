"""Execute the finite missing commissioned controls without writing supervisor status."""
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.commission_controls import execute


if __name__ == "__main__":
    result = execute(REPO/"results/v16")
    print({"commission_dependencies_verified": result["commission_dependencies_verified"],
        "condition_controls": result["condition_controls"]}, flush=True)
