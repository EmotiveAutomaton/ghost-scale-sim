"""Build the bounded V16 case index from completed sources without live status."""
import argparse
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.catalogue_runner_v2 import execute
from ghostscale.validation.soundingline.v16.records import write, now

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(REPO/"results/v16", resume=args.resume)
    except Exception as error:
        directory = REPO/"results/v16/case-catalogue-2/failures"
        number = len(list(directory.glob("attempt-*.json")))+1
        write(directory/f"attempt-{number}.json", {"execution_state": "failed", "instrument_state": "unresolved",
            "error": repr(error), "recorded_at": now(), "full_campaign_closeout": False})
        raise
    print({"case_count": result["case_count"], "instrument_state": result["instrument_state"]}, flush=True)

if __name__ == "__main__":
    main()
