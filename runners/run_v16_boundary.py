"""Execute the once-selected final finite discovery allocation."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO,supervisor
from ghostscale.validation.soundingline.v16.boundary_runner import execute


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--fixture",choices=["K01","R02"])
    args=parser.parse_args()
    with supervisor(args.root,"discovery") as heartbeat:
        result=execute(args.root,heartbeat,resume=args.resume,fixture=args.fixture)
        heartbeat(execution_state=result["execution_state"],result=result)
    print({key:value for key,value in result.items() if key!="cards"})


if __name__=="__main__":
    main()
