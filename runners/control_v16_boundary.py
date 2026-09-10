"""Run actual-source controls for the last finite expansion."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO,supervisor
from ghostscale.validation.soundingline.v16.boundary_control_runner import execute


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--fixture-source",type=Path)
    parser.add_argument("--fixture-name")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    with supervisor(args.root,"discovery") as heartbeat:
        result=execute(args.root,heartbeat,resume=args.resume,fixture_source=args.fixture_source,fixture_name=args.fixture_name)
        heartbeat(execution_state=result["execution_state"],result={key:value for key,value in result.items() if key!="cards"})
    print({key:value for key,value in result.items() if key!="cards"})


if __name__=="__main__":
    main()
