"""Thin, module-form entry point for the source-bound V17 continuation."""
import argparse
import json
from pathlib import Path
from ghostscale.validation.soundingline.v17.continuation_runtime import run
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--plan",required=True,type=Path)
    parser.add_argument("--admission",required=True,type=Path)
    parser.add_argument("--fixture-units",type=int)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_bytes());admission=json.loads(args.admission.read_bytes())
    if args.fixture_units is not None and plan.get("mode")!="discarded_development":
        raise ValueError("fixture stop is not a scientific run mode")
    raise SystemExit(run(args.root,plan,admission,fixture_units=args.fixture_units))
if __name__=="__main__":main()
