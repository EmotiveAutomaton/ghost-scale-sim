"""Separate analysis process for retained control and resource arithmetic."""
import argparse
from pathlib import Path
import os
import time
from ghostscale.validation.soundingline.v16.records import write, now, file_digest
from ghostscale.validation.soundingline.v16.aggregate_controls import run
REPO=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    began,cpu=time.perf_counter(),time.process_time()
    def report(label,value):
        write(args.output/"items"/(label+".json"),value)
        print({"recounted":label},flush=True)
    try:
        results=run(args.root,report)
    except Exception as error:
        write(args.output/"FAILURE.json",{"execution_state":"failed","instrument_state":"unresolved",
              "recorded_at":now(),"error":repr(error),"full_aggregate_regeneration":False})
        raise
    write(args.output/"RECEIPT.json",{"execution_state":"completed","instrument_state":"valid",
        "recorded_at":now(),"process_id":os.getpid(),"results":results,
        "scope":"Control fixture counts, hierarchical known-answer interval and saved measurement arithmetic",
        "full_aggregate_regeneration":False,"scientific_replay":False,
        "wall_seconds":time.perf_counter()-began,"cpu_seconds":time.process_time()-cpu,
        "sources":{p.relative_to(REPO).as_posix():file_digest(p) for p in
                   [Path(__file__).resolve(),REPO/"ghostscale/validation/soundingline/v16/aggregate_controls.py"]}})

if __name__=="__main__":
    main()

