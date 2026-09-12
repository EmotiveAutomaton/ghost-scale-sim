"""Module-form V17 queue/worker entry; resident Python environment stays unchanged."""
import argparse
import json
import os
from pathlib import Path
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
from ghostscale.validation.soundingline.v16.records import read
from ghostscale.validation.soundingline.v17.queue_runtime import packet,run_queue,watch_queue

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("packet","queue","watch"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--design",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--admission",type=Path)
    parser.add_argument("--stop-after-chunks",type=int)
    args=parser.parse_args()
    if args.command=="packet":
        if args.design is None: parser.error("packet requires --design")
        print(json.dumps(packet(args.root,read(args.design),stop_after_chunks=args.stop_after_chunks,admission=read(args.admission) if args.admission else None)))
        return 0
    if args.manifest is None: parser.error("queue requires --manifest")
    return watch_queue(args.root,args.manifest) if args.command=="watch" else run_queue(args.root,read(args.manifest))

if __name__=="__main__":
    raise SystemExit(main())
