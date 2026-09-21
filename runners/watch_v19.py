"""V19 adapter to the native queue/owner supervisor; no new orchestration framework."""
import argparse
from pathlib import Path
from runners.watch_v18_4 import supervise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True);parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args();supervise(args.campaign,args.queue,args.python,'runners.run_v19')
