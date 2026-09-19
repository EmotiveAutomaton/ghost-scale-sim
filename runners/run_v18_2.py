"""One source-frozen V18.2 worker; inherited environment, no dependency installs."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[variable]='1'
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v18_2.runtime import run

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    run(args.root,args.campaign)
