"""One source-frozen V21 packet; always launch in module form."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v21.runtime import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();run(a.root,a.campaign)
