"""One admitted V20 CPU packet; module-form launch preserves native ownership."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v20.runtime import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();run(a.root,a.campaign)
