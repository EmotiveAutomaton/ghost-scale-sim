"""Check every retained score and completion hash; never rewrite science."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,time,uuid
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now
from ghostscale.validation.soundingline.v21.metrics import check_scores

def verify(root):
    complete=read(root/'COMPLETE.json')
    if file_digest(root/'PLAN.json')!=complete['plan_sha256']:raise ValueError('changed plan')
    rows=0;blocks=0
    for name,h in complete['files'].items():
        if file_digest(root/name)!=h:raise ValueError('changed scientific file '+name)
        if name.startswith('raw/') and name.endswith('_points.npz'):
            with np.load(root/name) as a:
                if 'logp' in a and 'truth' in a:
                    # NpzFile is lazy: each a[key] decompresses the complete array.
                    # Cache once before scalar regrouping; score semantics unchanged.
                    arrays={key:a[key] for key in a.files}
                    check_scores(arrays['logp'],arrays['truth'],arrays);rows+=len(arrays['truth']);blocks+=1
    return dict(passed=True,packet=root.name,complete_sha256=file_digest(root/'COMPLETE.json'),
        files=len(complete['files']),forecast_blocks=blocks,forecast_rows=rows,
        scope='completion/input hashes and all saved probability scores; scientific criteria, full replay and documentary write-through are separate')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cpu=time.process_time();start=time.monotonic();identity=uuid.uuid4().hex;state='failed'
    try:write(a.out,verify(a.root));state='complete'
    finally:write(a.campaign/'attempts'/f'review-{identity}.json',dict(job=a.root.name,role='independent_review',state=state,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,at=now()))
