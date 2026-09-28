"""Numerical review entry point; never labels execution as scientific acceptance."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import read,write,now
from ghostscale.validation.soundingline.v20.analysis import review
from ghostscale.validation.soundingline.v18_4.priority import below_normal

def main(campaign,out):
    below_normal();start=time.process_time();queue=read(campaign/'QUEUE.json');roots=[]
    for j in queue['jobs']:
        root=Path(j['root'])
        if (root/'COMPLETE.json').exists():roots.append(root)
    if not roots:raise ValueError('no completed packets')
    result=review(roots);write(out,result)
    write(campaign/'attempts'/('review-'+out.stem+'.json'),dict(state='completed',packet=out.stem,accounting_card='numerical-review',cpu_seconds=time.process_time()-start,at=now()))
    print('Verified',len(roots),'complete packets; interpretation, replay and write-through remain explicit')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.campaign,a.out)
