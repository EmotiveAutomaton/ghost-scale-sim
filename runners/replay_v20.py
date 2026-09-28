"""Bounded full packet replay from an already extracted frozen source."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v20.studies import run_study
from ghostscale.validation.soundingline.v20.runtime import scientific_files

def replay(original,destination,pulse=lambda **kw:None):
    started=time.process_time();plan=read(original/'PLAN.json');destination.mkdir(parents=True,exist_ok=False)
    # Admission metadata is part of scientific_files for native packets. Rebuild
    # it from the bound plan; engineering fixtures may have no CARD.json.
    if (original/'CARD.json').exists():
        write(destination/'CARD.json',dict(id=original.name,design=plan['design'],
            plan_sha256=file_digest(original/'PLAN.json'),source_archive_sha256=plan['source_archive_sha256'],
            environment=plan['environment']))
    summary=run_study(destination,plan['design'],pulse)
    if summary!=read(original/'SUMMARY.json'):raise ValueError('replayed summary differs')
    expected={n:h for n,h in read(original/'COMPLETE.json')['files'].items() if n not in ('SUMMARY.json','EVIDENCE_ROLES.json')}
    actual=scientific_files(destination)
    if actual!=expected:raise ValueError('complete replay output identity differs')
    return dict(passed=True,files=len(actual),plan_sha256=file_digest(original/'PLAN.json'),cpu_seconds=time.process_time()-started)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();print(replay(a.original,a.destination))
