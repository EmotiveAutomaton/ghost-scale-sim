"""Complete small engineering consumers and replays, never scientific populations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v18_3.io import write,read,digest
from ghostscale.validation.soundingline.v21.runtime import scientific_files
from ghostscale.validation.soundingline.v21.studies import headroom,run_study

def scientific_value(value):
    if isinstance(value,dict):return {k:scientific_value(v) for k,v in value.items() if k not in ('fit_cpu_seconds','cpu_seconds','wall_seconds','headroom_sha256')}
    if isinstance(value,list):return [scientific_value(v) for v in value]
    return value

def run(base,soundingline):
    started=time.process_time();receipts=[]
    designs=dict(handler='headroom',card='G21-A01',tier='context',history=4,history_tier='complete',history_observed=True,
        shift='changed-tool',history_shift='native',training_lineages=list(range(71000500,71000508)),validation_lineages=list(range(71000600,71000608)),
        training_cases=512,validation_cases=512,lineages=[71000700,71000701],test_cases=32,law_count=4,fit_seeds=[0])
    for repeat in (0,1):
        start=time.process_time();root=base/str(repeat);source=root/'headroom';s=headroom(source,designs,lambda **kw:None)
        actual_gate=s['screen_admitted'];s['screen_admitted']=True
        # Isolated fixture: bypass admission ONLY to exercise complete consumers.
        write(source/'SUMMARY.json',s);write(source/'PLAN.json',dict(design=designs));write(source/'COMPLETE.json',dict(files=scientific_files(source)))
        results={'headroom':digest(scientific_value(s))}
        cases=[('learned',dict(designs,handler='learned',card='G21-B01',headroom_root=str(source))),
               ('adaptation',dict(designs,handler='adaptation',card='G21-B07',headroom_root=str(source))),
               ('factorization',dict(designs,handler='factorization',forecast_root=str(source))),
               ('support',dict(designs,handler='support',forecast_root=str(source))),
               ('records',dict(handler='records',card='G21-D02',lineages=[71000800],test_cases=8,rates=[0,.5])),
               ('aliases',dict(handler='aliases',card='G21-A02')),
               ('optimization',dict(designs,handler='optimization',budgets=[128,512])),
               ('transport',dict(handler='transport',soundingline=str(soundingline)))]
        for name,d in cases:
            target=root/name;target.mkdir(parents=True);result=run_study(target,d,lambda **kw:None);write(target/'SUMMARY.json',result)
            results[name]=digest(scientific_value(result));print(repeat,name,'complete',flush=True)
        arrays={p.relative_to(root).as_posix():__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in root.rglob('*_points.npz')}
        receipts.append(dict(results=results,arrays=arrays,cpu_seconds=time.process_time()-start,actual_headroom_admitted=actual_gate))
    if receipts[0]['results']!=receipts[1]['results'] or receipts[0]['arrays']!=receipts[1]['arrays']:raise ValueError('complete consumer replay differs')
    out=dict(passed=True,families=len(receipts[0]['results']),array_files=len(receipts[0]['arrays']),cpu_seconds=time.process_time()-started,
        scientific_hashes=receipts[0]['results'],scope='engineering fixtures; forced headroom gate only here, no scientific admission; all complete consumers replayed; timing fields excluded')
    write(base/'PREFLIGHT.json',out);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--soundingline',type=Path,required=True);a=p.parse_args();run(a.out,a.soundingline)
