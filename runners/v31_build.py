"""Prospective source-bound admission helpers; never alters an existing job."""
import time
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest,run_cell
from runners.run_v31 import equivalent
from runners.run_v27 import frozen_source

PINS=['runners/run_v31.py','runners/run_v29.py','runners/run_k3_redesign.py','runners/watch_v31.py','runners/v31_build.py']

def admit(repo,c,family,pool,module,settings,protocol,order,probe=None,extra=None,attempt=None):
    gatefile=c/'families'/family/'ADMISSION.json'
    if gatefile.exists():return read(gatefile)
    started=time.process_time();out=c/'preparation'/family
    if attempt:out=out/attempt
    out.mkdir(parents=True,exist_ok=True)
    controls=module.controls(out/'controls');assert controls['passed'],controls
    probe=dict(probe or settings[-1],worlds=8,blocks=8);t=time.process_time()
    run_cell(module,probe,out/'original',lane='excluded-cost-profile')
    run_cell(module,probe,out/'replay',lane='excluded-cost-profile');equivalent(out/'original',out/'replay',True)
    run_cell(module,probe,out/'reconstruct',mode='reconstruct',original=out/'original',lane='excluded-cost-profile');equivalent(out/'original',out/'reconstruct')
    used=time.process_time()-t;cap=max(120.,used/8*240*1.8)
    proof=dict(passed=True,controls=controls,full_seed_replay=True,independent_reconstruction=True,profile_worlds=8,profile_cpu_seconds=used,cpu_cap_seconds=cap)
    write(out/'VALIDATION.json',proof)
    source=frozen_source(c,PINS);gate=dict(passed=True,validation_sha256=digest(out/'VALIDATION.json'),protocol_sha256=digest(repo/protocol),source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),extra=extra)
    write(gatefile,gate);write(c/(family+'_ROSTER.json'),settings)
    for i,s in enumerate(settings):
        write(c/'jobs'/f'{family}-{i:04d}'/'PLAN.json',dict(family=family,pool=pool,module=module.__name__,setting=s,order=order+i,cpu_cap_seconds=cap,wall_cap_seconds=max(1800,cap*4),source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),acceptance_sha256=digest(c/'ACCEPTANCE.json'),admission_sha256=digest(gatefile),interpreter=str(repo/'.venv/Scripts/python.exe')))
    write(c/'accounting'/('prepare-'+family+'.json'),dict(pool=pool,cpu_seconds=time.process_time()-started,reservation=0,state='COMPLETE'))
    print(family,'admitted',len(settings),'profile',used,'cap',cap,flush=True)
    return gate
