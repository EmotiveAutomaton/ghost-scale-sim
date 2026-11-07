"""Finite serial excluded preflight with per-fixture durable receipts and CPU cap."""
from pathlib import Path
import argparse,sys,time,os
from datetime import datetime,timezone
source=Path(__file__).resolve().parents[1]
from ghostscale.validation.soundingline.v27 import setpoint_resolution as resolution,registry,setpoints,setpoint_particles
from ghostscale.validation.soundingline.v27.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--batch',required=True)
parser.add_argument('--counts',type=int,nargs=3,default=list(resolution.COUNTS));parser.add_argument('--reuse',type=Path)
parser.add_argument('--order',type=int,nargs='+')
parser.add_argument('--pool',choices=('preparation','followup_healing'),default='preparation')
parser.add_argument('--cpu-ceiling-seconds',type=int,default=21600)
args=parser.parse_args();c=args.campaign.resolve()
if not args.batch.replace('-','').isalnum():raise ValueError('invalid batch name')
root=c/args.batch;root.mkdir(exist_ok=True)
runtime.below_normal();start=time.process_time();opened=time.monotonic()
sources={n:digest(source/'ghostscale/validation/soundingline/v27'/n) for n in ('setpoints.py','setpoint_particles.py','setpoint_resolution.py')}
if args.counts!=sorted(set(args.counts)) or args.counts[0]<2048:raise ValueError('invalid increasing resolution ladder')
settings=registry.primary_settings('K3')[:15]
order=args.order if args.order is not None else list(range(len(settings)))
if sorted(order)!=list(range(len(settings))):raise ValueError('fixture order must contain every excluded setting exactly once')
if not 0<args.cpu_ceiling_seconds<=43200:raise ValueError('invalid CPU ceiling')
if args.reuse:
    reuse_protocol=read(args.reuse/'PROTOCOL.json')
    for name in ('setpoints.py','setpoint_particles.py'):
        if reuse_protocol['sources'][name]!=sources[name]:raise ValueError('retained likelihood source differs')
protocol=dict(sources=sources,counts=args.counts,limits=resolution.LIMITS,replicates=2,
    settings=settings,fixture_order=order,cpu_ceiling_seconds=args.cpu_ceiling_seconds,
    pool=args.pool,scientific_draws=0)
if args.reuse:protocol['retained_preflight_sha256']=digest(args.reuse/'COMPLETE.json')
write(root/'PROTOCOL.json',protocol)
def pulse(**kw):
    elapsed=time.process_time()-start
    write(root/'STATUS.json',dict(state='RUNNING',pid=os.getpid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=elapsed,**kw),immutable=False)
    kind='resolution_healing' if args.pool=='followup_healing' else 'resolution_preflight'
    write(c/'accounting'/(args.batch+'.json'),dict(pool=args.pool,kind=kind,state='RUNNING',cpu_seconds=elapsed,
        reserved_cpu_seconds=protocol['cpu_ceiling_seconds']),immutable=False)
    if elapsed>protocol['cpu_ceiling_seconds']:raise TimeoutError('Excluded resolution preparation cap reached')
try:
    pulse(phase='exact-controls')
    controls=setpoints.controls();checks=[]
    for k,point in enumerate(((8,8),(11,14),(14,8))):
        s=protocol['settings'][0]|dict(records=12,inference='checked_particles')
        raw=setpoints.generate(s,['excluded-static-resolution',k]);observed=resolution.hidden_observations(raw)
        exact,_=setpoints.likelihood(observed,s|dict(inference='exact'),point,'P0',0,0.)
        approx,_=setpoint_particles.likelihood(observed,s,point,'P0',0,0.,particles=32768)
        reference,_=setpoint_particles.likelihood(observed,s,point,'P0',0,0.,True,particles=32768)
        error=float(np.max(abs(exact-approx)));independent=float(np.max(abs(approx-reference)))
        checks.append(dict(point=point,exact_error=error,independent_error=independent,passed=error<=.1 and independent<=1e-9))
    write(root/'CONTROLS.json',dict(exact=controls,checks=checks,passed=all(x['passed'] for x in checks)))
    if not all(x['passed'] for x in checks):raise ArithmeticError('excluded static resolution controls failed')
    receipts=[]
    for index in protocol['fixture_order']:
        s=protocol['settings'][index]
        pulse(setting=index,phase='dynamic-resolution')
        retained_path=args.reuse/f'fixture-{index:02d}.json' if args.reuse else None
        retained=read(retained_path) if retained_path and retained_path.exists() else None
        if retained is not None:
            bound=next(row['sha256'] for row in read(args.reuse/'COMPLETE.json')['fixtures'] if row['setting']==index)
            if digest(retained_path)!=bound:raise ValueError('retained preflight changed')
        value=resolution.fixture(s,['excluded-dynamic-resolution-20261107',index],lambda **kw:pulse(setting=index,**kw),counts=tuple(args.counts),retained=retained)
        write(root/f'fixture-{index:02d}.json',value);receipts.append(dict(setting=index,passed=value['passed'],sha256=digest(root/f'fixture-{index:02d}.json')))
        # A failing grid is retained; stop before spending hours on the same
        # inadequate numerical resolution. The next operator diagnoses it.
        if not value['passed']:break
    write(root/'COMPLETE.json',dict(fixtures=receipts,passed=len(receipts)==len(protocol['settings']) and all(x['passed'] for x in receipts),
        source_hashes=sources,scientific_admission=False,reason='Resolution diagnostics only; scientific protocols and measured probes still required.'))
    state='COMPLETE'
except Exception as exc:
    state='FAILED';write(root/'FAILURE.json',dict(error=str(exc),error_type=type(exc).__name__))
finally:
    write(root/'STATUS.json',dict(state=state,pid=os.getpid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-opened),immutable=False)
    kind='resolution_healing' if args.pool=='followup_healing' else 'resolution_preflight'
    write(c/'accounting'/(args.batch+'.json'),dict(pool=args.pool,kind=kind,state='STOPPED',cpu_seconds=time.process_time()-start),immutable=False)
print(state,flush=True)
