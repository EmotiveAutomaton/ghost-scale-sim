"""One owner-authorized K3 repair, split into finite excluded fixture jobs."""
import argparse, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from ghostscale.validation.soundingline.v27.common import read, write, digest
from ghostscale.validation.soundingline.v27.contract import DRAIN, NEW_STARTS_CLOSE, POOLS
from ghostscale.validation.soundingline.v23.runtime import below_normal
from runners.watch_v23_events import event
from runners.queue_v27 import charges


def run(c, name):
    root=c/name;plan=read(root/'PLAN.json');start=time.process_time();began=time.monotonic()
    if plan['owner_authorization']!='owner-20261109-one-extra-per-family':raise ValueError('missing owner authority')
    if plan['fixture_order']!=[5,1,0,2,3,4,6,7,8,9,10,11,12,13,14]:raise ValueError('fixture order changed')
    if plan['counts']!=[131072,262144,524288] or plan['total_cpu_ceiling_seconds']!=129600:raise ValueError('resolution budget changed')
    source=Path(plan['source_root'])
    if digest(source/'SOURCE_MANIFEST.json')!=plan['source_manifest_sha256']:raise ValueError('source changed')
    if (root/'STATUS.json').exists():raise ValueError('no implicit repair restart')
    below_normal();used=0.;receipts=[];child=None;state='FAILED';reason=None
    def pulse(**kw):
        write(root/'STATUS.json',dict(state='RUNNING',pid=os.getpid(),heartbeat=datetime.now(timezone.utc).isoformat(),
            completed_fixtures=len(receipts),cpu_seconds=used+time.process_time()-start,**kw),immutable=False)
    try:
        for index in plan['fixture_order']:
            if datetime.now(timezone.utc)>=datetime.fromisoformat(NEW_STARTS_CLOSE):raise TimeoutError('new starts closed')
            remaining=plan['total_cpu_ceiling_seconds']-used-(time.process_time()-start)
            if remaining<600:raise TimeoutError('logical repair CPU ceiling')
            cap=int(min(43200,remaining));batch=f'{name}-f{index:02d}'
            if charges(c).get('followup_healing',0)+cap>POOLS['followup_healing']:raise TimeoutError('shared pool ceiling')
            pulse(fixture=index,phase='launching')
            argv=[sys.executable,'-B','-m','runners.check_v27_resolution','--campaign',str(c),'--batch',batch,
                '--counts',*[str(n) for n in plan['counts']],'--reuse',str(c/plan['reuse']),
                '--fixture-indices',str(index),'--pool','followup_healing','--cpu-ceiling-seconds',str(cap)]
            with (root/f'fixture-{index:02d}.log').open('xb') as log:
                child=subprocess.Popen(argv,cwd=source,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                opened=time.monotonic()
                while child.poll() is None:
                    pulse(fixture=index,phase='excluded-resolution',child_pid=child.pid)
                    if time.monotonic()-opened>18*3600 or datetime.now(timezone.utc)>=datetime.fromisoformat(DRAIN) or (c/'STOP').exists():
                        child.terminate();child.wait(timeout=30);raise TimeoutError('fixture wall or immutable drain boundary')
                    time.sleep(10)
            status=read(c/batch/'STATUS.json');used+=status['cpu_seconds']
            result=read(c/batch/'COMPLETE.json') if (c/batch/'COMPLETE.json').exists() else {}
            receipts.append(dict(fixture=index,batch=batch,cpu_seconds=status['cpu_seconds'],
                complete_sha256=digest(c/batch/'COMPLETE.json') if result else None,passed=result.get('passed',False)))
            if child.returncode or not result.get('passed'):raise ArithmeticError('excluded fixture failed; no scientific admission')
            child=None
        state='COMPLETE';reason='All 15 excluded settings passed unchanged numerical bars; scientific cost probes still required.'
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        write(root/'COMPLETE.json',dict(state=state,passed=state=='COMPLETE',fixtures=receipts,reason=reason,
            scientific_draws=0,scientific_admission=False,source_manifest_sha256=plan['source_manifest_sha256']))
        write(root/'STATUS.json',dict(state=state,pid=os.getpid(),heartbeat=datetime.now(timezone.utc).isoformat(),
            cpu_seconds=used+time.process_time()-start,wall_seconds=time.monotonic()-began,reason=reason),immutable=False)
        write(c/'accounting'/(name+'-controller.json'),dict(pool='followup_healing',kind='healing',family='K3',state='STOPPED',
            cpu_seconds=time.process_time()-start,child_costs_recorded_separately=True),immutable=False)
        event(c,name+'-terminal','build_required' if state=='COMPLETE' else 'packet_failure',
            dict(batch=name,complete_sha256=digest(root/'COMPLETE.json'),next_action='Inspect repair gate before any dynamic scientific admission.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--batch',required=True)
    a=p.parse_args();run(a.campaign.resolve(),a.batch)
