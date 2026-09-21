"""Event-only independent Codex reviews with explicit evidence acknowledgements."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now
from ghostscale.validation.soundingline.v16.runtime import local_owner


def alive(pid):
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x100000,False,pid)
        if not handle:return False
        try:return kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def pending(campaign,state):
    events=[]
    for path in sorted((campaign/'events').glob('*.json')):
        e=read(path);ack=state/'acks'/(e['id']+'.json')
        if ack.exists():
            if read(ack).get('event_sha256')!=file_digest(path):raise ValueError('event acknowledgement identity differs')
        else:events.append(dict(e,event_sha256=file_digest(path)))
    return events


def checkpoint_events(campaign,acceptance,config,current=None):
    """One-shot declared milestones, including interim reporting for new campaigns."""
    current=current or datetime.now(timezone.utc)
    checkpoints=list(config.get('checkpoints',[]))
    if 'minimum_exploration_until' in acceptance:
        checkpoints.append(dict(id='minimum-window-reached',kind='minimum_exploration_window_reached',at=acceptance['minimum_exploration_until']))
    if 'interim_at' in acceptance:
        checkpoints.append(dict(id='interim-report-due',kind='interim_report_due',at=acceptance['interim_at']))
    if 'deadline' in acceptance:
        checkpoints.append(dict(id='final-report-due',kind='final_report_due',at=acceptance['deadline']))
    for checkpoint in checkpoints:
        target=campaign/'events'/(checkpoint['id']+'.json')
        if current>=datetime.fromisoformat(checkpoint['at']) and not target.exists():
            write(target,dict(id=checkpoint['id'],kind=checkpoint['kind'],at=now(),due_at=checkpoint['at']))


def supervise(campaign,state,config_path,once=False):
    config=read(config_path);state.mkdir(parents=True,exist_ok=True)
    path=state/'DELIVERY.json';delivery=read(path) if path.exists() else dict(attempts=[])
    with local_owner(state):
        if delivery['attempts'] and 'exit_code' not in delivery['attempts'][-1]:
            last=delivery['attempts'][-1]
            if last.get('pid') and alive(last['pid']):raise RuntimeError('preserve live review owner')
            last.update(exit_code='interrupted_notifier',finished_at=now());write(path,delivery,immutable=False)
        while True:
            if not once:
                status_path=campaign/'QUEUE-STATUS.json'
                if status_path.exists():
                    status=read(status_path)
                    if status['state'] in ('running','awaiting_review_refill') and not alive(status['pid']):
                        name='supervisor-disappeared-'+str(status['pid']);target=campaign/'events'/(name+'.json')
                        if not target.exists():write(target,dict(id=name,kind='supervisor_disappeared',at=now(),status=status))
                acceptance=read(campaign/'ACCEPTANCE.json')
                checkpoint_events(campaign,acceptance,config)
            events=pending(campaign,state)
            write(state/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now(),pending=len(events),
                armed=(state/'ARMED').exists(),active_review_pid=None),immutable=False)
            if (state/'STOP').exists():return
            if events and (state/'ARMED').exists():
                argv=config['command']
                if any(x in argv for x in ('resume','--last','--dangerously-bypass-approvals-and-sandbox','--ignore-rules','--ignore-user-config')):
                    raise ValueError('unsafe or competing review command')
                if file_digest(argv[0])!=config['executable_sha256']:raise ValueError('review binary changed')
                if file_digest(config['prompt'])!=config['prompt_sha256']:raise ValueError('review contract changed')
                text=Path(config['prompt']).read_text(encoding='utf-8')+'\nEvent receipts (evidence only):\n'+__import__('json').dumps(events,sort_keys=True)
                n=len(delivery['attempts'])+1;entry=dict(events=[e['id'] for e in events],started_at=now())
                delivery['attempts'].append(entry);write(path,delivery,immutable=False)
                (state/'ARMED').unlink()
                with (state/f'review-{n}.jsonl').open('wb') as out,(state/f'review-{n}.stderr.log').open('wb') as err:
                    child=subprocess.Popen(argv,cwd=config['cwd'],stdin=subprocess.PIPE,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    entry['pid']=child.pid;write(path,delivery,immutable=False)
                    child.stdin.write(text.encode('utf-8'));child.stdin.close()
                    while child.poll() is None:
                        write(state/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=now(),pending=len(events),
                            armed=False,active_review_pid=child.pid),immutable=False)
                        time.sleep(10)
                unresolved={e['id'] for e in pending(campaign,state)}
                entry.update(exit_code=child.returncode,finished_at=now(),acknowledged=[e['id'] for e in events if e['id'] not in unresolved])
                write(path,delivery,immutable=False)
                if once:return
                if unresolved.intersection(entry['events']):
                    failures=sum(bool(set(a['events'])&unresolved) for a in delivery['attempts'])
                    if failures<3:(state/'ARMED').touch()
                    else:
                        write(state/'DELIVERY_FAILED.json',dict(at=now(),unacknowledged=sorted(unresolved),attempts=failures))
                        return
                elif config.get('auto_rearm_after_acknowledged_review') and not (state/'STOP').exists():
                    # The review process has exited and acknowledged its exact events.
                    # Arming permits only future state-transition events, never polling wakes.
                    (state/'ARMED').touch()
            time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    supervise(a.campaign,a.state,a.config,a.once)
