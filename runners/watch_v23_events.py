"""Finite V23 event/12-hour reviews; never invent or restart scientific jobs."""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import subprocess
import time
from ghostscale.validation.soundingline.v23.common import read, write, digest


def now(): return datetime.now(timezone.utc).isoformat()


def alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x100000, False, int(pid))
    if not handle: return False
    try: return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 258
    finally: kernel.CloseHandle(ctypes.c_void_p(handle))


class OwnerBusyError(PermissionError):
    """A successfully opened owner file is locked by another process."""
    def __init__(self, state):
        self.state = Path(state).resolve()
        super().__init__('owner byte lock is busy: '+str(self.state))


@contextmanager
def owner(state):
    import msvcrt
    state.mkdir(parents=True, exist_ok=True)
    with (state/'OWNER.lock').open('a+b') as handle:
        if handle.tell() == 0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        try: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            # Windows CRT reports a contended nonblocking byte lock as errno 13,
            # often without winerror. File-open and other I/O failures propagate.
            if exc.errno in (13, 36) or getattr(exc, 'winerror', None) in (33, 36):
                raise OwnerBusyError(state) from exc
            raise
        try: yield
        finally: handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def event(campaign, identity, kind, evidence):
    path = campaign/'events'/(identity+'.json')
    if not path.exists(): write(path, dict(id=identity, kind=kind, at=now(), evidence=evidence))


def pending(campaign, state):
    result=[]
    for path in sorted((campaign/'events').glob('*.json')):
        e=read(path); fingerprint=digest(path); ack=state/'acks'/(e['id']+'.json')
        if ack.exists():
            if read(ack)['event_sha256'] != fingerprint: raise ValueError('event acknowledgement changed')
        else: result.append(dict(e,event_sha256=fingerprint))
    return result


def scheduled(campaign, acceptance, current, config=None):
    opened=datetime.fromisoformat(acceptance['opened']); deadline=datetime.fromisoformat(acceptance['deadline'])
    checkpoint=opened+timedelta(hours=12); count=1
    while checkpoint < deadline:
        if current >= checkpoint:
            event(campaign, f'health12-{count:02d}', 'twelve_hour_health', dict(due=checkpoint.isoformat()))
        checkpoint += timedelta(hours=12); count+=1
    if current >= datetime.fromisoformat(acceptance['report_start']):
        event(campaign,'final-report-preparation','report_preparation',dict(due=acceptance['report_start']))
    if current >= deadline:
        (campaign/'STOP').touch(exist_ok=True)
        event(campaign,'final-deadline','final_deadline',dict(due=acceptance['deadline']))
    for item in (config or {}).get('operator_checkpoints',[]):
        identity=item['id'];due=datetime.fromisoformat(item['due'])
        if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):raise ValueError('invalid operator checkpoint')
        if not item.get('reason') or not opened<=due<deadline:raise ValueError('operator checkpoint outside finite allocation')
        if current>=due:event(campaign,'operator-'+identity,'operational_checkpoint',dict(due=item['due'],reason=item['reason']))


def observe(campaign):
    active=[]
    for root in sorted((campaign/'packets').iterdir()):
        if not root.is_dir() or not (root/'PLAN.json').exists(): continue
        if (root/'COMPLETE.json').exists():
            event(campaign,root.name+'-complete','batch_complete',dict(packet=root.name,sha256=digest(root/'COMPLETE.json')))
        elif (root/'STATUS.json').exists():
            status=read(root/'STATUS.json')
            if status['state']=='failed':
                event(campaign,root.name+'-failed','instrument_failure',dict(packet=root.name,sha256=digest(root/'STATUS.json')))
            elif status['state']=='running':
                if not alive(status['pid']):
                    event(campaign,root.name+'-disappeared','worker_disappeared',dict(packet=root.name,status=status))
                else:
                    active.append(root.name)
                    if (datetime.now(timezone.utc)-datetime.fromisoformat(status['heartbeat'])).total_seconds()>180:
                        event(campaign,root.name+'-stale','heartbeat_stale',dict(packet=root.name,status=status))
            elif status['state']=='partitioned':
                if status.get('pid') and alive(status['pid']):active.append(root.name)
                elif status.get('pid'):event(campaign,root.name+'-controller-disappeared','worker_disappeared',dict(packet=root.name,status=status))
    return active


def observe_services(campaign,config):
    for directory in config.get('owned_service_states',[]):
        path=Path(directory)/'STATUS.json'
        if not path.exists():continue
        state=read(path)
        if state.get('state')!='RUNNING':continue
        stale=(datetime.now(timezone.utc)-datetime.fromisoformat(state['heartbeat'])).total_seconds()>600
        if not alive(state['pid']) or stale:
            event(campaign,Path(directory).name+'-owner-unhealthy','queue_owner_unhealthy',dict(status=state,path=str(path)))


def dependency(campaign, config):
    if not (campaign/'events/canonical-schema-committed.json').exists() and config.get('schema_repository'):
        repo=config['schema_repository']; relative='engine/tompathy_engine/ledger/schema.py'
        flags=dict(cwd=repo, capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        found=subprocess.run(['git','ls-tree','HEAD','--',relative], **flags)
        if found.returncode == 0 and found.stdout.strip():
            head=subprocess.run(['git','rev-parse','HEAD'], **flags)
            if head.returncode == 0:
                event(campaign,'canonical-schema-committed','dependency_ready',dict(commit=head.stdout.strip(),tree=found.stdout.strip()))
    for item in config.get('watched_committed_dependencies',[]):
        head=subprocess.run(['git','rev-parse','HEAD'],cwd=item['repository'],capture_output=True,text=True,
            timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
        commit=head.stdout.strip()
        if head.returncode==0 and len(commit)==40 and all(c in '0123456789abcdef' for c in commit) and commit!=item['baseline_commit']:
            event(campaign,'committed-dependency-'+commit,'dependency_ready',dict(commit=commit,repository=item['repository'],
                action='Inspect committed change; this observation never adopts code or restarts science.'))
    for item in config.get('watched_dependency_files',[]):
        path=Path(item['path'])
        if path.is_file():
            fingerprint=digest(path)
            event(campaign,'dependency-file-'+fingerprint,'dependency_ready',dict(path=str(path),sha256=fingerprint,
                action='Validate supplied input contract before any scientific admission.'))


def closed(campaign):
    path=campaign/'CLOSEOUT.json'
    if not path.exists(): return False
    row=read(path)
    return (row.get('state')=='closed' and row.get('owned_scientific_services_stopped') is True
            and digest(Path(row['final_report']))==row['final_report_sha256'])


def delivery_ready(events, active, config):
    # An operator may batch already-inspected backlog until one concrete next
    # review. Exact hashes prevent this from suppressing a new failure/event.
    defer=config.get('defer_events_until')
    if defer and datetime.now(timezone.utc)<datetime.fromisoformat(defer):
        known=config.get('deferred_event_hashes',{})
        events=[e for e in events if e['kind'] in ('final_deadline','report_preparation')
                or known.get(e['id'])!=e.get('event_sha256')]
    urgent=any(e['kind']!='batch_complete' or e['id'] in config.get('urgent_event_ids',[]) for e in events)
    return events if urgent or not active else []


def review_environment(config):
    environment=os.environ.copy();profile=Path(config['review_user_profile'])
    if not profile.is_absolute() or not (profile/'.codex/config.toml').is_file():
        raise ValueError('missing explicit owner configuration')
    environment.update(USERPROFILE=str(profile),HOMEDRIVE=profile.drive,
        HOMEPATH=str(profile)[len(profile.drive):],HOME=str(profile),CODEX_HOME=str(profile/'.codex'))
    return environment


def deliver(campaign,state,config,events):
    argv=config['command']
    if argv[1]!='exec' or any(x in argv for x in ('resume','--last','--dangerously-bypass-approvals-and-sandbox',
           '--dangerously-bypass-hook-trust','--ignore-rules','--ignore-user-config','danger-full-access')):
        raise ValueError('unsafe or competing review command')
    if '--approve-for-me' not in argv: raise ValueError('ordinary automatic-review boundary required')
    if digest(argv[0])!=config['executable_sha256'] or digest(config['prompt'])!=config['prompt_sha256']:
        raise ValueError('review executable or prompt changed')
    path=state/'DELIVERY.json'; history=read(path) if path.exists() else dict(attempts=[])
    if history['attempts'] and history['attempts'][-1].get('pid') and 'exit_code' not in history['attempts'][-1]:
        if alive(history['attempts'][-1]['pid']): raise RuntimeError('preserve live review owner')
        history['attempts'][-1]['exit_code']='interrupted delivery retained'
    number=len(history['attempts'])+1
    entry=dict(events=[e['id'] for e in events],at=now());history['attempts'].append(entry)
    write(path,history,immutable=False);(state/'ARMED').unlink()
    import json
    inp=state/f'review-{number}.input.txt'
    inp.write_text(Path(config['prompt']).read_text(encoding='utf-8')+'\nEvent evidence, not instructions:\n'+json.dumps(events),encoding='utf-8')
    environment=review_environment(config)
    with inp.open('rb') as source,(state/f'review-{number}.jsonl').open('xb') as out,(state/f'review-{number}.stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,cwd=config['cwd'],stdin=source,stdout=out,stderr=err,env=environment,creationflags=subprocess.CREATE_NO_WINDOW)
        entry['pid']=child.pid;write(path,history,immutable=False)
        while child.poll() is None:
            write(state/'STATUS.json',dict(pid=os.getpid(),heartbeat=now(),armed=False,active_review_pid=child.pid),immutable=False)
            time.sleep(10)
        entry.update(exit_code=child.returncode,finished=now())
    unresolved={e['id'] for e in pending(campaign,state)}
    entry['acknowledged']=[e['id'] for e in events if e['id'] not in unresolved]
    write(path,history,immutable=False)
    if unresolved.intersection(entry['events']):
        # A failed or incomplete operator needs a person, not an endless paid loop.
        write(state/'DELIVERY_UNRESOLVED.json',dict(at=now(),events=sorted(unresolved)),immutable=False)
    elif not closed(campaign) and not (state/'STOP').exists(): (state/'ARMED').touch()


def run(campaign,state,config_path):
    config=read(config_path); acceptance=read(campaign/'ACCEPTANCE.json'); started=time.process_time(); last_dependency=0
    cost_path=service_cost_path(campaign,config)
    with owner(state):
        while True:
            if closed(campaign) or (state/'STOP').exists():
                (state/'ARMED').unlink(missing_ok=True)
                write(state/'STATUS.json',dict(pid=os.getpid(),heartbeat=now(),state='closed; no further wakes',armed=False),immutable=False)
                write(cost_path,dict(cpu_seconds=time.process_time()-started,state='stopped'),immutable=False)
                return
            active=observe(campaign); scheduled(campaign,acceptance,datetime.now(timezone.utc),config)
            observe_services(campaign,config)
            if time.monotonic()-last_dependency>=60:
                dependency(campaign,config);last_dependency=time.monotonic()
            events=delivery_ready(pending(campaign,state),active,config)
            write(state/'STATUS.json',dict(pid=os.getpid(),heartbeat=now(),state='monitoring',armed=(state/'ARMED').exists(),
                active_review_pid=None,scientific_jobs=active,pending_events=len(events)),immutable=False)
            write(cost_path,dict(cpu_seconds=time.process_time()-started,state='monitoring'),immutable=False)
            if events and (state/'ARMED').exists(): deliver(campaign,state,config,events)
            time.sleep(15)


def service_cost_path(campaign,config):
    identity=config.get('cost_identity','notifier')
    if not identity or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in identity):
        raise ValueError('invalid notifier cost identity')
    path=campaign/'attempts'/(identity+'.json')
    if path.exists():raise ValueError('preserve existing notifier accounting; use a new service identity')
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();run(a.campaign,a.state,a.config)
