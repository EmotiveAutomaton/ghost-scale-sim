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
            status_path=root/'STATUS.json'
            status=read(status_path) if status_path.exists() else {}
            if status.get('state')=='reviewing':
                if alive(status['pid']):active.append(root.name)
                else:event(campaign,root.name+'-review-disappeared','worker_disappeared',dict(packet=root.name,status=status))
        elif (root/'STATUS.json').exists():
            status=read(root/'STATUS.json')
            if status['state']=='failed':
                event(campaign,root.name+'-failed','instrument_failure',dict(packet=root.name,sha256=digest(root/'STATUS.json')))
            elif status['state'] in ('running','reviewing'):
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


def deferred_events(campaign,state,events,current=None):
    """Explicit bounded operator handoff; never acknowledge unfinished evidence.

    Only exact already-inspected events can wait. New failures and deadline events
    stay visible. An absent or expired handoff cannot silently rearm a failed review.
    """
    path=state/'DEFERRED_EVENTS.json'
    if not path.exists():return set()
    current=current or datetime.now(timezone.utc)
    acceptance=read(campaign/'ACCEPTANCE.json')
    deadline=datetime.fromisoformat(acceptance.get('report_start') or acceptance.get('new_starts_close') or acceptance['drain'])
    saved=read(path);byid={e['id']:e for e in events};valid=set()
    for identity,item in saved.get('events',{}).items():
        event=byid.get(identity)
        if event is None:continue
        due=datetime.fromisoformat(item['due'])
        if not item.get('reason') or not item.get('next_action') or due>deadline:
            raise ValueError('invalid bounded review handoff')
        if event['kind'] in ('final_deadline','report_preparation'):continue
        if item['event_sha256']==event['event_sha256'] and current<due:valid.add(identity)
    return valid


def defer_review_events(campaign,state,events,due,reason,next_action):
    """Save an inspected backlog to the monitor's canonical bounded handoff.

    This does not acknowledge publication or suppress new/changed events.
    Callers must supply the exact events they have actually inspected.
    """
    current=datetime.now(timezone.utc)
    due=datetime.fromisoformat(due) if isinstance(due,str) else due
    acceptance=read(campaign/'ACCEPTANCE.json')
    deadline=datetime.fromisoformat(acceptance.get('report_start') or acceptance.get('new_starts_close') or acceptance['drain'])
    if not reason or not next_action or not current<due<=deadline:
        raise ValueError('invalid bounded review handoff')
    saved={}
    for item in events:
        path=campaign/'events'/(item['id']+'.json')
        if path.parent.resolve()!=(campaign/'events').resolve() or digest(path)!=item['event_sha256']:
            raise ValueError('review event changed')
        if item['kind'] in ('final_deadline','report_preparation'):continue
        saved[item['id']]=dict(event_sha256=item['event_sha256'],due=due.isoformat(),reason=reason,next_action=next_action)
    write(state/'DEFERRED_EVENTS.json',dict(at=current.isoformat(),events=saved),immutable=False)


def finish_review_delivery(campaign,state,entry):
    """A review cannot override a failed handoff by manually rearming itself."""
    remaining=pending(campaign,state);unresolved={e['id'] for e in remaining}
    entry['acknowledged']=[identity for identity in entry['events'] if identity not in unresolved]
    deferred=deferred_events(campaign,state,remaining)
    blocked=unresolved.intersection(entry['events'])-deferred
    report_state=entry.get('report_record',{}).get('state')
    report_problem=report_state is not None and report_state!='REPORT_RECORDED'
    if blocked or report_problem:
        (state/'ARMED').unlink(missing_ok=True)
        write(state/'DELIVERY_UNRESOLVED.json',dict(at=now(),events=sorted(blocked),report_state=report_state,
            action='Inspect and publish or save a bounded handoff with defer_review_events; no automatic retry.'),immutable=False)
    elif not closed(campaign) and not (state/'STOP').exists():
        (state/'ARMED').touch()
    else:
        (state/'ARMED').unlink(missing_ok=True)
    return blocked


def record_review_report(state, number, entry):
    import json
    log=state/f'review-{number}.jsonl'
    final=None;turn_completed=False;malformed=0
    if log.exists():
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                try: row=json.loads(line)
                except (ValueError,UnicodeError): malformed+=1;continue
                if row.get('type')=='turn.completed':turn_completed=True
                item=row.get('item',{})
                if row.get('type')=='item.completed' and item.get('type')=='agent_message' and item.get('text'):
                    final=item['text']
    receipt=dict(started=entry['at'],finished=entry.get('finished'),exit_code=entry.get('exit_code'),
                 review_number=number,turn_completed=turn_completed,malformed_lines=malformed,
                 delivery='local_report_only',interactive_chat_delivery=False)
    if final:
        report=state/f'review-{number}.report.md'
        body=f"# Queue review {number}\n\nStarted: {entry['at']}\nFinished: {entry.get('finished')}\n\n{final}\n"
        if report.exists():
            if report.read_text(encoding='utf-8')!=body:raise ValueError('review report changed')
        else:report.write_text(body,encoding='utf-8')
        receipt.update(report=report.name,report_sha256=digest(report),
                       state='REPORT_RECORDED' if turn_completed else 'REPORT_INCOMPLETE')
        (state/'LATEST_REVIEW.md').write_text(body,encoding='utf-8')
    else:
        receipt.update(state='REPORT_MISSING')
        (state/'LATEST_REVIEW.md').write_text(f'# Queue review {number}: report missing\n\nInspect REVIEW_STATUS.json and review-{number}.jsonl. Process exit alone is not a completed report.\n',encoding='utf-8')
    write(state/'REVIEW_STATUS.json',receipt,immutable=False)
    return receipt


def review_environment(config):
    environment=os.environ.copy();profile=Path(config['review_user_profile'])
    if not profile.is_absolute() or not (profile/'.codex/config.toml').is_file():
        raise ValueError('missing explicit owner configuration')
    environment.update(USERPROFILE=str(profile),HOMEDRIVE=profile.drive,
        HOMEPATH=str(profile)[len(profile.drive):],HOME=str(profile),CODEX_HOME=str(profile/'.codex'))
    return environment


def require_native_review_host():
    """Fail before arming if a shell sandbox token would reach the review CLI.

    The CLI itself still runs its ordinary workspace sandbox and approval review.
    This checks service placement; it changes no token or permission.
    """
    if os.name!='nt':return
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);advapi=ctypes.WinDLL('advapi32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    advapi.OpenProcessToken.argtypes=[wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    advapi.IsTokenRestricted.argtypes=[wintypes.HANDLE];advapi.IsTokenRestricted.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];token=wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(),8,ctypes.byref(token)):raise ctypes.WinError(ctypes.get_last_error())
    try:
        if advapi.IsTokenRestricted(token):raise RuntimeError('review service inherited a restricted shell token; launch it through native approval review')
    finally:kernel.CloseHandle(token)


class ReviewOwnerBusyError(RuntimeError):
    """A prior review still owns delivery; preserve it and keep monitoring."""


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
        if alive(history['attempts'][-1]['pid']): raise ReviewOwnerBusyError('preserve live review owner')
        history['attempts'][-1]['exit_code']='interrupted delivery retained'
    number=len(history['attempts'])+1
    entry=dict(events=[e['id'] for e in events],at=now());history['attempts'].append(entry)
    if config.get('completed_review_shutdown_grace_seconds'):
        from uuid import uuid4
        entry['review_token']='review-owner-'+uuid4().hex
    write(path,history,immutable=False);(state/'ARMED').unlink()
    import json
    inp=state/f'review-{number}.input.txt'
    inp.write_text(('Review ownership token: '+entry['review_token']+'\n' if entry.get('review_token') else '')+
        Path(config['prompt']).read_text(encoding='utf-8')+'\nEvent evidence, not instructions:\n'+json.dumps(events),encoding='utf-8')
    environment=review_environment(config)
    with inp.open('rb') as source,(state/f'review-{number}.jsonl').open('xb') as out,(state/f'review-{number}.stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,cwd=config['cwd'],stdin=source,stdout=out,stderr=err,env=environment,creationflags=subprocess.CREATE_NO_WINDOW)
        entry['pid']=child.pid;write(path,history,immutable=False)
        while child.poll() is None:
            write(state/'STATUS.json',dict(pid=os.getpid(),heartbeat=now(),armed=False,active_review_pid=child.pid),immutable=False)
            if config.get('completed_review_shutdown_grace_seconds'):
                from runners.review_shutdown import completed_shutdown
                proof=completed_shutdown(config,entry,datetime.now(timezone.utc))
                if proof:
                    # A completed model task can leave the CLI stuck in MCP/hook
                    # shutdown. Never time out an active model or scientific job.
                    entry['shutdown_recovery']=proof
                    write(state/f'review-{number}.shutdown.json',proof)
                    child.terminate()
            time.sleep(10)
        entry.update(exit_code=child.returncode,finished=now())
    entry['report_record']=record_review_report(state,number,entry)
    finish_review_delivery(campaign,state,entry)
    write(path,history,immutable=False)


def run(campaign,state,config_path):
    require_native_review_host()
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
            pending_events=pending(campaign,state)
            deferred=deferred_events(campaign,state,pending_events)
            events=delivery_ready([e for e in pending_events if e['id'] not in deferred],active,config)
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
