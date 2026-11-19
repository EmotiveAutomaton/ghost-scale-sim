"""Verified cold-output relocation with resumable receipts and stable local paths.

Use only with an explicitly reviewed plan of immutable, closed output directories.
Never moves the repository, scientific sources, queue metadata or active output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


class ArchivePaused(Exception):pass


def lower_priority():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.SetPriorityClass.restype=wintypes.BOOL
        if not kernel.SetPriorityClass(wintypes.HANDLE(-1),0x4000):
            raise ctypes.WinError(ctypes.get_last_error())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    # A network-share reader or antivirus can briefly deny atomic replacement.
    # Retain the old file and pending new file if the bounded retry is exhausted.
    for attempt in range(6):
        try:
            os.replace(tmp,path)
            break
        except PermissionError:
            if attempt==5:raise
            time.sleep(0.2*(2**attempt))


def below(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    if path==root or not path.is_relative_to(root):raise ValueError('path outside approved root')
    return path


def listing(root):
    """Do not traverse any unexpected reparse point inside an output bundle."""
    found={};stack=[Path(root)]
    while stack:
        p=stack.pop()
        with os.scandir(p) as it:
            for entry in it:
                stat=entry.stat(follow_symlinks=False)
                if getattr(stat,'st_file_attributes',0)&0x400 or entry.is_symlink():
                    raise ValueError('nested link in output bundle')
                q=Path(entry.path)
                if entry.is_dir(follow_symlinks=False):stack.append(q)
                else:found[q.relative_to(root).as_posix()]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return found


def verify(root,files):
    actual=listing(root)
    if set(actual)!=set(files):raise ValueError('archive inventory mismatch')
    for name,row in files.items():
        if actual[name]['size']!=row['size'] or digest(root/name)!=row['sha256']:
            raise ValueError('archive content mismatch: '+name)


def relocate(workspace,destination,relative,state,pulse=lambda **kw:None):
    workspace=Path(workspace).resolve();destination=Path(destination).resolve();state=Path(state)
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='.local':
        raise ValueError('only reviewed local-data bundles can be relocated')
    source=workspace/relative;target=below(destination/'retained'/relative,destination)
    key=hashlib.sha256(relative.as_posix().encode()).hexdigest();receipt=state/'transactions'/(key+'.json')
    retired=source.with_name(source.name+'.archive-retired-'+key[:12])
    record=json.loads(receipt.read_text()) if receipt.exists() else None
    if record and (record['relative']!=relative.as_posix() or Path(record['target'])!=target):
        raise ValueError('receipt identity changed')
    if record and record['stage']=='EMPTY_RETAINED':
        below(source,workspace)
        if source.is_symlink() or not source.is_dir() or listing(source):
            raise ValueError('recorded empty source changed')
        mirror=destination/'manifests'/(key+'.json')
        if not mirror.exists() or json.loads(mirror.read_text())!=record:save(mirror,record)
        return record
    if record and record['stage']=='RECLAIMED':
        if not source.is_symlink() or source.resolve()!=target:raise ValueError('completed archive link changed')
        mirror=destination/'manifests'/(key+'.json')
        if not mirror.exists() or json.loads(mirror.read_text())!=record:
            # The local receipt may have committed before a remote replace failed.
            # This reconciles metadata only; no further evidence is removed.
            save(mirror,record)
        return record
    if record is None:
        below(source,workspace);below(retired,workspace)
        if source.is_symlink() or not source.is_dir() or retired.exists():raise ValueError('source is not an untouched directory')
        if target.exists():raise ValueError('unowned destination already exists')
        files=listing(source)
        if not files:
            # Excluded/failed jobs may have created an output directory without
            # writing evidence. Keep that directory untouched and record the
            # zero-byte disposition instead of stopping all later transfers.
            record=dict(stage='EMPTY_RETAINED',relative=relative.as_posix(),
                        target=str(target),files={},bytes=0,finished=time.time(),
                        original_path_preserved=True,empty_source_retained=True,
                        target_created=False)
            save(receipt,record);save(destination/'manifests'/(key+'.json'),record)
            return record
        record=dict(stage='INVENTORIED',relative=relative.as_posix(),target=str(target),files=files,
                    bytes=sum(x['size'] for x in files.values()),started=time.time())
        save(receipt,record);target.mkdir(parents=True,exist_ok=False)
    if record['relative']!=relative.as_posix() or Path(record['target'])!=target:raise ValueError('receipt identity changed')
    target_verified=False
    if record['stage']=='INVENTORIED':
        if listing(source)!={n:{k:v[k] for k in ('size','mtime_ns')} for n,v in record['files'].items()}:
            raise ValueError('source changed after inventory')
        for name,row in record['files'].items():
            src=source/name;dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True)
            before=src.stat()
            if dst.exists():
                sha=digest(src)
                if digest(dst)!=sha:raise ValueError('existing destination differs')
            else:
                tmp=dst.with_name(dst.name+'.archive-copy-part')
                h=hashlib.sha256()
                copied=0;last_pulse=time.monotonic()
                with src.open('rb') as inp,tmp.open('wb') as out:
                    for block in iter(lambda:inp.read(8*1024*1024),b''):
                        h.update(block);out.write(block);copied+=len(block)
                        if time.monotonic()-last_pulse>=15:
                            pulse(stage='copying',relative=relative.as_posix(),file=name,bytes=copied)
                            last_pulse=time.monotonic()
                    out.flush();os.fsync(out.fileno())
                sha=h.hexdigest()
                after=src.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('source changed during copy')
                shutil.copystat(src,tmp);os.replace(tmp,dst)
            row['sha256']=sha
            pulse(stage='copying',relative=relative.as_posix(),file=name,bytes=row['size'])
        verify(target,record['files'])
        target_verified=True
        record['stage']='VERIFIED';save(receipt,record)
        save(destination/'manifests'/(key+'.json'),record)
    if record['stage']=='VERIFIED':
        if not target_verified:verify(target,record['files']);target_verified=True
        if source.exists() and not source.is_symlink():
            if retired.exists():raise ValueError('retired-name collision')
            expected={n:{k:v[k] for k in ('size','mtime_ns')} for n,v in record['files'].items()}
            if listing(source)!=expected:raise ValueError('source changed before cutover')
            below(source,workspace);below(retired,workspace);source.rename(retired)
        if not source.exists():os.symlink(target,source,target_is_directory=True)
        if not source.is_symlink() or source.resolve()!=target:raise ValueError('cutover did not preserve path')
        record['stage']='LINKED';save(receipt,record)
    if record['stage']=='LINKED':
        # Revalidate the destination on recovery before reclaiming any source bytes.
        if not target_verified:verify(target,record['files'])
        if retired.exists():
            below(retired,workspace)
            remaining=listing(retired)
            if not set(remaining)<=set(record['files']):raise ValueError('unexpected retained files')
            for name in remaining:
                src=below(retired/name,retired)
                if digest(src)!=record['files'][name]['sha256']:raise ValueError('retired source changed')
                src.unlink()
            for directory,dirs,files in os.walk(retired,topdown=False,followlinks=False):
                if files:raise ValueError('unrecorded file during reclamation')
                candidate=Path(directory)
                if candidate.resolve()!=retired.resolve():below(candidate,retired)
                candidate.rmdir()
        record.update(stage='RECLAIMED',finished=time.time(),original_path_preserved=True)
        save(receipt,record);save(destination/'manifests'/(key+'.json'),record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());state=Path(plan['state']);began=time.process_time();start=time.monotonic();completed=[]
    lower_priority()
    def pulse(**kw):
        save(state/'STATUS.json',dict(at=time.time(),pid=os.getpid(),completed=len(completed),
            reclaimed_bytes=sum(x['bytes'] for x in completed),cpu_seconds=time.process_time()-began,
            wall_seconds=time.monotonic()-start,**kw))
        if kw.get('stage') not in ('paused','failed','complete'):
            if (state/'STOP').exists():raise ArchivePaused('operator pause; source evidence retained')
            if time.process_time()-began>plan.get('cpu_cap_seconds',7200):
                raise ArchivePaused('infrastructure CPU bound; retain resumable transfers')
    try:
        for relative in plan['sources']:
            pulse(stage='starting',relative=relative)
            done=relocate(plan['workspace'],plan['destination'],relative,state,pulse)
            completed.append(dict(relative=relative,bytes=done['bytes'],disposition=done['stage']))
        pulse(stage='complete')
        save(state/'COMPLETE.json',dict(at=time.time(),bundles=completed,bytes=sum(x['bytes'] for x in completed),
            cpu_seconds=time.process_time()-began,all_hashes_verified=True,paths_preserved=True,
            empty_sources_retained=sum(x['disposition']=='EMPTY_RETAINED' for x in completed)))
    except ArchivePaused as exc:
        pulse(stage='paused',reason=str(exc))
    except BaseException as exc:
        pulse(stage='failed',error=repr(exc));raise


if __name__=='__main__':main()
