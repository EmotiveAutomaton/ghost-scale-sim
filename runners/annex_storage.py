"""Register verified archive payloads with git-annex without copying raw bytes.

Consumes completed archive_storage receipts only. Existing scientific paths and
archive bytes remain unchanged. This command stages annex pointers; publication
and any site-specific command authorization are the caller's responsibility.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from datetime import datetime, timezone

DATA_SUFFIXES={'.npz','.npy','.parquet','.arrow','.feather','.csv','.tsv','.zip','.gz','.tar','.zst'}
SHA=re.compile(r'^[0-9a-f]{64}$')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,path)


def safe_relative(value):
    if '\\' in value or ':' in value:raise ValueError('nonportable relative path')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or any('\n' in x or '\r' in x or '\0' in x for x in p.parts):
        raise ValueError('unsafe relative path')
    return p


def is_raw(name):
    p=safe_relative(name);lower=p.name.lower()
    if any(x.lower() in ('sources','__pycache__','.git','notifications') for x in p.parts):return False
    if lower.startswith(('source','environment','config','plan','status','owner','launch','worker','failure')):return False
    return (p.suffix.lower() in DATA_SUFFIXES or lower.endswith('_points.json')
            or ('raw' in p.parts and p.suffix.lower() in ('.json','.jsonl')))


def entries(receipt,destination):
    if receipt.get('stage')!='RECLAIMED' or receipt.get('original_path_preserved') is not True:
        raise ValueError('archive transaction is not complete')
    relative=safe_relative(receipt['relative'])
    if relative.parts[0]!='.local' or len(relative.parts)<2:raise ValueError('unreviewed source root')
    expected=(Path(destination)/'retained'/str(relative)).resolve()
    if Path(receipt['target']).resolve()!=expected:raise ValueError('archive target binding mismatch')
    output=[]
    for name,record in sorted(receipt['files'].items()):
        sub=safe_relative(name)
        if not is_raw(name):continue
        size=record['size'];digest=record['sha256']
        if type(size) is not int or size<0 or not SHA.fullmatch(digest):raise ValueError('invalid content identity')
        output.append(dict(file=(PurePosixPath('raw/data')/PurePosixPath(*relative.parts[1:])/sub).as_posix(),
                           key=f'SHA256-s{size}--{digest}',sha256=digest,size=size,source=expected/str(sub),mtime_ns=record['mtime_ns']))
    return output


def link_object(row,target):
    source=row['source'];before=source.stat()
    if before.st_size!=row['size'] or before.st_mtime_ns!=row['mtime_ns']:
        raise ValueError('verified archive file changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if not os.path.samefile(source,target) and (target.stat().st_size!=row['size'] or sha(target)!=row['sha256']):
            raise ValueError('existing annex object differs')
    else:os.link(source,target)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('archive changed during registration')
    return target


def git(repo,args,data=None):
    p=subprocess.run(['git',*args],cwd=repo,input=data,text=True,encoding='utf-8',
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.strip() or p.stdout[-2000:])
    return p.stdout


def register(repo,destination,state,receipt_path,remote='ghost-archive'):
    return register_many(repo,destination,state,[receipt_path],remote)[0]


def register_many(repo,destination,state,receipt_paths,remote='ghost-archive'):
    repo=Path(repo);state=Path(state)
    pending=[];prior_results=[]
    for receipt_path in map(Path,receipt_paths):
        receipt=read(receipt_path);items=entries(receipt,destination)
        bundle=hashlib.sha256(receipt['relative'].encode()).hexdigest()
        done=state/'registered'/(bundle+'.json');source_sha=sha(receipt_path)
        if done.exists():
            prior=read(done)
            if prior['receipt_sha256']!=source_sha:raise ValueError('registered receipt changed')
            prior_results.append(prior)
        else:pending.append((receipt,items,bundle,done,source_sha))
    if not pending:return prior_results
    items=[item for _,rows,_,_,_ in pending for item in rows]
    if git(repo,['branch','--show-current']).strip()!='main':raise ValueError('registration requires main')
    remote_dir=Path(git(repo,['config','--get',f'remote.{remote}.annex-directory']).strip()).resolve()
    destination=Path(destination).resolve()
    if remote_dir!=destination/'annex-objects':raise ValueError('unexpected annex storage directory')
    uuid=git(repo,['config','--get',f'remote.{remote}.annex-uuid']).strip()
    if items:
        keys=list(dict.fromkeys(row['key'] for row in items))
        paths=git(repo,['annex','examinekey','--batch','--format=${hashdirlower}${key}/${key}\n'],'\n'.join(keys)+'\n').splitlines()
        if len(paths)!=len(keys):raise ValueError('incomplete key layout response')
        object_paths={key:remote_dir/str(safe_relative(path.replace('\\','/'))) for key,path in zip(keys,paths)}
        linked=set()
        for row in items:
            if row['key'] not in linked:link_object(row,object_paths[row['key']]);linked.add(row['key'])
            pointer=repo/row['file']
            if pointer.exists() and pointer.read_bytes()!=('/annex/objects/'+row['key']+'\n').encode():
                raise ValueError('existing repository path differs')
        result=git(repo,['annex','fromkey','--force','--batch','--json'],''.join(row['key']+' '+row['file']+'\n' for row in items))
        responses=[json.loads(line) for line in result.splitlines() if line.startswith('{')]
        if len(responses)!=len(items) or not all(x.get('success') for x in responses):raise ValueError('incomplete annex pointer admission')
        git(repo,['annex','setpresentkey','--batch'],'\n'.join(key+' '+uuid+' 1' for key in keys)+'\n')
        object_format=git(repo,['rev-parse','--show-object-format']).strip()
        index={}
        for line in git(repo,['ls-files','--stage','-z','--','raw/data']).split('\0'):
            if line:
                meta,name=line.split('\t',1);mode,blob,stage=meta.split();index[name]=(mode,blob,stage)
        for row in items:
            content=('/annex/objects/'+row['key']+'\n').encode()
            blob=hashlib.new(object_format,b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if index.get(row['file'])!=('100644',blob,'0'):raise ValueError('raw bytes entered Git index')
    results=[]
    for receipt,items,bundle,done,source_sha in pending:
        public=dict(schema=1,bundle_id=bundle,source_bundle=receipt['relative'].removeprefix('.local/'),
                migration_receipt_sha256=source_sha,storage_state='VERIFIED_ARCHIVE',
                scientific_scope='Storage integrity only; scientific validity and reporting remain in version result records.',
                remote=remote,files=[{k:row[k] for k in ('file','key','sha256','size')} for row in items])
        save(repo/'raw/catalog'/(bundle+'.json'),public)
        result=dict(at=datetime.now(timezone.utc).isoformat(),bundle_id=bundle,receipt_sha256=source_sha,
                files=len(items),logical_bytes=sum(x['size'] for x in items),catalog_sha256=sha(repo/'raw/catalog'/(bundle+'.json')))
        save(done,result);results.append(result)
    return prior_results+results


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--receipts',type=Path,nargs='+',required=True);a=p.parse_args()
    for result in register_many(a.repo,a.destination,a.state,a.receipts):print(json.dumps(result),flush=True)


if __name__=='__main__':main()
