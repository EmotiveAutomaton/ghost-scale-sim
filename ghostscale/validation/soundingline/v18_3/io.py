"""Immutable evidence and bounded retries for transient Windows reader sharing."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def retry(operation):
    for attempt in range(40):
        try:return operation()
        except PermissionError:
            if attempt==39:raise
            time.sleep(min(.005*2**min(attempt,4),.05))


def read(path):return json.loads(retry(lambda:Path(path).read_bytes()))
def file_digest(path):return hashlib.sha256(retry(lambda:Path(path).read_bytes())).hexdigest()
def replace(source,destination):return retry(lambda:os.replace(source,destination))


def write(path,value,*,immutable=True):
    path=Path(path);payload=canonical(value)+b'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if retry(path.read_bytes)!=payload:raise ValueError('immutable record differs: '+str(path))
        return hashlib.sha256(payload).hexdigest()
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    try:
        if immutable:retry(lambda:os.link(temporary,path))
        else:replace(temporary,path)
    finally:
        if temporary.exists():retry(temporary.unlink)
    return hashlib.sha256(payload).hexdigest()
