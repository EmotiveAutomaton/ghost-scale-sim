"""One admitted scientific worker at a time; native queue transitions, no LLM polling."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def stamp():return datetime.now(timezone.utc).isoformat()


def replace_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    for attempt in range(20):
        try:os.replace(temporary,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(0.05)


def watch(manifest,python):
    path=Path(manifest).resolve();jobs=json.loads(path.read_text())
    state=dict(pid=os.getpid(),started_at=stamp(),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),jobs=[],state='running')
    output=path.parent/(path.stem+'-STATUS.json')
    # Exclusive OS ownership is held across the entire queue; scientific runners
    # independently enforce global campaign ownership, including external launches.
    from ghostscale.validation.soundingline.v16.runtime import local_owner
    with local_owner(path.parent/(path.stem+'-owner')):
        for job in jobs['jobs']:
            root=Path(job['root']);source=Path(job['source'])
            expected=job['plan_sha256']
            if hashlib.sha256((root/'PLAN.json').read_bytes()).hexdigest()!=expected:
                raise ValueError('queued plan changed')
            command=[python,'-B','-m','runners.run_v18_1','run','--root',str(root),'--archive',job['archive']]
            log=root/'queued-run.log'
            with log.open('ab') as stream:
                child=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                entry=dict(root=str(root),source=str(source),worker_pid=child.pid,started_at=stamp(),command=command)
                state['jobs'].append(entry);replace_json(output,state)
                code=child.wait();entry.update(exit_code=code,finished_at=stamp())
            if code!=0 or not (root/'COMPLETE.json').exists():
                state.update(state='scientific_failure',finished_at=stamp());replace_json(output,state)
                replace_json(path.parent/(path.stem+'-EVENT.json'),dict(kind='scientific_failure',at=stamp(),job=entry))
                return 1
            entry['complete_sha256']=hashlib.sha256((root/'COMPLETE.json').read_bytes()).hexdigest()
            replace_json(output,state)
        state.update(state='queue_complete',finished_at=stamp());replace_json(output,state)
        replace_json(path.parent/(path.stem+'-EVENT.json'),dict(kind='queue_complete',at=stamp(),jobs=state['jobs']))
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--python',required=True)
    args=parser.parse_args();raise SystemExit(watch(args.manifest,args.python))


if __name__=='__main__':main()
