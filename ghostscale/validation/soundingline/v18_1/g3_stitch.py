"""One bounded existing Stitch adapter: native3 and graphic16 only.

Grounded learned fragments use the same explicit storage and online planner as G3.
The compiler's new child CPU is measured even on failure and retained separately.
"""
import ctypes
from itertools import product
import os
from pathlib import Path
import subprocess
import time
import uuid

from ..v16.records import write,file_digest,canonical
from ..v17 import stitch_adapter as stitch
from .common import solve,score_submission
from .g3 import contract

BINARY_SHA256='4fd644e2bc8f14de10f08b31c982079879d2d3ab181264996cfd547dfd8756c0'


def learn(training,kind,root):
    if file_digest(Path(os.environ['GS_V17_STITCH_EXE']))!=BINARY_SHA256:raise ValueError('Stitch binary pin mismatch')
    os.environ['GS_V17_STITCH_CACHE']=str((root/'stitch-cache').resolve())
    original=subprocess.Popen;children=[];start=time.monotonic()
    def capture(*args,**kwargs):
        child=original(*args,**kwargs);children.append(child);return child
    subprocess.Popen=capture
    try:return stitch.learn(training,kind)
    finally:
        subprocess.Popen=original
        for child in children:
            if child.poll() is None:raise RuntimeError('owned Stitch child unexpectedly still active')
            times=[ctypes.c_ulonglong() for _ in range(4)]
            ok=ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(child._handle)),*(ctypes.byref(x) for x in times))
            record=dict(pid=child.pid,returncode=child.returncode,wall_seconds=time.monotonic()-start,
                child_cpu_seconds=(times[2].value+times[3].value)/1e7 if ok else 0.,cpu_measured=bool(ok),binary_sha256=BINARY_SHA256)
            write(root/'child-attempts'/(uuid.uuid4().hex+'.json'),record)
            if not ok:raise RuntimeError('child CPU measurement unavailable; cannot admit unknown accounting')


def grounded(learned,training,kind,storage):
    definitions={d['name']:(d['body'],d['arity']) for d in learned['accepted']};counts={};expansion_work=0
    for definition in learned['accepted']:
        for args in product(range(3 if kind=='assembly' else 16),repeat=definition['arity']):
            fragment=tuple(stitch.expand(definition['body'],definitions,args,kind));expansion_work+=len(fragment)
            if len(fragment)<2:continue
            expansion_work+=sum(max(0,len(t['program'])-len(fragment)+1)*len(fragment) for t in training)
            count=sum(t['program'][i:i+len(fragment)]==list(fragment) for t in training
                      for i in range(len(t['program'])-len(fragment)+1))
            if count:counts[fragment]=max(counts.get(fragment,0),count)
    kept=[];used=0
    for fragment in sorted(counts,key=lambda p:(-counts[p],-len(p),p)):
        cost=2*len(fragment)+1
        if used+cost<=storage:kept.append(list(fragment));used+=cost
    return dict(fragments=kept,episodes=[],storage=used,grounding_operations=expansion_work,
                scope='grounded learned operators; every stored primitive argument charged, no free template instantiation')


def evaluate(case,root,budgets=(512,2048,8192),storage_caps=(32,128)):
    public=contract(canonical(case['public']));world=public['world'];kind=world['kind']
    if (kind=='assembly' and len(world['parents'])!=3) or (kind=='graphic' and world['cells']!=16):
        raise ValueError('only the existing native3/graphic16 codec is admitted')
    training=[t for t in public['training'] if t['feedback']]
    learned=(learn(training,kind,root) if training else dict(accepted=[],rejected=[],learning_operations=0,input_tokens=0,cache_key=None))
    rows=[]
    for storage in storage_caps:
        rep=grounded(learned,training,kind,storage)
        for budget in budgets:
            result=solve(world,public['initial'],public['target'],rep,budget,public['max_steps'],public['action_order'])
            result=score_submission(case['private']['true_world'],public['initial'],public['target'],result,public['max_steps'])
            rows.append(dict(method='stitch-grounded',budget=budget,storage_cap=storage,representation=rep,
                acquisition=dict(processed_trials=len(public['training']),source_execution_primitives=sum(t['source_primitive_cost'] for t in public['training']),
                    observed_actions=sum(t['observed_actions'] for t in public['training']),storage_tokens=rep['storage'],storage_envelope=storage,
                    failed_donor_trials=sum(not t['feedback'] for t in public['training']),censored_trials=sum(not t['complete'] for t in public['training']),
                    learning_operations=learned['learning_operations'],grounding_operations=rep['grounding_operations']),
                learned_cache_key=learned['cache_key'],accepted_definitions=len(learned['accepted']),rejected_definitions=len(learned['rejected']),**result))
    return rows,dict(binary_sha256=BINARY_SHA256,cache_key=learned['cache_key'],accepted=learned['accepted'],rejected=learned['rejected'])
