"""One fixed revision opportunity after retained native local-world episodes."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import time
import numpy as np
from ..v18_3.io import canonical, digest, file_digest, write
from . import local_world as L

ARMS = ('keep', 'native', 'request-adopted', 'visible-task', 'full-state-oracle')
METRICS = ('success', 'net_success', 'operation_cost', 'revision_rate',
           'repaired_failure', 'destroyed_success')


def success(artifact, request):
    return int(artifact[0] == 1 and artifact[1] == 1) if request == 0 else artifact[2]


def cost(op, amount):
    return 0. if op == 'inspect' else amount


def packet(artifact, previous, old_request, request, hidden):
    def mask(a): return [a[0], a[1], None if hidden else a[2]]
    return dict(artifact=mask(artifact), undo_artifact=mask(previous),
                original_request=old_request, current_request=request,
                observed_units=[0, 1] if hidden else [0, 1, 2])


def task_policy(artifact, previous, maker, request, hidden, operation_cost):
    current = [tuple(artifact[:2])+(b,) for b in range(2)] if hidden else [tuple(artifact)]
    buffers = [tuple(previous[:2])+(b,) for b in range(2)] if hidden else [tuple(previous)]
    legal = [op for op in L.OPERATIONS if maker[1] or op != 'accept-tool']
    scores = {op: math.fsum(success(L.execute(a,b,op,maker),request)
                          for a,b in product(current,buffers))/(len(current)*len(buffers))
                   - cost(op,operation_cost) for op in legal}
    best = max(scores.values()); selected = [op for op in legal if best-scores[op] <= 1e-12]
    return [(None,op,1/len(selected)) for op in selected]


def policy(world, maker, artifact, previous, old_request, request, hidden, arm, operation_cost):
    if arm == 'keep': return [(None,'inspect',1.)]
    if arm in ('visible-task','full-state-oracle'):
        return task_policy(artifact,previous,maker,request,hidden and arm=='visible-task',operation_cost)
    if arm not in ('native','request-adopted'): raise ValueError('unknown revision policy')
    adopted = (request, *maker[1:]) if arm=='request-adopted' else maker
    return list(L.choices(world,adopted,artifact,2,dict(requested_purpose=request)))


def evaluate(distribution, artifact, previous, maker, request, arm, operation_cost, inspection_cost):
    before=success(artifact,request); outcomes=[]
    for goal,op,p in distribution:
        after=L.execute(artifact,previous,op,maker); ok=success(after,request); fee=cost(op,operation_cost)
        outcomes.append(dict(goal=goal,operation=op,probability=p,after=list(after),success=ok,
            net_success=ok-fee-(0 if arm=='keep' else inspection_cost),operation_cost=fee,
            revision_rate=int(op!='inspect'),repaired_failure=int(not before and ok),
            destroyed_success=int(before and not ok)))
    if abs(math.fsum(x['probability'] for x in outcomes)-1)>1e-12:raise ValueError('policy normalization')
    return outcomes,{k:math.fsum(x['probability']*x[k] for x in outcomes) for k in METRICS}


def controls():
    w=L.law(190967);m=(0,1,0,0);a=(0,0,0);b=(1,1,1)
    native=policy(w,m,a,b,0,1,False,'native',.05)
    adopted=policy(w,m,a,b,0,1,False,'request-adopted',.05)
    _,keep=evaluate([(None,'inspect',1.)],a,b,m,1,'keep',.05,.01)
    best=task_policy(a,b,m,0,False,.05)
    _,v=evaluate(best,a,b,m,0,'full-state-oracle',.05,.01)
    illegal=False
    try:L.execute(a,b,'accept-tool',(0,0,0,0))
    except ValueError:illegal=True
    return dict(positive_undo=L.execute(a,b,'undo',m)==b,
        positive_illegal_tool=illegal,live_adoption=native!=adopted,
        placebo_native_hidden=native==policy(w,m,a,b,0,1,True,'native',.05),
        placebo_hidden_bits=task_policy(a,b,m,1,True,.05)==task_policy((0,0,1),(1,1,0),m,1,True,.05),
        positive_intact_oracle=policy(w,m,a,b,0,1,False,'visible-task',.05)==policy(w,m,a,b,0,1,False,'full-state-oracle',.05),
        positive_task_repair=abs(v['success']-1)<1e-12 and abs(v['net_success']-.94)<1e-12,
        placebo_keep=keep['success']==0 and keep['net_success']==0)


def group(records):
    groups={}
    for i,r in enumerate(records):
        key=(tuple(r['maker']),tuple(r['final']),tuple(r['steps'][-1]['before']),r['requested_purpose'])
        if key not in groups:groups[key]=dict(masses=[],source_indices=[])
        groups[key]['masses'].append(r['probability']);groups[key]['source_indices'].append(i)
    return [(key,dict(mass=math.fsum(v['masses']),source_indices=v['source_indices'])) for key,v in sorted(groups.items())]


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()) or cfg['arms']!=list(ARMS):raise ValueError('revision controls/admission failed')
    if cfg['operation_cost']!=.05 or cfg['inspection_cost']!=.01:raise ValueError('frozen costs differ')
    for name,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/name)!=h:raise ValueError('retained input differs')
    for folder in ('raw','evaluator'):(root/folder).mkdir()
    cells=[];public={};timing=[];paths=0;states=0;row_count=0
    for lineage in cfg['lineages']:
        pulse(phase='routine-revision',lineage=lineage);start=time.process_time()
        records=json.loads(gzip.decompress((root/'inputs'/f'lineage-{lineage}_points.json.gz').read_bytes()))
        if len(records)!=cfg['paths_per_lineage'] or abs(math.fsum(r['probability'] for r in records)-1)>1e-10:raise ValueError('native denominator')
        groups=group(records);world=L.law(lineage);rows=[];acc=defaultdict(list)
        evaluator=[]
        for state_index,((maker,a,b,old),g) in enumerate(groups):
            evaluator.append(dict(state_index=state_index,maker=list(maker),artifact=list(a),undo_artifact=list(b),original_request=old,**g))
            for changed,hidden in product((False,True),repeat=2):
                request=old^int(changed);visible=packet(a,b,old,request,hidden);ident=digest(visible)
                public[ident]=dict(inputs=visible,input_sha256=ident,passage_anchors=['unit-0','unit-1','unit-2'])
                arm_values={}
                for arm in ARMS:
                    choices=policy(world,maker,a,b,old,request,hidden,arm,.05)
                    outcomes,values=evaluate(choices,a,b,maker,request,arm,.05,.01);arm_values[arm]=values
                    rows.append(dict(state_index=state_index,changed_request=changed,hidden_presentation=hidden,
                        arm=arm,packet_sha256=ident,population_mass=g['mass'],outcomes=outcomes,**values))
                    key=(changed,hidden,maker[0],maker[1],maker[3],arm)
                    acc[key].append((g['mass'],values))
                if arm_values['full-state-oracle']['net_success']+1e-12 < max(arm_values[k]['net_success'] for k in ARMS if k!='keep'):
                    raise ValueError('oracle task dominance failed')
                if not hidden and arm_values['full-state-oracle']!=arm_values['visible-task']:raise ValueError('intact oracle identity')
        write(root/'evaluator'/f'{lineage}-states.json',evaluator)
        (root/'raw'/f'{lineage}-revision_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
        for key,rr in sorted(acc.items()):
            mass=math.fsum(p for p,v in rr)
            cells.append(dict(lineage=lineage,changed_request=key[0],hidden_presentation=key[1],purpose=key[2],skill=key[3],routine=key[4],arm=key[5],
                population_mass=mass,states=len(rr),**{k:math.fsum(p*v[k] for p,v in rr)/mass for k in METRICS}))
        paths+=len(records);states+=len(groups);row_count+=len(rows)
        timing.append(dict(lineage=lineage,cpu_seconds=time.process_time()-start,paths=len(records),states=len(groups),rows=len(rows),new_fits=0))
    write(root/'PUBLIC_PACKET.json',dict(schema='v19.revision.reader.1',cases=[public[k] for k in sorted(public)],
        semantics='observed current request; adoption unknown; hidden presentation is null; no maker or policy labels'))
    write(root/'TIMING.jsonl',dict(measurements=timing))
    checks['positive_complete_oracle_dominance']=True
    return dict(controls=checks,cells=cells,paths=paths,states=states,rows=row_count,public_packets=len(public),fits=0,
        scope='exact native-policy revision and supplied-mechanics task references; no learned competence or endorsement conclusion')
