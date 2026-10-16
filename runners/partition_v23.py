"""Service one logical batch through bounded, sequential scientific parts."""
import os,time
from pathlib import Path
from ghostscale.validation.soundingline.v23 import partition,runtime
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner


def step(campaign,root):
    from runners import queue_v23_families as queue
    plan=read(root/'PLAN.json');layout=plan['partition_layout']
    if not plan.get('scheduler_only'):raise ValueError('logical batch required')
    parts=partition.part_paths(root,plan)
    charged=0.;retained_cpu=0.;done=[];pending=[]
    for index,part in enumerate(parts):
        status=read(part/'STATUS.json') if (part/'STATUS.json').exists() else {}
        if str(index) in plan.get('retained_parts',{}):retained_cpu+=status.get('cpu_seconds',0.)
        else:charged+=status.get('cpu_seconds',0.)
        if status.get('state')=='failed':
            write(root/'STATUS.json',dict(state='failed',phase='partition-failed',packet=part.name,
                cpu_seconds=charged,heartbeat=runtime.now()),immutable=False);return
        if (part/'COMPLETE.json').exists():done.append(part)
        elif (part/'PLAN.json').exists():pending.append(part)
    write(root/'STATUS.json',dict(state='partitioned',phase='parts',cpu_seconds=charged,pid=os.getpid(),
        parts_complete=len(done),parts_total=len(parts),heartbeat=runtime.now()),immutable=False)
    if pending:return
    if len(done)==len(parts):
        # Assembly also consumes one scientist slot and has a bounded CPU cap.
        if queue.active_count(queue.inventory(campaign))>=read(campaign/'ACCEPTANCE.json')['max_scientific_workers']:return
        started=time.process_time()
        def pulse():
            if time.process_time()-started>layout['assembly_cpu_cap']:raise TimeoutError('assembly CPU cap reached')
        try:
            receipt=partition.assemble(root,parts,pulse)
            used=time.process_time()-started
            write(root/'COMPLETE.json',receipt)
            write(root/'STATUS.json',dict(state='complete',phase='review-needed',cpu_seconds=charged+used,
                scientific_batch_cpu_seconds=charged+used+retained_cpu,
                parts_complete=len(parts),parts_total=len(parts),heartbeat=runtime.now()),immutable=False)
        except BaseException as exc:
            write(root/'STATUS.json',dict(state='failed',phase='assembly',error=repr(exc),
                cpu_seconds=charged+time.process_time()-started,heartbeat=runtime.now()),immutable=False)
            raise
        return
    index=len(done);part=parts[index]
    with owner(campaign/'admission'):
        # Parent already reserves the entire logical batch. This is a slot check,
        # never a second charge of its child budget or a fresh budget decision.
        if queue.active_count(queue.inventory(campaign))>=read(campaign/'ACCEPTANCE.json')['max_scientific_workers']:return
        cap=layout['part_cpu_cap']
        if cap>14400 or cap*(len(parts)-len(plan.get('retained_parts',{})))+layout['assembly_cpu_cap']>plan['cpu_cap_seconds']:
            raise ValueError('partition reservation changed')
        child=dict(plan,scheduler_only=False,world_indices=layout['chunks'][index],
            logical_parent=str(root.resolve()),logical_plan_sha256=digest(root/'PLAN.json'),
            cpu_cap_seconds=cap,wall_cap_seconds=4*cap,second_burnin_reservation_seconds=0)
        child.pop('partition_layout');child.pop('burnin_reservations_seconds',None)
        source=Path(child['source_archive']).with_suffix('')
        for relative,expected in child['sources'].items():
            if digest(source/relative)!=expected:raise ValueError('partition source changed')
        write(part/'PLAN.json',child)
        write(part/'ADMISSION.json',dict(at=runtime.now(),logical_parent=root.name,
            logical_plan_sha256=digest(root/'PLAN.json'),plan_sha256=digest(part/'PLAN.json'),
            accounting='charged to existing logical reservation; no additional budget authority'))
        queue.launch(part,campaign,source)
