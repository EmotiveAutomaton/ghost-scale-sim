"""Service one logical batch through bounded, source-frozen scientific parts."""
import os,time
from pathlib import Path
from ghostscale.validation.soundingline.v23 import partition,runtime,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners.watch_v23_events import owner


def step(campaign,root):
    from runners import queue_v23_families as queue
    plan=read(root/'PLAN.json');layout=plan['partition_layout']
    if not plan.get('scheduler_only'):raise ValueError('logical batch required')
    parts=partition.part_paths(root,plan)
    charged=0.;retained_cpu=0.;done=[];pending=[];failed=[]
    for index,part in enumerate(parts):
        status=read(part/'STATUS.json') if (part/'STATUS.json').exists() else {}
        if str(index) in plan.get('retained_parts',{}):retained_cpu+=status.get('cpu_seconds',0.)
        else:charged+=status.get('cpu_seconds',0.)
        if status.get('state')=='failed':
            failed.append(part)
        elif (part/'COMPLETE.json').exists():done.append(part)
        elif (part/'PLAN.json').exists():pending.append(part)
    if failed:
        # Retain sibling reservations until already-running parts drain.
        write(root/'STATUS.json',dict(state='partitioned' if pending else 'failed',
            phase='failure-draining' if pending else 'partition-failed',failed_parts=[p.name for p in failed],
            cpu_seconds=charged,pid=os.getpid(),heartbeat=runtime.now()),immutable=False)
        return
    write(root/'STATUS.json',dict(state='partitioned',phase='parts',cpu_seconds=charged,pid=os.getpid(),
        parts_complete=len(done),parts_total=len(parts),heartbeat=runtime.now()),immutable=False)
    if len(done)==len(parts):
        # Assembly also consumes one scientist slot and has a bounded CPU cap.
        with owner(campaign/'admission'):
            if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign,plan):return
            write(root/'STATUS.json',dict(state='partitioned',phase='assembly',uses_scientific_slot=True,
                cpu_seconds=charged,pid=os.getpid(),heartbeat=runtime.now()),immutable=False)
        started=time.process_time()
        def pulse():
            if time.process_time()-started>layout['assembly_cpu_cap']:raise TimeoutError('assembly CPU cap reached')
        try:
            if plan['task']=='twin-batch':
                from ghostscale.validation.soundingline.v23.twin_partition import assemble
            else:assemble=partition.assemble
            receipt=assemble(root,parts,pulse)
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
    max_parallel=plan.get('max_parallel_parts',1)
    if type(max_parallel) is not int or not 1<=max_parallel<=3:raise ValueError('invalid bounded partition concurrency')
    if len(pending)>=max_parallel:return
    if runtime.now()>=read(campaign/'ACCEPTANCE.json')['report_start'] or (campaign/'STOP').exists():return
    unstarted=[i for i,part in enumerate(parts) if not (part/'PLAN.json').exists()]
    if not unstarted:return
    index=unstarted[0];part=parts[index]
    with owner(campaign/'admission'):
        # Parent already reserves the entire logical batch. This is a slot check,
        # never a second charge of its child budget or a fresh budget decision.
        if queue.active_count(queue.inventory(campaign))>=resource_policy.worker_limit(campaign,plan):return
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
