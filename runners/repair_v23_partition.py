"""Recover only failed execution partitions, retaining verified scientific work."""
from pathlib import Path
from ghostscale.validation.soundingline.v23 import family,partition,runtime,resource_policy
from ghostscale.validation.soundingline.v23.common import read,write,digest
from runners import queue_v23_families as queue
from runners.watch_v23_events import owner


def admit(campaign, original, identity, part_cap):
    campaign,original=Path(campaign).resolve(),Path(original).resolve()
    if Path(identity).name!=identity or not identity.startswith('v23-'):raise ValueError('unsafe recovery identity')
    if not 0<part_cap<=14400:raise ValueError('invalid bounded recovery cap')
    with owner(campaign/'admission'):
        original_plan=read(original/'PLAN.json')
        acceptance=resource_policy.resolve(campaign,original_plan)
        if not acceptance['opened']<=runtime.now()<acceptance['report_start']:raise ValueError('outside original window')
        authority=campaign/'CONTINUATION_AUTHORITY_20261016.json';read(authority)
        plan=read(original/'PLAN.json');status=read(original/'STATUS.json')
        if status['state']!='failed' or not plan.get('scheduler_only') or plan.get('retained_parts'):
            raise ValueError('one retained failed logical batch required')
        if (original/'COMPLETE.json').exists():raise ValueError('never repeat an accepted logical batch')
        retained={};failed=[];unstarted=[]
        for i,part in enumerate(partition.part_paths(original,plan)):
            if (part/'COMPLETE.json').exists():
                child,_=family.verify_packet(part);partition.verify_part_binding(original,plan,part,child,i)
                retained[str(i)]=dict(path=str(part.resolve()),complete_sha256=digest(part/'COMPLETE.json'))
            elif (part/'PLAN.json').exists():
                if read(part/'STATUS.json')['state']!='failed':raise ValueError('partition is still active')
                failed.append(part.name)
            else:unstarted.append(i)
        if not failed:raise ValueError('no failed partition to repair')
        cap=part_cap*(len(plan['partition_layout']['chunks'])-len(retained))+plan['partition_layout']['assembly_cpu_cap']
        # Recovery retains the original logical packet's source/runtime contract.
        # Do not grant a new budget that its frozen runtime cannot enforce.
        state=queue.capacity(campaign,queue.inventory(campaign),cap,budget_group=plan['budget_group'],
            needs_slot=False,policy_plan=original_plan)
        if state!='AVAILABLE':raise ValueError('recovery capacity held: '+state)
        root=campaign/'packets'/identity
        if root.exists():raise ValueError('preserve existing recovery identity')
        source=Path(plan['source_archive']).with_suffix('')
        for relative,expected in plan['sources'].items():
            if digest(source/relative)!=expected:raise ValueError('original frozen scientific source changed')
        new=dict(plan,cpu_cap_seconds=cap,wall_cap_seconds=4*cap,retained_parts=retained,
            repair_parent=str(original),repair_parent_plan_sha256=digest(original/'PLAN.json'),
            repair_parent_status_sha256=digest(original/'STATUS.json'),
            partition_layout=dict(plan['partition_layout'],part_cpu_cap=part_cap,total_cpu_cap=cap),
            second_burnin_reservation_seconds=0)
        write(root/'PLAN.json',new)
        write(root/'ADMISSION.json',dict(at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
            continuation_authority_sha256=digest(authority),retained_verified_parts=retained,
            failed_parts=failed,unstarted_parts=unstarted,new_cpu_cap_seconds=cap,
            changed_scientific_source=False,changed_seeds_fits_worlds_cases=False,
            original_failure_preserved=True,successor_requires_fresh_capacity_check=True))
        return root
