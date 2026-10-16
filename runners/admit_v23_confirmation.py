"""Source-bound repair confirmation admission after discovery order freezes.

No command in this module freezes the order or retries an executed identity.
The global reserve and confirmation group cap apply before a packet is created.
"""
import argparse
import secrets
import shutil
import zipfile
from pathlib import Path
from ghostscale.validation.soundingline.v23 import confirmation, ledger, runtime, support
from ghostscale.validation.soundingline.v23.common import read, write, digest
from ghostscale.validation.soundingline.v23.family import verify_packet
from runners.queue_v23_families import launch, inventory
from runners.watch_v23_events import owner


def capacity(campaign, entries, cap):
    acceptance=read(campaign/'ACCEPTANCE.json');reserved=0.;group=0.;active=0
    for root,plan,status in entries:
        unfinished=status.get('state') not in ('complete','failed')
        if plan.get('logical_parent'):
            if unfinished:active+=1
            continue
        if unfinished:
            reserved+=max(0,plan['cpu_cap_seconds']-status.get('cpu_seconds',0)) if plan.get('scheduler_only') else plan['cpu_cap_seconds']
            if not plan.get('scheduler_only'):active+=1
        if plan['budget_group']=='confirmation':
            group+=plan['cpu_cap_seconds'] if unfinished else status.get('cpu_seconds',0.)
        if status.get('state')!='failed' and plan.get('second_burnin_reservation_seconds'):
            second=campaign/'packets'/('v23-'+plan['family'].lower()+'-batch-2')
            if not (second/'PLAN.json').exists():reserved+=plan['second_burnin_reservation_seconds']
    if active>=acceptance['max_scientific_workers']:raise ValueError('no scientific worker slot')
    if runtime.cost(campaign)+reserved+cap>acceptance['cpu_seconds']-acceptance['protected_cpu_seconds']:
        raise ValueError('global protected reserve would be invaded')
    if group+cap>15*3600:raise ValueError('confirmation group cap exceeded')
    return reserved


def admit(campaign,order_path,index,card_path,review_path,first_packet,pilot,cap,do_launch=False):
    campaign=Path(campaign).resolve();order_path=Path(order_path).resolve();card_path=Path(card_path).resolve()
    with owner(campaign/'admission'):
        acceptance=read(campaign/'ACCEPTANCE.json')
        if not acceptance['opened']<=runtime.now()<acceptance['report_start']:
            raise ValueError('outside scientific admission window')
        entries=confirmation.validate_order(read(order_path))
        if not 1<=index<=len(entries):raise ValueError('invalid confirmation index')
        entry=entries[index-1];family=entry['family']
        if any(s.get('state') not in ('complete','failed') and p.get('task')!='confirmation' for _,p,s in inventory(campaign)):
            raise ValueError('discovery/setup still has unfinished admitted work')
        from ghostscale.validation.soundingline.v23.pipeline_new import BUILT
        from ghostscale.validation.soundingline.v23 import support_new
        if family not in ('R1','R2','R3','R4') and family not in BUILT:raise ValueError('confirmation pipeline is not implemented')
        if family.startswith('N') and not support_new.bound(family)['available']:raise ValueError('no finite raw support for confirmation')
        if not 0<cap<=14400:raise ValueError('bounded packet cap must be at most four CPU hours')
        root=campaign/'packets'/f'v23-{family.lower()}-confirmation-1'
        if root.exists():raise ValueError('confirmation identity already exists; never repeat valid science')
        review_path=Path(review_path).resolve();review=read(review_path)
        if (review.get('state')!='verified_discovery' or review.get('decision')!='NOMINATED'
                or review.get('nominated')!=entry['variant'] or review.get('family')!=family
                or not review.get('full_fresh_process_replay')
                or review['receipts'][-1]['machine_card_sha256']!=entry['machine_card_sha256']):
            raise ValueError('complete numerical discovery review required')
        verify_packet(card_path.parent)
        nominated=read(card_path)
        if (nominated['complete_sha256']!=digest(card_path.parent/'COMPLETE.json')
                or nominated['numerical_review_sha256']!=digest(card_path.parent/'NUMERICAL_REVIEW.json')):
            raise ValueError('nominated evidence bindings changed')
        first_packet=Path(first_packet).resolve();pilot=Path(pilot).resolve()
        first_plan,_=verify_packet(first_packet);pilot_plan,pilot_result=verify_packet(pilot)
        if first_plan['family']!=family or first_plan['batch']!=1 or pilot_plan['family']!=family:
            raise ValueError('wrong family cost evidence')
        controls=pilot_result['controls']
        controls_passed=controls.get('passed',controls.get('known_answer') and controls.get('mutation_refused'))
        if pilot_result['state']!='PILOT_VALID' or not controls_passed:
            raise ValueError('largest-cell known-answer pilot unresolved')
        worlds=256 if abs(entry['shrunk_mean']-.02)<=.02 else 128
        costs=[read(pilot/label/'pilot-cost.json') for label in ('original','replay')]
        cells=[p for row in costs for p in row.get('profile',[]) if p['variant']==entry['variant']]
        if len(cells)!=2:raise ValueError('nominated-cell timing required from both pilot passes')
        per_case=max((row['cpu_seconds']+1/64)/row['cases'] for row in cells)
        # Full first-batch cost covers fitting, setup and I/O in addition to its
        # science. Add all new largest-cell inference twice with a timing margin.
        estimated=read(first_packet/'STATUS.json')['cpu_seconds']+1.5*per_case*worlds*256*2
        if estimated>cap:raise ValueError('conservative full-fit and replay cost exceeds packet cap')
        reserved=capacity(campaign,inventory(campaign),cap)
        pin=read(campaign/'SCHEMA_PIN.json')
        if pin.get('committed_files_verified') is not True:raise ValueError('canonical schema not committed')
        ledger.canonical_schema(pin['directory'],pin)
        if runtime.environment()!=read(campaign/'ENVIRONMENT.json'):raise ValueError('environment changed')
        plan=dict(task='confirmation',family=family,batch='confirmation',variants=[entry['variant']],worlds=worlds,cases=256,
            nomination_order=str(order_path),nomination_order_sha256=digest(order_path),nomination_card=str(card_path),
            nomination_review=str(review_path),nomination_review_sha256=digest(review_path),
            confirmation_index=index,confirmation_support=support_new.bound(family) if family.startswith('N') else support.repair_bound(family),canary_key=secrets.token_hex(32),
            budget_group='confirmation',cpu_cap_seconds=cap,wall_cap_seconds=cap*4,protected=False,
            schema_directory=pin['directory'],schema_pin=pin,acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
            pilot_complete_sha256=digest(pilot/'COMPLETE.json'),first_batch_complete_sha256=digest(first_packet/'COMPLETE.json'))
        confirmation.verify_definition(plan)
        source=campaign/'sources'/root.name;source.mkdir(parents=True,exist_ok=False)
        repo=Path(__file__).resolve().parents[1]
        names=['ghostscale/__init__.py','ghostscale/validation/__init__.py','ghostscale/validation/soundingline/__init__.py',
               'runners/run_v23.py','runners/launch_background.py']
        names += [p.relative_to(repo).as_posix() for p in sorted((repo/'ghostscale/validation/soundingline/v23').glob('*.py'))]
        for name in names:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
        archive=source.with_suffix('.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(names):z.write(source/name,name)
        plan.update(source_archive=str(archive),source_archive_sha256=digest(archive),sources={name:digest(source/name) for name in names},
            dependency_files={str(p):digest(p) for p in (campaign/'dependencies').rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')})
        calibration=read(campaign/'CALIBRATION_ACCEPTANCE.json')
        if plan['sources']['ghostscale/validation/soundingline/v23/adaptive.py']!=calibration['adaptive_source_sha256']:
            raise ValueError('accepted calibrated source changed')
        write(root/'PLAN.json',plan)
        write(root/'ADMISSION.json',dict(at=runtime.now(),plan_sha256=digest(root/'PLAN.json'),
            estimated_cpu_seconds=estimated,new_cap_seconds=cap,reserved_cpu_seconds=reserved,
            confirmation_order_sha256=digest(order_path),registered_primary_unchanged=True))
        if do_launch:launch(root,campaign,source)
        return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--order',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--card',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--first-packet',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--cpu-cap',type=float,required=True);p.add_argument('--launch',action='store_true')
    a=p.parse_args();print(admit(a.campaign,a.order,a.index,a.card,a.review,a.first_packet,a.pilot,a.cpu_cap,a.launch))
