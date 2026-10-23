"""Freeze the owner's finite October 23 follow-ups without drawing pilot worlds."""
import argparse
from pathlib import Path
import shutil
import time
import zipfile
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v24 import contract,theory_pilot,gallery_pilot,n16_followup
from runners.run_v24_pilot import accounting

REPO=Path(__file__).resolve().parents[1]


def definitions():
    return [('N16','v24-n16-retained','v24-n16-retained-stress-1',n16_followup.configuration(),1800),
            ('PROMOTIONS','v24-gallery-pilot','v24-gallery-controls-pilot-1',gallery_pilot.configuration(),1800)]+[
        (family,'v24-theory-pilot','v24-'+family.lower().replace('_','-')+'-pilot-1',theory_pilot.configuration(family),3300)
        for family in theory_pilot.MODULES]


def prepare(campaign):
    if read(campaign/'ACCEPTANCE.json')!=contract.acceptance():raise ValueError('acceptance changed')
    if (campaign/'OWNER.lock').exists():raise ValueError('live or unreconciled science owner')
    previous=campaign/'packets/v24-n16-pilot-1';old=read(previous/'PLAN.json')
    complete=read(previous/'COMPLETE.json')
    for key,name in [('plan_sha256','PLAN.json'),('replay_sha256','REPLAY.json'),('review_sha256','INDEPENDENT_REVIEW.json')]:
        if complete[key]!=digest(previous/name):raise ValueError('retained pilot binding changed')
    queue=campaign/'followups-20261023-r1';queue.mkdir(exist_ok=False);packets=[]
    all_cap=sum(item[-1] for item in definitions());charged,_=accounting(campaign,'PROMOTIONS')
    if charged+all_cap+300>contract.TOTAL_SECONDS-contract.VERIFICATION_RESERVE:raise ValueError('follow-up reserve insufficient')
    for family,task,identity,cfg,cap in definitions():
        started=time.process_time();_,family_cost=accounting(campaign,family)
        if family_cost+cap+30>contract.FAMILY_SECONDS[family]:raise ValueError('family preparation headroom insufficient')
        physical=identity+'-r1' if task=='v24-n16-retained' else identity
        packet=campaign/'packets'/physical;packet.mkdir(exist_ok=False);source=packet/'source';source.mkdir()
        paths=list((REPO/'ghostscale').rglob('*.py'))+[REPO/'runners'/name for name in (
            'run_v24_pilot.py','launch_background.py','run_v24_followups.py','watch_v23_events.py')]
        for path in sorted(paths):
            target=source/path.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        (source/'runners/__init__.py').write_text('',encoding='utf-8')
        for directory in ('schema','dependencies'):
            for path in sorted((previous/'source'/directory).rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
                relative=path.relative_to(previous/'source')
                if digest(path)!=old['source_files'][relative.as_posix()]:raise ValueError('retained schema dependency changed')
                target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        protocol=REPO/'docs/versions/v24-powered-readers/FOLLOWUP_PROTOCOL_2026-10-23.md'
        shutil.copyfile(protocol,source/'PROTOCOL.md')
        intake=REPO/'results/v24/review-20261023/INTAKE.json';shutil.copyfile(intake,source/'SOUNDING_LINE_INTAKE.json')
        extra={}
        if task=='v24-n16-retained':
            inputs=source/'inputs';inputs.mkdir();raw=read(previous/'original/RAW_INDEX.json')
            for name in ('evidence_points.jsonl.gz','truth_points.jsonl.gz'):
                if digest(previous/'original'/name)!=raw[name]['sha256']:raise ValueError('retained stream changed')
                shutil.copyfile(previous/'original'/name,inputs/name)
            extra=dict(input_root=str(inputs),input_files={p.name:digest(p) for p in inputs.iterdir()},
                       retained_complete_sha256=digest(previous/'COMPLETE.json'))
        hashes={p.relative_to(source).as_posix():digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
        with zipfile.ZipFile(packet/'SOURCE.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in hashes:z.write(source/name,name)
        plan=dict(kind='pilot',task=task,family=family,identity=identity,configuration=cfg,
            confirmation=False,production=False,cpu_cap_seconds=cap,wall_cap_seconds=7200,
            numerical_threads=1,priority='below-normal',source_files=hashes,source_archive_sha256=digest(packet/'SOURCE.zip'),
            acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),protocol_sha256=digest(protocol),
            schema_directory=str(source/'schema'),schema_pin=old['schema_pin'],
            environment=runtime.environment(),interpreter=str(REPO/'.venv/Scripts/python.exe'),**extra)
        write(packet/'PLAN.json',plan);write(packet/'PREPARATION.json',dict(state='PREPARED_NOT_ADMITTED',plan_sha256=digest(packet/'PLAN.json')))
        write(packet/'STATUS.json',dict(state='PREPARED_NOT_ADMITTED'))
        write(campaign/'accounting'/(physical+'-preparation.json'),dict(kind='preparation',family=family,state='COMPLETE',cpu_seconds=time.process_time()-started))
        packets.append(dict(packet=physical,family=family,plan_sha256=digest(packet/'PLAN.json'),cpu_cap_seconds=cap))
    manifest=dict(kind='finite-owner-authorized-followups',packets=packets,automatic_retries=False,
        continue_independent_families_after_recorded_failure=True,production=False,
        protocol_sha256=digest(protocol),acceptance_sha256=digest(campaign/'ACCEPTANCE.json'))
    write(queue/'MANIFEST.json',manifest)
    print(queue)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    prepare(parser.parse_args().campaign.resolve())
