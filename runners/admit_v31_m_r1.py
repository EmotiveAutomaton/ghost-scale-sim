"""Admit the original M roster only after its retained-path R1 gate passes."""
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest


def admit(repo,c):
    started=time.process_time();job=c/'jobs/M_R1_VALIDATION'
    p=read(job/'PLAN.json');d=read(job/'COMPLETE.json');w=read(job/'WORKER_COMPLETE.json')
    assert d['state']==w['state']=='VERIFIED'
    assert d['plan_sha256']==w['plan_sha256']==digest(job/'PLAN.json')
    assert d['worker_receipt_sha256']==digest(job/'WORKER_COMPLETE.json')
    v=read(job/'output/VALIDATION.json')
    assert v['passed'] and v['full_seed_replay'] and v['independent_reconstruction']
    assert w['validation_sha256']==digest(job/'output/VALIDATION.json')
    source=Path(p['source_root']);assert digest(source/'SOURCE_MANIFEST.json')==p['source_manifest_sha256']
    for name,h in read(source/'SOURCE_MANIFEST.json').items():assert digest(source/name)==h
    roster=read(c/'M_ROSTER.json');assert len(roster)==552
    gate=c/'families/M_R1/ADMISSION.json'
    proof=dict(passed=True,validation_sha256=w['validation_sha256'],source_manifest_sha256=p['source_manifest_sha256'],
               protocol_sha256=digest(source/'docs/versions/v31-four-settings/protocols/M_R1.md'),
               scope='Corrected original settings and seeds; retained-path correction is not independent replication',
               supersedes_family='M_CORE')
    write(gate,proof)
    for i,s in enumerate(roster):
        write(c/'jobs'/f'M_R1-{i:04d}'/'PLAN.json',dict(family='M_R1',pool='M',module=v['production_module'],
              setting=s,order=20000+i,cpu_cap_seconds=v['cpu_cap_seconds'],wall_cap_seconds=max(1800,v['cpu_cap_seconds']*4),
              source_root=str(source),source_manifest_sha256=p['source_manifest_sha256'],
              acceptance_sha256=p['acceptance_sha256'],admission_sha256=digest(gate),interpreter=str(repo/'.venv/Scripts/python.exe')))
    write(c/'accounting/admit-m-r1.json',dict(pool='reserve',state='COMPLETE',cpu_seconds=time.process_time()-started,reservation=0))
    print('M_R1: 552 original settings admitted, cap',v['cpu_cap_seconds'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--campaign',type=Path,required=True)
    a=ap.parse_args();admit(a.repo.resolve(),a.campaign.resolve())
