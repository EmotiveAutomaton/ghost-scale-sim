"""Freeze and enqueue the first M/W/Q healings; never retries existing gates."""
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest
from runners.run_v27 import frozen_source
from runners.run_v31 import accounting
from runners.v31_build import PINS


def prepare(repo,c):
    start=time.process_time();roster=read(c/'M_ROSTER.json')
    probe=max(roster,key=lambda s:s['makers']*s['messages']**2)
    tasks=[('M_R1_VALIDATION','ghostscale.validation.soundingline.v31.m_r1_validation',
            dict(campaign=str(c),probe=probe),1800,'M_R1.md')]
    for family,cap in [('W',3600),('Q',10800)]:
        configs=[]
        for p in sorted((c/'jobs').glob(f'{family}_HIDDEN_V*-*/PLAN.json')):
            plan=read(p);configs.append(dict(parent=p.parent.name,setting=plan['setting']))
        assert len(configs)==(6 if family=='W' else 12)
        tasks.append((family+'_HIDDEN_R1','ghostscale.validation.soundingline.v31.hidden_r1_batch',
                      dict(family=family,campaign=str(c),configurations=configs),cap,'HIDDEN_RESOLUTION_R1.md'))
    a=read(c/'ACCEPTANCE.json');totals=accounting(c)
    assert totals['healing']+sum(x[3] for x in tasks)<=a['cpu_limits']['healing']
    source=frozen_source(c,PINS+['runners/prepare_v31_repairs_20261116.py',
        'docs/versions/v31-four-settings/protocols/M_R1.md',
        'docs/versions/v31-four-settings/protocols/HIDDEN_RESOLUTION_R1.md'])
    for order,(family,module,setting,cap,protocol) in enumerate(tasks):
        gate=c/'families'/family/'ADMISSION.json'
        assert not gate.exists(),family+' already frozen; no automatic retry'
        write(gate,dict(passed=True,scope='Excluded first healing only, not scientific admission',
              authorization='V31 packet section 7: one separately frozen repair per family',
              protocol_sha256=digest(repo/'docs/versions/v31-four-settings/protocols'/protocol),
              source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json')))
        write(c/'jobs'/family/'PLAN.json',dict(family=family,pool='healing',kind='validation',
              module=module,setting=setting,order=-30+order,cpu_cap_seconds=cap,
              wall_cap_seconds=cap*4,source_root=str(source),
              source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),
              acceptance_sha256=digest(c/'ACCEPTANCE.json'),admission_sha256=digest(gate),
              interpreter=str(repo/'.venv/Scripts/python.exe')))
    write(c/'accounting/prepare-repairs-20261116.json',dict(pool='reserve',state='COMPLETE',
          cpu_seconds=time.process_time()-start,reservation=0.))
    print('Queued three finite repair gates in existing Gear 2 slots',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();prepare(a.repo.resolve(),a.campaign.resolve())
