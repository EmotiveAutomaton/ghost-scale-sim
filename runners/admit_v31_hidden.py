"""Production admission after every frozen hidden-reader accuracy gate passes."""
import argparse,itertools,time
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest
from ghostscale.validation.soundingline.v31 import w_settings,q_rank
from runners.v31_build import admit

def run(repo,c):
    rows=[]
    for label,mod,pool in [('W_HIDDEN_V1',w_settings,'W'),('W_HIDDEN_V3',w_settings,'W'),('Q_HIDDEN_V1',q_rank,'Q'),('Q_HIDDEN_V2',q_rank,'Q')]:
        jobs=sorted((c/'jobs').glob(label+'-*'));complete=[j for j in jobs if (j/'COMPLETE.json').exists()]
        if any(read(j/'COMPLETE.json')['state']!='VERIFIED' for j in complete):
            rows.append(dict(reader=label,state='HELD_FAILED_ACCURACY',automatic_repair='One per family, only after diagnosis; never auto-retry here'));continue
        if len(complete)!=len(jobs) or not jobs:
            rows.append(dict(reader=label,state='WAITING_ACCURACY_GATES'));continue
        if any(read(j/'COMPLETE.json')['state']!='VERIFIED' or not read(j/'output/VALIDATION.json')['passed'] for j in jobs):
            rows.append(dict(reader=label,state='HELD_FAILED_ACCURACY',automatic_repair='One per family, only after diagnosis; never auto-retry here'));continue
        configurations=[]
        for j in jobs:
            done=read(j/'COMPLETE.json');worker=read(j/'WORKER_COMPLETE.json')
            assert done['plan_sha256']==worker['plan_sha256']==digest(j/'PLAN.json')
            assert done['worker_receipt_sha256']==digest(j/'WORKER_COMPLETE.json')
            assert worker['validation_sha256']==digest(j/'output/VALIDATION.json')
            p=read(j/'PLAN.json');v=read(j/'output/VALIDATION.json');configurations.append(dict(p['setting'],particles=v['production_particles']))
        if pool=='Q':
            settings=[dict(weight=weight,group_rate=s['group_rate'],move_rate=s['move_rate'],view=s['view'],particles=s['particles'],steps=2000,worlds=200,blocks=8) for s,weight in itertools.product(configurations,(0.,.5,1.))]
            admit(repo,c,label+'_SCIENCE','Q',mod,settings,'docs/versions/v31-four-settings/protocols/Q.md',52000,extra={'validation_sha256':[digest(j/'output/VALIDATION.json') for j in jobs]})
        else:
            settings=[]
            for j in sorted((c/'jobs').glob('W-MATCH-*')):
                if not (j/'COMPLETE.json').exists() or read(j/'COMPLETE.json')['state']!='VERIFIED':continue
                p=read(j/'PLAN.json');s=p['setting'];m=read(j/'output/MATCH.json')
                if not m['passed']:continue
                config=next((x for x in configurations if all(x[k]==s[k] for k in ('volatility','reliability','actionable'))),None)
                if config is None:continue
                # One admission per parent; additional matching completions never
                # mutate a roster whose scientific seeds have been frozen.
                settings=[dict(volatility=s['volatility'],reliability=s['reliability'],actionable=s['actionable'],view=config['view'],particles=config['particles'],models=m['models'],truth=t,steps=5000,worlds=200,blocks=8,parent_job=j.name) for t in (0,1)]
                admit(repo,c,label+'_SCIENCE_'+j.name,'W',mod,settings,'docs/versions/v31-four-settings/protocols/W.md',32000,extra={'matching_sha256':digest(j/'output/MATCH.json'),'validation_sha256':[digest(x/'output/VALIDATION.json') for x in jobs]})
            if label=='W_HIDDEN_V3':
                models=[w_settings.BASE.tolist() for _ in range(3)]
                for k in range(3):models[k][4]=float(k)
                settings=[dict(volatility=s['volatility'],reliability=s['reliability'],actionable=s['actionable'],view='V3',particles=s['particles'],models=models,truth=t,steps=5000,worlds=200,blocks=8) for s,t in itertools.product(configurations,range(3))]
                admit(repo,c,'W_BURST_HIDDEN_V3','W',mod,settings,'docs/versions/v31-four-settings/protocols/W_EXTENSIONS.md',35200,extra={'validation_sha256':[digest(x/'output/VALIDATION.json') for x in jobs]})
        rows.append(dict(reader=label,state='VALIDATED_PRODUCTION_PREPARED_WITH_RESOURCE_LIMITS'))
    write(c/'HIDDEN_ADMISSION_STATUS.json',rows,immutable=False);print(rows)
    for j in sorted((c/'jobs').glob('W_BLACKOUT_*')):
        if not (j/'COMPLETE.json').exists() or read(j/'COMPLETE.json')['state']!='VERIFIED':continue
        p=read(j/'PLAN.json')
        if p.get('kind')!='validation':continue
        result=read(j/'output/VALIDATION.json')
        if not result.get('matched') or not result['passed']:continue
        original=p['setting'];s={k:original[k] for k in ('volatility','reliability','actionable','view','blackout')}
        s.update(models=result['matching']['models'],particles=result['production_particles'],steps=5000,worlds=200,blocks=8)
        settings=[dict(s,truth=t) for t in (0,1)]
        admit(repo,c,j.name+'_SCIENCE','healing',w_settings,settings,'docs/versions/v31-four-settings/protocols/FOLLOWONS.md',64000,extra={'validation_sha256':digest(j/'output/VALIDATION.json')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();run(a.repo.resolve(),a.campaign.resolve())
