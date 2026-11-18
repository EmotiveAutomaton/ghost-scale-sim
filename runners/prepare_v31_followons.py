"""Evaluate only frozen V31 conditional triggers from verified primary receipts."""
import argparse,time,importlib
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest
from ghostscale.validation.soundingline.v31 import s_learning,q_event_followon
from runners.v31_build import admit,PINS
from runners.run_v27 import frozen_source

def prepare(repo,c):
    start=time.process_time();triggers=[];s_settings=[];q_settings=[];w_parents=[]
    for path in sorted((c/'jobs').glob('*/COMPLETE.json')):
        if read(path)['state']!='VERIFIED':continue
        job=path.parent;plan=read(job/'PLAN.json');family=plan['family'];s=plan['setting']
        if family not in ('D_CORE','D_HEAVY','S_LEARNING_R2','Q_EXACT','Q_HIDDEN_V1_SCIENCE') and not family.startswith('W_HIDDEN_V1_SCIENCE_'):continue
        src=job/'output/study-original/SUMMARY.json'
        if not src.exists():continue
        receipt=read(job/'WORKER_COMPLETE.json');assert receipt['summary_sha256']==digest(src)
        result=read(src);metrics={x['measure']:x for x in result['rows']}
        if family in ('D_CORE','D_HEAVY') and s['people']==2272 and s['days']==10 and s['prompts']==6:
            ratio=metrics['random_slope']['spread']/metrics['split']['spread']
            triggers.append(dict(parent=job.name,kind='D_optional_recipe',eligible=ratio<=2/3,spread_ratio=ratio,summary_sha256=digest(src)))
        if family=='S_LEARNING_R2' and s['truth']=='SHIFT':
            row=metrics['shift_wait_gap_at50'];ci=row['interval95'];eligible=max(abs(ci[0]),abs(ci[1]))<=.2
            triggers.append(dict(parent=job.name,kind='S_cost_learning',eligible=eligible,summary_sha256=digest(src)))
            if eligible:
                for truth in ('SHIFT','COST'):s_settings.append(dict(s,truth=truth,kinds=['BASE','SHIFT','SHARPEN','SLOW','COST'],parent_job=job.name))
        if family in ('Q_EXACT','Q_HIDDEN_V1_SCIENCE') and s['view']=='V1' and s['weight']==1:
            eligible=metrics['shows_without_own_event']['interval95'][0]>.2
            triggers.append(dict(parent=job.name,kind='Q_exact_events',eligible=eligible,summary_sha256=digest(src)))
            if eligible:q_settings.append(dict(s,parent_job=job.name))
        if family.startswith('W_HIDDEN_V1_SCIENCE_') and s['volatility']==.02 and s['reliability']==.9:
            parent=read(c/'jobs'/s['parent_job']/'PLAN.json')['setting']
            if (parent['left'],parent['right'])==(0,1):
                eligible=metrics['separation']['unresolved']['interval95'][0]>.5
                triggers.append(dict(parent=job.name,kind='W_blackout',eligible=eligible,summary_sha256=digest(src)))
                if eligible:w_parents.append((s['parent_job'],dict(parent,view='V1',low_particles=512,high_particles=2048)))
    # Admit each parent separately so a later qualifying result never modifies a
    # previously frozen roster or seed identity. Profiles execute serially.
    for module,prefix,settings in ((s_learning,'S_COST_FOLLOWON',s_settings),(q_event_followon,'Q_EVENTS_FOLLOWON',q_settings)):
        parents=sorted(set(s['parent_job'] for s in settings))
        for parent in parents:
            rows=[s for s in settings if s['parent_job']==parent]
            actual=importlib.import_module(read(c/'jobs'/parent/'PLAN.json')['module']) if prefix=='S_COST_FOLLOWON' else module
            admit(repo,c,prefix+'_'+parent,'healing',actual,rows,'docs/versions/v31-four-settings/protocols/FOLLOWONS.md',62000)
    for parent,s in {p:(p,s) for p,s in w_parents}.values():
        family='W_BLACKOUT_'+parent;gatepath=c/'families'/family/'ADMISSION.json'
        if gatepath.exists():continue
        source=frozen_source(c,PINS);write(gatepath,dict(passed=True,scope='Conditional excluded matching and hidden accuracy only',source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),protocol_sha256=digest(repo/'docs/versions/v31-four-settings/protocols/FOLLOWONS.md')))
        for length in (1,2,4,8,16,32):
            setting=dict(s,blackout=length)
            write(c/'jobs'/f'{family}-{length:02d}'/'PLAN.json',dict(family=family,pool='healing',kind='validation',module='ghostscale.validation.soundingline.v31.w_blackout_validation',setting=setting,order=63000+length,cpu_cap_seconds=1800,wall_cap_seconds=7200,source_root=str(source),source_manifest_sha256=digest(source/'SOURCE_MANIFEST.json'),acceptance_sha256=digest(c/'ACCEPTANCE.json'),admission_sha256=digest(gatepath),interpreter=str(repo/'.venv/Scripts/python.exe')))
    write(c/'TRIGGER_STATUS.json',dict(rows=triggers),immutable=False)
    write(c/'accounting'/('followon-build-'+str(time.time_ns())+'.json'),dict(pool='reserve',cpu_seconds=time.process_time()-start,reservation=0,state='COMPLETE'))
    print('Conditional triggers evaluated',len(triggers),'eligible',sum(x['eligible'] for x in triggers))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();prepare(a.repo.resolve(),a.campaign.resolve())
