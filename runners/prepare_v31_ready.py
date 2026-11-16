"""Admit only completed matching dependencies to previously verified readers."""
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v31.common import read,write,digest

def prepare(c):
    started=time.process_time();status=[];corners={(.05,.75,.5),(.02,.9,.8),(.1,.6,.2)}
    for job in sorted((c/'jobs').glob('W-MATCH-*')):
        if not (job/'COMPLETE.json').exists():continue
        if read(job/'COMPLETE.json')['state']!='VERIFIED':continue
        match=read(job/'output/MATCH.json');setting=read(job/'PLAN.json')['setting']
        if not match['passed']:status.append(dict(job=job.name,state='MATCH_INFEASIBLE'));continue
        idx=int(job.name.rsplit('-',1)[1]);world=tuple(setting[k] for k in ('volatility','reliability','actionable'))
        for view in ('V4','V2'):
            if view=='V2' and world not in corners:continue
            gatepath=c/'families'/('W_READER_'+view)/'ADMISSION.json'
            if not gatepath.exists():continue
            gate=read(gatepath);profile=read(c/'preparation'/('W_READER_'+view)/'VALIDATION.json');source=Path(gate['source_root'])
            for truth in (0,1):
                name=f'W-MATCHED-{idx:04d}-{view}-{truth}';planpath=c/'jobs'/name/'PLAN.json'
                if planpath.exists():continue
                s={k:setting[k] for k in ('volatility','reliability','actionable')};s.update(models=match['models'],view=view,truth=truth,steps=5000,worlds=200,blocks=8)
                cap=profile['cpu_cap_seconds']
                write(planpath,dict(family='W_READER_'+view,pool='W',module='ghostscale.validation.soundingline.v31.w_settings',setting=s,order=31000+idx*10+truth,cpu_cap_seconds=cap,wall_cap_seconds=max(1800,cap*4),source_root=str(source),source_manifest_sha256=gate['source_manifest_sha256'],acceptance_sha256=digest(c/'ACCEPTANCE.json'),admission_sha256=digest(gatepath),matching_sha256=digest(job/'output/MATCH.json'),matching_job=job.name,interpreter=read(job/'PLAN.json')['interpreter']))
                status.append(dict(job=name,state='PREPARED'))
    write(c/'READY_BUILD_STATUS.json',dict(rows=status,scope='Exact matched W readers; hidden readers require separate numerical admission'),immutable=False)
    write(c/'accounting'/('ready-build-'+str(time.time_ns())+'.json'),dict(pool='reserve',cpu_seconds=time.process_time()-started,reservation=0,state='COMPLETE'))
    print('New exact matched W cells',sum(r['state']=='PREPARED' for r in status))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();prepare(a.campaign.resolve())
