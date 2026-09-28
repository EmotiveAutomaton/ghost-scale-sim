"""Freeze the finite V20 forest; admission never invents work from a deadline."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,itertools,json,shutil
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,now,digest
from ghostscale.validation.soundingline.v20.runtime import REPO,source_files,freeze,fingerprint,validate_design
from ghostscale.validation.soundingline.v20.world import law

def designs():
    base=dict(training_lineages=list(range(8)),lineages=list(range(24,56)),test_cases=2048,fit_seed=0,epochs=120,history=0,tier='context',train_cases=8192,shift='native',split='discovery',reserve_eligible=False)
    opening=[dict(base,branch='G0'),dict(base,branch='G3')]
    opening += [dict(base,branch=b) for b in ('G1','G2','G4','G5','G6')]
    # G2 opening emphasizes complete observed records; it is not a duplicate G1 fit.
    opening[3]['tier']='complete'
    finite=[]
    for tier,h,n,shift,fit in itertools.product(('artifact','context','sparse','complete'),(0,1,4,16,64,256),(128,512,2048,8192,32768,131072),('native','changed-tool','presentation-shift'),(0,1)):
        if h==256 and n>32768:continue
        d=dict(base,branch='G1',tier=tier,history=h,train_cases=n,shift=shift,fit_seed=fit)
        if any(d==o for o in opening):continue
        if tier=='complete' and h==0 and n==8192 and shift=='native' and fit==0:continue
        finite.append(d)
    # Broad native sample/history curves first, shifted-world challenges after review.
    finite.sort(key=lambda d:(d['shift']!='native',d['fit_seed'],d['train_cases']>32768,d['tier']!='context',d['history'],d['train_cases']))
    for rel,copied,irrelevant,fee,n,fit in itertools.product((.5,.65,.8,.95),(False,True),(False,True),(.005,.02,.08),(2048,32768),(0,1)):
        # The irrelevant arm has actual reliability 1/2; avoid redundant rel labels.
        if irrelevant and rel!=.5:continue
        d=dict(base,branch='G4',reliability=rel,copied=copied,irrelevant=irrelevant,fee=fee,train_cases=n,fit_seed=fit)
        finite.append(d)
    for budget,n,fit in itertools.product((0,1,4,16,32,64),(2048,8192,32768),(0,1)):
        if budget==16 and n==8192 and fit==0:continue
        finite.append(dict(base,branch='G5',candidate_budget=budget,train_cases=n,fit_seed=fit))
    for n,fit in itertools.product((128,512,2048,8192,32768,131072),(0,1)):
        if n==8192 and fit==0:continue
        finite.append(dict(base,branch='G6',train_cases=n,fit_seed=fit))
    # Fixed diagnostic reserve, regardless of discovery sign; untouched outcomes.
    confirmation=[dict(base,branch='G7',lineages=list(range(96,128)),split='confirmation',reserve_eligible=True,tier=tier,history=h,train_cases=32768,fit_seed=fit)
        for tier,h,fit in itertools.product(('context','artifact','complete'),(4,64),(0,1))]
    return opening,finite,confirmation

def estimator(timings):
    import numpy as np
    measured={r['design']['history']:r for r in timings};keys=sorted(measured)
    rates=[measured[h]['fit_cpu_seconds']/measured[h]['design']['train_cases'] for h in keys]
    residual=[max(1.,measured[h]['cpu_seconds']-measured[h]['fit_cpu_seconds'])/len(measured[h]['design']['lineages']) for h in keys]
    def estimate(d):
        if d['branch'] in ('G0','G3'):return 60.,90.
        h=d['history'] if d['branch'] in ('G1','G2','G7') else 0
        fit=float(np.interp(h,keys,rates))*d['train_cases']
        # Interpolate measured complete residual costs; conservatively scale all
        # by output rows even though exact-world checking itself is fixed cost.
        consumer=float(np.interp(h,keys,residual))*len(d['lineages'])*d['test_cases']/512
        cpu=fit+consumer
        if d['branch']=='G6':cpu*=2.2
        return max(60.,cpu),max(90.,cpu*1.35)

    return estimate

def prepare(campaign,controls,timing_root):
    campaign=campaign.resolve();setup=read(campaign/'SETUP.json')
    if (campaign/'ACCEPTANCE.json').exists():raise ValueError('immutable acceptance already exists; use bounded refill')
    gate=read(controls);files={n:file_digest(REPO/n) for n in source_files()}
    if not gate['passed'] or gate['sources']!=files:raise ValueError('current controls do not bind sources')
    timings=[dict(read(p),fit_cpu_seconds=read(p.parent/'FIT_COST.json')['fit_cpu_seconds']) for p in timing_root.glob('*/TIMING.json')];estimate=estimator(timings)
    laws={};collisions=[]
    for lineage in list(range(8))+list(range(24,56))+list(range(96,160)):
        effective={k:v for k,v in law(lineage).items() if k not in ('lineage','presentation')};h=digest(effective)
        if h in laws:collisions.append([laws[h],lineage])
        laws[h]=lineage
    if collisions:raise ValueError('effective coefficient collision '+repr(collisions))
    opening,finite,confirmation=designs()
    for d in opening+finite+confirmation:
        cpu,wall=estimate(d);d.update(cpu_cap_seconds=max(600.,cpu*2.5),estimated_cpu_seconds=cpu,estimated_wall_seconds=wall)
        validate_design(d)
    artifact=REPO/'results/v20/packets';artifact.mkdir(parents=True,exist_ok=True)
    acceptance=dict(schema='v20.acceptance.1',accepted_at=now(),setup_started_at=setup['setup_started_at'],deadline=setup['deadline'],report_start=setup['report_start'],
        interim_at='2026-09-30T12:00:00+00:00',confirmation_start='2026-10-01T08:00:00+00:00',
        cumulative_cpu_ceiling_seconds=90*3600,exploratory_cpu_ceiling_seconds=72*3600,protected_cpu_seconds=18*3600,
        worker_memory_ceiling_bytes=8*2**30,minimum_free_disk_bytes=200*2**30,artifact_ceiling_bytes=64*2**30,artifact_root=str(artifact),
        gear=1,workers=1,numerical_threads=1,gpu=False,clock_rule='Fixed Friday deadline; shortened shared window; never reset',
        confirmation_lineages=list(range(96,128)),source_admission_sha256=file_digest(controls))
    write(campaign/'ACCEPTANCE.json',acceptance)
    source=campaign/'capsules/core-1/source';archive=campaign/'capsules/core-1/SOURCE.zip';archive.parent.mkdir(parents=True,exist_ok=True)
    freeze(source,archive,gate)
    # Admit a measured 30-hour scientific wave, including every implemented branch.
    selected=list(opening);remaining=[];wall=sum(d['estimated_wall_seconds'] for d in selected)
    for d in finite:
        if d['shift']=='native' and wall<30*3600:selected.append(d);wall+=d['estimated_wall_seconds']
        else:remaining.append(d)
    jobs=[];since_review=0.
    for index,d in enumerate(selected+confirmation):
        jid=f"v20-{index:04d}-{d['branch'].lower()}";root=artifact/jid;root.mkdir(exist_ok=False)
        os.link(archive,root/'SOURCE.zip')
        plan=dict(schema='v20.plan.1',design=d,sources=files,environment=fingerprint(),admission_sha256=file_digest(controls),source_archive_sha256=file_digest(archive))
        write(root/'PLAN.json',plan);write(root/'CARD.json',dict(id=jid,design=d,plan_sha256=file_digest(root/'PLAN.json'),source_archive_sha256=file_digest(archive),environment=fingerprint()))
        since_review+=d['estimated_wall_seconds'];boundary=index==len(opening)-1 or since_review>=4*3600 or index==len(selected)-1
        if boundary:since_review=0
        jobs.append(dict(id=jid,root=str(root),source=str(source),plan_sha256=file_digest(root/'PLAN.json'),requires=[] if index<2 else ['v20-0000-g0'],
            not_before=acceptance['confirmation_start'] if d['reserve_eligible'] else acceptance['accepted_at'],reserve_eligible=d['reserve_eligible'],
            cpu_estimate_seconds=d['estimated_cpu_seconds'],wall_estimate_seconds=d['estimated_wall_seconds'],review_boundary=boundary))
    queue=dict(schema='v20.queue.1',revision=1,jobs=jobs);write(campaign/'QUEUE.json',queue)
    write(campaign/'FOREST.json',dict(designs=remaining,rule='Agent admits useful registered successors after completed-result review; no new hyperparameters or seed padding',source=str(source),archive=str(archive),admission_sha256=file_digest(controls)))
    setup_cpu=sum(read(p)['cpu_seconds'] for p in campaign.glob('CONTROL_ATTEMPT_*.json'))+read(timing_root/'TOTAL.json')['cpu_seconds']+gate['cpu_seconds']
    write(campaign/'attempts/setup.json',dict(state='completed',packet='setup-controls-and-pilots',accounting_card='setup',cpu_seconds=setup_cpu,uncertainty_cpu_seconds=setup_cpu+900,scope='900 additional CPU seconds conservatively charged for unmetered early engineering and native fixtures'))
    public={k:v for k,v in acceptance.items() if k!='artifact_root'};write(REPO/'results/v20/ACCEPTANCE.json',public)
    workload=sum(d['estimated_wall_seconds'] for d in opening+finite+confirmation);remaining_hours=(datetime.fromisoformat(acceptance['report_start'])-datetime.now(timezone.utc)).total_seconds()/3600
    forecast=dict(admitted=len(jobs),discovery=len(selected),confirmation=len(confirmation),conditional=len(remaining),opening_wall_hours=wall/3600,
        conditional_plus_admitted_wall_hours=workload/3600,remaining_science_window_hours=remaining_hours,required_backlog_ratio=1.25,
        measured_backlog_ratio=(workload/3600)/remaining_hours,runway_requirement_met=wall>=24*3600,backlog_requirement_met=workload>=1.25*remaining_hours*3600,
        forecast='Conservative complete-block extrapolation from four measured calibration workloads; recalibrate at first named batch; not a runtime guarantee',timings=timings)
    write(REPO/'results/v20/OPENING_FORECAST.json',forecast);write(REPO/'results/v20/FOREST.json',dict(admitted=[dict(id=j['id'],design=d) for j,d in zip(jobs,selected+confirmation)],conditional=remaining))
    write(REPO/'results/v20/SOURCE_MANIFEST.json',dict(files=files,archive_sha256=file_digest(archive),admission_sha256=file_digest(controls)))
    return forecast

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);p.add_argument('--timing-root',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.campaign,a.controls,a.timing_root)))
