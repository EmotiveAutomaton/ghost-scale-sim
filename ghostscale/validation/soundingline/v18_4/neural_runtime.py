"""One Ghost evaluator owns a waiting, explicitly metered CPU PyTorch child."""
from datetime import datetime,timezone
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
from ..v18_3.io import read,write,file_digest,canonical,now
from ..v16.runtime import local_owner
from ..v18_3.native import ProcessClock
from ..v18_3 import world as W
from . import neural_data as D


def proper_scores(truth,p):
    if p.shape!=truth.shape or np.any(p<0) or not np.all(np.isfinite(p)) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('invalid held-out forecast')
    log=np.zeros_like(p,dtype=float)
    np.log(p,out=log,where=p>0)
    loss=-np.sum(truth*log,axis=1);loss[np.any((truth>0)&(p==0),axis=1)]=np.inf
    return loss,np.sum((p-truth)**2,axis=1),np.sum(abs(p-truth),axis=1)/2


def score_E(root,pulse=lambda **kw:None):
    from .runtime import stats
    data=root/'data';child=root/'neural';manifest=read(data/'reader/INPUTS.json')
    evaluator=read(data/'EVALUATOR.json');completed=read(child/'COMPLETE.json')
    rows=[];reference_checks=0;scalar_checks=0
    for condition,entry in evaluator['truth'].items():
        path=data/entry['name']
        if file_digest(path)!=entry['sha256']:raise ValueError('test truth changed')
        with np.load(path,allow_pickle=False) as z:
            truth=z['target'].astype(float);exact=z['exact'];ids=z['ids']
            summaries={name:z[name].copy() for name in ('passive_summary','intervention_summary')}
        public_path=data/'reader'/manifest['tests'][condition]['name']
        if file_digest(public_path)!=manifest['tests'][condition]['sha256']:raise ValueError('test features changed')
        with np.load(public_path,allow_pickle=False) as z:public={k:z[k].copy() for k in z.files}
        if file_digest(data/f'{condition}-points.json.gz')!=entry['points_sha256']:raise ValueError('retained evaluator points changed')
        retained=json.loads(gzip.decompress((data/f'{condition}-points.json.gz').read_bytes()))
        queries=D.TEST_QUERIES[condition]
        # Fixed evenly spaced cases, chosen before outcomes. Reconstruct features and exact forecasts.
        for i in sorted({int(j*(len(retained)-1)/15) for j in range(16)}):
            case=retained[i];w=case['world'];state=case['state'];payload=W.packet(w,case['history'])
            h,n,wf=D.features(payload)
            if not np.array_equal(h,public['history'][i]) or n!=public['length'][i]:raise ValueError('feature reconstruction differs')
            posterior=W.posterior(payload)
            for q,c in enumerate(queries):
                from ..v18_3.verify import reference_policy
                raw=reference_policy(w,state,c);expected=np.zeros(16)
                for program,prob in zip(W.PROGRAMS,raw):expected[sum(1<<v for v in program)]+=prob
                row=i*len(queries)+q
                if not np.allclose(expected,truth[row],atol=1e-7,rtol=0):raise ValueError('independent neural target differs')
                if not np.allclose(posterior@W.artifact_matrix(w,c),exact[row],atol=1e-12,rtol=0):raise ValueError('exact control differs')
                reference_checks+=1
        forecasts={'exact':[(None,exact)]}
        forecasts.update({name.replace('_','-'):[(None,p)] for name,p in summaries.items()})
        for fit_name,tests in completed['predictions'].items():
            receipt=tests[condition];pfile=child/receipt['file']
            if file_digest(pfile)!=receipt['sha256']:raise ValueError('neural forecast changed')
            with np.load(pfile,allow_pickle=False) as z:p=z['probabilities'].astype(float)
            kind,seed=fit_name.rsplit('-seed',1);forecasts.setdefault(kind,[]).append((int(seed),p))
        for method,fits in forecasts.items():
            for seed,p in fits:
                losses,brier,tv=proper_scores(truth,p)
                for j in (0,len(p)//2,len(p)-1):
                    terms=[-float(q)*math.log(float(v)) for q,v in zip(truth[j],p[j]) if q>0 and v>0]
                    if np.isfinite(losses[j]) and abs(math.fsum(terms)-losses[j])>1e-10:raise ValueError('scalar proper-score mismatch')
                    scalar_checks+=1
                for cell in sorted(set(ids[:,0])):
                    for lineage in sorted(set(ids[:,1])):
                        chosen=(ids[:,0]==cell)&(ids[:,1]==lineage)
                        rows.append(dict(condition=condition,method=method,seed=seed,cell=int(cell),lineage=int(lineage),
                            expected_loss=W.loss_record(float(np.mean(losses[chosen]))),brier=float(np.mean(brier[chosen])),
                            total_variation=float(np.mean(tv[chosen])),queries=int(chosen.sum())))
        pulse(phase='scoring',condition=condition)
    grouped={}
    for row in rows:
        key=(row['condition'],row['method'],row['lineage'])
        grouped.setdefault(key,[]).append(row)
    clusters=[]
    for (condition,method,lineage),points in grouped.items():
        values={m:float(np.mean([r[m] for r in points])) for m in ('brier','total_variation')}
        values['expected_loss']=None if any(r['expected_loss']['infinite'] for r in points) else float(np.mean([r['expected_loss']['value'] for r in points]))
        clusters.append(dict(condition=condition,method=method,lineage=lineage,**values))
    summary={};paired={}
    for condition in manifest['tests']:
        for method in sorted({r['method'] for r in clusters}):
            selected=[r for r in clusters if r['condition']==condition and r['method']==method]
            summary[condition+'|'+method]={m:stats([r[m] for r in selected],('E',condition,method,m)) for m in ('expected_loss','brier','total_variation')}
            baseline={r['lineage']:r for r in clusters if r['condition']==condition and r['method']=='direct'}
            paired[condition+'|'+method+' minus direct']={m:stats([None if r[m] is None or baseline[r['lineage']][m] is None else r[m]-baseline[r['lineage']][m] for r in selected],('E-paired',condition,method,m)) for m in ('expected_loss','brier','total_variation')}
    path=root/'neural_points.json.gz';path.write_bytes(gzip.compress(canonical(dict(rows=rows,clusters=clusters)),mtime=0))
    result=dict(family='L',cells=summary,paired=paired,raw_sha256=file_digest(path),
        checks=dict(independent_target_and_exact_reconstructions=reference_checks,independent_scalar_scores=scalar_checks),
        fits=completed['fits'],benchmarks=completed['benchmarks'],environment=completed['environment'],
        learning_control_failures=[k for k,v in completed['fits'].items() if not v['learning_control_passed']],
        independent_unit='coefficient-draw lineage; query probes, fit seeds and 16 architecture cells averaged within it',
        scope='equal full-distribution simulator supervision; finite architecture discovery, no identified psychological slots')
    write(root/'SUMMARY.json',result);return result


def run(root,campaign):
    from .runtime import REPO,fingerprint
    with local_owner(campaign/'scientific-worker-owner'),local_owner(root):
        # A surviving orphan keeps its own lock even if its waiting parent disappeared.
        with local_owner(campaign/'neural-child-owner'):pass
        plan=read(root/'PLAN.json');acceptance=read(campaign/'ACCEPTANCE.json');design=plan['design']
        if fingerprint()!=plan['environment']:raise ValueError('Ghost runtime changed')
        if any(file_digest(REPO/name)!=value for name,value in plan['sources'].items()):raise ValueError('source identity changed')
        if file_digest(root/'SOURCE.zip')!=plan['source_archive_sha256']:raise ValueError('source archive changed')
        if (root/'COMPLETE.json').exists():
            completed=read(root/'COMPLETE.json')
            for name,value in completed['files'].items():
                if file_digest(root/name)!=value:raise ValueError('completed neural packet changed')
            return 'complete'
        if os.name=='nt':
            from .priority import below_normal
            below_normal()
        start=time.monotonic();cpu=time.process_time();attempt=uuid.uuid4().hex;child_cpu=0.;child=None
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('unreconciled interrupted attempt')
        old=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        def emit(state='running',**detail):
            write(root/'STATUS.json',dict(state=state,pid=os.getpid(),parent_pid=os.getppid(),attempt=attempt,heartbeat=now(),**detail),immutable=False)
            write(campaign/'attempts'/f'{attempt}.json',dict(packet=root.name,state=state,cpu_seconds=time.process_time()-cpu,
                child_cpu_seconds=child_cpu,wall_seconds=time.monotonic()-start),immutable=False)
        def limited():
            return ((campaign/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])
                or old+time.process_time()-cpu+child_cpu>=acceptance['cumulative_cpu_ceiling_seconds'])
        def pulse(**detail):
            emit(**detail)
            if limited():raise TimeoutError('absolute campaign resource cutoff')
        emit()
        try:
            pulse(phase='before-generation')
            if design['study']=='E':D.prepare(root/'data',**design['data'],pulse=pulse)
            elif design['study']=='G':
                from .intervention_data import prepare
                prepare(root/'data',**design['data'],pulse=pulse)
            elif design['study']=='E-purpose':
                from .purpose_data import prepare
                parent=campaign/design['parent_packet']
                if file_digest(parent/'COMPLETE.json')!=design['parent_complete_sha256']:raise ValueError('frozen behavior parent changed')
                prepare(root/'data',parent,pulse)
            else:raise ValueError('unimplemented neural study')
            pulse(phase='before-training')
            config=dict(design['training'],report_start=acceptance['report_start'],cpu_ceiling_seconds=acceptance['cumulative_cpu_ceiling_seconds'])
            write(root/'TRAINING.json',config)
            python=Path(os.environ['GHOST_V18_TORCH_PYTHON']).resolve()
            if not python.is_absolute() or not python.is_file():raise ValueError('explicit shared interpreter missing')
            env=os.environ.copy();env['GHOST_V18_CHILD_LOCK']=str(campaign/'neural-child-owner')
            # Reserve evaluator time and include failed previous attempts in the new allowance.
            env['GHOST_V18_CHILD_CPU_LIMIT']=str(max(0,acceptance['cumulative_cpu_ceiling_seconds']-old-(time.process_time()-cpu)-60))
            module=design.get('worker_module','ghostscale.validation.soundingline.v18_3.'+{'E':'torch_worker','G':'intervention_worker','E-purpose':'purpose_worker'}[design['study']])
            child_started=time.time();clock=None;child_dir=root/'neural';child_dir.mkdir(exist_ok=True)
            with (root/'neural.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([str(python),'-B','-m',module,
                    '--inputs',str(root/'data/reader/INPUTS.json'),'--output',str(child_dir),'--config',str(root/'TRAINING.json')],
                    cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while child.poll() is None:
                    status=read(child_dir/'STATUS.json') if (child_dir/'STATUS.json').exists() else {}
                    if status.get('heartbeat') and datetime.fromisoformat(status['heartbeat']).timestamp()>=child_started:
                        if clock is None and os.name=='nt' and (status['pid']==child.pid or status.get('parent_pid')==child.pid):
                            try:clock=ProcessClock(status['pid'])
                            except OSError:pass
                        child_cpu=max(child_cpu,status.get('cpu_seconds',0),clock.seconds() if clock else 0)
                    if limited():(child_dir/'STOP').touch()
                    emit(phase='waiting-for-CPU-child',child_pid=child.pid,child_cpu_seconds=child_cpu)
                    time.sleep(2)
            status=read(child_dir/'STATUS.json') if (child_dir/'STATUS.json').exists() else {}
            if clock:
                child_cpu=max(child_cpu,clock.seconds());clock.close()
            elif status.get('state')=='complete':child_cpu=max(child_cpu,status.get('cpu_seconds',0))
            else:child_cpu=max(child_cpu,time.time()-child_started)
            write(root/f'CHILD-{attempt}.json',dict(pid=child.pid,exit_code=child.returncode,cpu_seconds=child_cpu,
                accounting='retained native handle' if clock else 'self report on complete; conservative wall charge otherwise'))
            emit(phase='child-exited')
            if child.returncode!=0:raise RuntimeError('CPU child failed; retained log and checkpoint')
            if status.get('state')!='complete':emit('resource_cutoff');return 'resource_cutoff'
            pulse(phase='independent-scoring')
            if design['study']=='E':score_E(root,pulse)
            elif design['study']=='G':
                from .intervention_data import score
                score(root,pulse)
            else:
                from .purpose_data import score
                score(root,pulse)
            files={p.relative_to(root).as_posix():file_digest(p) for p in root.rglob('*') if p.is_file()
                and p.suffix in ('.npz','.gz','.pt','.json') and p.name not in ('STATUS.json','CURRENT.json','COMPLETE.json')}
            files['neural/COMPLETE.json']=file_digest(root/'neural/COMPLETE.json')
            write(root/'COMPLETE.json',dict(files=files,completed_at=now(),validity='scoped numerical, input and retained prediction checks; source replay still required'))
            emit('complete');return 'complete'
        except TimeoutError as exc:
            emit('resource_cutoff',error=repr(exc));return 'resource_cutoff'
        except BaseException as exc:
            if child is not None and child.poll() is None:
                (root/'neural/STOP').touch()
                # Preserve running accounting: a live child forbids any replacement owner.
                emit('running',error=repr(exc),reconciliation_required=True)
            else:emit('failed',error=repr(exc))
            raise
