"""Reuse the existing single parent/waiting CPU-child pattern for V19 capability."""
from pathlib import Path
import os,subprocess,time,gzip,shutil
import numpy as np
from ..v18_3.io import read,write,file_digest,canonical
from ..v18_3.native import ProcessClock
from ..v16.runtime import local_owner
from .readout_model import save_arrays

def prepare(root,cfg):
    data=root/'data';reader=data/'reader';reader.mkdir(parents=True,exist_ok=True)
    manifest=dict(train={},development={});sources=root/'inputs';means={}
    for draw in cfg['training_draws']:
        labels=[]
        for split in ('train','development'):
            manifest[split][str(draw)]={}
            for condition in ('independent','copied'):
                name=f'{split}-{draw}-{condition}.npz'
                with np.load(sources/'reader'/name,allow_pickle=False) as z:
                    values={k:z[k] for k in ('codes','novel')}
                    if split=='train':
                        values['target']=z['target'];labels.append(z['target'][z['novel']])
                    save_arrays(reader/name,**values)
                manifest[split][str(draw)][condition]=dict(file=name,sha256=file_digest(reader/name))
        means[draw]=np.concatenate(labels).mean(0)
    write(reader/'INPUTS.json',manifest);return means

def score(root,cfg,means):
    rows=[];capability=[];child=read(root/'neural/RESULT.json')
    for draw in cfg['training_draws']:
        for condition in ('independent','copied'):
            with np.load(root/'inputs/evaluator'/f'development-{draw}-{condition}.npz',allow_pickle=False) as z:
                target=z['target'].reshape(-1,32,4,8);ids=z['ids']
            n=len(target);pred={'no-history':np.broadcast_to(means[draw].reshape(1,1,4,8),target.shape),'exact-filter':target}
            for seed in cfg['fit_seeds']:
                for kind in cfg['kinds']:
                    with np.load(root/'neural'/f'{draw}-{seed}-{kind}'/f'{condition}.npz',allow_pickle=False) as z:pred[f'{kind}-{seed}']=z['probabilities'].astype(float)
            for method,prob in pred.items():
                if np.any(prob<0) or not np.isfinite(prob).all() or not np.allclose(prob.sum(-1),1,atol=1e-6):raise ValueError('bad saved probability')
                # Literal zero on a possible outcome is infinite, never silently floored.
                logs=np.zeros_like(prob);np.log(prob,out=logs,where=prob>0)
                loss=-(target*logs).sum(-1).mean(-1);loss[np.any((target>0)&(prob==0),axis=(2,3))]=np.inf
                brier=((target-prob)**2).sum(-1).mean(-1)
                if not np.isfinite(loss).all():raise ValueError('nonfinite categorical score')
                for length in cfg['lengths']:
                    for lineage in cfg['development_lineages']:
                        chosen=ids[:,0]==lineage
                        rows.append(dict(draw=draw,condition=condition,method=method,length=length,lineage=lineage,streams=int(chosen.sum()),loss=float(loss[chosen,length-1].mean()),brier=float(brier[chosen,length-1].mean()),prediction_std=float(prob[chosen,length-1].std(0).mean())))
            for seed in cfg['fit_seeds']:
                for kind in cfg['kinds']:
                    for length in cfg['lengths']:
                        def mean(method,key='loss'):return float(np.mean([r[key] for r in rows if (r['draw'],r['condition'],r['length'],r['method'])==(draw,condition,length,method)]))
                        no=mean('no-history');exact=mean('exact-filter');learned=mean(f'{kind}-{seed}');gain=no-learned;headroom=no-exact;spread=mean(f'{kind}-{seed}','prediction_std')
                        capability.append(dict(draw=draw,seed=seed,kind=kind,condition=condition,length=length,loss=learned,mean_loss=no,exact_loss=exact,gain=gain,headroom=headroom,prediction_std=spread,passed=bool(headroom>0 and gain>=.02 and gain>=.5*headroom and spread>1e-5)))
    (root/'raw').mkdir(exist_ok=True);(root/'raw/tiny_points.json.gz').write_bytes(gzip.compress(canonical(rows),mtime=0))
    return dict(controls=dict(public_fields=True,normalized_forecasts=True,finite_scores=True,child_controls=all(all(v[k] for k in ('causal_prefix','no_op_decode','duplicate_passthrough','normalized','learning_passed','uniform_null')) for v in child['controls'].values())),capability=capability,rows=len(rows),fits=len(child['fits']),settings=2,scope='observable fresh-episode capability pilot; no local-process primary or causal-interchange admission',warrant='exploratory constructed method; miniature — architecture untested')

def run(root,plan,pulse):
    from .runtime import REPO
    cfg=plan['design'];campaign=Path(plan['campaign'])
    owner=read(campaign/'TINY_TRAINING_CONFIG.json');receipt=read(campaign/'TINY_SETTINGS.json')
    if receipt['owner']!='Ghost' or receipt['settings']!=cfg['settings'] or file_digest(campaign/'TINY_SETTINGS.json')!=cfg['settings_receipt_sha256']:raise ValueError('shared setting ownership mismatch')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('frozen tiny input changed')
    with local_owner(campaign/'neural-child-owner'):pass
    means=prepare(root,cfg);acceptance=read(campaign/'ACCEPTANCE.json')
    config=dict(kinds=cfg['kinds'],draws=cfg['training_draws'],seeds=cfg['fit_seeds'],epochs=cfg['epochs'],batch_size=32,learning_rate=.001,weight_decay=.0001,gradient_clip=1.,torch_version=cfg['torch_version'],numpy_version=cfg['torch_numpy_version'],deadline=acceptance['deadline'])
    write(root/'FIT_CONFIG.json',config)
    python=Path(owner['python']).resolve()
    if not python.is_file() or file_digest(python)!=owner['python_sha256']:raise ValueError('installed interpreter changed')
    childdir=root/'neural';childdir.mkdir(exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    env['GHOST_V19_CHILD_CPU_LIMIT']=str(max(0,pulse(phase='before-child')-60))
    clock=None;child_cpu=0.;began=time.monotonic();error=None
    with (root/'neural.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([str(python),'-B','-m','ghostscale.validation.soundingline.v19.tiny_worker','--inputs',str(root/'data/reader/INPUTS.json'),'--output',str(childdir),'--config',str(root/'FIT_CONFIG.json'),'--campaign',str(campaign)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        while child.poll() is None:
            status=read(childdir/'STATUS.json') if (childdir/'STATUS.json').exists() else {}
            if clock is None and status.get('pid') and (status['pid']==child.pid or status.get('parent_pid')==child.pid):
                try:clock=ProcessClock(status['pid'])
                except OSError:pass
            child_cpu=max(child_cpu,status.get('cpu_seconds',0),clock.seconds() if clock else 0)
            try:pulse(phase='waiting-for-CPU-child',child_pid=child.pid,child_cpu_seconds=child_cpu)
            except BaseException as exc:
                error=exc;(childdir/'STOP').touch()
            # Native wait only; the parent does no concurrent numerical work.
            time.sleep(1)
    status=read(childdir/'STATUS.json') if (childdir/'STATUS.json').exists() else {}
    child_cpu=max(child_cpu,clock.seconds() if clock else status.get('cpu_seconds',0))
    if clock:clock.close()
    elif status.get('state')!='complete':child_cpu=max(child_cpu,time.monotonic()-began)
    write(root/'CHILD_ACCOUNTING.json',dict(pid=child.pid,exit_code=child.returncode,cpu_seconds=child_cpu,native_handle=bool(clock),parent_waited=True))
    pulse(phase='child-exited',child_cpu_seconds=child_cpu)
    if error:raise error
    if child.returncode or status.get('state')!='complete':raise RuntimeError('tiny child failed; log and checkpoints retained')
    return score(root,cfg,means)
