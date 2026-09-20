"""One CPU child extracts frozen states and fits a bounded decoder ladder."""
from . import torch_worker as T,decoder_readout as R
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch


def features(data,best=None,question_only=False):
    if question_only:state=np.empty((len(data['history']),0))
    elif best is None:state=data['history'].numpy().reshape(len(data['history']),-1).astype(float)
    else:
        saved=torch.load(best,map_location='cpu',weights_only=True)
        model=T.Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval();states=[]
        with torch.no_grad():
            for first in range(0,len(data['history']),128):
                states.append(model.encode(data['history'][first:first+128],data['length'][first:first+128]).numpy())
        state=np.concatenate(states).astype(float)
    return np.concatenate((state[data['sample'].numpy()],data['query'].numpy()),axis=1)


def forecast(best,data,path,base,identity):
    with np.load(best,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    entry=identity['encoder'];encoder=None if entry is None else T.confined(base,entry['name'])
    if encoder is not None and T.sha(encoder)!=entry['sha256']:raise ValueError('frozen forecast encoder changed')
    x=features(data,encoder,identity['representation']=='question-only');p=R.predict(model,x)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-8):raise ValueError('invalid probe forecast')
    np.savez_compressed(path,probabilities=p)
    return dict(rows=len(p),sha256=T.sha(path))


def run(inputs,output,config_path):
    output.mkdir(parents=True,exist_ok=True);manifest=T.read(inputs);config=T.read(config_path);base=inputs.parent
    from .priority import below_normal
    below_normal()
    environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if environment!=config['environment'] or torch.cuda.is_initialized():raise ValueError('decoder CPU environment differs')
    identity=dict(input_sha256=T.sha(inputs),config_sha256=T.sha(config_path),source_sha256=T.sha(__file__),readout_sha256=T.sha(R.__file__))
    if (output/'IDENTITY.json').exists() and T.read(output/'IDENTITY.json')!=identity:raise ValueError('decoder identity changed')
    T.write(output/'IDENTITY.json',identity)
    def pulse(**detail):T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='running',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU,**detail))
    def check():
        if datetime.now(timezone.utc)>=datetime.fromisoformat(config['report_start']) or time.process_time()-T.START_CPU>=min(config['cpu_ceiling_seconds'],float(os.environ.get('GHOST_V18_CHILD_CPU_LIMIT','inf'))) or (output/'STOP').exists():
            raise TimeoutError('decoder resource cutoff')
    pulse();check()
    for entry in (manifest['train'],*manifest['tests'].values(),*manifest['encoders'].values()):
        if T.sha(T.confined(base,entry['name']))!=entry['sha256']:raise ValueError('decoder capsule changed')
    controls=R.controls();T.write(output/'CONTROLS.json',controls)
    if not all(controls.values()):raise ValueError('decoder known-answer control failed')
    train=T.dataset(base/manifest['train']['name'],'train_');dev=T.dataset(base/manifest['train']['name'],'dev_')
    tests={name:T.dataset(base/entry['name']) for name,entry in manifest['tests'].items()}
    fits={};predictions={};train_per_cell=manifest['train_histories']//16
    for representation,entry in {'raw-history-seed0':None,'question-only-seed0':None,**manifest['encoders']}.items():
        check();pulse(representation=representation);started=time.process_time()
        method,seed=representation.rsplit('-seed',1);seed=int(seed)
        best=None if entry is None else T.confined(base,entry['name'])
        x=features(train,best,method=='question-only');v=features(dev,best,method=='question-only')
        test_x={name:features(data,best,method=='question-only') for name,data in tests.items()}
        extraction=time.process_time()-started
        for per_cell in config['label_budgets']:
            if per_cell>train_per_cell or per_cell%128:raise ValueError('budget must preserve the full crossed query rotation')
            rows=np.flatnonzero(train['sample'].numpy()%train_per_cell<per_cell)
            y=train['target'].numpy()[rows];query=train['query'].numpy()[rows]
            for kind in ('linear','nonlinear'):
                for control in ('aligned','permuted'):
                    check();name=f'{method}-{kind}-n{per_cell}-{control}-seed{seed}';folder=output/name;folder.mkdir(exist_ok=True)
                    fit_id=dict(run=identity,encoder=entry,representation=method,kind=kind,per_cell=per_cell,control=control,seed=seed)
                    if (folder/'COMPLETE.json').exists():
                        receipt=T.read(folder/'COMPLETE.json')
                        if receipt['identity']!=fit_id or T.sha(folder/'READOUT.npz')!=receipt['readout_sha256']:raise ValueError('decoder fit changed')
                        with np.load(folder/'READOUT.npz',allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
                    else:
                        # No test outcome enters fitting. Permutation is within question context.
                        labels=y if control=='aligned' else R.permute_targets(y,query,manifest['query_context_features'],180404)
                        dev_labels=dev['target'].numpy()
                        if control=='permuted':dev_labels=R.permute_targets(dev_labels,dev['query'].numpy(),manifest['query_context_features'],180405)
                        started=time.process_time()
                        model,receipt=R.fit(x[rows],labels,v,dev_labels,kind,seed=180404,width=config['random_features'],
                            alphas=config['alphas'],temperatures=config['temperatures'],check=check)
                        receipt.update(identity=fit_id,fit_cpu_seconds=time.process_time()-started,extraction_cpu_seconds=extraction,
                            learning_control_passed=True,learning_control_scope='known-answer controls; no scientific ranking or improvement gate')
                        np.savez_compressed(folder/'READOUT.npz',**model);receipt['readout_sha256']=T.sha(folder/'READOUT.npz')
                        T.write(folder/'COMPLETE.json',receipt)
                    fits[name]=receipt;predictions[name]={}
                    for condition,a in test_x.items():
                        check();p=R.predict(model,a);path=output/f'{name}-{condition}-PREDICTIONS.npz';np.savez_compressed(path,probabilities=p)
                        predictions[name][condition]=dict(selected=name,file=path.name,sha256=T.sha(path),rows=len(p))
                    pulse(completed_reader=name)
    T.write(output/'COMPLETE.json',dict(identity=identity,environment=environment,fits=fits,predictions=predictions,benchmarks={},cpu_seconds=time.process_time()-T.START_CPU,cuda_initialized=torch.cuda.is_initialized()))
    T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='complete',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from ..v16.runtime import local_owner
    with local_owner(Path(os.environ['GHOST_V18_CHILD_LOCK'])):
        try:run(a.inputs,a.output,a.config)
        except BaseException as exc:
            T.write(a.output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='resource_cutoff' if isinstance(exc,TimeoutError) else 'failed',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU,error=repr(exc)))
            if not isinstance(exc,TimeoutError):raise
