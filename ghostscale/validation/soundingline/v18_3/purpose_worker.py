"""Frozen memory extraction and equal-supervision role readout on CPU."""
from . import torch_worker as T,purpose_readout as R
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch


def dataset(path,prefix='',public=False):
    with np.load(path,allow_pickle=False) as z:
        keys={k[len(prefix):] for k in z.files if k.startswith(prefix)}
        required={'history','length','world'};allowed=required if public else required|{'target'}
        if not required<=keys or not keys<=allowed:raise ValueError('hidden role truth or undeclared public field')
        data={k:z[prefix+k].copy() for k in keys}
    if 'target' in data and (np.any(data['target']<0) or not np.allclose(data['target'].sum(1),1,atol=1e-7)):
        raise ValueError('invalid supervised role distribution')
    return data


def features(data,best=None):
    if best is None:return np.concatenate((data['history'].reshape(len(data['history']),-1),data['world']),axis=1).astype(float)
    saved=torch.load(best,weights_only=True,map_location='cpu');model=T.Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval();states=[]
    with torch.no_grad():
        for first in range(0,len(data['history']),128):
            h=torch.from_numpy(data['history'][first:first+128]);length=torch.from_numpy(data['length'][first:first+128])
            states.append(model.encode(h,length).numpy())
    return np.concatenate((np.concatenate(states),data['world']),axis=1).astype(float)


def run(inputs,output,config_path):
    output.mkdir(parents=True,exist_ok=True);manifest=T.read(inputs);config=T.read(config_path);base=inputs.parent
    environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if environment!=config['environment'] or torch.cuda.is_initialized():raise ValueError('purpose CPU environment differs')
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    identity=dict(input_sha256=T.sha(inputs),config_sha256=T.sha(config_path),source_sha256=T.sha(__file__),readout_sha256=T.sha(R.__file__))
    if (output/'IDENTITY.json').exists() and T.read(output/'IDENTITY.json')!=identity:raise ValueError('purpose identity changed')
    T.write(output/'IDENTITY.json',identity)
    def pulse(**detail):T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='running',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU,**detail))
    def limited():return datetime.now(timezone.utc)>=datetime.fromisoformat(config['report_start']) or time.process_time()-T.START_CPU>=min(config['cpu_ceiling_seconds'],float(os.environ.get('GHOST_V18_CHILD_CPU_LIMIT','inf'))) or (output/'STOP').exists()
    def check():
        if limited():raise TimeoutError('purpose resource cutoff')
    pulse();check()
    for item in (manifest['train'],*manifest['tests'].values(),*manifest['encoders'].values()):
        if T.sha(T.confined(base,item['name']))!=item['sha256']:raise ValueError('purpose capsule changed')
    train=dataset(base/manifest['train']['name'],'train_');dev=dataset(base/manifest['train']['name'],'dev_')
    tests={name:dataset(base/entry['name'],public=True) for name,entry in manifest['tests'].items()}
    fits={};predictions={}
    for name,entry in {'raw-history':None,**manifest['encoders']}.items():
        check();pulse(phase='extracting-frozen-memory',reader=name);started=time.process_time()
        best=None if entry is None else base/entry['name'];folder=output/name;folder.mkdir(exist_ok=True)
        train_x=features(train,best);dev_x=features(dev,best);test_x={condition:features(data,best) for condition,data in tests.items()}
        extraction_cpu=time.process_time()-started;check()
        if (folder/'FIT.json').exists():
            receipt=T.read(folder/'FIT.json')
            if T.sha(folder/'READOUT.npz')!=receipt['readout_sha256'] or receipt['identity']!=identity:raise ValueError('completed purpose fit changed')
            with np.load(folder/'READOUT.npz',allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
        else:
            started=time.process_time();model,receipt=R.fit(train_x,train['target'],dev_x,dev['target'],config['alphas'],config['temperatures'])
            receipt.update(fit_cpu_seconds=time.process_time()-started,extraction_cpu_seconds=extraction_cpu,identity=identity)
            np.savez_compressed(folder/'READOUT.npz',**model);receipt['readout_sha256']=T.sha(folder/'READOUT.npz');T.write(folder/'FIT.json',receipt)
        fits[name]=receipt;predictions[name]={}
        for condition,x in test_x.items():
            check();p=R.predict(model,x);path=output/f'{name}-{condition}-PREDICTIONS.npz';np.savez_compressed(path,probabilities=p)
            predictions[name][condition]=dict(file=path.name,sha256=T.sha(path),rows=len(p))
        pulse(completed_reader=name)
    T.write(output/'COMPLETE.json',dict(identity=identity,environment=environment,fits=fits,predictions=predictions,cpu_seconds=time.process_time()-T.START_CPU,cuda_initialized=torch.cuda.is_initialized()))
    T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='complete',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from ..v16.runtime import local_owner
    with local_owner(Path(os.environ['GHOST_V18_CHILD_LOCK'])):
        try:run(a.inputs,a.output,a.config)
        except BaseException as exc:
            T.write(a.output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='resource_cutoff' if isinstance(exc,TimeoutError) else 'failed',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU,error=repr(exc)))
            if not isinstance(exc,TimeoutError):raise
