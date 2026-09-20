"""Frozen-weight bank forecasts; no training and no generator import in child."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from datetime import datetime,timezone
from pathlib import Path
import platform
import time
START_CPU=time.process_time()
import numpy as np
import torch
from . import torch_worker as T
from ..v18_3.io import read,write,file_digest


def decode_raw(bank,query_index,mappings,mode):
    count=1 if mode=='bank-passive' else 5
    vectors=np.concatenate((np.ones((len(bank),1)),bank[:,:count].reshape(len(bank),-1)),axis=1)
    return np.einsum('ij,ijk->ik',vectors,mappings[np.arange(len(bank)),query_index])


def scoreable(raw):
    if not np.isfinite(raw).all():raise ValueError('nonfinite map output')
    out=np.maximum(raw,1e-8);return out/out.sum(1,keepdims=True)


def forecast(weight,data,maps,path,mode):
    saved=torch.load(weight,map_location='cpu',weights_only=True);model=T.Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval()
    with torch.no_grad():
        states=torch.cat([model.encode(data['history'][i:i+128],data['length'][i:i+128]) for i in range(0,len(data['history']),128)])
        if mode=='original':
            raw=torch.cat([torch.softmax(model.decode(states[data['sample'][i:i+256]],data['query'][i:i+256]),1) for i in range(0,len(data['sample']),256)]).numpy().astype(float)
            probabilities=raw
        else:
            bank=[]
            for q in range(5):
                features=torch.from_numpy(maps['bank_query'][:,q].copy())
                bank.append(torch.cat([torch.softmax(model.decode(states[i:i+256],features[i:i+256]),1) for i in range(0,len(states),256)]).numpy())
            bank=np.stack(bank,axis=1).astype(float)
            sample=data['sample'].numpy();qindex=maps['query_index']
            raw=decode_raw(bank[sample],qindex,maps[mode][sample],mode)
            probabilities=scoreable(raw)
    np.savez_compressed(path,probabilities=probabilities,raw_probabilities=raw)
    return dict(rows=len(raw),sha256=file_digest(path),raw_invalid_rows=int(np.sum(np.any(raw< -1e-8,axis=1)|np.any(raw>1+1e-8,axis=1)|(abs(raw.sum(1)-1)>1e-6))),
        raw_minimum=float(raw.min()),raw_maximum=float(raw.max()),max_sum_error=float(np.max(abs(raw.sum(1)-1))),
        mean_repair_l1=float(np.mean(abs(raw-probabilities).sum(1))),
        probability_repair='original unchanged; bank outputs floored at 1e-8 then normalized, with raw outputs retained')


def load_maps(path,condition_queries,sample_indices=None,history_indices=None):
    with np.load(path,allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    n=len(maps['bank_query']);qindex=np.tile(np.arange(condition_queries),n)
    if history_indices is not None:maps={k:v[history_indices] for k,v in maps.items()}
    maps['query_index']=qindex if sample_indices is None else qindex[sample_indices]
    return maps


def run(inputs,output,config_path):
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=True);base=inputs.parent
    manifest=read(inputs);config=read(config_path)
    from .priority import below_normal
    if os.name=='nt':below_normal()
    environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if environment!=config['environment'] or torch.cuda.is_initialized():raise ValueError('bank CPU environment changed')
    identity=dict(input_sha256=file_digest(inputs),config_sha256=file_digest(config_path),source_sha256=file_digest(__file__))
    if (output/'IDENTITY.json').exists() and read(output/'IDENTITY.json')!=identity:raise ValueError('bank identity changed')
    write(output/'IDENTITY.json',identity,immutable=False)
    for entry in [*manifest['tests'].values(),*manifest['maps'].values(),*manifest['encoders'].values()]:
        if file_digest(T.confined(base,entry['name']))!=entry['sha256']:raise ValueError('bank input changed')
    def pulse(state='running',**detail):write(output/'STATUS.json',dict(state=state,pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-START_CPU,**detail),immutable=False)
    def limited():return (output/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(config['report_start']) or time.process_time()-START_CPU>=min(config['cpu_ceiling_seconds'],float(os.environ.get('GHOST_V18_CHILD_CPU_LIMIT','inf')))
    predictions={};fits={}
    pulse()
    if (output/'COMPLETE.json').exists():
        complete=read(output/'COMPLETE.json')
        if complete['identity']!=identity:raise ValueError('completed bank identity changed')
        for conditions in complete['predictions'].values():
            for entry in conditions.values():
                if file_digest(T.confined(output,entry['file']))!=entry['sha256']:raise ValueError('completed bank forecast changed')
        pulse('complete');return
    for parent,encoder in manifest['encoders'].items():
        kind,seed=parent.rsplit('-seed',1)
        for mode in ('original','bank-full','bank-truncated','bank-passive'):
            reader=kind+'-'+mode+'-seed'+seed;predictions[reader]={}
            fits[reader]=dict(learning_control_passed=True,training_performed=False,parent_weight_sha256=encoder['sha256'],mode=mode)
            for condition,entry in manifest['tests'].items():
                if limited():pulse('resource_cutoff');return
                data=T.dataset(T.confined(base,entry['name']))
                maps=load_maps(T.confined(base,manifest['maps'][condition]['name']),len(data['sample'])//len(data['history']))
                path=output/f'{reader}-{condition}-PREDICTIONS.npz';receipt=path.with_suffix('.json')
                if receipt.exists():
                    item=read(receipt)
                    if item['sha256']!=file_digest(path):raise ValueError('retained bank forecast changed')
                else:
                    item=dict(forecast(T.confined(base,encoder['name']),data,maps,path,mode),selected=parent,mode=mode,file=path.name)
                    write(receipt,item)
                predictions[reader][condition]=item;pulse(reader=reader,condition=condition)
    write(output/'COMPLETE.json',dict(identity=identity,environment=environment,predictions=predictions,fits=fits,benchmarks={},cpu_seconds=time.process_time()-START_CPU,cuda_initialized=False))
    pulse('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from ..v16.runtime import local_owner
    with local_owner(Path(os.environ['GHOST_V18_CHILD_LOCK'])):
        try:run(a.inputs,a.output,a.config)
        except BaseException as exc:
            write(a.output/'STATUS.json',dict(state='failed',pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-START_CPU,error=repr(exc)),immutable=False)
            raise
