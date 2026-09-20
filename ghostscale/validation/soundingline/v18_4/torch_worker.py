"""Isolated CPU neural worker: serialized public inputs, no generator import.

Sounding Line's immutable identity, development-selection and resumable optimizer
patterns are adapted to small Ghost-owned models. No model service or GPU access.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import time
START_CPU=time.process_time()
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
from torch import nn
from ..v18_3.io import replace,write as durable_write

torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def write(path,data):
    return durable_write(path,data,immutable=False)


def confined(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('input path escapes public capsule')
    return path


def count(model):return sum(p.numel() for p in model.parameters())


class Reader(nn.Module):
    def __init__(self,kind,input_size,query_size,width,head=64,direct_hidden=8,max_history=32):
        super().__init__();self.kind=kind;self.width=width;self.spec=dict(kind=kind,input_size=input_size,
            query_size=query_size,width=width,head=head,direct_hidden=direct_hidden,max_history=max_history)
        if kind=='flat':self.encoder=nn.GRU(input_size,width,batch_first=True)
        elif kind=='split':
            if width%3:raise ValueError('three equal declared role slots required')
            self.encoders=nn.ModuleList(nn.GRU(input_size,width//3,batch_first=True) for _ in range(3))
        elif kind in ('direct','question-only'):
            self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(max_history*input_size,direct_hidden),nn.Tanh(),nn.Linear(direct_hidden,width),nn.Tanh())
        else:raise ValueError('unknown reader')
        self.decoder=nn.Sequential(nn.Linear(width+query_size,head),nn.Tanh(),nn.Linear(head,16))

    def encode(self,h,length):
        if self.kind in ('direct','question-only'):
            # The no-history rival retains the same nominal architecture/budget.
            # Every history entry, including length/source flags, is inaccessible.
            return self.encoder(torch.zeros_like(h) if self.kind=='question-only' else h)
        encoders=self.encoders if self.kind=='split' else [self.encoder]
        return torch.cat([encoder(h)[0][torch.arange(len(h)),length-1] for encoder in encoders],dim=1)

    def decode(self,state,q):return self.decoder(torch.cat((state,q),dim=1))
    def forward(self,h,length,q):return self.decode(self.encode(h,length),q)


def architecture(kind,input_size,query_size,width):
    # Equal target parameter budgets, actual integer differences are reported.
    reference=count(Reader('flat',input_size,query_size,width))
    if kind=='split':
        candidates=[Reader(kind,input_size,query_size,width,head=h) for h in range(32,161,4)]
    elif kind in ('direct','question-only'):
        candidates=[Reader(kind,input_size,query_size,width,direct_hidden=h) for h in range(2,33)]
    else:return Reader(kind,input_size,query_size,width)
    return min(candidates,key=lambda m:abs(count(m)-reference))


def dataset(path,prefix=''):
    with np.load(path,allow_pickle=False) as z:
        required={'history','length','sample','query'}
        available={name[len(prefix):] for name in z.files if name.startswith(prefix)}
        if not required<=available:raise ValueError('incomplete public neural input')
        allowed=required|{'world','target'}
        if not available<=allowed:raise ValueError('undeclared neural input field')
        result={name:torch.from_numpy(z[prefix+name].copy()) for name in available}
    if result['history'].ndim!=3 or result['query'].ndim!=2:raise ValueError('bad feature shape')
    if len(result['sample'])!=len(result['query']):raise ValueError('bad query alignment')
    if torch.any(result['length']<=0) or torch.any(result['length']>result['history'].shape[1]):raise ValueError('invalid history support')
    if torch.any(result['sample']<0) or torch.any(result['sample']>=len(result['history'])):raise ValueError('bad history index')
    for name in ('history','query'):
        if not torch.isfinite(result[name]).all():raise ValueError('nonfinite input')
    if 'target' in result:
        p=result['target']
        if torch.any(p<0) or not torch.allclose(p.sum(1),torch.ones(len(p)),atol=1e-6,rtol=0):raise ValueError('invalid supervision')
    return result


def batch(data,indices):
    history_index=data['sample'][indices]
    return data['history'][history_index],data['length'][history_index],data['query'][indices]


def loss(logits,target):return -(target*torch.log_softmax(logits,dim=1)).sum(1).mean()


def evaluate(model,data,batch_size=256):
    model.eval();total=0.;n=len(data['sample'])
    with torch.no_grad():
        for first in range(0,n,batch_size):
            indices=torch.arange(first,min(n,first+batch_size))
            value=loss(model(*batch(data,indices)),data['target'][indices])
            if not torch.isfinite(value):raise ValueError('nonfinite evaluation loss')
            total+=float(value)*len(indices)
    return total/n


def positive_control(kind,input_size,query_size,width):
    torch.manual_seed(180300)
    model=architecture(kind,input_size,query_size,width)
    h=torch.zeros(32,32,input_size);h[:,0,:16]=torch.eye(16).repeat(2,1)
    length=torch.ones(32,dtype=torch.long);q=torch.zeros(32,query_size);target=torch.eye(16).repeat(2,1)
    if kind=='question-only':
        if query_size<16:raise ValueError('query-only identity control needs sixteen query features')
        q[:,:16]=target
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    initial=float(loss(model(h,length,q),target).detach())
    for _ in range(100):
        optimizer.zero_grad();value=loss(model(h,length,q),target);value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('invalid positive-control gradient')
        optimizer.step()
    final=float(loss(model(h,length,q),target).detach())
    return dict(initial_loss=initial,final_loss=final,passed=final<.5,parameters=count(model))


def fit(root,identity,train,dev,config,pulse,limited):
    fit_cpu=time.process_time()
    root.mkdir(parents=True,exist_ok=True)
    identity_hash=hashlib.sha256(encoded(identity)).hexdigest()
    if (root/'IDENTITY.json').exists() and read(root/'IDENTITY.json')!=identity:raise ValueError('fit identity changed')
    write(root/'IDENTITY.json',identity)
    if (root/'COMPLETE.json').exists():
        receipt=read(root/'COMPLETE.json')
        if receipt['identity_sha256']!=identity_hash or sha(root/'BEST.pt')!=receipt['best_sha256']:raise ValueError('completed fit changed')
        return receipt
    torch.manual_seed(identity['seed'])
    model=architecture(identity['kind'],train['history'].shape[2],train['query'].shape[1],identity['width'])
    # Architecture search consumes RNG; reset actual parameters from a fixed seed.
    torch.manual_seed(identity['seed']);model=Reader(**model.spec)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001)
    progress=dict(epoch=0,offset=0,steps=0,best_loss=1e100,curve=[],initial_train_loss=evaluate(model,train))
    if (root/'CURRENT.json').exists():
        receipt=read(root/'CURRENT.json');path=confined(root,receipt['file'])
        if sha(path)!=receipt['sha256']:raise ValueError('checkpoint hash changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['identity_sha256']!=identity_hash:raise ValueError('checkpoint recipe changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']);progress=state['progress']
    n=len(train['sample'])
    def save():
        progress['checkpoint_generation']=progress.get('checkpoint_generation',0)+1
        name=f"checkpoint-{progress['checkpoint_generation']%2}.pt";path=root/name
        temporary=path.with_suffix('.tmp')
        torch.save(dict(identity_sha256=identity_hash,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        rng=torch.get_rng_state(),progress=progress),temporary)
        replace(temporary,path)
        write(root/'CURRENT.json',dict(file=name,sha256=sha(path),steps=progress['steps']))
    while progress['epoch']<config['epochs']:
        generator=torch.Generator().manual_seed(identity['seed']+progress['epoch'])
        order=torch.randperm(n,generator=generator);model.train()
        while progress['offset']<n:
            if limited():save();return None
            indices=order[progress['offset']:progress['offset']+config['batch_size']]
            optimizer.zero_grad();value=loss(model(*batch(train,indices)),train['target'][indices])
            if not torch.isfinite(value):save();raise ValueError('nonfinite training loss retained')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            progress['steps']+=1;progress['offset']+=len(indices)
            if progress['steps']%50==0:save();pulse(fit=root.name,steps=progress['steps'])
        dev_loss=evaluate(model,dev)
        progress['curve'].append(dict(epoch=progress['epoch'],dev_loss=dev_loss))
        if dev_loss<progress['best_loss']:
            progress['best_loss']=dev_loss
            temporary=root/'BEST.tmp';torch.save(dict(spec=model.spec,state=model.state_dict(),identity_sha256=identity_hash),temporary)
            replace(temporary,root/'BEST.pt')
        progress['epoch']+=1;progress['offset']=0;save();pulse(fit=root.name,epoch=progress['epoch'])
    final_train=evaluate(model,train)
    receipt=dict(identity_sha256=identity_hash,best_sha256=sha(root/'BEST.pt'),parameters=count(model),
                 initial_train_loss=progress['initial_train_loss'],final_train_loss=final_train,
                 learning_control_passed=final_train<progress['initial_train_loss']-.02,
                 best_dev_loss=progress['best_loss'],curve=progress['curve'],steps=progress['steps'],spec=model.spec,
                 final_attempt_cpu_seconds=time.process_time()-fit_cpu)
    write(root/'COMPLETE.json',receipt);return receipt


def forecast(best,data,path):
    if 'target' in data:raise ValueError('test truth in reader inputs')
    saved=torch.load(best,map_location='cpu',weights_only=True);model=Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval()
    start_cpu=time.process_time();start_wall=time.perf_counter();outputs=[]
    with torch.no_grad():
        # Every reader, including direct history, may cache its current encoding.
        states=torch.cat([model.encode(data['history'][i:i+128],data['length'][i:i+128]) for i in range(0,len(data['history']),128)])
        for first in range(0,len(data['sample']),256):
            indices=torch.arange(first,min(len(data['sample']),first+256))
            logits=model.decode(states[data['sample'][indices]],data['query'][indices])
            outputs.append(torch.softmax(logits,dim=1).numpy())
    p=np.concatenate(outputs)
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('invalid forecast')
    np.savez_compressed(path,probabilities=p)
    return dict(rows=len(p),sha256=sha(path),cpu_seconds=time.process_time()-start_cpu,
                wall_seconds=time.perf_counter()-start_wall,state_floats=model.width,
                history_floats=int(np.prod(data['history'].shape[1:])))


def update_costs(best,data):
    saved=torch.load(best,map_location='cpu',weights_only=True);model=Reader(**saved['spec']);model.load_state_dict(saved['state']);model.eval()
    h=data['history'][:64];rows=[]
    with torch.no_grad():
        for n in (8,16,32):
            prefix=h.clone();prefix[:,n:]=0;length=torch.full((len(h),),n,dtype=torch.long)
            start=time.perf_counter()
            for _ in range(20):full=model.encode(prefix,length)
            full_seconds=(time.perf_counter()-start)/(20*len(h))
            if model.kind in ('direct','question-only'):incremental_seconds=full_seconds;error=0.
            else:
                encoders=model.encoders if model.kind=='split' else [model.encoder]
                previous=[e(prefix[:,:n-1])[1] for e in encoders]
                start=time.perf_counter()
                for _ in range(20):updated=torch.cat([e(prefix[:,n-1:n],s)[1][0] for e,s in zip(encoders,previous)],1)
                incremental_seconds=(time.perf_counter()-start)/(20*len(h));error=float(torch.max(abs(updated-full)))
                if error>2e-6:raise ValueError('incremental recurrent state differs')
            rows.append(dict(history_length=n,full_encoding_seconds_per_history=full_seconds,
                update_seconds_per_history=incremental_seconds,incremental_max_error=error,
                maintained_floats=(0 if model.kind=='question-only' else model.width if model.kind!='direct' else int(np.prod(h.shape[1:])))))
    return dict(rows=rows,repetitions=20,batch_histories=len(h),scope='single-thread CPU microbenchmark; training and query decoding reported separately')


def run(inputs,output,config_path):
    inputs=Path(inputs);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=read(inputs);config=read(config_path);base=inputs.parent
    if torch.cuda.is_initialized():raise RuntimeError('CUDA must remain uninitialized')
    if os.name=='nt':
        from .priority import below_normal
        below_normal()
    fingerprint=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if fingerprint!=config['environment']:raise ValueError('shared environment fingerprint changed')
    identity=dict(input_sha256=sha(inputs),config_sha256=sha(config_path),source_sha256=sha(__file__))
    if (output/'IDENTITY.json').exists() and read(output/'IDENTITY.json')!=identity:raise ValueError('neural run identity changed')
    write(output/'IDENTITY.json',identity)
    if (output/'COMPLETE.json').exists():
        complete=read(output/'COMPLETE.json')
        if complete['identity']!=identity:raise ValueError('completed neural identity changed')
        for tests in complete['predictions'].values():
            for receipt in tests.values():
                if sha(confined(output,receipt['file']))!=receipt['sha256']:raise ValueError('completed predictions changed')
        for name,receipt in complete['fits'].items():
            if sha(confined(output,name)/'BEST.pt')!=receipt['best_sha256']:raise ValueError('completed fit changed')
        write(output/'STATUS.json',dict(state='complete',pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-START_CPU))
        return
    for value in [manifest['train'],*manifest['tests'].values()]:
        if sha(confined(base,value['name']))!=value['sha256']:raise ValueError('input capsule changed')
    deadline=datetime.fromisoformat(config['report_start'])
    def pulse(**detail):
        write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),
              cpu_seconds=time.process_time()-START_CPU,state='running',**detail))
    def limited():
        ceiling=min(config['cpu_ceiling_seconds'],float(os.environ.get('GHOST_V18_CHILD_CPU_LIMIT','inf')))
        return datetime.now(timezone.utc)>=deadline or time.process_time()-START_CPU>=ceiling or (output/'STOP').exists()
    pulse()
    if limited():
        write(output/'STATUS.json',dict(state='resource_cutoff',cpu_seconds=time.process_time()-START_CPU,pid=os.getpid()));return
    train=dataset(confined(base,manifest['train']['name']),'train_');dev=dataset(confined(base,manifest['train']['name']),'dev_')
    controls={}
    for kind in config['models']:
        controls[kind]=positive_control(kind,manifest['history_features'],manifest['query_features'],min(config['widths']))
        if not controls[kind]['passed']:
            write(output/'CONTROLS.json',controls);raise ValueError('known-answer neural learning control failed')
    write(output/'CONTROLS.json',controls);fits={};predictions={};benchmarks={}
    for kind in config['models']:
        for seed in config['seeds']:
            candidates=[]
            for width in config['widths']:
                name=f'{kind}-seed{seed}-width{width}'
                fit_id=dict(kind=kind,seed=seed,width=width,**identity)
                receipt=fit(output/name,fit_id,train,dev,config,pulse,limited)
                if receipt is None:
                    write(output/'STATUS.json',dict(state='resource_cutoff',cpu_seconds=time.process_time()-START_CPU,pid=os.getpid()));return
                fits[name]=receipt;candidates.append((receipt['best_dev_loss'],name))
            selected=min(candidates)[1];predictions[f'{kind}-seed{seed}']={}
            for name,value in manifest['tests'].items():
                data=dataset(confined(base,value['name']))
                target=output/f'{kind}-seed{seed}-{name}-PREDICTIONS.npz'
                receipt=forecast(output/selected/'BEST.pt',data,target)
                predictions[f'{kind}-seed{seed}'][name]=dict(selected=selected,file=target.name,**receipt)
            benchmarks[f'{kind}-seed{seed}']=update_costs(output/selected/'BEST.pt',data)
            pulse(completed_reader=f'{kind}-seed{seed}')
    if torch.cuda.is_initialized():raise RuntimeError('unexpected CUDA initialization')
    complete=dict(identity=identity,environment=fingerprint,fits=fits,predictions=predictions,benchmarks=benchmarks,
                  cpu_seconds=time.process_time()-START_CPU,cuda_initialized=False,scope='training and forecasts; evaluator must score separately')
    write(output/'COMPLETE.json',complete)
    write(output/'STATUS.json',dict(state='complete',pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=complete['cpu_seconds']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    from ..v16.runtime import local_owner
    with local_owner(Path(os.environ['GHOST_V18_CHILD_LOCK'])):
        try:run(args.inputs,args.output,args.config)
        except BaseException as exc:
            write(args.output/'STATUS.json',dict(state='failed',pid=os.getpid(),parent_pid=os.getppid(),
                heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-START_CPU,error=repr(exc)))
            raise
