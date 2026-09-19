"""CPU interchange-intervention training with equal counterfactual supervision."""
from . import torch_worker as T
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch import nn

METHODS=('split-behavior','split-IIT','flat-IIT','direct-pair','split-shuffled-IIT')


class Model(nn.Module):
    def __init__(self,method,input_size,query_size,width=24,head=64,direct_hidden=4):
        super().__init__();self.method=method;self.width=width
        self.spec=dict(method=method,input_size=input_size,query_size=query_size,width=width,head=head,direct_hidden=direct_hidden)
        kind='direct' if method=='direct-pair' else ('flat' if method=='flat-IIT' else 'split')
        self.reader=T.Reader(kind,input_size,query_size,width,head=head,direct_hidden=direct_hidden)
        if method=='direct-pair':self.cf=nn.Sequential(nn.Linear(width*2+query_size+3,head),nn.Tanh(),nn.Linear(head,16))

    def transplant(self,base,source,role):
        mask=(torch.arange(self.width)[None,:]//(self.width//3))==role[:,None]
        return torch.where(mask,source,base)

    def counterfactual(self,base,source,query,role):
        if self.method=='direct-pair':return self.cf(torch.cat((base,source,query,torch.nn.functional.one_hot(role,3).float()),1))
        return self.reader.decode(self.transplant(base,source,role),query)

    def incompatible_partition(self,base,source,query,role):
        if self.method=='direct-pair':raise ValueError('direct pair predictor has no internal role partition')
        mask=torch.arange(self.width)[None,:]%3==role[:,None]
        return self.reader.decode(torch.where(mask,source,base),query)

    def forward(self,bh,bl,sh,sl,q,role):
        b=self.reader.encode(bh,bl);s=self.reader.encode(sh,sl)
        return self.counterfactual(b,s,q,role),self.reader.decode(b,q)


def architecture(method,input_size,query_size,width):
    target=T.count(T.Reader('flat',input_size,query_size,width))
    if method=='direct-pair':
        options=[Model(method,input_size,query_size,width,head=h,direct_hidden=d) for h in range(16,97,4) for d in range(2,9)]
    else:options=[Model(method,input_size,query_size,width,head=h) for h in range(32,161,4)]
    return min(options,key=lambda m:abs(T.count(m)-target))


def dataset(path,prefix='',public=False):
    with np.load(path,allow_pickle=False) as z:
        available={name[len(prefix):] for name in z.files if name.startswith(prefix)}
        required={'history','length','base','source','role','query'}
        allowed=required if public else required|{'target','behavior_target'}
        if not required<=available or not available<=allowed:raise ValueError('undeclared intervention input or hidden test label')
        data={k:torch.from_numpy(z[prefix+k].copy()) for k in available}
    if data['history'].ndim!=3 or data['query'].ndim!=2:raise ValueError('bad intervention features')
    n=len(data['base'])
    if any(len(data[k])!=n for k in ('source','role','query')):raise ValueError('mismatched pair roster')
    for k in ('base','source'):
        if torch.any(data[k]<0) or torch.any(data[k]>=len(data['history'])):raise ValueError('invalid pair index')
    if torch.any(data['role']<0) or torch.any(data['role']>2):raise ValueError('invalid role instruction')
    for k in ('target','behavior_target'):
        if k in data and (torch.any(data[k]<0) or not torch.allclose(data[k].sum(1),torch.ones(n),atol=1e-6,rtol=0)):raise ValueError('invalid counterfactual supervision')
    return data


def batch(data,indices):
    b=data['base'][indices];s=data['source'][indices]
    return data['history'][b],data['length'][b],data['history'][s],data['length'][s],data['query'][indices],data['role'][indices]


def objective(cf,behavior,data,indices,method,permutation=None):
    normal=T.loss(behavior,data['behavior_target'][indices])
    if method=='split-behavior':return normal
    label_indices=permutation[indices] if permutation is not None else indices
    return .5*(normal+T.loss(cf,data['target'][label_indices]))


def evaluate(model,data,permutation=None):
    total=0.;n=len(data['base']);model.eval()
    with torch.no_grad():
        for first in range(0,n,256):
            idx=torch.arange(first,min(n,first+256));cf,b=model(*batch(data,idx))
            total+=float(objective(cf,b,data,idx,model.method,permutation))*len(idx)
    return total/n


def fit(root,identity,train,dev,config,pulse,limited):
    root.mkdir(parents=True,exist_ok=True);identity_hash=hashlib.sha256(T.encoded(identity)).hexdigest();started=time.process_time()
    if (root/'IDENTITY.json').exists() and T.read(root/'IDENTITY.json')!=identity:raise ValueError('intervention fit identity changed')
    T.write(root/'IDENTITY.json',identity)
    if (root/'COMPLETE.json').exists():
        result=T.read(root/'COMPLETE.json')
        if result['identity_sha256']!=identity_hash or T.sha(root/'BEST.pt')!=result['best_sha256']:raise ValueError('completed intervention fit changed')
        return result
    torch.manual_seed(identity['seed']);m=architecture(identity['method'],train['history'].shape[2],train['query'].shape[1],identity['width'])
    torch.manual_seed(identity['seed']);model=Model(**m.spec);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.0001)
    shuffled=identity['method']=='split-shuffled-IIT'
    perm=torch.randperm(len(train['base']),generator=torch.Generator().manual_seed(1803301)) if shuffled else None
    dev_perm=torch.randperm(len(dev['base']),generator=torch.Generator().manual_seed(1803302)) if shuffled else None
    progress=dict(epoch=0,offset=0,steps=0,best_loss=1e100,curve=[],initial_train_loss=evaluate(model,train,perm))
    if (root/'CURRENT.json').exists():
        receipt=T.read(root/'CURRENT.json');path=T.confined(root,receipt['file'])
        if T.sha(path)!=receipt['sha256']:raise ValueError('intervention checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity_sha256']!=identity_hash:raise ValueError('intervention recipe changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);progress=saved['progress']
    def save():
        path=root/f"step-{progress['steps']:07d}-{time.time_ns()}.pt";tmp=path.with_suffix('.tmp')
        torch.save(dict(identity_sha256=identity_hash,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),progress=progress),tmp);T.replace(tmp,path)
        T.write(root/'CURRENT.json',dict(file=path.name,sha256=T.sha(path)))
    n=len(train['base'])
    while progress['epoch']<config['epochs']:
        order=torch.randperm(n,generator=torch.Generator().manual_seed(identity['seed']+progress['epoch']));model.train()
        while progress['offset']<n:
            if limited():save();return None
            idx=order[progress['offset']:progress['offset']+config['batch_size']]
            optimizer.zero_grad();cf,b=model(*batch(train,idx));value=objective(cf,b,train,idx,model.method,perm)
            if not torch.isfinite(value):save();raise ValueError('nonfinite intervention fit')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            progress['offset']+=len(idx);progress['steps']+=1
            if progress['steps']%100==0:save();pulse(fit=root.name,steps=progress['steps'])
        score=evaluate(model,dev,dev_perm);progress['curve'].append(dict(epoch=progress['epoch'],dev_loss=score))
        if score<progress['best_loss']:
            progress['best_loss']=score;tmp=root/'BEST.tmp';torch.save(dict(spec=model.spec,state=model.state_dict(),identity_sha256=identity_hash),tmp);T.replace(tmp,root/'BEST.pt')
        progress['epoch']+=1;progress['offset']=0;save();pulse(fit=root.name,epoch=progress['epoch'])
    final=evaluate(model,train,perm)
    result=dict(identity_sha256=identity_hash,best_sha256=T.sha(root/'BEST.pt'),parameters=T.count(model),spec=model.spec,
        initial_train_loss=progress['initial_train_loss'],final_train_loss=final,learning_control_passed=final<progress['initial_train_loss']-.02,
        curve=progress['curve'],best_dev_loss=progress['best_loss'],steps=progress['steps'],final_attempt_cpu_seconds=time.process_time()-started)
    T.write(root/'COMPLETE.json',result);return result


def permutation_check(model,b,s,q,role):
    # A coherent change from contiguous to interleaved coordinate groups must commute.
    perm=torch.arange(model.width).reshape(3,-1).T.reshape(-1);inverse=torch.argsort(perm)
    labels=torch.arange(model.width)//(model.width//3)
    mixed=torch.where(labels[perm][None,:]==role[:,None],s[:,perm],b[:,perm])[:,inverse]
    expected=model.reader.decode(model.transplant(b,s,role),q)
    actual=model.reader.decode(mixed,q)
    error=float(torch.max(abs(actual-expected)))
    if error>1e-6:raise ValueError('compatible coordinate partition changed computation')
    return error


def forecast(best,data,path,pulse=lambda **kw:None,limited=lambda:False):
    saved=torch.load(best,map_location='cpu',weights_only=True);model=Model(**saved['spec']);model.load_state_dict(saved['state']);model.eval()
    start=time.process_time();outputs={k:[] for k in ('counterfactual','behavior','wrong_mapping')};error=0.
    if model.method!='direct-pair':outputs['incompatible_partition']=[]
    with torch.no_grad():
        states=torch.cat([model.reader.encode(data['history'][i:i+128],data['length'][i:i+128]) for i in range(0,len(data['history']),128)])
        for first in range(0,len(data['base']),512):
            if limited():raise TimeoutError('intervention forecast reached resource cutoff')
            idx=torch.arange(first,min(len(data['base']),first+512));b=states[data['base'][idx]];s=states[data['source'][idx]];q=data['query'][idx];role=data['role'][idx]
            cf=model.counterfactual(b,s,q,role);normal=model.reader.decode(b,q);wrong=model.counterfactual(b,s,q,(role+1)%3)
            values=[cf,normal,wrong]
            if model.method!='direct-pair':values.append(model.incompatible_partition(b,s,q,role))
            for name,value in zip(outputs,values):outputs[name].append(torch.softmax(value,1).numpy())
            if first==0:error=permutation_check(model,b,s,q,role)
            if first%65536==0:pulse(forecast_rows=first)
    arrays={k:np.concatenate(v) for k,v in outputs.items()}
    for p in arrays.values():
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):raise ValueError('invalid interchange forecast')
    np.savez_compressed(path,**arrays)
    return dict(file=path.name,sha256=T.sha(path),rows=len(data['base']),cpu_seconds=time.process_time()-start,coordinate_permutation_error=error,
        incompatible_partition_available=model.method!='direct-pair')


def run(inputs,output,config_path):
    output.mkdir(parents=True,exist_ok=True);manifest=T.read(inputs);config=T.read(config_path);base=inputs.parent
    environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    if environment!=config['environment'] or torch.cuda.is_initialized():raise ValueError('intervention CPU environment changed')
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    identity=dict(input_sha256=T.sha(inputs),config_sha256=T.sha(config_path),source_sha256=T.sha(__file__),reader_sha256=T.sha(T.__file__))
    if (output/'IDENTITY.json').exists() and T.read(output/'IDENTITY.json')!=identity:raise ValueError('intervention identity changed')
    T.write(output/'IDENTITY.json',identity)
    def pulse(**detail):T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),state='running',heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-T.START_CPU,**detail))
    def limited():return datetime.now(timezone.utc)>=datetime.fromisoformat(config['report_start']) or time.process_time()-T.START_CPU>=min(config['cpu_ceiling_seconds'],float(os.environ.get('GHOST_V18_CHILD_CPU_LIMIT','inf'))) or (output/'STOP').exists()
    pulse()
    if limited():raise TimeoutError('intervention deadline before dispatch')
    for entry in (manifest['train'],manifest['test']):
        if T.sha(T.confined(base,entry['name']))!=entry['sha256']:raise ValueError('counterfactual inputs changed')
    if (output/'COMPLETE.json').exists():
        complete=T.read(output/'COMPLETE.json')
        for name,receipt in complete['fits'].items():
            if T.sha(output/name/'BEST.pt')!=receipt['best_sha256']:raise ValueError('intervention weights changed')
        for receipt in complete['predictions'].values():
            if T.sha(output/receipt['file'])!=receipt['sha256']:raise ValueError('intervention predictions changed')
        T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),state='complete',cpu_seconds=time.process_time()-T.START_CPU))
        return
    train=dataset(base/manifest['train']['name'],'train_');dev=dataset(base/manifest['train']['name'],'dev_');test=dataset(base/manifest['test']['name'],public=True)
    fits={};predictions={};checks={}
    for method in config['models']:
        if method not in METHODS:raise ValueError('unfrozen method')
        for seed in config['seeds']:
            name=f'{method}-seed{seed}';fit_id=dict(method=method,seed=seed,width=config['width'],**identity)
            receipt=fit(output/name,fit_id,train,dev,config,pulse,limited)
            if receipt is None:raise TimeoutError('intervention training resource cutoff')
            fits[name]=receipt;predictions[name]=forecast(output/name/'BEST.pt',test,output/(name+'-PREDICTIONS.npz'),pulse,limited)
            checks[name]=predictions[name]['coordinate_permutation_error'];pulse(completed_reader=name)
    T.write(output/'COMPLETE.json',dict(identity=identity,environment=environment,fits=fits,predictions=predictions,permutation_checks=checks,cuda_initialized=torch.cuda.is_initialized(),cpu_seconds=time.process_time()-T.START_CPU))
    T.write(output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),state='complete',cpu_seconds=time.process_time()-T.START_CPU))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from ..v16.runtime import local_owner
    with local_owner(Path(os.environ['GHOST_V18_CHILD_LOCK'])):
        try:run(a.inputs,a.output,a.config)
        except BaseException as exc:
            T.write(a.output/'STATUS.json',dict(pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),state='resource_cutoff' if isinstance(exc,TimeoutError) else 'failed',cpu_seconds=time.process_time()-T.START_CPU,error=repr(exc)))
            if not isinstance(exc,TimeoutError):raise
