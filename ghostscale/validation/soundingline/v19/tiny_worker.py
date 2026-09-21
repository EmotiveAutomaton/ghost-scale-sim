"""CPU-only public-input sequence learner; no generator or evaluator imports."""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_n]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,json,time,math,io
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from ..v18_3.io import write,replace
from ..v16.runtime import local_owner
from ..v18_4.priority import below_normal
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def arrays(p,**v):
    p.parent.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();np.savez_compressed(buffer,**v);p.write_bytes(buffer.getvalue())
def encode(codes,novel):
    if codes.shape!=novel.shape or codes.ndim!=2 or codes.shape[1]!=32:raise ValueError('bad stream shape')
    if np.any((codes<0)|(codes>31)) or novel.dtype!=np.bool_:raise ValueError('bad public tokens')
    n=len(codes);tokens=np.full((n,33),32,np.int64);positions=np.cumsum(novel,1)
    for i in range(n):tokens[i,1:1+novel[i].sum()]=codes[i,novel[i]]
    return torch.from_numpy(tokens),torch.from_numpy(positions)

class Reader(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.embedding=nn.Embedding(33,32)
        if kind=='transformer':
            self.encoder=nn.TransformerEncoderLayer(32,4,64,dropout=0.,activation='gelu',batch_first=True,norm_first=False)
            pos=torch.arange(33)[:,None];rates=torch.exp(torch.arange(0,32,2)*(-math.log(10000)/32))
            pe=torch.zeros(33,32);pe[:,0::2]=torch.sin(pos*rates);pe[:,1::2]=torch.cos(pos*rates);self.register_buffer('positions',pe)
            width=32
        elif kind=='recurrent':self.encoder=nn.GRU(32,38,batch_first=True);width=38
        else:raise ValueError('unadmitted setting')
        self.head=nn.Linear(width,32)
    def encode(self,tokens):
        x=self.embedding(tokens)
        if self.kind=='transformer':
            x=x+self.positions[:x.shape[1]]
            mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),1)
            return self.encoder(x,src_mask=mask)
        return self.encoder(x)[0]
    def decode(self,state):return self.head(state).reshape(*state.shape[:-1],4,8)
    def forward(self,tokens,positions):
        h=self.encode(tokens);state=h.gather(1,positions[...,None].expand(-1,-1,h.shape[-1]));return self.decode(state),state

def objective(logits,target,novel):
    loss=-(target*torch.log_softmax(logits,-1)).sum(-1).mean(-1)
    return (loss*novel).sum()/novel.sum()

def structural_controls(kind):
    torch.manual_seed(196001);m=Reader(kind);m.eval()
    codes=np.tile(np.arange(32),(2,1));novel=np.ones_like(codes,dtype=bool);tokens,pos=encode(codes,novel)
    altered=tokens.clone();altered[:,17:]=torch.flip(altered[:,17:],[1])
    with torch.no_grad():
        y,h=m(tokens,pos);z,_=m(altered,pos)
        causal=bool(torch.allclose(y[:,:16],z[:,:16],atol=1e-6,rtol=0))
        noop=bool(torch.equal(m.decode(h),y));p=torch.softmax(y,-1)
        copied=codes.copy();copied[:,1::2]=copied[:,::2];flags=novel.copy();flags[:,1::2]=False
        ct,cp=encode(copied,flags);cy,_=m(ct,cp)
        copy=bool(torch.equal(cy[:,1::2],cy[:,::2]))
    return dict(causal_prefix=causal,no_op_decode=noop,duplicate_passthrough=copy,normalized=bool(torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=1e-6)),parameters=sum(x.numel() for x in m.parameters()))

def learning_control(kind):
    # This fixed known-answer fixture is included in each reserved architecture setting.
    torch.manual_seed(196002);m=Reader(kind);opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.0001)
    tokens=torch.full((32,2),32,dtype=torch.long);tokens[:,1]=torch.arange(32)%8;pos=torch.ones((32,1),dtype=torch.long)
    target=torch.eye(8)[tokens[:,1]][:,None,None,:].expand(-1,1,4,-1);mask=torch.ones((32,1),dtype=torch.bool)
    initial=float(objective(m(tokens,pos)[0],target,mask).detach())
    for _ in range(80):
        opt.zero_grad();loss=objective(m(tokens,pos)[0],target,mask);loss.backward();opt.step()
    final=float(objective(m(tokens,pos)[0],target,mask).detach())
    # Known no-information target has its strict minimum at the constant uniform forecast.
    null=torch.ones_like(target)/8;null_loss=float(objective(torch.zeros_like(target),null,mask))
    return dict(initial_loss=initial,final_loss=final,learning_passed=final<.20,uniform_null=abs(null_loss-math.log(8))<1e-6)

def run(inputs,output,config,campaign):
    below_normal();start=time.process_time();cfg=read(config);manifest=read(inputs);output.mkdir(exist_ok=True)
    with local_owner(campaign/'neural-child-owner'):
        def pulse(**details):
            write(output/'STATUS.json',dict(state='running',pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-start,**details),immutable=False)
            if (output/'STOP').exists() or datetime.now(timezone.utc)>=datetime.fromisoformat(cfg['deadline']) or time.process_time()-start>=float(os.environ['GHOST_V19_CHILD_CPU_LIMIT']):raise TimeoutError('tiny child safe cutoff')
        pulse(phase='controls')
        if torch.__version__!=cfg['torch_version'] or np.__version__!=cfg['numpy_version']:raise ValueError('child fingerprint changed')
        controls={kind:dict(structural_controls(kind),**learning_control(kind)) for kind in cfg['kinds']}
        for v in controls.values():
            if not all(v[k] for k in ('causal_prefix','no_op_decode','duplicate_passthrough','normalized','learning_passed','uniform_null')):raise ValueError('tiny instrument fixture failed')
        fits=[]
        for draw in cfg['draws']:
            train=[]
            for condition in ('independent','copied'):
                entry=manifest['train'][str(draw)][condition];path=inputs.parent/entry['file']
                if sha(path)!=entry['sha256']:raise ValueError('public training bytes changed')
                with np.load(path,allow_pickle=False) as z:
                    if set(z.files)!={'codes','novel','target'}:raise ValueError('undeclared train fields')
                    train.append({k:z[k].copy() for k in z.files})
            a={k:np.concatenate([d[k] for d in train]) for k in train[0]};tokens,pos=encode(a['codes'],a['novel'])
            targets=torch.from_numpy(a['target'].reshape(-1,32,4,8).astype('float32'));novel=torch.from_numpy(a['novel'])
            if not torch.allclose(targets.sum(-1),torch.ones_like(targets.sum(-1)),atol=1e-6) or torch.any(targets<0):raise ValueError('invalid target')
            for seed in cfg['seeds']:
                for kind in cfg['kinds']:
                    pulse(phase='fit-start',draw=draw,seed=seed,kind=kind);fit=output/f'{draw}-{seed}-{kind}';fit.mkdir(exist_ok=True)
                    identity=dict(draw=draw,seed=seed,kind=kind,inputs_sha256=sha(inputs),config_sha256=sha(config));ident=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                    torch.manual_seed(seed);model=Reader(kind);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                    progress=dict(epoch=0,offset=0,steps=0,curve=[],permutation=None)
                    if (fit/'CURRENT.json').exists():
                        receipt=read(fit/'CURRENT.json');path=fit/receipt['file']
                        if sha(path)!=receipt['sha256']:raise ValueError('checkpoint changed')
                        ck=torch.load(path,map_location='cpu',weights_only=True)
                        if ck['identity']!=ident:raise ValueError('checkpoint identity mismatch')
                        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);progress=ck['progress']
                    def save():
                        path=fit/f"checkpoint-{progress['epoch']%2}.pt";temp=path.with_suffix('.tmp')
                        torch.save(dict(identity=ident,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),progress=progress),temp);replace(temp,path)
                        write(fit/'CURRENT.json',dict(file=path.name,sha256=sha(path),steps=progress['steps']),immutable=False)
                    try:
                        while progress['epoch']<cfg['epochs']:
                            if progress['permutation'] is None:progress['permutation']=torch.randperm(len(tokens))
                            while progress['offset']<len(tokens):
                                pulse(phase='training',draw=draw,seed=seed,kind=kind,epoch=progress['epoch'],steps=progress['steps'])
                                take=progress['permutation'][progress['offset']:progress['offset']+cfg['batch_size']]
                                model.train();optimizer.zero_grad();logits,_=model(tokens[take],pos[take]);loss=objective(logits,targets[take],novel[take]);loss.backward()
                                norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                                if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('nonfinite fit')
                                optimizer.step();progress['steps']+=1;progress['offset']+=len(take)
                            model.eval()
                            with torch.no_grad():
                                losses=[];weights=[]
                                for first in range(0,len(tokens),32):
                                    sl=slice(first,first+32);logits,_=model(tokens[sl],pos[sl]);losses.append(float(objective(logits,targets[sl],novel[sl])));weights.append(int(novel[sl].sum()))
                            progress['curve'].append(float(np.average(losses,weights=weights)));progress['epoch']+=1;progress['offset']=0;progress['permutation']=None;save()
                    except BaseException:save();raise
                    arrays(fit/'PARAMETERS.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
                    for condition in ('independent','copied'):
                        entry=manifest['development'][str(draw)][condition];path=inputs.parent/entry['file']
                        if sha(path)!=entry['sha256']:raise ValueError('development inputs changed')
                        with np.load(path,allow_pickle=False) as z:
                            if set(z.files)!={'codes','novel'}:raise ValueError('undeclared development field')
                            dt,dp=encode(z['codes'],z['novel'])
                        model.eval()
                        with torch.no_grad():logits,state=model(dt,dp);prob=torch.softmax(logits,-1).numpy();hidden=state.numpy()
                        arrays(fit/f'{condition}.npz',probabilities=prob,hidden=hidden)
                    fits.append(dict(identity=identity,steps=progress['steps'],epochs=progress['epoch'],curve=progress['curve'],parameters=sum(p.numel() for p in model.parameters()),training_unique_prefixes=int(novel.sum()),target_contexts=4))
        result=dict(controls=controls,fits=fits,environment=dict(torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),device='cpu'))
        write(output/'RESULT.json',result)
        write(output/'STATUS.json',dict(state='complete',pid=os.getpid(),parent_pid=os.getppid(),heartbeat=datetime.now(timezone.utc).isoformat(),cpu_seconds=time.process_time()-start),immutable=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path);p.add_argument('--output',type=Path);p.add_argument('--config',type=Path);p.add_argument('--campaign',type=Path);a=p.parse_args()
    run(a.inputs,a.output,a.config,a.campaign)
