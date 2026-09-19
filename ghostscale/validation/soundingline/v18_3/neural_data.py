"""Ghost-owned public feature construction; hidden test labels never enter Torch.

The training/development response distributions are equal simulator supervision
for every model. Test outcomes, latent states and exact references stay evaluator-only.
"""
from itertools import product
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from . import world as W
from .io import read,write,file_digest,digest

MAX_HISTORY=32
TRAIN_QUERIES=tuple(W.QUERIES[:5])
NEW_QUERIES=(W.context(goal=0,signal=1),W.context(goal=1,signal=0),W.context(goal=1,signal=1,budget=1))
NEURAL_STATES=tuple(i for i,s in enumerate(W.STATES) if s[0]>0)


def parity(state):
    s=W.STATES[state]
    return (s[0]-1)^s[1]^s[2]^s[3]


def onehot(value,values):
    return [float(value==x) for x in values]


def context_features(c):
    return (onehot(c['goal'],(None,0,1))+onehot(c['signal'],(None,0,1))+
            onehot(c['budget'],(1,2,3))+[float(c['offered'] is None or j in c['offered']) for j in range(4)]+
            [c['price_scale'],float(c['uninformative'])])


def world_features(w):
    return ([float(j in w['groups'][i]) for i in range(2) for j in range(4)]+[w['price'],w['temperature'],w['noise']]+
            onehot(w['rule'],('softmax','satisficing','lexicographic'))+
            [float(w[k]) for k in ('coupled','endogenous','shared')])


def features(payload):
    public=W.parse(payload);history=public['history'];seen=set();rows=[]
    for obs in history:
        rows.append(onehot(tuple(obs['program']),W.PROGRAMS)+context_features(obs['context'])+[float(obs['source'] not in seen)])
        seen.add(obs['source'])
    if not rows:raise ValueError('neural history must be nonempty')
    if len(rows)>MAX_HISTORY:raise ValueError('no silent history truncation')
    array=np.zeros((MAX_HISTORY,len(rows[0])),dtype=np.float32);array[:len(rows)]=rows
    return array,len(rows),np.asarray(world_features(public['world']),np.float32)


def history(w,state,r,length):
    out=[]
    for t in range(length):
        c=TRAIN_QUERIES[int(r.integers(len(TRAIN_QUERIES)))]
        obs=W.observe(w,state,c,r,f'episode-{t}')
        out.append(dict(out[-1]) if w['shared'] and t%3 else obs)
    return out


def predictive_check(w):
    """Linear span and Bayesian-update closure for each exact finite test bank."""
    rows=[]
    for name,bank in (('passive',(W.context(),)),('intervention-bank',TRAIN_QUERIES)):
        matrix=np.concatenate([np.ones((len(W.STATES),1))]+[W.artifact_matrix(w,c) for c in bank],axis=1)
        pinv=np.linalg.pinv(matrix,rcond=1e-10)
        projector=matrix@pinv
        future=np.concatenate([W.artifact_matrix(w,c) for c in NEW_QUERIES],axis=1)
        residual=float(np.max(abs(projector@future-future)))
        update_residual=0.
        for c in TRAIN_QUERIES:
            likelihood=W.matrix(w,c)
            for j in range(likelihood.shape[1]):
                # Unnormalized numerator and normalizer must both be representable.
                transformed=likelihood[:,j,None]*matrix
                update_residual=max(update_residual,float(np.max(abs(projector@transformed-transformed))))
        rows.append(dict(bank=name,rank=int(np.linalg.matrix_rank(matrix,tol=1e-10)),
                         new_query_span_residual=residual,update_closure_residual=update_residual,
                         sufficient_for_declared_updates=residual<1e-8 and update_residual<1e-8))
    return rows


@lru_cache(maxsize=1024)
def predictive_projector(world_json,bank):
    w=json.loads(world_json);queries=(W.context(),) if bank=='passive-summary' else TRAIN_QUERIES
    tests=np.concatenate([np.ones((len(W.STATES),1))]+[W.artifact_matrix(w,c) for c in queries],axis=1)
    return tests@np.linalg.pinv(tests,rcond=1e-10),int(np.linalg.matrix_rank(tests,tol=1e-10))


def summary_state(w,posterior,bank):
    projector,rank=predictive_projector(W.canonical(w).decode(),bank)
    state=posterior@projector
    # A predeclared simple decoder, not an exact PSR claim when span/closure fails.
    projected=np.maximum(state,0.);projected/=projected.sum()
    return projected,dict(storage_floats=rank,negative_mass_removed=float(-np.minimum(state,0.).sum()),
        decoder='orthogonal minimum-norm state projection followed by nonnegative simplex normalization')


def make_split(split,per_cell,parity_kind='even',query_kind='old',pulse=lambda **kw:None,pilot=False):
    histories=[];lengths=[];worlds=[];query=[];sample_index=[];targets=[];exact=[];ids=[];truth=[]
    summaries={name:[] for name in ('passive_summary','intervention_summary')}
    for cell in range(16):
        for i in range(per_cell):
            candidates=[s for s in NEURAL_STATES if parity(s)==(1 if parity_kind=='odd' else 0)]
            state=candidates[i%len(candidates)]
            offset={'train':40000,'dev':50000,'test':60000,'pilot':90000}[split]
            w=W.make_world(cell,offset+i+(100000 if pilot else 0))
            r=W.rng('neural',split,parity_kind,cell,i,pilot)
            length=(8,16,32)[(i//len(candidates))%3]
            h=history(w,state,r,length);payload=W.packet(w,h)
            x,n,wf=features(payload);weights=W.posterior(payload)
            summary_weights={name:summary_state(w,weights,name.replace('_','-')) for name in summaries} if split=='test' else {}
            number=len(histories);histories.append(x);lengths.append(n);worlds.append(wf)
            queries=TRAIN_QUERIES if query_kind=='old' else NEW_QUERIES
            for q,c in enumerate(queries):
                sample_index.append(number);query.append(context_features(c)+world_features(w))
                targets.append(W.artifact_matrix(w,c)[state]);exact.append(weights@W.artifact_matrix(w,c))
                for name,(summary,_) in summary_weights.items():summaries[name].append(summary@W.artifact_matrix(w,c))
                ids.append([cell,i,q,length])
            truth.append(dict(state=state,world=w,history=h,summary_decoders={name:meta for name,(_,meta) in summary_weights.items()}))
            if i%8==7:pulse(phase='generating',split=split,cell=cell,histories=len(histories))
    return dict(history=np.asarray(histories,np.float32),length=np.asarray(lengths,np.int64),
                world=np.asarray(worlds,np.float32),sample=np.asarray(sample_index,np.int64),
                query=np.asarray(query,np.float32),target=np.asarray(targets,np.float32),
                exact=np.asarray(exact,np.float64),ids=np.asarray(ids,np.int64),
                **{name:np.asarray(values,np.float64) for name,values in summaries.items() if values}),truth


def prepare(root,train_per_cell=128,dev_per_cell=16,test_per_cell=32,pilot=False,pulse=lambda **kw:None):
    import gzip
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    public_root=root/'reader';public_root.mkdir(exist_ok=True)
    config=dict(train_per_cell=train_per_cell,dev_per_cell=dev_per_cell,test_per_cell=test_per_cell,pilot=pilot)
    write(root/'BUILD_PLAN.json',config)
    if (public_root/'INPUTS.json').exists():
        manifest=read(public_root/'INPUTS.json')
        for value in [manifest['train'],*manifest['tests'].values()]:
            if file_digest(public_root/value['name'])!=value['sha256']:raise ValueError('input changed on resume')
        return manifest
    # Each per-cell count is a multiple of eight, so every role marginal/pair is balanced.
    if any(n%8 for n in (train_per_cell,dev_per_cell,test_per_cell)):
        raise ValueError('balanced role-combination count must be a multiple of eight')
    def part(name,split,n,parity_kind='even',query_kind='old'):
        receipt=root/f'{name}-PART.json';data_path=root/f'{name}-DATA.npz';points=root/f'{name}-points.json.gz'
        if receipt.exists():
            saved=read(receipt)
            if file_digest(data_path)!=saved['data_sha256'] or file_digest(points)!=saved['points_sha256']:
                raise ValueError('retained generation shard changed')
            with np.load(data_path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
            return data,json.loads(gzip.decompress(points.read_bytes()))
        data,truth=make_split(split,n,parity_kind,query_kind,pulse,pilot)
        np.savez_compressed(data_path,**data);points.write_bytes(gzip.compress(W.canonical(truth),mtime=0))
        write(receipt,dict(data_sha256=file_digest(data_path),points_sha256=file_digest(points)))
        return data,truth
    train,_=part('train','pilot' if pilot else 'train',train_per_cell)
    dev,_=part('dev','dev',dev_per_cell)
    train_payload={f'train_{k}':v for k,v in train.items() if k not in ('exact','ids')}
    train_payload.update({f'dev_{k}':v for k,v in dev.items() if k not in ('exact','ids')})
    np.savez_compressed(public_root/'TRAIN.npz',**train_payload)
    cases={};input_paths={};truth_paths={}
    for name,parity_kind,query_kind in (('in-support','even','old'),('new-combinations','odd','old'),
                                       ('new-queries','even','new'),('both-new','odd','new')):
        data,truth=part(name,'test',test_per_cell,parity_kind,query_kind)
        # Truth and public histories are retained separately for independent reconstruction.
        np.savez_compressed(public_root/f'{name}-INPUT.npz',**{k:v for k,v in data.items() if k in ('history','length','world','sample','query')})
        np.savez_compressed(root/f'{name}-TRUTH.npz',**{k:data[k] for k in ('target','exact','ids','passive_summary','intervention_summary')})
        input_paths[name]=dict(name=f'{name}-INPUT.npz',sha256=file_digest(public_root/f'{name}-INPUT.npz'))
        truth_paths[name]=dict(name=f'{name}-TRUTH.npz',sha256=file_digest(root/f'{name}-TRUTH.npz'),points_sha256=file_digest(root/f'{name}-points.json.gz'))
        cases[name]=dict(histories=len(data['history']),queries=len(data['query']))
    checks=[dict(cell=cell,checks=predictive_check(W.make_world(cell,70000))) for cell in range(16)]
    write(root/'PREDICTIVE_CLOSURE.json',dict(worlds=checks,scope='declared finite observation/update/query families'))
    public=dict(schema='v18.3.neural-input.1',train=dict(name='TRAIN.npz',sha256=file_digest(public_root/'TRAIN.npz')),
                tests=input_paths,cases=cases,training_supervision='same exact simulator behavioral distributions for all models',
                max_history=MAX_HISTORY,history_features=train['history'].shape[2],query_features=train['query'].shape[1],
                train_histories=len(train['history']),dev_histories=len(dev['history']),pilot=pilot)
    write(root/'EVALUATOR.json',dict(truth=truth_paths,scope='must never be passed to training child'))
    write(public_root/'INPUTS.json',public)
    return public
