"""Counterfactual supervision with separate generator truth and public capsules."""
import gzip
import json
from pathlib import Path
import numpy as np
from . import world as W,neural_data as D
from .io import read,write,file_digest,canonical

ROLES=((0,3),(1,),(2,))
QUERIES=(W.context(),W.context(goal=0),W.context(signal=0))


def mixed_state(base,source,role):
    state=list(W.STATES[base])
    for axis in ROLES[role]:state[axis]=W.STATES[source][axis]
    result=tuple(state)
    if result not in W.STATES or result[0]==0:raise ValueError('counterfactual outside declared support')
    return W.STATES.index(result)


def cell_data(split,cell,draws,pilot=False,pulse=lambda **kw:None):
    states=D.NEURAL_STATES;histories=[];lengths=[];base=[];source=[];roles=[];queries=[]
    targets=[];behavior=[];exact=[];ids=[];points=[]
    mapping=np.array([[[mixed_state(a,b,r) for b in states] for a in states] for r in range(3)])
    for draw in range(draws):
        offset={'train':210000,'dev':220000,'test':230000}[split]+(1000000 if pilot else 0)
        w=W.make_world(cell,offset+draw);world_hist=[];post=[];first=len(histories)
        for i,state in enumerate(states):
            h=D.history(w,state,W.rng('intervention',split,pilot,cell,draw,i),32)
            payload=W.packet(w,h);x,n,_=D.features(payload);histories.append(x);lengths.append(n);world_hist.append(h)
            p=W.posterior(payload)[list(states)];p/=p.sum();post.append(p)
        post=np.asarray(post)
        matrices=np.stack([W.artifact_matrix(w,q) for q in QUERIES],axis=1)
        exact_all=np.einsum('bi,sj,rijqk->rbsqk',post,post,matrices[mapping],optimize=True) if split=='test' else None
        if exact_all is not None:
            for b in range(16):exact_all[:,b,b]=np.einsum('i,iqk->qk',post[b],matrices[list(states)])
        for b in range(16):
            sources=range(16) if split=='test' else [(b+3*j)%16 for j in range(4)]
            for s in sources:
                for role in range(3):
                    for q,c in enumerate(QUERIES):
                        base.append(first+b);source.append(first+s);roles.append(role)
                        queries.append(D.context_features(c)+D.world_features(w))
                        targets.append(matrices[mapping[role,b,s],q]);behavior.append(matrices[states[b],q])
                        if exact_all is not None:exact.append(exact_all[role,b,s,q])
                        ids.append([cell,draw,b,s,role,q])
        points.append(dict(world=w,histories=world_hist,states=list(states),cell=cell,draw=draw))
        pulse(phase='counterfactual-generation',split=split,cell=cell,draw=draw)
    arrays=dict(history=np.asarray(histories,np.float32),length=np.asarray(lengths,np.int64),
        base=np.asarray(base,np.int64),source=np.asarray(source,np.int64),role=np.asarray(roles,np.int64),
        query=np.asarray(queries,np.float32),target=np.asarray(targets,np.float32),
        behavior_target=np.asarray(behavior,np.float32),ids=np.asarray(ids,np.int64))
    if exact:arrays['exact']=np.asarray(exact,np.float64)
    return arrays,points


def prepare(root,train_draws=8,dev_draws=2,test_draws=20,pilot=False,pulse=lambda **kw:None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);public=root/'reader';public.mkdir(exist_ok=True)
    write(root/'BUILD_PLAN.json',dict(train_draws=train_draws,dev_draws=dev_draws,test_draws=test_draws,pilot=pilot))
    if (public/'INPUTS.json').exists():
        manifest=read(public/'INPUTS.json')
        for value in (manifest['train'],manifest['test']):
            if file_digest(public/value['name'])!=value['sha256']:raise ValueError('intervention input changed')
        return manifest
    assembled={}
    for split,draws in (('train',train_draws),('dev',dev_draws),('test',test_draws)):
        all_data=[];all_points=[];offset=0
        for cell in range(16):
            name=f'{split}-cell{cell}';path=root/(name+'.npz');points_path=root/(name+'-points.json.gz');receipt=root/(name+'.json')
            if receipt.exists():
                saved=read(receipt)
                if file_digest(path)!=saved['data_sha256'] or file_digest(points_path)!=saved['points_sha256']:raise ValueError('intervention shard corruption')
                with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
                points=json.loads(gzip.decompress(points_path.read_bytes()))
            else:
                data,points=cell_data(split,cell,draws,pilot,pulse)
                np.savez_compressed(path,**data);points_path.write_bytes(gzip.compress(canonical(points),mtime=0))
                write(receipt,dict(data_sha256=file_digest(path),points_sha256=file_digest(points_path)))
            data['base']+=offset;data['source']+=offset;offset+=len(data['history'])
            all_data.append(data);all_points.extend(points)
        assembled[split]={k:np.concatenate([d[k] for d in all_data]) for k in all_data[0]}
        if split=='test':
            (root/'test-points.json.gz').write_bytes(gzip.compress(canonical(all_points),mtime=0))
        pulse(phase='assembled-counterfactual-split',split=split)
    train={f'{split}_{k}':v for split in ('train','dev') for k,v in assembled[split].items() if k!='ids'}
    np.savez_compressed(public/'TRAIN.npz',**train)
    test=assembled['test']
    np.savez_compressed(public/'TEST_INPUT.npz',**{k:v for k,v in test.items() if k not in ('target','behavior_target','exact','ids')})
    np.savez_compressed(root/'TRUTH.npz',**{k:test[k] for k in ('target','behavior_target','exact','ids')})
    write(root/'EVALUATOR.json',dict(truth_sha256=file_digest(root/'TRUTH.npz'),points_sha256=file_digest(root/'test-points.json.gz')))
    manifest=dict(schema='v18.3.intervention-input.1',train=dict(name='TRAIN.npz',sha256=file_digest(public/'TRAIN.npz')),
        test=dict(name='TEST_INPUT.npz',sha256=file_digest(public/'TEST_INPUT.npz')),pilot=pilot,
        history_features=test['history'].shape[2],query_features=test['query'].shape[1],test_pairs=len(test['base']),
        training_pairs=len(assembled['train']['base']),training_supervision='equal exact behavior and generator-defined counterfactual distributions',
        role_instructions=['acquired curriculum plus persistent tradeoff','current goal','private belief'])
    write(public/'INPUTS.json',manifest);return manifest


def score(root,pulse=lambda **kw:None):
    from .neural_runtime import proper_scores
    from .runtime import stats
    data=root/'data';child=root/'neural';evaluator=read(data/'EVALUATOR.json')
    if file_digest(data/'TRUTH.npz')!=evaluator['truth_sha256'] or file_digest(data/'test-points.json.gz')!=evaluator['points_sha256']:
        raise ValueError('intervention evaluator changed')
    with np.load(data/'TRUTH.npz',allow_pickle=False) as z:truth={k:z[k].copy() for k in z.files}
    ids=truth['ids'];completed=read(child/'COMPLETE.json');rows=[];reference_checks=0
    retained=json.loads(gzip.decompress((data/'test-points.json.gz').read_bytes()))
    # Fixed source-independent selection across every architecture and twenty draws.
    for point in retained[::max(1,len(retained)//32)]:
        w=point['world'];cell=point['cell'];draw=point['draw']
        from .verify import reference_policy
        for b,s,role in ((0,15,0),(3,10,1),(6,9,2)):
            mixed=mixed_state(D.NEURAL_STATES[b],D.NEURAL_STATES[s],role)
            selected=np.flatnonzero((ids[:,0]==cell)&(ids[:,1]==draw)&(ids[:,2]==b)&(ids[:,3]==s)&(ids[:,4]==role))
            for j in selected:
                c=QUERIES[ids[j,5]];p=reference_policy(w,mixed,c);artifact=np.zeros(16)
                for program,prob in zip(W.PROGRAMS,p):artifact[sum(1<<v for v in program)]+=prob
                if not np.allclose(artifact,truth['target'][j],atol=1e-7,rtol=0):raise ValueError('counterfactual target mismatch')
                reference_checks+=1
    def forecasts():
        yield 'exact-public-history',(None,truth['exact'],None,None,None)
        yield 'known-state-ceiling',(None,truth['target'],truth['behavior_target'],None,None)
        for name,receipt in completed['predictions'].items():
            path=child/receipt['file']
            if file_digest(path)!=receipt['sha256']:raise ValueError('intervention predictions changed')
            with np.load(path,allow_pickle=False) as z:
                yield name,(int(name.rsplit('-seed',1)[1]),z['counterfactual'].copy(),z['behavior'].copy(),z['wrong_mapping'].copy(),z['incompatible_partition'].copy() if 'incompatible_partition' in z else None)
    for name,(seed,cf,behavior,wrong,incompatible) in forecasts():
        method=name.rsplit('-seed',1)[0] if seed is not None else name
        cf_loss,cf_brier,cf_tv=proper_scores(truth['target'],cf)
        behavior_loss=proper_scores(truth['behavior_target'],behavior)[0] if behavior is not None else None
        wrong_loss=proper_scores(truth['target'],wrong)[0] if wrong is not None else None
        incompatible_loss=proper_scores(truth['target'],incompatible)[0] if incompatible is not None else None
        for cell in range(16):
            for draw in sorted(set(ids[:,1])):
                for role in range(3):
                    chosen=(ids[:,0]==cell)&(ids[:,1]==draw)&(ids[:,4]==role)
                    rows.append(dict(method=method,seed=seed,cell=cell,lineage=int(draw),role=role,
                        counterfactual_loss=W.loss_record(float(cf_loss[chosen].mean())),brier=float(cf_brier[chosen].mean()),
                        total_variation=float(cf_tv[chosen].mean()),
                        behavior_loss=None if behavior_loss is None else W.loss_record(float(behavior_loss[chosen].mean())),
                        wrong_mapping_loss=None if wrong_loss is None else W.loss_record(float(wrong_loss[chosen].mean())),
                        incompatible_partition_loss=None if incompatible_loss is None else W.loss_record(float(incompatible_loss[chosen].mean())),probes=int(chosen.sum())))
        pulse(phase='scoring-counterfactual',method=name)
    clusters=[]
    for method in sorted({r['method'] for r in rows}):
        for draw in sorted(set(ids[:,1])):
            chosen=[r for r in rows if r['method']==method and r['lineage']==draw]
            result={}
            for metric in ('counterfactual_loss','behavior_loss','wrong_mapping_loss','incompatible_partition_loss'):
                values=[r[metric] for r in chosen]
                result[metric]=None if any(v is None or v['infinite'] for v in values) else float(np.mean([v['value'] for v in values]))
            for metric in ('brier','total_variation'):result[metric]=float(np.mean([r[metric] for r in chosen]))
            clusters.append(dict(method=method,lineage=int(draw),**result))
    summary={method:{m:stats([r[m] for r in clusters if r['method']==method],('G',method,m)) for m in ('counterfactual_loss','behavior_loss','wrong_mapping_loss','incompatible_partition_loss','brier','total_variation')} for method in sorted({r['method'] for r in rows})}
    path=root/'neural_points.json.gz';path.write_bytes(gzip.compress(canonical(dict(rows=rows,clusters=clusters)),mtime=0))
    write(root/'SUMMARY.json',dict(family='G',cells=summary,raw_sha256=file_digest(path),fits=completed['fits'],
        checks=dict(independent_counterfactual_targets=reference_checks,coordinate_permutation=completed['permutation_checks']),
        independent_unit='coefficient-draw lineage; architectures, exhaustive pairs, roles, queries and fit seeds averaged within draw',
        scope='explicit counterfactual-supervision experiment on valid finite generator states; no uniquely identified internal ontology'))
