"""Generator truth and exact references are kept outside learner feature arrays."""
from collections import defaultdict
import gzip
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import canonical,write,digest
from ..v18_4 import neural_data as D
from . import local_world as L
from .readout_model import save_arrays

TIERS=('E0','E1','E2-sparse','E2-full')

def old_features(history):
    seen=set();x=[]
    for obs in history:
        # Program and rule are deliberately absent.
        x.extend(D.onehot(obs['artifact'],range(16))+D.context_features(obs['context'])+[float(obs['source'] not in seen)])
        seen.add(obs['source'])
    return x

def old_posterior(w,history):
    p=np.ones(24)/24;seen=set()
    for obs in history:
        if obs['source'] in seen:continue
        seen.add(obs['source']);p*=W.artifact_matrix(w,obs['context'])[:,obs['artifact']];p/=p.sum()
    return p

def local_features(packet):
    L.validate_public(packet);a=packet['inputs'];x=list(a['artifact'])
    x += [float('initial' in a)]+a.get('initial',[0,0,0])+[a.get('requested_purpose',0)]
    for i in range(3):
        obs=next((r for r in a.get('observations',[]) if r['step']==i),None)
        x += [float(obs is not None)]
        x += ([float(obs['operation']==op) for op in L.OPERATIONS]+obs['before']+obs['after']+(obs['tool_proposal'] or [0,0,0])) if obs else [0.]*15
    return x

def local_targets(record):
    return [float(e['goal']==g) for e in record['steps'] for g in L.GOALS]+[float(e['operation']==op) for e in record['steps'] for op in L.OPERATIONS]

def local_cache(records):
    old=np.zeros((16,4,8))
    for r in records:
        artifact=sum(v<<i for i,v in enumerate(r['final']))
        old[r['maker_index'],r['context_index'],artifact]+=r['probability']*64
    if not np.allclose(old.sum(2),1):raise ValueError('old local outcome teacher unnormalized')
    return old.reshape(16,32)

def local_reference(records,tier,old):
    groups=defaultdict(list)
    for r in records:groups[digest(L.project(r,tier))].append(r)
    out={}
    for key,rr in groups.items():
        w=np.array([r['probability'] for r in rr]);w/=w.sum()
        out[key]=(w@np.array([old[r['maker_index']] for r in rr]),w@np.array([local_targets(r) for r in rr]))
    return out

def generate(root,cfg,pulse):
    kind=cfg['domain'];datasets={};evaluator=[]
    for split,lineages,count in [('train',cfg['train_lineages'],16),('development',cfg['development_lineages'],32)]:
        for draw in cfg['training_draws']:
            store={tier:defaultdict(list) for tier in (TIERS if kind=='local' else ('artifact-history',))}
            for lineage in lineages:
                pulse(phase='readout-data',split=split,draw=draw,lineage=lineage)
                random=W.rng('v19-readout-data',kind,split,draw,lineage)
                if kind=='local':
                    records=L.enumerate_world(L.law(lineage));old=local_cache(records)
                    refs={tier:local_reference(records,tier,old) for tier in TIERS}
                    weights=np.array([r['probability'] for r in records]);weights/=weights.sum()
                    choices=random.choice(len(records),size=count,p=weights)
                    prior=np.mean(np.array([local_targets(r) for r in records])*np.array([r['probability'] for r in records])[:,None],axis=0)*len(records)
                    # Full enumerated support retained for independent regrouping.
                    raw=root/'evaluator'/f'{split}-{draw}-{lineage}_points.json.gz';raw.parent.mkdir(parents=True,exist_ok=True)
                    raw.write_bytes(gzip.compress(canonical(records),mtime=0))
                    for i,index in enumerate(choices):
                        r=records[int(index)]
                        for tier in TIERS:
                            packet=L.project(r,tier);x=local_features(packet);bank,exact=refs[tier][digest(packet)]
                            s=store[tier];s['x'].append(x);s['old'].append(old[r['maker_index']]);s['bank'].append(bank)
                            s['target'].append(local_targets(r));s['exact'].append(exact);s['prior'].append(prior)
                            s['ids'].append([lineage,i,0])
                        evaluator.append(dict(split=split,draw=draw,lineage=lineage,index=i,trajectory_index=int(index)))
                else:
                    for i in range(count):
                        cell=i%16;w=W.make_world(cell,lineage);state=int(random.integers(24))
                        history=D.history(w,state,random,8);posterior=old_posterior(w,history)
                        old=np.concatenate([W.artifact_matrix(w,q) for q in D.TRAIN_QUERIES],1)
                        target=np.concatenate([W.artifact_matrix(w,q) for q in D.FAR_QUERIES],1)
                        s=store['artifact-history'];s['x'].append(old_features(history));s['old'].append(old[state]);s['bank'].append(posterior@old)
                        s['target'].append(target[state]);s['exact'].append(posterior@target);s['prior'].append(target.mean(0))
                        rule,_,endogenous,_=W.FACTORS[cell];s['ids'].append([lineage,i,2*rule+endogenous])
                        evaluator.append(dict(split=split,draw=draw,lineage=lineage,index=i,world=w,state=state,history=history))
            for tier,s in store.items():
                arrays={k:np.asarray(v,dtype=np.int64 if k=='ids' else np.float64) for k,v in s.items()}
                # Deterministic training permutation makes nested budgets broad across lineages.
                if split=='train':
                    order=W.rng('v19-nested',kind,draw).permutation(len(arrays['x']))
                    arrays={k:v[order] for k,v in arrays.items()}
                role=root/'reader'/(f'{split}-{draw}-{tier}.npz')
                allowed=('x','old','target','ids') if split=='train' else ('x','ids')
                save_arrays(role,**{k:arrays[k] for k in allowed})
                save_arrays(root/'evaluator'/f'{split}-{draw}-{tier}.npz',**arrays)
                datasets[(split,draw,tier)]=arrays
    write(root/'INPUT_SCHEMA.json',dict(domain=kind,reader_train=['x','old observable response teacher','new target teacher','ids for pairing only'],
        reader_development=['x','ids for pairing only'],evaluator_only=['state','local goals','selected operations when unwitnessed','exact bank','exact new answers','prior'],
        prohibited_features=['rule','program','policy matrix','maker','local goal'],tiers=list(store)))
    (root/'evaluator/selection_points.json.gz').write_bytes(gzip.compress(canonical(evaluator),mtime=0))
    return datasets

def controls():
    packet=dict(schema='v19.local.public.1',tier='E0',inputs=dict(artifact=[0,1,0]))
    passed=False
    try:L.validate_public(dict(packet,private_state=2))
    except ValueError:passed=True
    a=local_features(packet);b=local_features(dict(packet,inputs=dict(artifact=[1,1,0])))
    return dict(placebo_private_field_rejected=passed,live_visible_feature_changes=a!=b,
        artifact_only_feature_width=len(a)<512)
