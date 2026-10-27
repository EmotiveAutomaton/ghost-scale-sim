"""Prospective W5 calibration from complete, aggregate-only WB1 effects.

No corpus reader and no scientific admission without the received aggregate.
"""
import math,re
from datetime import datetime,timezone
import numpy as np
from . import pieces
from .common import rng,softmax,posterior,scores,record,close

CUTOFF=datetime.fromisoformat('2026-10-28T19:00:00+00:00')
SCHEMA='ghost-v25-wb1-aggregate-1'


def calibrated_library(payload,observed):
    """Validate a reviewed receipt and deterministically construct its library."""
    if observed.tzinfo is None or observed>CUTOFF:raise ValueError('WB1 receipt after cutoff or naive clock')
    if set(payload)!={'schema','complete','controls_passed','effect_basis','source','classes'}:raise ValueError('aggregate-only schema fields required')
    if payload['schema']!=SCHEMA or payload['complete'] is not True or payload['controls_passed'] is not True:raise ValueError('complete passing WB1 result required')
    if payload['effect_basis']!='standardized_mean':raise ValueError('standardized class means required')
    source=payload['source']
    if set(source)!={'project','claim_id','result_sha256'} or source['project']!='SoundingLine':raise ValueError('source receipt required')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',source['claim_id']) or not re.fullmatch(r'[0-9a-f]{64}',source['result_sha256']):raise ValueError('invalid source identity')
    rows=payload['classes']
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty class roster required')
    ids=[];features=None
    for row in rows:
        if set(row)!={'id','events','effects'} or not isinstance(row['id'],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',row['id']):raise ValueError('aggregate class fields required')
        if type(row['events']) is not int or row['events']<=0:raise ValueError('positive event count required')
        ids.append(row['id']);effects=row['effects']
        if not isinstance(effects,dict) or len(effects)<3:raise ValueError('at least three common effect measures required')
        if features is None:features=set(effects)
        if set(effects)!=features:raise ValueError('incomplete effect roster')
        for key,value in effects.items():
            if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',key) or type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite numeric effect means required')
    if len(set(ids))!=len(ids):raise ValueError('duplicate class')
    selected=sorted(rows,key=lambda row:(-row['events'],row['id']))[:6]
    spread={key:max(row['effects'][key] for row in selected)-min(row['effects'][key] for row in selected) for key in features}
    measures=sorted(features,key=lambda key:(-spread[key],key))[:3]
    values=np.array([[row['effects'][key] for key in measures] for row in selected],float)
    cuts=np.quantile(values,[1/3,2/3],axis=0,method='linear')
    quantized=np.where(values<=cuts[0],0.,np.where(values<=cuts[1],.5,1.))
    return dict(class_ids=[row['id'] for row in selected],events=[row['events'] for row in selected],
        measures=measures,spread=[spread[k] for k in measures],terciles=cuts.tolist(),effects=quantized.tolist(),
        original_class_count=len(rows),source=source,
        convention='Top six by event count; ties by class ID. Spread=max-min on retained classes; feature ties lexical. Linear quantiles, equal values stay together; <=lower maps0, <=upper maps.5, above maps1.')


def assumptions(e,available,kind,hypotheses,independent=False):
    count=len(e);mapping={k:[k] for k in range(count)};effects={k:e[k].copy() for k in range(count)}
    if kind=='merged':mapping[1]=[0];effects[0]=(e[0]+e[1])/2
    elif kind=='missing':effects[count-1]=np.full(3,.5)
    elif kind=='split':
        mapping[0]=[0,count];axis=int(np.argmax(e[0]));effects[count]=e[0].copy()
        effects[0][axis]=min(1,e[0,axis]+.25);effects[count][axis]=max(0,e[0,axis]-.25)
    labels=sorted({label for k in available for label in mapping[k]})
    utilities=[[b*(math.fsum(float(pieces.WEIGHTS[w,j]*effects[k][j]) for j in range(3))+int(h>=0 and k in mapping[h])) for k in labels] for w,b,h in hypotheses]
    return labels,softmax(utilities,independent),mapping


def block(setting,identity,independent=False):
    library=setting['calibrated_library'];e=np.asarray(library['effects'],float);count=len(e)
    if not 1<=count<=6 or e.shape!=(count,3) or not np.isin(e,[0,.5,1]).all():raise ValueError('invalid frozen calibrated library')
    hs=[(w,b,h) for w in range(15) for b in (1,3) for h in range(-1,count)]
    situations=list(dict.fromkeys(tuple(k for k in a if k<count) for a in pieces.TRAIN))
    situations=[a for a in situations if a]
    kinds=['true','missing']+(['merged'] if count>=2 else [])+([] if setting.get('omit_split') else ['split'])
    laws={(kind,i):assumptions(e,a,kind,hs,independent) for kind in kinds for i,a in enumerate(situations)}
    # Independent scoring cannot change generated observations through tiny arithmetic differences.
    generated=[assumptions(e,a,'true',hs,False) for a in situations]
    random=rng(*identity);out=[];raw=[]
    for maker in range(8):
        truth=int(random.choice([i for i,h in enumerate(hs) if h[1]==setting['beta']]));weight=hs[truth][0]
        ll={kind:np.zeros(len(hs)) for kind in kinds};history=[]
        for t in range(512):
            si=t%len(situations);labels,law,_=generated[si];piece=int(random.choice(labels,p=law[truth]));coin=int(random.integers(2));observed={}
            for kind in kinds:
                labels,law,mapping=laws[kind,si];group=mapping[piece];label=group[coin] if len(group)==2 else group[0]
                ll[kind]+=np.log(law[:,labels.index(label)]);observed[kind]=label
            history.append(dict(situation=si,piece=piece,split_coin=coin,reader_labels=observed))
            if t+1 in (8,32,128,512):
                ps={kind:posterior(value,independent=independent).reshape(15,2*(count+1)).sum(1) for kind,value in ll.items()}
                for kind,p in ps.items():
                    metrics=scores(p,weight);metrics.update(primary=float(math.log(ps['true'][weight])-math.log(p[weight])) if p[weight]>0 and ps['true'][weight]>0 else None,zero_truth_support=float(p[weight]==0),false_confident=float(any(p[j]>.9 for j in range(15) if j!=weight)))
                    out.append(record(metrics,maker=maker,reader=kind,history=t+1))
        raw.append(dict(truth=truth,observations=history))
    return dict(public=dict(calibration=library,effects=e.tolist(),training=situations,
        interpretation='WB1 calibrates the constructed world library; true/merged/missing/split readers are tested within it. No empirical mechanism is inferred.',
        merge_available=count>=2,reader=f'uniform {len(hs)}-state prior over15 weights,2 betas and{count+1} habits'),
        truth_and_observations=raw,readouts=out)


def controls():
    e=np.array([[0,.5,1],[1,.5,0],[.5,0,1.]])
    hs=[(w,b,h) for w in range(15) for b in (1,3) for h in range(-1,3)]
    labels,law,_=assumptions(e,(0,1,2),'true',hs)
    close(law.tolist(),pieces.laws(e,(0,1,2),hs,independent=True).tolist(),tol=1e-12)
    permutation=[2,0,1];inverse=np.argsort(permutation)
    remapped=[(w,b,-1 if h<0 else int(inverse[h])) for w,b,h in hs]
    close(law.tolist(),pieces.laws(e[permutation],tuple(int(inverse[k]) for k in labels),remapped,independent=True).tolist(),tol=1e-12)
    flat=np.full((3,3),.5);nohabit=[(w,3,-1) for w in range(15)]
    _,null,_=assumptions(flat,(0,1,2),'true',nohabit)
    close(null.tolist(),np.full((15,3),1/3).tolist(),tol=1e-12)
    assert np.max(abs(law[0]-law[-1]))>.01
    return dict(passed=True,independent_likelihood=True,relabelled_exact_zero_loss=True,uniform_null=True,nonconstant_positive=True)
