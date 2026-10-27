"""W5: identifiable original labels versus merged, split and unknown pieces."""
import math
import numpy as np
from . import pieces
from .common import rng,softmax,posterior,scores,record,close

def assumed_laws(e,available,kind,independent=False):
    mapping={k:[k] for k in range(6)};effects={k:e[k].copy() for k in range(6)}
    if kind=='merged':mapping[1]=[0];effects[0]=(e[0]+e[1])/2
    if kind=='missing':effects[5]=np.full(3,.5)
    if kind=='split':
        mapping[0]=[0,6];axis=int(np.argmax(e[0]));effects[6]=e[0].copy();effects[0][axis]=min(1,e[0,axis]+.25);effects[6][axis]=max(0,e[0,axis]-.25)
    labels=sorted({label for k in available for label in mapping[k]});values=[]
    for wi,b,h in pieces.HYPOTHESES:
        values.append([b*(math.fsum(float(pieces.WEIGHTS[wi,j]*effects[label][j]) for j in range(3))+int(h>=0 and label in mapping[h])) for label in labels])
    return labels,softmax(np.array(values),independent),mapping

def block(setting,identity,independent=False):
    if 'calibrated_library' in setting:
        from . import wb1
        return wb1.block(setting,identity,independent)
    r=rng(*identity);e=pieces.library('partial',r);kinds=['true','merged','missing']+([] if setting.get('omit_split') else ['split']);out=[];raw=[]
    laws={(kind,si):assumed_laws(e,a,kind,independent) for kind in kinds for si,a in enumerate(pieces.TRAIN)}
    for maker in range(8):
        truth=int(r.choice([i for i,h in enumerate(pieces.HYPOTHESES) if h[1]==setting['beta']]));weight=pieces.HYPOTHESES[truth][0];ll={k:np.zeros(210) for k in kinds};history=[]
        for t in range(512):
            si=t%4;labels,law,_=laws['true',si];piece=int(r.choice(labels,p=law[truth]));coin=int(r.integers(2));observed={}
            for kind in kinds:
                labels,law,mapping=laws[kind,si];group=mapping[piece];label=group[coin] if len(group)==2 else group[0];ll[kind]+=np.log(law[:,labels.index(label)]);observed[kind]=label
            history.append(dict(situation=si,piece=piece,split_coin=coin,reader_labels=observed))
            if t+1 in (8,32,128,512):
                ps={k:posterior(v,independent=independent).reshape(15,14).sum(1) for k,v in ll.items()}
                for kind,p in ps.items():
                    metrics=scores(p,weight);metrics.update(primary=float(math.log(ps['true'][weight])-math.log(p[weight])) if p[weight]>0 and ps['true'][weight]>0 else None,zero_truth_support=float(p[weight]==0),false_confident=float(any(p[j]>.9 for j in range(15) if j!=weight)))
                    out.append(record(metrics,maker=maker,reader=kind,history=t+1))
        raw.append(dict(truth=truth,observations=history))
    return dict(public=dict(effects=e.tolist(),merged='0 and 1, average assumed vector',split='piece 0, fair observation coin, largest effect plus/minus .25',missing='piece 5 represented as neutral other',reader='original 210-state prior; nuisance habit label follows the mapping'),truth_and_observations=raw,readouts=out)

def controls():
    e=pieces.library('partial',rng('label-control'));a=(0,1,2,3,4,5);labels,law,_=assumed_laws(e,a,'true')
    close(law.tolist(),pieces.laws(e,a).tolist(),tol=1e-12)
    permutation=[3,1,5,0,4,2];inverse=np.argsort(permutation);permuted=e[permutation]
    remapped=[(w,b,-1 if h<0 else int(inverse[h])) for w,b,h in pieces.HYPOTHESES]
    p=pieces.laws(permuted,tuple(int(inverse[k]) for k in a),remapped)
    close(p.tolist(),law.tolist(),tol=1e-12)
    from . import wb1
    return dict(passed=True,relabelled_library_exact_zero_loss=True,true_likelihood_equivalence=True,calibrated=wb1.controls())
