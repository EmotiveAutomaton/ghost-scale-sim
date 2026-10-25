"""W3: joint effect/source inference, omission, global trust and outcomes only."""
import itertools,math
import numpy as np
from . import pieces
from .common import rng,softmax,posterior,record,close

LEARNER=np.array([1.,1.,-.5])
SOURCE=np.array([[1,1,-.5],[1,1,0],[1,1,.25],[1,1,.5],[-1,-1,.5],[-1,-1,0],[.5,1,-.5],[1,.5,-.5]])

def libraries(bundling,r):
    e=np.array([[1,.5,0],[.5,1,0],[.5,0,0],[0,.5,0],[0,0,1],[0,0,.5]],float)
    if bundling=='bundled':e[:2]=[[1,0,.5],[0,1,.5]]
    e=e[r.permutation(6)];pairs=list(itertools.combinations(range(6),2));chosen=r.choice(len(pairs),7,replace=False);candidates=[e]
    for index in chosen:
        swapped=e.copy();i,j=pairs[index];swapped[[i,j]]=swapped[[j,i]];candidates.append(swapped)
    assert len({a.tobytes() for a in candidates})==8
    return np.array(candidates)

def plan_policy(libraries,belief,available,independent=False):
    if independent:
        expected=[math.fsum(float(belief[h])*math.fsum(float(libraries[h,k,j]*LEARNER[j]) for j in range(3)) for h in range(len(belief))) for k in available]
    else:expected=(belief@np.einsum('hkj,j->hk',libraries,LEARNER))[list(available)]
    return softmax(3*np.array(expected),independent)

def block(setting,identity,independent=False):
    r=rng(*identity);out=[];raw=[];truth={'aligned':0,'partial':3,'opposed':4}[setting['alignment']]
    for maker in range(8):
        lib=libraries(setting['bundling'],r);source_laws=[softmax(3*np.einsum('lkj,sj->lsk',lib[:,list(a)],SOURCE),independent) for a in pieces.TRAIN]
        observed_ll=np.zeros(8);joint=np.zeros((8,8));counts=np.zeros(6);success=1.;failure=1.;history=[]
        prior_utility=np.mean(lib@LEARNER,axis=0);threshold=float(max(prior_utility));keep=np.array([i for i,w in enumerate(SOURCE) if w[2]!=SOURCE[truth,2]])
        for t in range(max(setting.get('prefixes',[8,32,128]))):
            si=t%4;a=pieces.TRAIN[si];local=int(r.choice(len(a),p=source_laws[si][0,truth]));piece=a[local];counts[piece]+=1;joint+=np.log(source_laws[si][:,:,local]);observation=None
            if (t+1)%4==0:
                observation=lib[0,piece]+r.normal(0,.5,3)
                if independent:ll=np.array([math.fsum(-.5*((float(observation[j]-e[piece,j])/.5)**2+math.log(2*math.pi*.25)) for j in range(3)) for e in lib])
                else:ll=-.5*np.sum((observation-lib[:,piece])**2/.25+math.log(2*math.pi*.25),axis=1)
                observed_ll+=ll;joint+=ll[:,None]
                if float(observation@LEARNER)>threshold:success+=1
                else:failure+=1
            history.append(dict(situation=si,piece=piece,outcome=None if observation is None else observation.tolist()))
            if t+1 not in setting.get('prefixes',[8,32,128]):continue
            beliefs=dict(reject=np.full(8,1/8),outcomes=posterior(observed_ll,independent=independent),joint=posterior(joint.ravel(),independent=independent).reshape(8,8).sum(1),omitted=posterior(joint[:,keep].ravel(),independent=independent).reshape(8,len(keep)).sum(1))
            vals={k:[] for k in ('adopt','reject','trust','joint','outcomes','omitted')}
            for available in pieces.SWITCH:
                oracle=plan_policy(lib,np.eye(8)[0],available,independent);own=plan_policy(lib,beliefs['reject'],available,independent)
                copy=counts[list(available)];copy=copy/copy.sum() if copy.sum() else np.full(len(available),1/len(available))
                policies={k:plan_policy(lib,b,available,independent) for k,b in beliefs.items()};policies.update(adopt=copy,trust=success/(success+failure)*copy+failure/(success+failure)*own)
                utility=lib[0,list(available)]@LEARNER;negative=lib[0,list(available),2]
                for name,p in policies.items():vals[name].append((float((oracle-p)@utility),float((p-oracle)@negative)))
            means={k:np.mean(v,axis=0) for k,v in vals.items()}
            for name,(regret,capture) in means.items():out.append(record(dict(regret=float(regret),capture=float(capture)),maker=maker,reader=name,history=t+1))
            out.append(record(dict(primary=float(means['omitted'][1]-means['outcomes'][1]),joint_minus_outcomes_capture=float(means['joint'][1]-means['outcomes'][1]),joint_minus_trust_capture=float(means['joint'][1]-means['trust'][1]),joint_minus_trust_regret=float(means['joint'][0]-means['trust'][0])),maker=maker,reader='contrasts',history=t+1))
        raw.append(dict(libraries=lib.tolist(),source_weight=SOURCE[truth].tolist(),omitted_weight_indices=keep.tolist(),observations=history))
    return dict(public=dict(learner_weights=LEARNER.tolist(),source_weights=SOURCE.tolist(),choice_rule='beta 3 softmax',noise_sd=.5,heldout=pieces.SWITCH),truth_and_observations=raw,readouts=out)

def controls():
    lib=libraries('bundled',rng('known-library-fixture'));point=np.eye(8)[0];a=(0,1,2,3,4,5)
    oracle=softmax(3*lib[0]@LEARNER)
    close(plan_policy(lib,point,a).tolist(),oracle.tolist(),tol=1e-12)
    ll=np.arange(8,dtype=float);outcome=posterior(ll,point)
    close(plan_policy(lib,outcome,a).tolist(),oracle.tolist(),tol=1e-12)
    close(plan_policy(lib,np.full(8,1/8),a).tolist(),plan_policy(lib,np.full(8,1/8),a,True).tolist(),tol=1e-12)
    return dict(passed=True,point_truth_joint_oracle=True,point_truth_reject_outcomes_equal=True,independent_utility=True,
        limitation='Own-weight planning does not generally guarantee less capture than global trust at finite evidence; that comparison is measured, not asserted as a theorem.')
