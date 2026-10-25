"""W4: leave-one-out priors with independent near-mean threshold panels."""
import math
import numpy as np
from . import pieces
from .common import rng,simplex,softmax,posterior,record,close

WEIGHTS=simplex(10,interior=True);TENTHS=np.rint(10*WEIGHTS).astype(int);MU=np.array([5,3,2]);DEVIATION=np.max(abs(TENTHS-MU),axis=1)
TAILS=np.stack([TENTHS[:,j]>MU[j] for j in range(3)]+[TENTHS[:,j]<MU[j] for j in range(3)],axis=1).astype(float)

def flags(post):return np.max(post@TAILS,axis=1)

def threshold(values):
    values=np.round(np.asarray(values),8);cut=float(np.sort(values)[math.ceil(.95*len(values))-1]);above=np.sum(values>cut);ties=np.sum(values==cut)
    fraction=(.05*len(values)-above)/ties
    if not -1e-12<=fraction<=1+1e-12:raise ValueError('invalid matched-specificity tie mass')
    return cut,float(np.clip(fraction,0,1))

def detect(values,cut):
    value,tie=cut;values=np.round(np.asarray(values),8)
    return (values>value).astype(float)+tie*(values==value)

def block(setting,identity,independent=False):
    r=rng(*identity);e=pieces.library('partial',r);alpha=setting['alpha'];logprior=np.sum((alpha*np.array([.5,.3,.2])-1)*np.log(WEIGHTS),axis=1);prior=softmax(logprior)
    near=prior*(DEVIATION<=1);near/=near.sum();population_prior=near if setting['placebo'] else prior
    main=r.choice(36,64,p=population_prior);truth=np.r_[main,r.choice(36,256,p=near),np.full(256,int(np.flatnonzero(DEVIATION==0)[0])),main]
    excluded=r.integers(0,64,512);habits=np.r_[np.zeros(576,dtype=int),r.integers(1,7,64)];planted=7*truth+habits
    count=np.zeros((640,4,6),dtype=np.int16);hypotheses=[(w,3,h) for w in range(36) for h in range(-1,6)]
    joint_laws=[pieces.laws(e,a,hypotheses,weights=WEIGHTS,independent=independent) for a in pieces.TRAIN]
    generator_laws=[pieces.laws(e,a,hypotheses,weights=WEIGHTS) for a in pieces.TRAIN]
    laws=[v[::7] for v in joint_laws];ll=np.zeros((640,36));joint_ll=np.zeros((640,252));out=[];increments=[];previous=0
    for n in setting.get('prefixes',[8,32,128,512,1024]):
        delta=np.zeros_like(count)
        for si,available in enumerate(pieces.TRAIN):
            trials=(n+3-si)//4-(previous+3-si)//4
            for h in sorted(set(planted.tolist())):
                idx=np.flatnonzero(planted==h);delta[idx,si,:len(available)]=r.multinomial(trials,generator_laws[si][h],size=len(idx))
            loglaw=np.log(laws[si]);logjoint=np.log(joint_laws[si]);counts=delta[:,si,:len(available)]
            if independent:
                for option in range(len(available)):
                    ll+=counts[:,option,None]*loglaw[:,option];joint_ll+=counts[:,option,None]*logjoint[:,option]
            else:ll+=counts@loglaw.T;joint_ll+=counts@logjoint.T
        count+=delta;increments.append(dict(records=n,category_counts=delta.tolist()));previous=n
        individual=softmax(ll,independent);total=individual[:64].sum(0)
        empirical=np.empty_like(individual);empirical[:64]=(total-individual[:64])/63
        empirical[64:576]=(total-individual[excluded])/63;empirical[576:]=(total-individual[:64])/63
        with np.errstate(divide='ignore'):inferred=posterior(ll,empirical,independent)
        oracle=posterior(ll,population_prior,independent);rates={}
        for name,post in [('individual',individual),('estimated',inferred),('oracle',oracle)]:
            calibration=flags(post[64:320]);cut=threshold(calibration);population_flags=flags(post[:64]);detections=detect(population_flags,cut);exact=detect(flags(post[320:576]),cut);dev=DEVIATION[main]
            rates[name]=float(np.mean(detections[dev>=2])) if np.any(dev>=2) else None
            out.append(record(dict(detection_ge_two=rates[name],population_near_mean_false_detection=float(np.mean(detections[dev<=1])) if np.any(dev<=1) else None,
                exact_mean_false_detection=float(np.mean(exact)),calibration_false_detection=float(np.mean(detect(calibration,cut))),raw_point95_detection=float(np.mean(population_flags>.95+1e-12)),raw_point95_near_mean_false_detection=float(np.mean(calibration>.95+1e-12)),raw_point95_exact_mean_false_detection=float(np.mean(flags(post[320:576])>.95+1e-12)),threshold=cut[0],threshold_tie_mass=cut[1]),reader=name,history=n))
            for d in sorted(set(dev.tolist())):out.append(record(dict(detection=float(np.mean(detections[dev==d])),makers=int(np.sum(dev==d))),reader=name,history=n,deviation_tenths=int(d)))
            correct=post[np.arange(64),main];prior_prob=(np.full(64,1/36) if name=='individual' else population_prior[main] if name=='oracle' else empirical[np.arange(64),main])
            gains=[math.log(x/y) for x,y in zip(correct,prior_prob) if x>0 and y>0]
            out.append(record(dict(true_mass=float(np.mean(correct)),log_gain=math.fsum(gains)/len(gains) if gains else None,unavailable_log_scores=64-len(gains),half_brier=float(np.mean((np.sum(post[:64]**2,axis=1)-2*correct+1)/2))),reader=name,history=n,measure='posterior'))
        out.append(record(dict(primary=rates['estimated']-rates['individual'] if rates['estimated'] is not None else None),reader='matched_false_detection_contrast',history=n))
        habit_post=softmax(joint_ll,independent).reshape(640,36,7).sum(1);habit_flag=1-habit_post[:,0];habit_cut=threshold(habit_flag[64:320])
        out.append(record(dict(habit_detection=float(np.mean(detect(habit_flag[576:],habit_cut))),habit_false_detection=float(np.mean(detect(habit_flag[:64],habit_cut))),habit_raw_point95_detection=float(np.mean(habit_flag[576:]>.95)),habit_raw_point95_false_detection=float(np.mean(habit_flag[:64]>.95))),reader='own_ends_vs_habit',history=n))
    return dict(public=dict(effects=e.tolist(),grid_tenths=TENTHS.tolist(),center_tenths=MU.tolist(),population_prior=population_prior.tolist(),threshold_rule='Exactly .05 empirical near-mean calibration rate using expected randomized tie credit; probability scores rounded to8 decimal places before exact ranking/ties. Null makers set thresholds only.',roster='64 population,256 near-mean threshold,256 exact-mean validation,64 paired-weight habit makers. Every null uses a fixed random63 of the original64 population makers.',habit_reader='Uniform36 weights times7 habits; own-ends deviation reported separately at matched near-mean null specificity.'),truth=dict(weights=truth.tolist(),habits=habits.tolist(),null_excluded_population_member=excluded.tolist()),observations=increments,readouts=out)

def controls():
    assert len(WEIGHTS)==36 and np.sum(DEVIATION<=1)==7
    ll=np.arange(72,dtype=float).reshape(2,36)/10
    a=posterior(ll);b=posterior(ll,np.full(36,1/36),True)
    close(a.tolist(),b.tolist(),tol=1e-12);close(flags(a).tolist(),flags(b).tolist(),tol=1e-12)
    values=np.r_[np.zeros(244),np.ones(12)];cut=threshold(values)
    assert abs(np.mean(detect(values,cut))-.05)<1e-12
    return dict(passed=True,integer_grid_and_seven_point_null=True,uniform_empirical_prior_equivalence=True,exact_five_percent_tie_calibration=True)
