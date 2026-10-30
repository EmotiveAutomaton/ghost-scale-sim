"""Prospective generation-one population-size and affect-channel diagnostics."""
import math
import numpy as np
from . import population,pieces,sacred,population_precision
from .common import rng,softmax,posterior,scores,record,close


def population_size(setting,identity,independent=False):
    """Parent W4 with its population size varied; both null panels remain 256."""
    size=setting['population_size']
    if type(size) is not int or size<2:raise ValueError('population needs at least two members')
    weights=population.WEIGHTS;deviation=population.DEVIATION
    r=rng(*identity);e=pieces.library('partial',r);alpha=setting['alpha']
    prior=softmax(np.sum((alpha*np.array([.5,.3,.2])-1)*np.log(weights),axis=1))
    near=prior*(deviation<=1);near/=near.sum();generating_prior=near if setting['placebo'] else prior
    main=r.choice(36,size,p=generating_prior)
    truth=np.r_[main,r.choice(36,256,p=near),np.full(256,int(np.flatnonzero(deviation==0)[0])),main]
    excluded=r.integers(0,size,512);habits=np.r_[np.zeros(size+512,dtype=int),r.integers(1,7,size)];planted=7*truth+habits
    count=np.zeros((2*size+512,4,6),dtype=np.int16);hs=[(w,3,h) for w in range(36) for h in range(-1,6)]
    joint_laws=[pieces.laws(e,a,hs,weights=weights,independent=independent) for a in pieces.TRAIN]
    generated=[pieces.laws(e,a,hs,weights=weights) for a in pieces.TRAIN];laws=[v[::7] for v in joint_laws]
    ll=np.zeros((2*size+512,36));joint_ll=np.zeros((2*size+512,252));out=[];increments=[];previous=0
    for n in setting.get('prefixes',[8,32,128,512,1024]):
        delta=np.zeros_like(count)
        for si,available in enumerate(pieces.TRAIN):
            trials=(n+3-si)//4-(previous+3-si)//4
            for h in sorted(set(planted.tolist())):
                idx=np.flatnonzero(planted==h);delta[idx,si,:len(available)]=r.multinomial(trials,generated[si][h],size=len(idx))
            loglaw=np.log(laws[si]);logjoint=np.log(joint_laws[si]);counts=delta[:,si,:len(available)]
            if independent:
                for option in range(len(available)):
                    ll+=counts[:,option,None]*loglaw[:,option];joint_ll+=counts[:,option,None]*logjoint[:,option]
            else:ll+=counts@loglaw.T;joint_ll+=counts@logjoint.T
        count+=delta;increments.append(dict(records=n,category_counts=delta.tolist()));previous=n
        individual=softmax(ll,independent);total=individual[:size].sum(0);empirical=np.empty_like(individual)
        empirical[:size]=(total-individual[:size])/(size-1)
        empirical[size:size+512]=(total-individual[excluded])/(size-1)
        empirical[size+512:]=(total-individual[:size])/(size-1)
        with np.errstate(divide='ignore'):inferred=posterior(ll,empirical,independent)
        oracle=posterior(ll,generating_prior,independent);rates={}
        for name,post in [('individual',individual),('estimated',inferred),('oracle',oracle)]:
            ranked=population_precision.rounded_flags(post,e,count,main,excluded,alpha,setting['placebo'],name)
            calibration=ranked[size:size+256];cut=population.threshold(calibration)
            pop_flags=population.flags(post[:size]);detections=population.detect(ranked[:size],cut)
            exact=population.detect(ranked[size+256:size+512],cut);dev=deviation[main]
            rates[name]=float(np.mean(detections[dev>=2])) if np.any(dev>=2) else None
            out.append(record(dict(detection_ge_two=rates[name],population_near_mean_false_detection=float(np.mean(detections[dev<=1])) if np.any(dev<=1) else None,
                exact_mean_false_detection=float(np.mean(exact)),calibration_false_detection=float(np.mean(population.detect(calibration,cut))),raw_point95_detection=float(np.mean(pop_flags>.95+1e-12)),raw_point95_near_mean_false_detection=float(np.mean(population.flags(post[size:size+256])>.95+1e-12)),raw_point95_exact_mean_false_detection=float(np.mean(population.flags(post[size+256:size+512])>.95+1e-12)),threshold=cut[0],threshold_tie_mass=cut[1]),reader=name,history=n))
            for d in sorted(set(dev.tolist())):out.append(record(dict(detection=float(np.mean(detections[dev==d])),makers=int(np.sum(dev==d))),reader=name,history=n,deviation_tenths=int(d)))
            correct=post[np.arange(size),main];initial=np.full(size,1/36) if name=='individual' else generating_prior[main] if name=='oracle' else empirical[np.arange(size),main]
            gains=[math.log(x/y) for x,y in zip(correct,initial) if x>0 and y>0]
            out.append(record(dict(true_mass=float(np.mean(correct)),log_gain=math.fsum(gains)/len(gains) if gains else None,unavailable_log_scores=size-len(gains),half_brier=float(np.mean((np.sum(post[:size]**2,axis=1)-2*correct+1)/2))),reader=name,history=n,measure='posterior'))
        out.append(record(dict(primary=rates['estimated']-rates['individual'] if rates['estimated'] is not None else None),reader='matched_false_detection_contrast',history=n))
        habit_post=softmax(joint_ll,independent).reshape(2*size+512,36,7).sum(1);flag=1-habit_post[:,0];cut=population.threshold(flag[size:size+256])
        out.append(record(dict(habit_detection=float(np.mean(population.detect(flag[size+512:],cut))),habit_false_detection=float(np.mean(population.detect(flag[:size],cut))),habit_raw_point95_detection=float(np.mean(flag[size+512:]>.95)),habit_raw_point95_false_detection=float(np.mean(flag[:size]>.95))),reader='own_ends_vs_habit',history=n))
    return dict(public=dict(effects=e.tolist(),grid_tenths=population.TENTHS.tolist(),center_tenths=population.MU.tolist(),population_prior=generating_prior.tolist(),population_size=size,
        threshold_rule='Independent256 near-mean calibration, expected randomized ties at5%; independent256 exact-mean validation. Main priors leave self out; nulls each exclude one fixed randomly chosen population member.',
        roster=f'{size} population,256 near-mean threshold,256 exact-mean validation,{size} paired-weight habit makers.'),
        truth=dict(weights=truth.tolist(),habits=habits.tolist(),null_excluded_population_member=excluded.tolist()),observations=increments,readouts=out)


CHANNELS={'neither':(),'pleasure':(0,),'distress':(1,),'both':(0,1)}


def gaussian_logs(signal,means,sd,channels,independent=False):
    if not channels:return np.zeros(len(means))
    if independent:
        return np.array([math.fsum(-.5*((float(signal[j]-mean[j])/sd)**2+math.log(2*math.pi*sd*sd)) for j in channels) for mean in means])
    return -.5*np.sum((signal[list(channels)]-means[:,list(channels)])**2/(sd*sd)+math.log(2*math.pi*sd*sd),axis=1)


def affect_channels(setting,identity,independent=False):
    r=rng(*identity);sd=setting['noise_sd'];out=[];raw=[]
    if sd<=0 or not math.isfinite(sd):raise ValueError('finite positive affect noise required')
    truths=[sacred.HYPOTHESES.index((kind,w)) for kind in ('habit','end') for w in sacred.WEIGHTS]
    for maker,truth in enumerate(truths):
        ll={name:np.zeros(12) for name in CHANNELS};first={name:None for name in CHANNELS};history=[];kind=truth//3
        for t in range(32):
            quantity=sacred.TRAIN[t%3];price=(.25,.5,1.,2.,4.,8.)[t%6]
            generated=sacred.laws(quantity,price);law=sacred.laws(quantity,price,independent=independent)
            choice=int(r.choice(2,p=generated[truth]));blocked=bool(r.random()<.25 and choice==0)
            means=sacred.affect_means(choice,blocked,setting);signal=means[truth]+r.normal(0,sd,2)
            for name,channels in CHANNELS.items():
                ll[name]+=np.log(law[:,choice])+gaussian_logs(signal,means,sd,channels,independent)
                p=posterior(ll[name],independent=independent).reshape(4,3).sum(1)
                # Stable log odds are aggregated in log space, including very strong signals.
                def logsum(x):
                    top=float(np.max(x));return top+math.log(math.fsum(math.exp(float(v-top)) for v in x))
                odds=(logsum(ll[name][6:9])-logsum(ll[name][3:6]))*(1 if kind==2 else -1)
                if first[name] is None and odds>=1:first[name]=t+1
                if t+1 in (4,8,16,32):
                    metrics=scores(p,kind);metrics.update(primary=odds,false_confident=float(any(p[j]>.9 for j in range(4) if j!=kind)))
                    out.append(record(metrics,maker=maker,mechanism=sacred.TYPES[kind],reader=name,history=t+1))
            history.append(dict(quantity=quantity,price=price,choice=choice,blocked=blocked,affect=signal.tolist()))
        for name in CHANNELS:out.append(record(dict(records_needed=first[name],not_within_32=float(first[name] is None)),maker=maker,mechanism=sacred.TYPES[kind],reader=name,history='threshold'))
        raw.append(dict(truth=truth,observations=history))
    return dict(public=dict(hypotheses=sacred.HYPOTHESES,channels=CHANNELS,noise_sd=sd,known_flags={k:setting[k] for k in ('habit_affect','end_distress')},
        choice_rule='Native beta3/.02 lapse; trained-range trade only; paired observations across four reader ablations.',limitation='Pleasure is a planted end-only signal, not an empirical psychological identification.'),truth_and_observations=raw,readouts=out)


def controls():
    assert population.controls()['passed'] and sacred.controls()['passed']
    means=np.array([[0.,0.],[1.,0.]]);signal=np.array([.4,.3])
    close(gaussian_logs(signal,means,.5,(0,1)).tolist(),gaussian_logs(signal,means,.5,(0,1),True).tolist(),tol=1e-12)
    close(gaussian_logs(signal,means,.5,(1,)).tolist(),[gaussian_logs(signal,means,.5,(1,))[0]]*2,tol=1e-12)
    # Exact expected log ratio for equal-variance Gaussian means differs by d^2/(2 sigma^2).
    for sd in (.25,.5,1.,2.):
        expected=gaussian_logs(np.array([1.,0.]),means,sd,(0,))[1]-gaussian_logs(np.array([1.,0.]),means,sd,(0,))[0]
        assert abs(expected-1/(2*sd*sd))<1e-12 and expected>0
    close(gaussian_logs(signal,means,.5,()).tolist(),[0.,0.],tol=1e-12)
    return dict(passed=True,parent_known_answers=True,gaussian_density_independent=True,equal_channel_exact_alias=True,no_channel_zero_information=True,gaussian_positive_information=True)
