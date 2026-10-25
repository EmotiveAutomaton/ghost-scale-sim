"""W12: generated mechanisms, cached stakes, lapse and observed affect."""
import math,itertools
import numpy as np
from .common import rng,softmax,posterior,scores,record,close

TYPES=('instrumental','habit','end','sacred');WEIGHTS=(.5,1.,2.);HYPOTHESES=list(itertools.product(TYPES,WEIGHTS));TRAIN=(1.,2.,4.);OUTSIDE=(.25,.5,8.,16.)

def laws(q,price,devalue=False,substitute=False,novel=False,independent=False):
    values=[]
    for kind,w in HYPOTHESES:
        if kind=='instrumental':value=0. if devalue else q
        elif kind=='habit':value=0. if novel else w*min(TRAIN,key=lambda x:abs(x-q))
        elif kind=='end':value=w*q
        else:value=100*w*q+20.
        alternative=q+.25 if substitute else price
        values.append([value,alternative])
    return .98*softmax(3*np.array(values),independent)+.01

def affect_means(choice,blocked,setting):
    return np.array([[float(kind=='end' and choice==0 and not blocked),float(blocked and ((kind=='habit' and setting['habit_affect']) or (kind=='end' and setting['end_distress'])))] for kind,w in HYPOTHESES])

def block(setting,identity,independent=False):
    r=rng(*identity);out=[];raw=[];truths=[HYPOTHESES.index((kind,setting['weight'])) for kind in TYPES]
    mixes=('trade','devalue','substitute','affect','novel','outside','single-large','many-small')
    for truth in truths:
        for mi,mix in enumerate(mixes):
            ll=np.zeros(12);history=[];hit=None
            for t in range(32):
                q=TRAIN[t%3] if mi<5 else OUTSIDE[t%4] if mix=='outside' else 16. if mix=='single-large' else 1.
                price=(.25,.5,1.,2.,4.,8.)[t%6]
                devalue=1<=mi<=4 and t%4==0;substitute=2<=mi<=4 and t%4==1;novel=mi==4 and t%4==3;observe_affect=3<=mi<=4
                law=laws(q,price,devalue,substitute,novel,independent);choice=int(r.choice(2,p=law[truth]));ll+=np.log(law[:,choice]);blocked=bool(r.random()<.25 and choice==0);signal=None
                if observe_affect:
                    means=affect_means(choice,blocked,setting);signal=means[truth]+r.normal(0,.5,2)
                    if independent:ll+=np.array([math.fsum(-.5*((float(signal[j]-m[j])/.5)**2+math.log(2*math.pi*.25)) for j in range(2)) for m in means])
                    else:ll+=-.5*np.sum((signal-means)**2/.25+math.log(2*math.pi*.25),axis=1)
                p=posterior(ll,independent=independent).reshape(4,3).sum(1);kind=truth//3;contrast=math.log(p[2]/p[1])*(1 if kind==2 else -1 if kind==1 else 0)
                if kind in (1,2) and hit is None and contrast>=1:hit=t+1
                metrics=scores(p,kind);metrics.update(end_vs_habit_evidence=contrast if kind in (1,2) else None)
                for rival in range(4):metrics['evidence_vs_'+TYPES[rival]]=math.log(p[kind]/p[rival])
                out.append(record(metrics,truth=TYPES[kind],mechanism=TYPES[kind],reader=mix,history=t+1))
                history.append(dict(quantity=q,price=price,devalue=devalue,substitute=substitute,novel=novel,choice=choice,blocked=blocked,affect=None if signal is None else signal.tolist()))
            if truth//3 in (1,2):out.append(record(dict(records_needed=hit,not_within_32=float(hit is None)),truth=TYPES[truth//3],mechanism=TYPES[truth//3],reader=mix,history='threshold'))
            raw.append(dict(truth=truth,mix=mix,observations=history))
    return dict(public=dict(hypotheses=HYPOTHESES,trained_quantities=TRAIN,untrained_quantities=OUTSIDE,affect_noise_sd=.5,known_affect_flags=setting,choice_rule='beta3 plus .02 uniform lapse',limitation='Trained-range end/cache choice equality and outside-range separation are built into this declared cache.'),truth_and_observations=raw,readouts=out)

def controls():
    for q in TRAIN:
        for price in (.25,.5,1.,2.,4.,8.):
            p=laws(q,price)
            close(p[3:6].tolist(),p[6:9].tolist(),tol=1e-12)
    for q in (*TRAIN,*OUTSIDE):
        p=laws(q,8.);assert np.max(abs(p[9:,0]-.99))<1e-12
    close(laws(2.,.5).tolist(),laws(2.,.5,independent=True).tolist(),tol=1e-12)
    return dict(passed=True,matched_end_habit_trade_alias=True,sacred_refusal_lapse_limit=True,independent_softmax=True)
