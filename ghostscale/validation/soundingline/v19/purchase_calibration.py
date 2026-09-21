"""Whole-stream calibration only; evaluator truth never enters a sensor."""
from itertools import product
import gzip,math
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import canonical,write,file_digest,read
from ..v18_4 import distinct_families as D

SENSORS=('raw','centered')
PRIORS=('equal-class','duplicate-supplied')

def unique_history(history):
    seen={}
    for obs in history:
        if obs['source'] in seen and obs!=seen[obs['source']]:raise ValueError('conflicting source copy')
        seen.setdefault(obs['source'],obs)
    return list(seen.values())

def rolling(surprise,entropy):
    if len(surprise)!=32 or len(entropy)!=32:raise ValueError('32 unique observations required')
    s=np.asarray(surprise);e=np.asarray(entropy)
    values={key:[float(np.mean((s if key=='raw' else s-e)[t-4:t])) for t in range(8,32)] for key in SENSORS}
    return dict(values=values,maxima={key:max(v) for key,v in values.items()})

def sensor(public):
    history=unique_history(public['history']);worlds=D.catalog(public['world'])[:2]
    if len(history)!=32 or public['prior_mode'] not in PRIORS:raise ValueError('invalid sensor input')
    prior=np.array([2.,1.]) if public['prior_mode']=='duplicate-supplied' else np.ones(2)
    weights=np.ones((2,len(W.STATES)))/len(W.STATES);scores=np.zeros(2);predictions=[];surprises=[];entropies=[]
    for obs in history:
        matrix=np.array([W.matrix(w,obs['context']) for w in worlds])
        p=np.einsum('m,ms,msp->p',D.mixture(scores+np.log(prior)),weights,matrix)
        index=W.PROGRAMS.index(tuple(obs['program']))
        predictions.append(p.tolist());surprises.append(-math.log(float(p[index])));entropies.append(float(-np.sum(p[p>0]*np.log(p[p>0]))))
        weights,scores=D.update(weights,scores,matrix[:,:,index])
    return dict(predictions=predictions,surprises=surprises,entropies=entropies,**rolling(surprises,entropies),likelihood_evaluations=2*len(W.STATES)*32)

def verify(public,result):
    """Independent scalar batch-product filter, proper-alphabet sensor and maxima."""
    history=unique_history(public['history']);worlds=D.catalog(public['world'])[:2];prior=[2.,1.] if public['prior_mode']=='duplicate-supplied' else [1.,1.]
    laws=[[W.matrix(w,o['context']) for o in history] for w in worlds]
    observed=[W.PROGRAMS.index(tuple(o['program'])) for o in history];surprises=[];entropies=[]
    for t in range(32):
        logs=[[math.log(prior[k]/len(W.STATES))+math.fsum(math.log(float(laws[k][j][s,observed[j]])) for j in range(t)) for s in range(len(W.STATES))] for k in range(2)]
        top=max(x for row in logs for x in row);mass=[[math.exp(x-top) for x in row] for row in logs];den=math.fsum(x for row in mass for x in row)
        p=[math.fsum(mass[k][s]*float(laws[k][t][s,a])/den for k in range(2) for s in range(len(W.STATES))) for a in range(len(W.PROGRAMS))]
        if max(abs(x-y) for x,y in zip(p,result['predictions'][t]))>1e-10 or abs(math.fsum(p)-1)>1e-10:raise ValueError('independent predictive distribution differs')
        surprises.append(-math.log(p[observed[t]]));entropies.append(-math.fsum(x*math.log(x) for x in p if x>0))
    if max(abs(x-y) for x,y in zip(surprises,result['surprises']))>1e-10 or max(abs(x-y) for x,y in zip(entropies,result['entropies']))>1e-10:raise ValueError('scalar sensor differs')
    for key in SENSORS:
        values=[math.fsum(surprises[j]-(entropies[j] if key=='centered' else 0) for j in range(t-4,t))/4 for t in range(8,32)]
        if max(abs(x-y) for x,y in zip(values,result['values'][key]))>1e-10 or abs(max(values)-result['maxima'][key])>1e-10:raise ValueError('eligible maximum differs')
    return True

def threshold(maxima):
    a=np.sort(np.asarray(maxima,float))
    if len(a)<10 or not np.isfinite(a).all():raise ValueError('invalid threshold population')
    allowed=len(a)//10;bar=float(a[len(a)-allowed-1]);purchases=int(np.sum(a>bar))
    return dict(threshold=bar,streams=len(a),strict_exceedances=purchases,rate=purchases/len(a),ties=int(np.sum(a==bar)),allowed_exceedances=allowed,comparison='strict greater than')

def controls():
    p=np.array([[.5,.5]]);q,_=D.update(p,np.zeros(1),np.array([[.1,.9]]))
    inert,_=D.update(p,np.zeros(1),np.array([[.5,.5]]))
    no=rolling([math.log(2)]*32,[math.log(2)]*32)
    return {'live:informative_filter':bool(np.allclose(q,[[.1,.9]])), 'placebo:uniform_filter':bool(np.array_equal(p,inert)), 'placebo:entropy_centered_zero':no['maxima']['centered']==0., 'live:threshold_ties':threshold([1.]*128)['strict_exceedances']==0}

def run(root,plan,pulse):
    cfg=plan['design'];indices=cfg['calibration_indices'];evaluation=cfg['evaluation_indices']
    if set(indices)&set(evaluation) or len(indices)!=128 or len(evaluation)!=20:raise ValueError('invalid disjoint frozen roster')
    checks=controls();(root/'raw').mkdir(exist_ok=True);(root/'reader').mkdir(exist_ok=True)
    strata={(cell,order,prior,key):[] for cell,order,prior,key in product(cfg['cells'],cfg['orders'],cfg['priors'],SENSORS)}
    raw_files={};reader_files={};count=0
    for index in indices:
        pulse(phase='calibration',completed_streams=count);records=[]
        for cell,order,prior in product(cfg['cells'],cfg['orders'],cfg['priors']):
            public,truth=D.make_case(index,cell,'in-family',order,1,32);public['prior_mode']=prior
            result=sensor(public);verify(public,result)
            data=canonical(public);import hashlib
            h=hashlib.sha256(data).hexdigest();path=root/'reader'/f'{h}.json'
            if not path.exists():path.write_bytes(data)
            reader_files[path.relative_to(root).as_posix()]=file_digest(path)
            records.append(dict(index=index,cell=cell,order=order,prior=prior,public_sha256=h,evaluator=truth,sensor=result))
            for key in SENSORS:strata[cell,order,prior,key].append(dict(index=index,maximum=result['maxima'][key]))
            count+=1
        path=root/'raw'/f'calibration-{index}_points.json.gz';path.write_bytes(gzip.compress(canonical(records),mtime=0));raw_files[path.relative_to(root).as_posix()]=file_digest(path)
    bars=[]
    for (cell,order,prior,key),rows in strata.items():
        bar=threshold([r['maximum'] for r in rows]);bars.append(dict(cell=cell,order=order,prior=prior,sensor=key,**bar))
    write(root/'THRESHOLDS.json',dict(schema='v19.purchase-thresholds.1',calibration_indices=indices,evaluation_indices=evaluation,strata=bars,calibration_raw=raw_files,scope='frozen finite calibration only; evaluation not generated'))
    write(root/'READER_MANIFEST.json',dict(fields=['world','history','queries','prior_mode'],privilege='supplied law and realized executable programs; no evaluator truth or pairing IDs',files=reader_files))
    checks['independent_scalar_reconstruction']=True;checks['calibration_evaluation_disjoint']=not bool(set(indices)&set(evaluation))
    return dict(controls=checks,streams=count,strata=len(bars),thresholds_sha256=file_digest(root/'THRESHOLDS.json'),stage='calibration; independent threshold review required before evaluation admission',scope='constructed privileged finite-world method; no missing-law or held-out-rate result',tiny_settings=0)
