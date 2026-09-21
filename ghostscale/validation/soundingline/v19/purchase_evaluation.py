"""Frozen-threshold paid expansion with explicit sensor and predictor costs."""
from copy import deepcopy
from itertools import product
import gzip,hashlib,math,time
import numpy as np
from ..v18_3 import world as W
from ..v18_3.io import canonical,write,read,file_digest
from ..v18_4 import distinct_families as D
from . import purchase_calibration as C

METHODS=('fixed','select','mixture','expanded-select','expanded-mixture',
         'paid-raw-1.5','paid-raw-3','calibrated-raw','calibrated-centered')

def trigger(values,threshold):
    if len(values)!=24 or not np.isfinite(values).all() or not math.isfinite(threshold):raise ValueError('invalid trigger')
    return next((8+j for j,v in enumerate(values) if v>threshold),None)

def priced(match,expanded,cost,length=32):
    return dict(net_match_low=match-(expanded+1e-5*cost)/length,
                net_match_high=match-(4*expanded+1e-4*cost)/length)

def read_stream(public,method,bars):
    if method not in METHODS:raise ValueError('undeclared method')
    sensor=None;step=None;threshold=None;key=None
    if method.startswith(('paid-','calibrated-')):
        sensor=C.sensor(public)
        key='centered' if method=='calibrated-centered' else 'raw'
        threshold=float(method.rsplit('-',1)[1]) if method.startswith('paid-') else bars[key]
        step=trigger(sensor['values'][key],threshold)
        # The predictor only receives the first past-only crossing. No purchase
        # when absent; it must not fall back to the inherited automatic threshold.
        row=D.read_stream(public,'paid-mixture-1.5',allow_expansion=step is not None,force_at=step)
    else:row=D.read_stream(public,method)
    row['method']=method;row['predictor_likelihood_evaluations']=row['likelihood_evaluations']
    row['sensor_likelihood_evaluations']=0 if sensor is None else sensor['likelihood_evaluations']
    row['likelihood_evaluations']+=row['sensor_likelihood_evaluations']
    row['sensor_probability_terms']=0 if sensor is None else 2*len(W.STATES)*len(W.PROGRAMS)*32
    row['sensor']=sensor;row['threshold']=threshold;row['sensor_kind']=key
    row['near_threshold_opportunities']=0 if sensor is None else sum(abs(v-threshold)<=1e-12 for v in sensor['values'][key])
    return row

def verify(unit,bars):
    """Full independent native batch forecasts/actions, plus decisions and costs."""
    base=deepcopy(unit)
    for row in base['rows']:
        row['likelihood_evaluations']=row['predictor_likelihood_evaluations']
        row.update(priced(row['expected_match'],row['expanded'],row['likelihood_evaluations']))
    D.verify(base)
    for row in unit['rows']:
        sensor=row['sensor'];extra=0
        if sensor is not None:
            C.verify(unit['public'],sensor);extra=2*len(W.STATES)*32
            key='centered' if row['method']=='calibrated-centered' else 'raw'
            threshold=float(row['method'].rsplit('-',1)[1]) if row['method'].startswith('paid-') else bars[key]
            # Scalar reconstruction verifies sensor values to tolerance. Exact
            # decisions retain frozen arithmetic; report sensitivity separately.
            values=sensor['values'][key]
            candidates=[t for t,v in zip(range(8,32),values) if v>threshold]
            expected=candidates[0] if candidates else 33
            if row['threshold']!=threshold or row['purchase_step']!=expected:raise ValueError('purchase differs')
            if any(x['expanded']!=(t>=expected) for t,x in enumerate(row['trace'])):raise ValueError('purchase prefix differs')
            if row['near_threshold_opportunities']!=sum(abs(v-threshold)<=1e-12 for v in values):raise ValueError('threshold sensitivity differs')
        if row['sensor_likelihood_evaluations']!=extra or row['likelihood_evaluations']!=row['predictor_likelihood_evaluations']+extra:raise ValueError('sensor cost differs')
        terms=extra*len(W.PROGRAMS)
        if row['sensor_probability_terms']!=terms:raise ValueError('sensor alphabet work differs')
        for key,value in priced(row['expected_match'],row['expanded'],row['likelihood_evaluations']).items():
            if abs(row[key]-value)>1e-12:raise ValueError('total priced score differs')
    return True

def evaluate(public,truth,bars):
    rows=[];timing=[]
    for method in METHODS:
        start=time.process_time();row=read_stream(public,method,bars);elapsed=time.process_time()-start
        rows.append(D.score(row,truth));timing.append(dict(method=method,reader_cpu_seconds=elapsed))
    return rows,timing

def run(root,plan,pulse):
    cfg=plan['design'];path=root/'inputs/THRESHOLDS.json'
    if file_digest(path)!=cfg['thresholds_sha256']:raise ValueError('threshold artifact changed')
    artifact=read(path)
    proof=root/'inputs/INDEPENDENT_REVIEW.json'
    if file_digest(proof)!=cfg['calibration_review_sha256'] or not read(proof)['passed'] or read(proof)['thresholds_sha256']!=cfg['thresholds_sha256']:raise ValueError('review identity changed')
    if artifact['evaluation_indices']!=cfg['evaluation_indices'] or set(artifact['calibration_indices'])&set(cfg['evaluation_indices']):raise ValueError('roster changed')
    mapping={(b['cell'],b['order'],b['prior'],b['sensor']):b['threshold'] for b in artifact['strata']}
    checks=C.controls();count=0;files={};groups={}
    (root/'reader').mkdir(exist_ok=True);(root/'raw').mkdir(exist_ok=True)
    for index in cfg['evaluation_indices']:
        pulse(phase='purchase-evaluation',completed_streams=count);records=[]
        for cell,kind,order,prior in product(cfg['cells'],D.KINDS,D.ORDERS,C.PRIORS):
            public,truth=D.make_case(index,cell,kind,order,1,32);public['prior_mode']=prior
            bars={key:mapping[cell,order,prior,key] for key in C.SENSORS}
            rows,timing=evaluate(public,truth,bars)
            unit=dict(index=index,cell=cell,kind=kind,order=order,prior_mode=prior,copy_span=1,length=32,public=public,evaluator=truth,rows=rows)
            verify(unit,bars)
            data=canonical(public);h=hashlib.sha256(data).hexdigest();dest=root/'reader'/f'{h}.json'
            if not dest.exists():dest.write_bytes(data)
            files[dest.relative_to(root).as_posix()]=h
            unit.pop('public');unit['public_sha256']=h;records.append(unit)
            with (root/'TIMING.jsonl').open('ab') as f:f.write(canonical(dict(index=index,cell=cell,kind=kind,order=order,prior=prior,methods=timing)))
            for row in rows:
                key=(cell,kind,order,prior,row['method']);groups.setdefault(key,[]).append({m:row[m] for m in D.METRICS})
            count+=1
        dest=root/'raw'/f'evaluation-{index}_points.json.gz';dest.write_bytes(gzip.compress(canonical(records),mtime=0))
    strata=[dict(cell=k[0],kind=k[1],order=k[2],prior=k[3],method=k[4],streams=len(v),**{m:float(np.mean([x[m] for x in v])) for m in D.METRICS}) for k,v in groups.items()]
    write(root/'STRATA.json',dict(strata=strata,aggregation='twenty paired coefficient indices per cell/truth/order/prior/method; report equal-cell/truth weights separately; no question-level independence'))
    write(root/'READER_MANIFEST.json',dict(fields=['world','history','queries','prior_mode'],privilege='supplied candidate law and executable-program observations; truth and pairing labels excluded',files=files))
    checks['independent_forecasts_actions_sensors_decisions_costs']=True
    return dict(controls=checks,streams=count,methods=list(METHODS),strata=len(strata),thresholds_sha256=cfg['thresholds_sha256'],scope='constructed supplied-law/program method; catalog purchase, not law invention or local-process correspondence',cost_scope='standalone full sensor pass charged in addition to predictor likelihood calls for each paid arm; full-alphabet contraction terms separately counted; actual CPU separate; verification CPU globally charged',tiny_settings=0)
