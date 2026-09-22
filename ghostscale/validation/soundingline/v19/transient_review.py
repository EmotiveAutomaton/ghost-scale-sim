"""Independent stream, hypothesis-product, forecast and population reconstruction.

No producer filtering, stream generation, endpoint-law or scoring functions are used.
Completed native paths use the separate crossed-law execution/probability ruler.
"""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest
from ..v18_3.world import rng
from .crossed_review import population, near

MAKERS = tuple(product(range(2), repeat=4))
ARMS = ('static', 'reset-16', 'coherent-mixture')
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')
AXES = ('length', 'switched', 'duplicates', 'step')


def complement(m):
    return MAKERS.index((1-MAKERS[m][0], *MAKERS[m][1:]))


def endpoint_law(records):
    weights = defaultdict(list)
    for r in records:
        weights[r['maker_index'], r['context_index'], sum(v*2**i for i,v in enumerate(r['final']))].append(r['probability'])
    table = np.array([[[math.fsum(weights[m,c,e])*64 for e in range(8)] for c in range(4)] for m in range(16)])
    near(table.sum(2), np.ones((16,4)), 1e-12)
    return table/table.sum(2, keepdims=True)


def distinct(observations):
    identities = {}; result = []
    for r in observations:
        identity = (r['source_step'],r['context'],r['endpoint'])
        if r['source_id'] in identities:
            if identities[r['source_id']] != identity: raise ValueError('conflicting source')
        else:
            identities[r['source_id']] = identity; result.append(r)
    return result


def reconstruct_stream(table, lineage, draw, maker, length, switched, duplicates):
    uniforms = rng('v19-transient-filter',lineage,draw,maker,length).random(length)
    observations = []
    for i,u in enumerate(uniforms):
        step = i+1
        if duplicates and step%4 == 0:
            observations.append({**observations[-1], 'step':step}); continue
        state = complement(maker) if switched and step > length//2 else maker
        context = (i+i//4)%4
        cumulative = np.cumsum(table[state,context])
        endpoint = next((e for e,p in enumerate(cumulative) if u < p),7)
        observations.append(dict(step=step,source_step=step,source_id=f'source-{step:03d}',context=context,endpoint=endpoint))
    return observations


def hypothesis_product(table, observations, arm, switch_at, step):
    if arm not in ARMS: raise ValueError('unknown arm')
    obs = distinct(observations)
    if arm == 'reset-16': obs = obs[-16:]
    hypotheses = tuple(product(range(2 if arm == 'coherent-mixture' else 1),range(16)))
    weights = np.full(len(hypotheses),1/len(hypotheses))
    # Sequential products and normalization independently check the log-sum producer.
    for r in obs:
        likelihood = np.array([table[complement(m) if flip and r['source_step']>switch_at else m,r['context'],r['endpoint']] for flip,m in hypotheses])
        weights *= likelihood
        total = math.fsum(weights)
        if total <= 0 or not np.isfinite(total): raise ValueError('empty support')
        weights /= total
    current = np.zeros(16)
    for (flip,m),w in zip(hypotheses,weights):
        current[complement(m) if flip and step>switch_at else m] += w
    return current,weights.reshape(-1,16)


def metrics(table, current, actual):
    forecast = np.array([[math.fsum(current[m]*table[m,c,e] for m in range(16)) for e in range(8)] for c in range(4)])
    loss = -math.fsum(table[actual,c,e]*math.log(max(forecast[c,e],1e-300)) for c in range(4) for e in range(8))/4
    ordered = sorted(current,reverse=True); mass = 0.
    for boundary in ordered:
        mass += boundary
        if mass >= .9: break
    selected = [i for i,p in enumerate(current) if p >= boundary-1e-14]
    return dict(endpoint_loss=loss,state_loss=-math.log(max(current[actual],1e-300)),true_state_mass=float(current[actual]),coverage90=float(actual in selected),size90=float(len(selected))),forecast


def checkpoints(length):
    if length == 32: return (8,16,17,20,32)
    if length == 128: return (16,32,64,65,80,128)
    raise ValueError('unadmitted length')


def controls():
    neutral = np.full((16,4,8),1/8)
    table = np.full((16,4,8),.1/7)
    for m in range(16): table[m,:,MAKERS[m][0]] = .9
    a = dict(step=1,source_step=1,source_id='a',context=0,endpoint=0)
    b = dict(step=2,source_step=2,source_id='b',context=0,endpoint=1)
    current,joint = hypothesis_product(table,[a,b],'coherent-mixture',1,2)
    weights = [[table[m,0,0]*table[complement(m) if f else m,0,1] for m in range(16)] for f in (0,1)]
    expected = np.asarray(weights)/math.fsum(v for row in weights for v in row)
    p,_ = hypothesis_product(table,[a],'static',1,1)
    duplicate,_ = hypothesis_product(table,[a,dict(a,step=2)],'static',1,2)
    empty = conflict = False
    try: hypothesis_product(np.zeros_like(table),[a],'static',1,1)
    except ValueError: empty=True
    try: distinct([a,dict(a,endpoint=1)])
    except ValueError: conflict=True
    flat,_ = hypothesis_product(neutral,[a,b],'coherent-mixture',1,2)
    scored,_ = metrics(neutral,flat,0)
    observations=[dict(a,step=i,source_step=i,source_id=str(i)) for i in range(1,20)]
    recent,_ = hypothesis_product(table,observations,'reset-16',9,19)
    last,_ = hypothesis_product(table,observations[-16:],'static',9,19)
    return {'live:informative_observation':bool(p[:8].sum()>.9),
        'positive:enumerated_hypotheses':bool(np.allclose(joint,expected,atol=1e-14,rtol=0)),
        'positive:current_state_remap':bool(np.allclose(current,expected[0]+expected[1,[complement(m) for m in range(16)]],atol=1e-14,rtol=0)),
        'positive:empty_support_rejected':empty,'positive:conflicting_source_rejected':conflict,
        'positive:tie_inclusive_credible_set':scored['coverage90']==1 and scored['size90']==16,
        'positive:known_log_score':abs(scored['endpoint_loss']-math.log(8))<1e-14,
        'placebo:uninformative':bool(np.allclose(flat,1/16,atol=1e-14,rtol=0)),
        'placebo:duplicate_identity':bool(np.array_equal(p,duplicate)),
        'positive:last_sixteen_sources':bool(np.array_equal(recent,last))}


def regroup(cells,cfg):
    index = {(r['lineage'],r['draw'],*(r[k] for k in AXES),r['arm']):r for r in cells}
    if len(index)!=len(cells): raise ValueError('duplicate stratum')
    lineages = sorted({r['lineage'] for r in cells}); draws = sorted({r['draw'] for r in cells})
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages),size=(cfg['bootstrap_resamples'],len(lineages)))
    strata = sorted({tuple(r[k] for k in AXES) for r in cells}); means=[]; contrasts=[]
    for stratum in strata:
        by_arm={}
        for arm in ARMS:
            values = np.array([[math.fsum(index[l,d,*stratum,arm][metric] for d in draws)/len(draws) for metric in METRICS] for l in lineages])
            by_arm[arm]=values
            means.append(dict(zip(AXES,stratum),arm=arm,**{m:float(values[:,i].mean()) for i,m in enumerate(METRICS)}))
        for arm,base in (('reset-16','static'),('coherent-mixture','static'),('coherent-mixture','reset-16')):
            delta=by_arm[arm]-by_arm[base]; bs=delta[samples].mean(1)
            for i,metric in enumerate(METRICS):
                contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=base,metric=metric,mean=float(delta[:,i].mean()),low=float(np.quantile(bs[:,i],.025)),high=float(np.quantile(bs[:,i],.975)),lineage_values=delta[:,i].tolist()))
    return dict(cells=cells,means=means,contrasts=contrasts,lineages=lineages,draws=draws,
        population='all sixteen makers equally within each draw; both observation draws averaged within paired coefficient lineage; each length/switch/source/checkpoint retained',
        uncertainty='conditional eight-lineage resampling; two observation draws do not establish universal sampling uncertainty; no fitted model')


def zipped(path): return json.loads(gzip.decompress(path.read_bytes()))


def run(root,plan,pulse):
    checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg=plan['design']; original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design=read(original/'PLAN.json')['design']; summary=read(original/'SUMMARY.json')
    if design['arms']!=list(ARMS) or design['lengths']!=[32,128]: raise ValueError('design differs')
    contexts=[{'initial':list(a),'requested_purpose':p} for a in ((0,1,0),(1,0,1)) for p in range(2)]
    all_cells=[]; public={}; error=0.; paths=rows_count=streams_count=observations_count=0
    for lineage in design['lineages']:
        pulse(phase='independent-native-paths',lineage=lineage)
        records=zipped(original/'inputs'/f'lineage-{lineage}_points.json.gz')
        _,_,e=population(records,lineage,'original'); error=max(error,e); paths+=len(records)
        table=endpoint_law(records); error=max(error,near(table,read(original/'evaluator'/f'{lineage}-law.json'),1e-12))
        streams=zipped(original/'raw'/f'{lineage}-observations_points.json.gz')
        roster=list(product(design['draws'],range(16),design['lengths'],(False,True),(False,True)))
        if len(streams)!=len(roster): raise ValueError('stream denominator differs')
        rows=zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed={(r['stream'],r['step'],r['arm']):r for r in rows}
        if len(indexed)!=len(rows): raise ValueError('duplicate forecast row')
        consumed=set();acc=defaultdict(list)
        for si,(stream,key) in enumerate(zip(streams,roster)):
            draw,maker,length,switched,duplicates=key
            pulse(phase='independent-hypothesis-products',lineage=lineage,stream=si)
            observations=reconstruct_stream(table,lineage,*key)
            expected=dict(zip(('draw','maker','length','switched','duplicates'),key),observations=observations)
            if stream!=expected: raise ValueError('stream/source assignment differs')
            if len(distinct(observations))!=(length*3//4 if duplicates else length): raise ValueError('unique-source denominator')
            visible=dict(contexts=contexts,observations=observations); ident=digest(visible)
            public[ident]=dict(input_sha256=ident,inputs=visible)
            streams_count+=1; observations_count+=length
            for step,arm in product(checkpoints(length),ARMS):
                row=indexed[si,step,arm]; consumed.add((si,step,arm))
                actual=complement(maker) if switched and step>length//2 else maker
                metadata=dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,switched=switched,duplicates=duplicates,step=step,arm=arm,unique_sources=len(distinct(observations[:step])))
                if any(row[k]!=v for k,v in metadata.items()): raise ValueError('forecast assignment differs')
                current,joint=hypothesis_product(table,observations[:step],arm,length//2,step)
                values,forecast=metrics(table,current,actual)
                error=max(error,near(current,row['posterior'],1e-10),near(joint,row['joint'],1e-10),near(forecast,row['forecast'],1e-10),near([values[k] for k in METRICS],[row[k] for k in METRICS],1e-10))
                acc[draw,length,switched,duplicates,step,arm].append(values)
        if consumed!=set(indexed): raise ValueError('extra/missing forecast rows')
        rows_count+=len(rows)
        for key,values in sorted(acc.items()):
            if len(values)!=16: raise ValueError('maker denominator differs')
            all_cells.append(dict(lineage=lineage,**dict(zip(('draw',*AXES,'arm'),key)),makers=16,**{m:math.fsum(v[m] for v in values)/16 for m in METRICS}))
    key=lambda r:(r['lineage'],r['draw'],*(r[k] for k in AXES),r['arm'])
    expected={key(r):r for r in all_cells};actual={key(r):r for r in summary['cells']}
    if len(actual)!=len(summary['cells']) or set(actual)!=set(expected): raise ValueError('summary denominator differs')
    for k,r in expected.items():
        if actual[k]['makers']!=16: raise ValueError('summary maker denominator')
        error=max(error,near([r[m] for m in METRICS],[actual[k][m] for m in METRICS],1e-10))
    packet=read(original/'PUBLIC_PACKET.json')
    if packet['schema']!='v19.transient.reader.1' or packet['cases']!=[public[k] for k in sorted(public)]: raise ValueError('reader projection differs')
    for k,v in dict(rows=rows_count,streams=streams_count,observations=observations_count,public_packets=len(public),fits=0).items():
        if summary[k]!=v: raise ValueError('overall denominator differs')
    checks.update(positive_native_path_reconstruction=True,positive_complete_streams=True,positive_complete_forecasts=True,positive_public_projection=True)
    write(root/'INDEPENDENT_REGROUP.json',regroup(all_cells,cfg))
    return dict(controls=checks,paths=paths,rows=rows_count,streams=streams_count,observations=observations_count,cells=len(all_cells),public_packets=len(public),max_error=error,
        scope='independent exact-law and supplied-time filtering review; current-state coverage separate from endpoint prediction; no learned capability or historical path conclusion')
