"""Independent stream, hypothesis-product, forecast and population reconstruction.

No producer filtering, stream generation, endpoint-law or scoring functions are used.
Completed native paths use the separate crossed-law execution/probability ruler.
"""
from collections import defaultdict
from itertools import product
from functools import lru_cache
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest
from ..v18_3.world import rng
from .crossed_review import population, near

MAKERS = tuple(product(range(2), repeat=4))
ARMS = ('static', 'reset-16', 'known-time-type', 'unknown-time', 'unknown-time-type')
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')
AXES = ('length', 'kind', 'switched', 'duplicates', 'step')


def complement(m, kind='purpose'):
    axis = {'purpose': 0, 'skill': 1}[kind]
    values = list(MAKERS[m]); values[axis] = 1-values[axis]
    return MAKERS.index(tuple(values))


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


def reconstruct_stream(table, lineage, draw, maker, length, kind, switched, duplicates):
    uniforms = rng('v19-transient-filter',lineage,draw,maker,length).random(length)
    observations = []
    for i,u in enumerate(uniforms):
        step = i+1
        if duplicates and step%4 == 0:
            observations.append({**observations[-1], 'step':step}); continue
        state = complement(maker,kind) if switched and step > length//2 else maker
        context = (i+i//4)%4
        cumulative = np.cumsum(table[state,context])
        endpoint = next((e for e,p in enumerate(cumulative) if u < p),7)
        observations.append(dict(step=step,source_step=step,source_id=f'source-{step:03d}',context=context,endpoint=endpoint))
    return observations


def hypothesis_roster(arm, length, kind):
    if arm not in ARMS or kind not in ('purpose', 'skill'):
        raise ValueError('unknown comparison')
    hs = [('none', 0, m) for m in range(16)]
    if arm in ('static', 'reset-16'):
        return hs, np.full(16, 1/16)
    kinds = ('purpose', 'skill') if arm == 'unknown-time-type' else (kind,)
    times = (length//2,) if arm == 'known-time-type' else tuple(range(8,length-7))
    hs.extend((k,t,m) for k in kinds for t in times for m in range(16))
    prior = np.empty(len(hs)); prior[:16] = 1/32
    prior[16:] = 1/(2*(len(hs)-16))
    return hs,prior


def current_indices(hs, step):
    return np.array([complement(m,k) if k != 'none' and t < step else m for k,t,m in hs])


@lru_cache(maxsize=None)
def state_schedule(arm, length, kind):
    hs,prior=hypothesis_roster(arm,length,kind)
    initial=np.array([m for k,t,m in hs])
    changed=np.array([m if k=='none' else complement(m,k) for k,t,m in hs])
    times=np.array([length+1 if k=='none' else t for k,t,m in hs])
    schedule=np.where(np.arange(length+1)[:,None]>times[None,:],changed[None,:],initial[None,:])
    return hs,prior,schedule


def product_checkpoints(table, observations, arm, length, kind, steps):
    hs,prior,schedule = state_schedule(arm,length,kind)
    unique = distinct(observations)
    if any(not 1<=r['source_step']<=length for r in unique): raise ValueError('source time outside horizon')
    # Scaled products independently check the producer's accumulated log weights.
    weights = prior.copy(); used=0; result={}
    for step in steps:
        available = distinct(observations[:step])
        if arm == 'reset-16':
            weights = prior.copy(); batch=available[-16:]
        else:
            batch=available[used:]
        for row in batch:
            weights *= table[schedule[row['source_step']],row['context'],row['endpoint']]
            total = math.fsum(weights)
            if total <= 0 or not np.isfinite(total): raise ValueError('empty support')
            weights /= total
        used=len(available)
        current=np.zeros(16)
        np.add.at(current,schedule[step],weights)
        result[step]=(current,weights.copy(),hs)
    return result


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
    from .transient_review import controls as previous_controls
    checks=previous_controls()
    neutral=np.full((16,4,8),1/8)
    obs=[dict(step=i,source_step=i,source_id=str(i),context=0,endpoint=i%2) for i in (1,9)]
    hs,prior=hypothesis_roster('unknown-time-type',32,'purpose')
    flat,joint,_=product_checkpoints(neutral,obs,'unknown-time-type',32,'purpose',[2])[2]
    checks.update({
        'positive:complete_unknown_roster':len(hs)==560,
        'positive:type_prior':all(abs(math.fsum(w for (x,t,m),w in zip(hs,prior) if x==k)-v)<1e-14 for k,v in (('none',.5),('purpose',.25),('skill',.25))),
        'positive:future_state':bool(np.array_equal(current_indices(hs,8),[m for k,t,m in hs])),
        'positive:both_involutions':all(complement(complement(m,k),k)==m for m in range(16) for k in ('purpose','skill')),
        'placebo:uninformative_unknown':bool(np.allclose(flat,1/16,rtol=0,atol=1e-14) and np.allclose(joint,prior,rtol=0,atol=1e-14))})
    return checks


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
        for ai,arm in enumerate(ARMS):
            for_base = ARMS[:ai]
            for base in for_base:
                delta=by_arm[arm]-by_arm[base]; bs=delta[samples].mean(1)
                for i,metric in enumerate(METRICS):
                    contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=base,metric=metric,mean=float(delta[:,i].mean()),low=float(np.quantile(bs[:,i],.025)),high=float(np.quantile(bs[:,i],.975)),lineage_values=delta[:,i].tolist()))
    return dict(cells=cells,means=means,contrasts=contrasts,lineages=lineages,draws=draws,
        population='all sixteen makers equally within each draw; both observation draws averaged within paired coefficient lineage; each length/type/switch/source/checkpoint retained; no-change type cells are paired duplicates, never pooled',
        uncertainty='conditional eight-lineage resampling; two observation draws do not establish universal sampling uncertainty; no fitted model')


def zipped(path): return json.loads(gzip.decompress(path.read_bytes()))


def check_packet(packet, public):
    if packet['schema']!='v19.unknown-change.reader.1' or packet['cases']!=[public[k] for k in sorted(public)]:
        raise ValueError('reader projection differs')


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
        roster=list(product(design['draws'],range(16),design['lengths'],('purpose','skill'),(False,True),(False,True)))
        if len(streams)!=len(roster): raise ValueError('stream denominator differs')
        rows=zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed={(r['stream'],r['step'],r['arm']):r for r in rows}
        if len(indexed)!=len(rows): raise ValueError('duplicate forecast row')
        consumed=set();acc=defaultdict(list);mapped=defaultdict(list)
        mapping=read(original/'evaluator'/f'{lineage}-joint-map.json')
        with np.load(original/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as stored:
            arrays={name:stored[name] for name in stored.files}
        expected_keys={f'{length}-{kind}-{arm}' for length,kind,arm in product(design['lengths'],('purpose','skill'),ARMS)}
        if set(arrays)!=expected_keys or set(mapping)!=expected_keys: raise ValueError('joint array roster differs')
        for length,kind,arm in product(design['lengths'],('purpose','skill'),ARMS):
            name=f'{length}-{kind}-{arm}'; hs,prior=hypothesis_roster(arm,length,kind)
            if mapping[name]['hypotheses']!=[list(h) for h in hs]: raise ValueError('hypothesis mapping differs')
            error=max(error,near(prior,mapping[name]['prior'],1e-14))
            if arrays[name].dtype!=np.float64 or arrays[name].shape!=(len(mapping[name]['rows']),len(hs)):
                raise ValueError('joint array dimensions or dtype differ')
        for si,(stream,key) in enumerate(zip(streams,roster)):
            draw,maker,length,kind,switched,duplicates=key
            pulse(phase='independent-hypothesis-products',lineage=lineage,stream=si)
            observations=reconstruct_stream(table,lineage,*key)
            expected=dict(zip(('draw','maker','length','kind','switched','duplicates'),key),observations=observations)
            if stream!=expected: raise ValueError('stream/source assignment differs')
            if len(distinct(observations))!=(length*3//4 if duplicates else length): raise ValueError('unique-source denominator')
            visible=dict(contexts=contexts,observations=observations); ident=digest(visible)
            public[ident]=dict(input_sha256=ident,inputs=visible)
            streams_count+=1; observations_count+=length
            outputs={arm:product_checkpoints(table,observations,arm,length,kind,checkpoints(length)) for arm in ARMS}
            for step,arm in product(checkpoints(length),ARMS):
                row=indexed[si,step,arm]; consumed.add((si,step,arm))
                actual=complement(maker,kind) if switched and step>length//2 else maker
                metadata=dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,kind=kind,switched=switched,duplicates=duplicates,step=step,arm=arm,unique_sources=len(distinct(observations[:step])))
                if any(row[k]!=v for k,v in metadata.items()): raise ValueError('forecast assignment differs')
                current,joint,hs=outputs[arm][step]
                name=f'{length}-{kind}-{arm}'; ji=len(mapped[name]);mapped[name].append([si,step])
                if row['joint_array']!=name or row['joint_row']!=ji: raise ValueError('joint row mapping differs')
                if mapping[name]['rows'][ji]!=[si,step]: raise ValueError('joint index mapping differs')
                types={k:math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x==k) for k in ('none','purpose','skill')}
                if set(row['type_mass'])!=set(types): raise ValueError('type labels differ')
                error=max(error,near([types[k] for k in types],[row['type_mass'][k] for k in types],1e-10))
                values,forecast=metrics(table,current,actual)
                error=max(error,near(current,row['posterior'],1e-10),near(joint,arrays[name][ji],1e-10),near(forecast,row['forecast'],1e-10),near([values[k] for k in METRICS],[row[k] for k in METRICS],1e-10))
                acc[draw,length,kind,switched,duplicates,step,arm].append(values)
        for name in expected_keys:
            if mapped[name]!=mapping[name]['rows']: raise ValueError('unconsumed joint rows')
        # No-change observations are the same paired controls for both change types.
        stream_index={(s['draw'],s['maker'],s['length'],s['kind'],s['switched'],s['duplicates']):s for s in streams}
        for draw,maker,length,duplicates in product(design['draws'],range(16),design['lengths'],(False,True)):
            if stream_index[draw,maker,length,'purpose',False,duplicates]['observations']!=stream_index[draw,maker,length,'skill',False,duplicates]['observations']:
                raise ValueError('no-change type identity differs')
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
    check_packet(packet,public)
    for k,v in dict(rows=rows_count,streams=streams_count,observations=observations_count,public_packets=len(public),fits=0).items():
        if summary[k]!=v: raise ValueError('overall denominator differs')
    checks.update(positive_native_path_reconstruction=True,positive_complete_streams=True,positive_complete_forecasts=True,positive_public_projection=True)
    write(root/'INDEPENDENT_REGROUP.json',regroup(all_cells,cfg))
    return dict(controls=checks,paths=paths,rows=rows_count,streams=streams_count,observations=observations_count,cells=len(all_cells),public_packets=len(public),max_error=error,
        scope='independent exact-law unknown-time/type filtering review; current-state coverage separate from endpoint prediction; no learned capability or historical path conclusion')
