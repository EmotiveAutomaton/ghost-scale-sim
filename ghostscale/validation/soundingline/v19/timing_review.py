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
from . import unknown_review as P

MAKERS = tuple(product(range(2), repeat=4))
ARMS = ('static', 'reset-16', 'known-time-type', 'unknown-time', 'unknown-time-type')
METRICS = ('endpoint_loss', 'state_loss', 'true_state_mass', 'coverage90', 'size90')
AXES = ('length', 'kind', 'quarter', 'duplicates', 'step')


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


def reconstruct_stream(table, lineage, draw, maker, length, kind, quarter, duplicates):
    uniforms = rng('v19-transient-filter',lineage,draw,maker,length).random(length)
    observations = []
    for i,u in enumerate(uniforms):
        step = i+1
        if duplicates and step%4 == 0:
            observations.append({**observations[-1], 'step':step}); continue
        state = complement(maker,kind) if step > length*quarter//4 else maker
        context = (i+i//4)%4
        cumulative = np.cumsum(table[state,context])
        endpoint = next((e for e,p in enumerate(cumulative) if u < p),7)
        observations.append(dict(step=step,source_step=step,source_id=f'source-{step:03d}',context=context,endpoint=endpoint))
    return observations


def hypothesis_roster(arm, length, kind, switch_at):
    if arm not in ARMS or kind not in ('purpose', 'skill'):
        raise ValueError('unknown comparison')
    hs = [('none', 0, m) for m in range(16)]
    if arm in ('static', 'reset-16'):
        return hs, np.full(16, 1/16)
    kinds = ('purpose', 'skill') if arm == 'unknown-time-type' else (kind,)
    times = (switch_at,) if arm == 'known-time-type' else tuple(range(8,length-7))
    hs.extend((k,t,m) for k in kinds for t in times for m in range(16))
    prior = np.empty(len(hs)); prior[:16] = 1/32
    prior[16:] = 1/(2*(len(hs)-16))
    return hs,prior


def current_indices(hs, step):
    return np.array([complement(m,k) if k != 'none' and t < step else m for k,t,m in hs])


@lru_cache(maxsize=None)
def state_schedule(arm, length, kind, switch_at):
    hs,prior=hypothesis_roster(arm,length,kind,switch_at)
    initial=np.array([m for k,t,m in hs])
    changed=np.array([m if k=='none' else complement(m,k) for k,t,m in hs])
    times=np.array([length+1 if k=='none' else t for k,t,m in hs])
    schedule=np.where(np.arange(length+1)[:,None]>times[None,:],changed[None,:],initial[None,:])
    return hs,prior,schedule


def product_checkpoints(table, observations, arm, length, kind, switch_at, steps):
    hs,prior,schedule = state_schedule(arm,length,kind,switch_at)
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
    checks=P.controls()
    law=np.full((16,4,8),1/8)
    obs=[dict(step=i,source_step=i,source_id=str(i),context=0,endpoint=0) for i in range(1,33)]
    for at in (8,24):
        hs,prior=hypothesis_roster('known-time-type',32,'purpose',at)
        cur,joint,_=product_checkpoints(law,obs,'known-time-type',32,'purpose',at,[32])[32]
        checks[f'placebo:neutral-{at}']=bool(np.allclose(cur,1/16) and np.allclose(joint,prior))
        checks[f'positive:time-roster-{at}']=set(t for k,t,m in hs if k!='none')=={at}
        checks[f'positive:boundary-{at}']=all(current_indices([h],at)[0]==h[2] for h in hs)
    checks['positive:unknown-time-invariance']=hypothesis_roster('unknown-time-type',32,'skill',8)[0]==hypothesis_roster('unknown-time-type',32,'skill',24)[0]
    return checks


def parent_cells(parent,lineage,table):
    """Independently rebuild retained identities/scores; these are not new replicates."""
    streams=zipped(parent/'raw'/f'{lineage}-observations_points.json.gz')
    rows=zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
    error=near(table,read(parent/'evaluator'/f'{lineage}-law.json'),1e-12)
    identities=set();acc=defaultdict(list);seen=set()
    for stream in streams:
        key=tuple(stream[k] for k in ('draw','maker','length','kind','switched','duplicates'))
        if key in identities:raise ValueError('duplicate parent stream')
        identities.add(key)
        if stream['observations']!=P.reconstruct_stream(table,lineage,*key):raise ValueError('parent stream differs')
        if stream['switched'] and stream['observations']!=reconstruct_stream(table,lineage,*key[:4],2,key[-1]):
            raise ValueError('midpoint identity differs')
    for r in rows:
        ident=r['stream'],r['step'],r['arm']
        if ident in seen:raise ValueError('duplicate parent row')
        seen.add(ident);s=streams[r['stream']]
        actual=complement(s['maker'],s['kind']) if s['switched'] and r['step']>s['length']//2 else s['maker']
        if actual!=r['actual_maker']:raise ValueError('parent actual maker differs')
        values,forecast=metrics(table,np.asarray(r['posterior']),actual)
        error=max(error,near(forecast,r['forecast'],1e-10),near([values[m] for m in METRICS],[r[m] for m in METRICS],1e-10))
        key=tuple(r[k] for k in ('draw',*P.AXES,'arm'))
        acc[key].append(values)
    expected_rows=sum(len(checkpoints(s['length']))*len(ARMS) for s in streams)
    if len(rows)!=expected_rows:raise ValueError('parent row denominator')
    saved=[r for r in read(parent/'SUMMARY.json')['cells'] if r['lineage']==lineage]
    index={tuple(r[k] for k in ('draw',*P.AXES,'arm')):r for r in saved}
    if len(index)!=len(saved) or set(index)!=set(acc):raise ValueError('parent stratum denominator')
    result=[]
    for key,values in sorted(acc.items()):
        if len(values)!=16 or index[key]['makers']!=16:raise ValueError('parent maker denominator')
        scores={m:math.fsum(v[m] for v in values)/16 for m in METRICS}
        error=max(error,near([scores[m] for m in METRICS],[index[key][m] for m in METRICS],1e-10))
        result.append(dict(lineage=lineage,**dict(zip(('draw',*P.AXES,'arm'),key)),makers=16,**scores))
    return result,error


def regroup(cells,cfg,parents):
    index = {(r['lineage'],r['draw'],*(r[k] for k in AXES),r['arm']):r for r in cells}
    if len(index)!=len(cells): raise ValueError('duplicate stratum')
    lineages = sorted({r['lineage'] for r in cells}); draws = sorted({r['draw'] for r in cells})
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages),size=(cfg['bootstrap_resamples'],len(lineages)))
    strata = sorted({tuple(r[k] for k in AXES) for r in cells}); means=[]; contrasts=[]
    parent_index={(r['lineage'],r['draw'],*(r[k] for k in P.AXES),r['arm']):r for r in parents}
    def estimate(values):
        paired=values.mean(1); boot=paired[samples].mean(1)
        return dict(mean=float(paired.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),lineage_values=paired.tolist(),draw_means=values.mean(0).tolist())
    for stratum in strata:
        by_arm={}
        for arm in ARMS:
            values=np.array([[[index[l,d,*stratum,arm][m] for m in METRICS] for d in draws] for l in lineages])
            by_arm[arm]=values
            means.append(dict(zip(AXES,stratum),arm=arm,**{m:estimate(values[:,:,i]) for i,m in enumerate(METRICS)}))
        for ai,arm in enumerate(ARMS):
            for base in ARMS[:ai]:
                for i,metric in enumerate(METRICS):
                    contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=base,metric=metric,comparison='same-actual-time',**estimate(by_arm[arm][:,:,i]-by_arm[base][:,:,i])))
            length,kind,quarter,duplicates,step=stratum
            old=np.array([[[parent_index[l,d,length,kind,True,duplicates,step,arm][m] for m in METRICS] for d in draws] for l in lineages])
            for i,metric in enumerate(METRICS):
                contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=arm,metric=metric,comparison='actual-time-minus-midpoint',**estimate(by_arm[arm][:,:,i]-old[:,:,i])))
    return dict(cells=cells,parent_cells=parents,means=means,contrasts=contrasts,lineages=lineages,draws=draws,
        population='all sixteen makers equally within each draw; both observation draws averaged within paired coefficient lineage; each actual quarter/length/type/source/checkpoint retained; parent stationary and midpoint cells retained once as identities; no pooling across quarters',
        uncertainty='conditional eight-lineage resampling; two observation draws do not establish universal sampling uncertainty; no fitted model')


def zipped(path): return json.loads(gzip.decompress(path.read_bytes()))


def check_packet(packet, public):
    if packet['schema']!='v19.change-timing.reader.1' or packet['cases']!=[public[k] for k in sorted(public)]:
        raise ValueError('reader projection differs')


def run(root,plan,pulse):
    checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg=plan['design']; original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design=read(original/'PLAN.json')['design']; summary=read(original/'SUMMARY.json')
    if design['arms']!=list(ARMS) or any(n not in (32,128) for n in design['lengths']) or design['quarters']!=[1,3]: raise ValueError('design differs')
    contexts=[{'initial':list(a),'requested_purpose':p} for a in ((0,1,0),(1,0,1)) for p in range(2)]
    all_cells=[]; parents=[]; public={}; error=0.; paths=rows_count=streams_count=observations_count=0
    for lineage in design['lineages']:
        pulse(phase='independent-native-paths',lineage=lineage)
        records=zipped(original/'inputs'/f'lineage-{lineage}_points.json.gz')
        _,_,e=population(records,lineage,'original'); error=max(error,e); paths+=len(records)
        table=endpoint_law(records); error=max(error,near(table,read(original/'evaluator'/f'{lineage}-law.json'),1e-12))
        old,e=parent_cells(original/'inputs/parent',lineage,table);parents.extend(old);error=max(error,e)
        streams=zipped(original/'raw'/f'{lineage}-observations_points.json.gz')
        roster=list(product(design['draws'],range(16),design['lengths'],('purpose','skill'),design['quarters'],(False,True)))
        if len(streams)!=len(roster): raise ValueError('stream denominator differs')
        rows=zipped(original/'raw'/f'{lineage}-forecasts_points.json.gz')
        indexed={(r['stream'],r['step'],r['arm']):r for r in rows}
        if len(indexed)!=len(rows): raise ValueError('duplicate forecast row')
        consumed=set();acc=defaultdict(list);mapped=defaultdict(list)
        mapping=read(original/'evaluator'/f'{lineage}-joint-map.json')
        with np.load(original/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as stored:
            arrays={name:stored[name] for name in stored.files}
        expected_keys={f'{length}-{kind}-{quarter}-{arm}' for length,kind,quarter,arm in product(design['lengths'],('purpose','skill'),design['quarters'],ARMS)}
        if set(arrays)!=expected_keys or set(mapping)!=expected_keys: raise ValueError('joint array roster differs')
        for length,kind,quarter,arm in product(design['lengths'],('purpose','skill'),design['quarters'],ARMS):
            name=f'{length}-{kind}-{quarter}-{arm}'; hs,prior=hypothesis_roster(arm,length,kind,length*quarter//4)
            if mapping[name]['hypotheses']!=[list(h) for h in hs]: raise ValueError('hypothesis mapping differs')
            error=max(error,near(prior,mapping[name]['prior'],1e-14))
            if arrays[name].dtype!=np.float64 or arrays[name].shape!=(len(mapping[name]['rows']),len(hs)):
                raise ValueError('joint array dimensions or dtype differ')
        for si,(stream,key) in enumerate(zip(streams,roster)):
            draw,maker,length,kind,quarter,duplicates=key;switch_at=length*quarter//4
            pulse(phase='independent-hypothesis-products',lineage=lineage,stream=si)
            observations=reconstruct_stream(table,lineage,*key)
            expected=dict(zip(('draw','maker','length','kind','quarter','duplicates'),key),switch_at=switch_at,observations=observations)
            if stream!=expected: raise ValueError('stream/source assignment differs')
            if len(distinct(observations))!=(length*3//4 if duplicates else length): raise ValueError('unique-source denominator')
            visible=dict(contexts=contexts,observations=observations); ident=digest(visible)
            public[ident]=dict(input_sha256=ident,inputs=visible)
            streams_count+=1; observations_count+=length
            outputs={arm:product_checkpoints(table,observations,arm,length,kind,switch_at,checkpoints(length)) for arm in ARMS}
            for step,arm in product(checkpoints(length),ARMS):
                row=indexed[si,step,arm]; consumed.add((si,step,arm))
                actual=complement(maker,kind) if step>length*quarter//4 else maker
                metadata=dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,kind=kind,quarter=quarter,switch_at=switch_at,duplicates=duplicates,step=step,arm=arm,unique_sources=len(distinct(observations[:step])))
                if any(row[k]!=v for k,v in metadata.items()): raise ValueError('forecast assignment differs')
                current,joint,hs=outputs[arm][step]
                name=f'{length}-{kind}-{quarter}-{arm}'; ji=len(mapped[name]);mapped[name].append([si,step])
                if row['joint_array']!=name or row['joint_row']!=ji: raise ValueError('joint row mapping differs')
                if mapping[name]['rows'][ji]!=[si,step]: raise ValueError('joint index mapping differs')
                types={k:math.fsum(float(w) for (x,t,m),w in zip(hs,joint) if x==k) for k in ('none','purpose','skill')}
                if set(row['type_mass'])!=set(types): raise ValueError('type labels differ')
                error=max(error,near([types[k] for k in types],[row['type_mass'][k] for k in types],1e-10))
                values,forecast=metrics(table,current,actual)
                error=max(error,near(current,row['posterior'],1e-10),near(joint,arrays[name][ji],1e-10),near(forecast,row['forecast'],1e-10),near([values[k] for k in METRICS],[row[k] for k in METRICS],1e-10))
                acc[draw,length,kind,quarter,duplicates,step,arm].append(values)
        for name in expected_keys:
            if mapped[name]!=mapping[name]['rows']: raise ValueError('unconsumed joint rows')
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
    write(root/'INDEPENDENT_REGROUP.json',regroup(all_cells,cfg,parents))
    return dict(controls=checks,paths=paths,rows=rows_count,streams=streams_count,observations=observations_count,cells=len(all_cells),public_packets=len(public),max_error=error,
        scope='independent actual-change-timing filtering and midpoint comparison; current-state coverage separate from endpoint prediction; no learned capability or historical path conclusion')
