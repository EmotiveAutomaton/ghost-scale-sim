"""Independent source-identity omission reconstruction using scaled products.

Only independent verification modules supply likelihood products and scoring.
No producer filter, stream, score or identity-omission function is imported.
"""
from collections import defaultdict
from itertools import product
import gzip
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, digest, canonical
from .crossed_review import population, near
from .unknown_review import (ARMS, METRICS, AXES, complement, endpoint_law,
    distinct, reconstruct_stream, hypothesis_roster, product_checkpoints,
    metrics, checkpoints, zipped)
from . import unknown_review as P
from .timing_review import parent_cells as score_parent_cells


def omit_identity(observations):
    distinct(observations)
    if [r['step'] for r in observations] != list(range(1,len(observations)+1)):
        raise ValueError('nonconsecutive observation times')
    return [{**r,'source_id':f'source-{i:03d}'} for i,r in enumerate(observations,1)]


def controls():
    checks=P.controls()
    law=np.full((16,4,8),1/8)
    rows=[dict(step=1,source_step=1,source_id='a',context=0,endpoint=0),
          dict(step=2,source_step=1,source_id='a',context=0,endpoint=0)]
    supplied=omit_identity(rows)
    flat=product_checkpoints(law,supplied,'unknown-time-type',32,'purpose',[2])[2][0]
    law[:8,0]=[.8,.2,0,0,0,0,0,0];law[8:,0]=[.2,.8,0,0,0,0,0,0]
    twice=product_checkpoints(law,supplied,'static',32,'purpose',[2])[2][0]
    checks.update({'placebo:neutral_omission':bool(np.allclose(flat,1/16)),
        'positive:counted_twice':bool(np.allclose(twice[:8],(.8**2/(.8**2+.2**2))/8)),
        'positive:timestamps_contents_preserved':all({k:v for k,v in x.items() if k!='source_id'}=={k:v for k,v in y.items() if k!='source_id'} for x,y in zip(rows,supplied))})
    return checks


def parent_cells(parent,lineage,table):
    cells,error=score_parent_cells(parent,lineage,table)
    streams=zipped(parent/'raw'/f'{lineage}-observations_points.json.gz')
    rows=zipped(parent/'raw'/f'{lineage}-forecasts_points.json.gz')
    indexed={(r['stream'],r['step'],r['arm']):r for r in rows}
    consumed=set()
    for si,s in enumerate(streams):
        for arm in ARMS:
            outputs=product_checkpoints(table,s['observations'],arm,s['length'],s['kind'],checkpoints(s['length']))
            for step,(current,joint,hs) in outputs.items():
                r=indexed[si,step,arm];consumed.add((si,step,arm))
                expected=dict(draw=s['draw'],initial_maker=s['maker'],length=s['length'],kind=s['kind'],switched=s['switched'],duplicates=s['duplicates'],unique_sources=len(distinct(s['observations'][:step])))
                if any(r[k]!=v for k,v in expected.items()):raise ValueError('parent assignment differs')
                error=max(error,near(current,r['posterior'],1e-10))
    if consumed!=set(indexed):raise ValueError('parent row roster differs')
    return cells,error,streams,indexed


def regroup(cells,cfg,parents):
    index = {(r['lineage'],r['draw'],*(r[k] for k in AXES),r['arm']):r for r in cells}
    if len(index)!=len(cells): raise ValueError('duplicate stratum')
    lineages = sorted({r['lineage'] for r in cells}); draws = sorted({r['draw'] for r in cells})
    samples = np.random.default_rng(cfg['bootstrap_seed']).integers(len(lineages),size=(cfg['bootstrap_resamples'],len(lineages)))
    strata = sorted({tuple(r[k] for k in AXES) for r in cells}); means=[]; contrasts=[]
    parent_index={(r['lineage'],r['draw'],*(r[k] for k in AXES),r['arm']):r for r in parents}
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
                    contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=base,metric=metric,comparison='within-omission',**estimate(by_arm[arm][:,:,i]-by_arm[base][:,:,i])))
            length,kind,switched,duplicates,step=stratum
            old=np.array([[[parent_index[l,d,length,kind,switched,duplicates,step,arm][m] for m in METRICS] for d in draws] for l in lineages])
            for i,metric in enumerate(METRICS):
                contrasts.append(dict(zip(AXES,stratum),arm=arm,baseline=arm,metric=metric,comparison='omission-minus-identity-aware',**estimate(by_arm[arm][:,:,i]-old[:,:,i])))
    return dict(cells=cells,parent_cells=parents,means=means,contrasts=contrasts,lineages=lineages,draws=draws,
        population='all sixteen makers equally within each draw; both observation draws averaged within paired coefficient lineage; each length/reference factor/actual change/source/checkpoint retained; identity-aware controls paired, never added as replicates; stationary reference-factor cells never pooled',
        uncertainty='conditional eight-lineage resampling; two observation draws do not establish universal sampling uncertainty; no fitted model')


def check_packet(packet, public):
    if packet['schema']!='v19.source-omission.reader.1' or packet['cases']!=[public[k] for k in sorted(public)]:
        raise ValueError('reader projection differs')


def run(root,plan,pulse):
    checks=controls(); write(root/'CONTROLS.json',checks)
    if not all(checks.values()): raise ValueError('independent controls failed')
    cfg=plan['design']; original=root/'inputs/original'
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h: raise ValueError('review input differs')
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']: raise ValueError('target plan differs')
    design=read(original/'PLAN.json')['design']; summary=read(original/'SUMMARY.json')
    if design['arms']!=list(ARMS) or any(n not in (32,128) for n in design['lengths']): raise ValueError('design differs')
    contexts=[{'initial':list(a),'requested_purpose':p} for a in ((0,1,0),(1,0,1)) for p in range(2)]
    all_cells=[]; parents=[]; public={}; source_counts=[]; independent_identities=0; error=0.; paths=rows_count=streams_count=observations_count=0
    for lineage in design['lineages']:
        pulse(phase='independent-native-paths',lineage=lineage)
        records=zipped(original/'inputs'/f'lineage-{lineage}_points.json.gz')
        _,_,e=population(records,lineage,'original'); error=max(error,e); paths+=len(records)
        table=endpoint_law(records); error=max(error,near(table,read(original/'evaluator'/f'{lineage}-law.json'),1e-12))
        old,e,parent_streams,parent_rows=parent_cells(original/'inputs/parent',lineage,table);parents.extend(old);error=max(error,e)
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
            true_observations=reconstruct_stream(table,lineage,*key)
            if parent_streams[si]['observations']!=true_observations:raise ValueError('parent stream order differs')
            observations=omit_identity(true_observations)
            expected=dict(zip(('draw','maker','length','kind','switched','duplicates'),key),observations=observations)
            if stream!=expected: raise ValueError('stream/source assignment differs')
            if len(distinct(observations))!=length or len(distinct(true_observations))!=(length*3//4 if duplicates else length): raise ValueError('unique-source denominator')
            visible=dict(contexts=contexts,observations=observations); ident=digest(visible)
            public[ident]=dict(input_sha256=ident,inputs=visible)
            streams_count+=1; observations_count+=length
            outputs={arm:product_checkpoints(table,observations,arm,length,kind,checkpoints(length)) for arm in ARMS}
            for step,arm in product(checkpoints(length),ARMS):
                row=indexed[si,step,arm]; consumed.add((si,step,arm))
                actual=complement(maker,kind) if switched and step>length//2 else maker
                metadata=dict(stream=si,draw=draw,initial_maker=maker,actual_maker=actual,length=length,kind=kind,switched=switched,duplicates=duplicates,step=step,arm=arm,supplied_sources=len(distinct(observations[:step])),true_sources=len(distinct(true_observations[:step])))
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
                if not duplicates:
                    oldrow=parent_rows[si,step,arm]
                    if row['posterior']!=oldrow['posterior'] or row['forecast']!=oldrow['forecast'] or any(row[m]!=oldrow[m] for m in METRICS):
                        raise ValueError('independent-stream exact identity differs')
                    independent_identities+=1
                retained=observations[:step][-16:] if arm=='reset-16' else observations[:step]
                original_ids={true_observations[r['step']-1]['source_id'] for r in retained}
                source_counts.append(dict(lineage=lineage,stream=si,step=step,arm=arm,
                    supplied=len(retained),true=len(original_ids)))
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
    write(root/'INDEPENDENT_REGROUP.json',regroup(all_cells,cfg,parents))
    if independent_identities*2!=rows_count:raise ValueError('independent identity denominator')
    (root/'RETAINED_SOURCE_COUNTS_points.json.gz').write_bytes(gzip.compress(canonical(source_counts),mtime=0))
    return dict(independent_identity_rows=independent_identities,controls=checks,paths=paths,rows=rows_count,streams=streams_count,observations=observations_count,cells=len(all_cells),public_packets=len(public),max_error=error,
        scope='independent identity-only source omission and paired identity-aware filtering review; timestamps retained; current-state coverage separate from endpoint prediction; no learned capability or historical path conclusion')
