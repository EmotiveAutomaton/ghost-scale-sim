"""Independent joint-state reconstruction; no producer kernels are imported."""
from collections import Counter, defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, canonical

TOL = 1e-12


def structure(spec):
    hs = spec['hypotheses']; cp = spec['checkpoint']; horizon = spec['length']
    if not 1 <= cp <= horizon: raise ValueError('time bounds')
    def state(h, t):
        kind, change, initial = h
        if kind not in ('none', 'purpose', 'skill') or not 0 <= initial < 16:
            raise ValueError('hypothesis')
        bits = list(map(int, f'{initial:04b}'))
        if kind != 'none' and t > change:
            i = 0 if kind == 'purpose' else 1
            bits[i] = 1 - bits[i]
        return sum(b * 2**(3-i) for i, b in enumerate(bits))
    paths = [tuple(state(h, t) for t in range(cp, horizon+1)) for h in hs]
    signatures = [tuple(s) for s in spec['signatures']]
    mapping = np.array(spec['membership'], int)
    if len(mapping) != len(hs) or set(mapping) != set(range(len(signatures))):
        raise ValueError('membership')
    if len(set(signatures)) != len(signatures) or any(signatures[g] != p for g,p in zip(mapping, paths)):
        raise ValueError('future schedule')
    members = [np.flatnonzero(mapping == g) for g in range(len(signatures))]
    mixed = [g for g, indices in enumerate(members) if len(indices) > 1]
    single = [int(indices[0]) for indices in members if len(indices) == 1]
    past = []; supports = []
    for t in range(1, cp+1):
        states = np.array([state(h,t) for h in hs], int)
        pairs = sorted(set(zip(mapping.tolist(), states.tolist())))
        lookup = {p:i for i,p in enumerate(pairs)}
        ids = [lookup[g,s] for g,s in zip(mapping,states)]
        mask = [len(members[g]) > 1 for g,s in pairs]
        supports.append(dict(pairs=[list(p) for p in pairs], mapping=ids, mixed=mask))
        past.append(states)
    return dict(mapping=mapping, members=members, mixed=mixed, single=single,
                past=past, supports=supports, signatures=np.array(signatures,int))


def close(a, b, label):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label+' shape/finite')
    if np.max(abs(a-b), initial=0) > TOL: raise ValueError(label)


def verify_batch(W, st, law, raw):
    """Bincount joint masses and independent full-hypothesis matrix products.

    Singleton equality is verified exactly. Only their common numerator is
    shared in the TV expression; both normalizers are computed independently.
    """
    W=np.asarray(W,float); law=np.asarray(law,float).reshape(16,32)
    if W.ndim!=2 or W.shape[1]!=len(st['mapping']) or (W<0).any(): raise ValueError('weights')
    close(W.sum(-1), np.ones(len(W)), 'posterior mass')
    if (law<0).any(): raise ValueError('law')
    close(law.reshape(16,4,8).sum(-1),np.ones((16,4)),'law mass')
    G=np.stack([np.bincount(st['mapping'],weights=w,minlength=len(st['members'])) for w in W])
    close(G,raw['group_weights'],'group mass')
    single=st['single']; mixed=st['mixed']; single_groups=st['mapping'][single]
    if not np.array_equal(raw['group_weights'][:,single_groups],W[:,single]):
        raise ValueError('singleton identity')
    expected_fields={'group_weights'}
    summaries=[]; max_independent=0.; report_count=np.zeros(len(W),int)
    for t,(states,s) in enumerate(zip(st['past'],st['supports']),1):
        fields={f'{t}-{k}' for k in ('mixed_joint_mass','report_probability','possible','updated_group_tv','report_probability_error')}
        expected_fields |= fields
        pairs=np.array(s['pairs'],int); mask=np.array(s['mixed'],bool)
        joint=np.stack([np.bincount(s['mapping'],weights=w,minlength=len(pairs)) for w in W])
        retained=raw[f'{t}-mixed_joint_mass']
        close(joint[:,mask],retained,'joint mass')
        # Independent direct full-hypothesis path, not producer reduceat.
        direct=np.stack([W[:,st['members'][g]] @ law[states[st['members'][g]]] for g in mixed],axis=-1) if mixed else np.empty((len(W),32,0))
        # Reconstruct every compressed-state numerator from the retained values.
        factored=np.zeros_like(direct)
        mixed_pairs=pairs[mask]
        for j,g in enumerate(mixed):
            ix=np.flatnonzero(mixed_pairs[:,0]==g)
            factored[:,:,j]=retained[:,ix] @ law[mixed_pairs[ix,1]]
        singleton=W[:,single] @ law[states[single]]
        p=singleton+direct.sum(-1); q=singleton+factored.sum(-1)
        past_mass=np.stack([np.bincount(states,weights=w,minlength=16) for w in W])
        close(p,past_mass @ law,'independent denominator')
        close(q,raw[f'{t}-report_probability'],'saved denominator')
        if not np.array_equal(p>0,q>0) or not np.array_equal(p>0,raw[f'{t}-possible']):
            raise ValueError('report support')
        close(p.reshape(-1,4,8).sum(-1),np.ones((len(W),4)),'report mass')
        invp=np.divide(1.,p,out=np.zeros_like(p),where=p>0)
        invq=np.divide(1.,q,out=np.zeros_like(q),where=q>0)
        tv=.5*(abs(direct*invp[:,:,None]-factored*invq[:,:,None]).sum(-1)+singleton*abs(invp-invq))
        if tv.max(initial=0)>TOL: raise ValueError('updated distribution')
        for field in ('updated_group_tv','report_probability_error'):
            value=raw[f'{t}-{field}']
            if value.shape!=p.shape or not np.isfinite(value).all() or (value<0).any() or value.max(initial=0)>TOL:
                raise ValueError('reported '+field)
        # Rounding-scale maxima are checked at the frozen tolerance, not claimed
        # bit-identical across genuinely different summation algorithms.
        close(tv,raw[f'{t}-updated_group_tv'],'recorded update discrepancy')
        close(abs(p-q),raw[f'{t}-report_probability_error'],'recorded denominator discrepancy')
        max_independent=max(max_independent,float(tv.max(initial=0)))
        report_count+=(p>0).sum(-1)
        summaries.append(dict(full_joint_cells=len(pairs),mixed_joint_cells=int(mask.sum()),
                              shared_support_int32_bytes=4*(2*len(pairs)+len(W[0]))))
    if set(raw)!=expected_fields: raise ValueError('raw field roster')
    storage=dict(full_weight_values=len(W[0]),group_weight_values=len(G[0]),
                 full_joint_values=sum(s['full_joint_cells'] for s in summaries),
                 factored_joint_values=len(G[0])+sum(s['mixed_joint_cells'] for s in summaries),
                 shared_support_int32_bytes=sum(s['shared_support_int32_bytes'] for s in summaries),
                 shared_future_schedule_int32_bytes=4*st['signatures'].size,
                 shared_hypothesis_to_group_int32_bytes=4*len(W[0]),reports=32*len(st['past']))
    return storage,report_count,max_independent


def controls():
    # Analytic contraction and dependence rulers, independent of the producer.
    a=np.array([.8,.2]);b=np.array([.3,.7]);f=np.array([1.,0.])
    return {'live:forecast_contraction':bool(abs((a-b)@f)<=.5*abs(a-b).sum()+1e-15),
            'placebo:identical_distribution':float(.5*abs(a-a).sum())==0.,
            'positive:tight_bound':bool(abs((a-b)@f)==.5*abs(a-b).sum())}


def run(root,plan,pulse):
    cfg=plan['design'];checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    original=root/'inputs/original';parent=root/'inputs/parent'
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    design=read(original/'PLAN.json')['design'];specs=read(parent/'SCHEDULES.json')
    expected_specs={f'{h}-{cp}' for h in design['lengths'] for cp in design['checkpoints'] if cp<=h}
    if set(specs)!=expected_specs or specs!=read(original/'evaluator/SCHEDULES.json'):raise ValueError('schedule roster')
    structures={};storage_rows=[];regroup=defaultdict(list);all_rows=[];unavailable=[];paired={}
    (root/'reconstructed').mkdir(exist_ok=True)
    for key,spec in specs.items():
        st=structure(spec);structures[key]=st
        if st['supports']!=read(original/'evaluator'/f'{key}-support.json'):raise ValueError('stored support')
        write(root/'reconstructed'/f'{key}-support.json',st['supports'])
    recorded=json.loads(gzip.decompress((original/'raw/sufficient_summary_points.json.gz').read_bytes()))
    keyfields=('lineage','evidence','length','checkpoint','draw','initial_maker','kind','switched','duplicates')
    keyed={tuple(r[k] for k in keyfields):r for r in recorded}
    if len(keyed)!=len(recorded):raise ValueError('duplicate summary')
    timings=[json.loads(s) for s in (original/'TIMING.jsonl').read_text().splitlines()]
    timing_keys={(r['lineage'],r['evidence'],r['length'],r['checkpoint'],r['first']):r for r in timings}
    if len(timing_keys)!=len(timings):raise ValueError('duplicate timing')
    used_timing=set();used_raw=set();max_error=0.
    for lineage in design['lineages']:
        law=np.asarray(read(parent/'aware/evaluator'/f'{lineage}-law.json'))
        for evidence in ('aware','omitted'):
            base=parent/evidence
            if not np.array_equal(law,read(base/'evaluator'/f'{lineage}-law.json')):raise ValueError('law pairing')
            bindings=read(base/'evaluator'/f'{lineage}-joint-map.json')
            rows=json.loads(gzip.decompress((base/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            rows=[r for r in rows if r['arm']=='unknown-time-type']
            with np.load(base/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:
                arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                cp=spec['checkpoint'];length=spec['length'];st=structures[key]
                chosen=[r for r in rows if r['length']==length and r['step']==cp]
                if not chosen:
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                ids=[tuple(r[k] for k in keyfields[4:]) for r in chosen]
                expected=set(product(design['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                W=[]
                for r,identity in zip(chosen,ids):
                    n,j=r['joint_array'],r['joint_row']
                    if bindings[n]['hypotheses']!=spec['hypotheses'] or bindings[n]['rows'][j]!=[r['stream'],cp]:raise ValueError('posterior row binding')
                    k=(lineage,key,*identity);witness=(r['stream'],r['actual_maker'])
                    if evidence=='aware':paired[k]=witness
                    elif paired[k]!=witness:raise ValueError('evidence pairing')
                    W.append(arrays[n][j])
                W=np.asarray(W)
                current=st['signatures'][st['mapping'],0]
                prior_forecast=np.stack([np.bincount(current,weights=w,minlength=16) for w in W])@law.reshape(16,32)
                close(prior_forecast.reshape(-1,4,8),[r['forecast'] for r in chosen],'parent forecast')
                for first in range(0,len(W),design['batch_rows']):
                    pulse(phase='independent-joint-state-reconstruction',lineage=lineage,evidence=evidence,checkpoint=key,first=first)
                    filename=f'{lineage}-{evidence}-{key}-{first:03d}_points.npz';used_raw.add(filename)
                    with np.load(original/'raw'/filename,allow_pickle=False) as z:raw={n:z[n] for n in z.files}
                    block=W[first:first+design['batch_rows']]
                    storage,possible,error=verify_batch(block,st,law,raw);max_error=max(max_error,error)
                    tk=(lineage,evidence,length,cp,first);tr=timing_keys[tk];used_timing.add(tk)
                    if tr['rows']!=len(block) or tr['reports']!=32*cp or not math.isfinite(tr['cpu_seconds']) or tr['cpu_seconds']<0:raise ValueError('timing coverage')
                    for offset,r in enumerate(chosen[first:first+len(block)]):
                        out=dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,
                                 **{k:r[k] for k in keyfields[4:]},**storage,possible_reports=int(possible[offset]))
                        sk=tuple(out[k] for k in keyfields);old=keyed.pop(sk)
                        for k,v in out.items():
                            if old[k]!=v:raise ValueError('summary '+k)
                        for k in ('stream','actual_maker','joint_array','joint_row'):
                            if old[k]!=r[k]:raise ValueError('summary binding '+k)
                        for field,rawfield in [('max_update_tv','updated_group_tv'),('all_future_coordinate_error_upper_bound','updated_group_tv'),('max_report_probability_error','report_probability_error')]:
                            value=max(float(raw[f'{t}-{rawfield}'][offset].max()) for t in range(1,cp+1))
                            if old[field]!=value:raise ValueError('recorded maximum')
                            out[field]=value
                        all_rows.append(out);regroup[(lineage,evidence,length,cp,r['draw'])].append(out)
    if keyed or used_timing!=set(timing_keys):raise ValueError('complete summary/timing coverage')
    if used_raw!={p.name for p in (original/'raw').glob('*.npz')}:raise ValueError('raw roster')
    summary=read(original/'SUMMARY.json')
    if summary['rows']!=len(all_rows) or summary['reports']!=sum(r['reports'] for r in all_rows) or summary['unavailable_checkpoints']!=unavailable:raise ValueError('complete summary')
    if summary['max_update_tv']!=max(r['max_update_tv'] for r in all_rows):raise ValueError('summary maximum')
    strata=[]
    for key,rows in sorted(regroup.items()):
        if len(rows)!=128:raise ValueError('draw stratum denominator')
        counts={k:sum(r[k] for r in rows) for k in ('reports','possible_reports')}
        sizes={k:sorted(set(r[k] for r in rows)) for k in ('full_weight_values','group_weight_values','full_joint_values','factored_joint_values','shared_support_int32_bytes','shared_future_schedule_int32_bytes','shared_hypothesis_to_group_int32_bytes')}
        if any(len(v)!=1 for v in sizes.values()):raise ValueError('structural storage consistency')
        strata.append(dict(zip(keyfields[:5],key),rows=len(rows),**counts,**{k:v[0] for k,v in sizes.items()},max_update_tv=max(r['max_update_tv'] for r in rows)))
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows),mtime=0))
    write(root/'PAIRED_STRATA.json',strata)
    write(root/'TIMING_REVIEW.json',dict(batches=len(timings),rows=sum(r['rows'] for r in timings),cpu_seconds=sum(r['cpu_seconds'] for r in timings),scope='producer joint construction plus all32reports at every past time;fixed32row batches;amortized query cost only;excludes serialization and parent loading'))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='independently reconstructed complete supports;all summary rows and paired law/source/checkpoint/draw strata;all-future contraction bound,not measured native forecast maximum'))
    return dict(passed=True,controls=checks,rows=len(all_rows),reports=sum(r['reports'] for r in all_rows),strata=len(strata),structures=len(structures),batches=len(used_raw),max_independent_update_tv=max_error,numerical_acceptance=False,scope='independent complete joint mass,denominator,update,storage and timing coverage;event-owned separate regroup and numerical adjudication remain required')
