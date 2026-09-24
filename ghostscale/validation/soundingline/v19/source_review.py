"""Independent retrospective provenance reconstruction; no producer imports."""
from collections import defaultdict
from itertools import product
import gzip
import json
import math
import numpy as np
from ..v18_3.io import read, write, file_digest, canonical
from .sufficient_review import structure, close, TOL

ALPHAS = (0., .25, .5, .75, 1.)
IDENTITY = ('draw', 'initial_maker', 'kind', 'switched', 'duplicates')
KEY = ('lineage', 'evidence', 'length', 'checkpoint', 'alpha') + IDENTITY


def source_inventory(observations, checkpoint):
    if not 1 <= checkpoint <= len(observations): raise ValueError('checkpoint')
    registry = {}; inventory = []
    for step, row in enumerate(observations, 1):
        if row['step'] != step: raise ValueError('observation order')
        source = row['source_id']
        content = (row['source_step'], row['context'], row['endpoint'])
        if not 1 <= content[0] <= step or not 0 <= content[1] < 4 or not 0 <= content[2] < 8:
            raise ValueError('source bounds')
        if source not in registry:
            if content[0] != step: raise ValueError('missing original')
            registry[source] = content
            if step <= checkpoint: inventory.append((source, *content))
        elif registry[source] != content: raise ValueError('changed source')
    return inventory


def future_mass(delta, st):
    """Direct indexed accumulations at bit-state changes, then prefix sums.

    Each hypothesis is inspected for its actual schedule change. There is no
    sparse matrix, producer transition table, or producer grouping kernel.
    """
    paths = st['signatures'][st['mapping']]
    count, times = paths.shape
    h, offset = np.nonzero(paths[:, 1:] != paths[:, :-1]); offset = offset + 1
    plus = offset * 16 + paths[h, offset]
    minus = offset * 16 + paths[h, offset-1]
    result = []
    for weights in delta:
        base = np.bincount(paths[:, 0], weights=weights, minlength=times*16)
        base += np.bincount(plus, weights=weights[h], minlength=times*16)
        base -= np.bincount(minus, weights=weights[h], minlength=times*16)
        result.append(np.cumsum(base.reshape(times, 16), axis=0))
    return np.asarray(result)


def verify_batch(W, st, law, ids, raw):
    """Rebuild source likelihoods, every future coordinate, and scalar factors."""
    W=np.asarray(W,float); law=np.asarray(law,float); ids=np.asarray(ids,int)
    if W.ndim != 2 or W.shape[1] != len(st['mapping']) or (W < 0).any(): raise ValueError('weights')
    close(W.sum(-1), np.ones(len(W)), 'weight mass')
    if law.shape != (16,4,8) or (law < 0).any(): raise ValueError('law')
    close(law.sum(-1),np.ones((16,4)), 'law mass')
    if ids.ndim != 2 or ids.shape[1] != 4 or not len(ids): raise ValueError('sources')
    if not np.array_equal(raw['source_rows'],ids): raise ValueError('source rows')
    delta=[]; probabilities=[]
    for row,t,ctx,end in ids:
        if not 0<=row<len(W) or not 1<=t<=len(st['past']) or not 0<=ctx<4 or not 0<=end<8: raise ValueError('source bounds')
        weights=W[row]; past=st['past'][t-1]
        mass=np.bincount(past,weights=weights,minlength=16)
        p=np.array([math.fsum(float(mass[s])*float(law[s,ctx,e]) for s in range(16)) for e in range(8)])
        if p[end] <= 0: raise ValueError('observed source unsupported')
        updated=weights*law[past,ctx,end]/p[end]
        close(updated.sum(),1.,'independent update mass')
        delta.append(updated-weights); probabilities.append(p)
    delta=np.asarray(delta); p=np.asarray(probabilities)
    close(p,raw['independent_report_probability'],'independent probability')
    group_delta=np.stack([np.bincount(st['mapping'],weights=d,minlength=len(st['members'])) for d in delta])
    group_tv=.5*abs(group_delta).sum(-1)
    forecasts=future_mass(delta,st)@law.reshape(16,32)
    maximum=abs(forecasts).max(axis=(1,2))
    mean_tv=.5*abs(forecasts.reshape(len(ids),-1,4,8)).sum(-1).mean(axis=(1,2))
    base={'group_tv':group_tv,'max_future_difference':maximum,'mean_future_tv':mean_tv}
    if (maximum>group_tv+TOL).any(): raise ValueError('contraction')
    for name, value in base.items(): close(value,raw['independent_vs_copy_'+name],name)
    correct=np.empty((len(ids),5,8));fraction=np.zeros_like(correct)
    independent_possible=p>0; copy_possible=np.zeros_like(p,dtype=bool)
    metrics={}
    for arm in ('independent','copy'):
        for name in ('support_failure_mass','supported_mass'):
            metrics[arm+'_'+name]=np.empty((len(ids),5))
        for name in base:
            for metric in ('supported_error_mass_','max_supported_'):
                metrics[arm+'_'+metric+name]=np.empty((len(ids),5))
    for n,(_,_,_,old) in enumerate(ids):
        copy_possible[n,old]=True
        for ai,alpha in enumerate(ALPHAS):
            # Scalar Bayes factors and exact cancellation of weighted errors.
            for endpoint in range(8):
                copied=alpha if endpoint==old else 0.
                denominator=math.fsum((copied,(1-alpha)*float(p[n,endpoint])))
                correct[n,ai,endpoint]=denominator
                fraction[n,ai,endpoint]=copied/denominator if denominator>0 else 0.
            old_mass=correct[n,ai,old]
            fail=math.fsum(float(correct[n,ai,e]) for e in range(8) if e!=old)
            metrics['copy_support_failure_mass'][n,ai]=fail
            metrics['copy_supported_mass'][n,ai]=old_mass
            metrics['independent_support_failure_mass'][n,ai]=math.fsum(float(correct[n,ai,e]) for e in range(8) if not independent_possible[n,e])
            metrics['independent_supported_mass'][n,ai]=math.fsum(float(correct[n,ai,e]) for e in range(8) if independent_possible[n,e])
            for name,value in base.items():
                metrics['independent_supported_error_mass_'+name][n,ai]=alpha*value[n]
                metrics['independent_max_supported_'+name][n,ai]=alpha/old_mass*value[n]
                metrics['copy_supported_error_mass_'+name][n,ai]=(1-alpha)*p[n,old]*value[n]
                metrics['copy_max_supported_'+name][n,ai]=(1-alpha)*p[n,old]/old_mass*value[n]
    close(correct,raw['correct_report_probability'],'mixture probability')
    close(fraction,raw['copy_fraction'],'copy fraction')
    for key,value in [('independent_possible',independent_possible),('copy_possible',copy_possible),('correct_possible',correct>0)]:
        if not np.array_equal(raw[key],value): raise ValueError(key)
    close(correct.sum(-1),np.ones((len(ids),5)),'correct report mass')
    expected={'source_rows','independent_report_probability','correct_report_probability','copy_fraction','independent_possible','copy_possible','correct_possible'}|{'independent_vs_copy_'+n for n in base}
    if set(raw)!=expected:raise ValueError('raw field roster')
    return metrics


def controls():
    p=.2; alpha=.5; denominator=alpha+(1-alpha)*p
    return {'live:copied_evidence_changes_weight':alpha/denominator>alpha,
            'placebo:zero_copy_weight':0./p==0.,
            'positive:copy_rival_loses_support':(1-alpha)*(1-p)>.1}


def run(root,plan,pulse):
    cfg=plan['design']; checks=controls();write(root/'CONTROLS.json',checks)
    if not all(checks.values()):raise ValueError('controls')
    for n,h in cfg['input_files'].items():
        if file_digest(root/'inputs'/n)!=h:raise ValueError('input binding')
    original=root/'inputs/original';parent=root/'inputs/parent'
    if file_digest(original/'PLAN.json')!=cfg['target_plan_sha256']:raise ValueError('target plan')
    design=read(original/'PLAN.json')['design'];specs=read(parent/'SCHEDULES.json')
    if design['alphas']!=list(ALPHAS):raise ValueError('alpha roster')
    if set(specs)!={f'{h}-{cp}' for h in design['lengths'] for cp in design['checkpoints'] if cp<=h}:raise ValueError('schedule roster')
    if specs!=read(original/'evaluator/SCHEDULES.json'):raise ValueError('saved schedules')
    structures={key:structure(spec) for key,spec in specs.items()}
    recorded=json.loads(gzip.decompress((original/'raw/source_summary_points.json.gz').read_bytes()))
    keyed={tuple(r[k] for k in KEY):r for r in recorded}
    if len(keyed)!=len(recorded):raise ValueError('duplicate summary')
    timings=[json.loads(s) for s in (original/'TIMING.jsonl').read_text().splitlines()]
    timing_keys={(r['lineage'],r['evidence'],r['length'],r['checkpoint']):r for r in timings}
    if len(timing_keys)!=len(timings):raise ValueError('duplicate timing')
    used_timing=set();used_raw=set();paired={};all_rows=[];unavailable=[];total_sources=0;strata=defaultdict(list)
    (root/'reconstructed').mkdir(exist_ok=True)
    for lineage in design['lineages']:
        law=np.asarray(read(parent/'aware/evaluator'/f'{lineage}-law.json'))
        streams=json.loads(gzip.decompress((parent/'aware/raw'/f'{lineage}-observations_points.json.gz').read_bytes()))
        for evidence in ('aware','omitted'):
            folder=parent/evidence
            if not np.array_equal(law,read(folder/'evaluator'/f'{lineage}-law.json')):raise ValueError('paired law')
            mapping=read(folder/'evaluator'/f'{lineage}-joint-map.json')
            rows=json.loads(gzip.decompress((folder/'raw'/f'{lineage}-forecasts_points.json.gz').read_bytes()))
            rows=[r for r in rows if r['arm']=='unknown-time-type']
            with np.load(folder/'raw'/f'{lineage}-joint_points.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files if n.endswith('-unknown-time-type')}
            for key,spec in specs.items():
                cp=spec['checkpoint'];length=spec['length'];st=structures[key]
                chosen=[r for r in rows if r['length']==length and r['step']==cp]
                if not chosen:
                    if (length,cp) not in ((128,8),(128,17),(128,20)):raise ValueError('missing checkpoint')
                    unavailable.append(dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp));continue
                ids=[tuple(r[k] for k in IDENTITY) for r in chosen]
                expected=set(product(design['draws'],range(16),('purpose','skill'),(False,True),(False,True)))
                if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('paired roster')
                W=[];bindings=[];sources=[]
                for row,(r,identity) in enumerate(zip(chosen,ids)):
                    n,j=r['joint_array'],r['joint_row'];stream=streams[r['stream']]
                    if mapping[n]['hypotheses']!=spec['hypotheses'] or mapping[n]['rows'][j]!=[r['stream'],cp]:raise ValueError('posterior binding')
                    if tuple(stream[k] for k in ('draw','maker','kind','switched','duplicates'))!=identity or stream['length']!=length:raise ValueError('stream binding')
                    selected=source_inventory(stream['observations'],cp)
                    pair=(lineage,key,*identity);witness=(r['stream'],r['actual_maker'],selected)
                    if evidence=='aware':paired[pair]=witness
                    elif paired[pair]!=witness:raise ValueError('paired sources')
                    W.append(arrays[n][j]);bindings.append(dict(**r,report_sources=len(selected)))
                    sources.extend((row,t,ctx,end,source_id) for source_id,t,ctx,end in selected)
                saved=read(original/'evaluator'/f'{lineage}-{evidence}-{key}-bindings.json')
                if saved['rows']!=bindings or saved['sources']!=[list(s) for s in sources]:raise ValueError('saved source binding')
                W=np.asarray(W);current=st['signatures'][st['mapping'],0]
                forecasts=np.stack([np.bincount(current,weights=w,minlength=16) for w in W])@law.reshape(16,32)
                close(forecasts.reshape(-1,4,8),[r['forecast'] for r in chosen],'parent forecast')
                accumulated=defaultdict(lambda:defaultdict(list))
                for first in range(0,len(sources),design['source_batch_size']):
                    pulse(phase='independent-source-mixture-review',lineage=lineage,evidence=evidence,checkpoint=key,first_source=first)
                    filename=f'{lineage}-{evidence}-{key}-{first:05d}_points.npz';used_raw.add(filename)
                    with np.load(original/'raw'/filename,allow_pickle=False) as z:raw={n:z[n] for n in z.files}
                    block=np.asarray([s[:4] for s in sources[first:first+design['source_batch_size']]],int)
                    metrics=verify_batch(W,st,law,block,raw)
                    for offset,row in enumerate(block[:,0]):
                        for name,values in metrics.items():accumulated[int(row)][name].append(values[offset])
                for row,r in enumerate(bindings):
                    for ai,alpha in enumerate(ALPHAS):
                        out=dict(lineage=lineage,evidence=evidence,length=length,checkpoint=cp,alpha=alpha,**{k:r[k] for k in IDENTITY},**{k:r[k] for k in ('stream','actual_maker','joint_array','joint_row','report_sources')})
                        for name,values in accumulated[row].items():
                            if len(values)!=r['report_sources']:raise ValueError('source denominator')
                            numbers=[float(v[ai]) for v in values]
                            out[name]=max(numbers) if '_max_supported_' in name else math.fsum(numbers)/len(numbers)
                        old=keyed.pop(tuple(out[k] for k in KEY))
                        if set(old)!=set(out):raise ValueError('summary fields')
                        for name,value in out.items():
                            if name in accumulated[row]:close(value,old[name],'summary '+name)
                            elif old[name]!=value:raise ValueError('summary identity')
                        all_rows.append(out);strata[(lineage,evidence,length,cp,r['draw'],alpha)].append(out)
                tk=(lineage,evidence,length,cp);tr=timing_keys[tk];used_timing.add(tk)
                if tr['posterior_rows']!=len(W) or tr['sources']!=len(sources) or not math.isfinite(tr['cpu_seconds']) or tr['cpu_seconds']<0:raise ValueError('timing coverage')
                total_sources+=len(sources)
    if keyed or used_timing!=set(timing_keys):raise ValueError('summary/timing coverage')
    if used_raw!={p.name for p in (original/'raw').glob('*.npz')}:raise ValueError('raw roster')
    summary=read(original/'SUMMARY.json')
    for name,value in dict(rows=len(all_rows),posterior_rows=len(all_rows)//5,sources=total_sources,report_queries=total_sources*40,unavailable_checkpoints=unavailable).items():
        if summary[name]!=value:raise ValueError('summary '+name)
    grouped=[]
    for key,rows in sorted(strata.items()):
        if len(rows)!=128:raise ValueError('paired stratum denominator')
        measures={k:math.fsum(float(r[k]) for r in rows)/len(rows) for k in rows[0] if k.startswith(('copy_','independent_'))}
        grouped.append(dict(zip(('lineage','evidence','length','checkpoint','draw','alpha'),key),rows=len(rows),**measures))
    (root/'reconstructed/summary_points.json.gz').write_bytes(gzip.compress(canonical(all_rows),mtime=0))
    write(root/'PAIRED_STRATA.json',grouped)
    write(root/'TIMING_REVIEW.json',dict(batches=len(timings),posterior_rows=sum(r['posterior_rows'] for r in timings),sources=total_sources,cpu_seconds=math.fsum(r['cpu_seconds'] for r in timings),scope='producer source batches,all future differences,raw serialization and source accumulation;excludes parent loading;amortized cost not latency'))
    write(root/'EVIDENCE_ROLES.json',dict(reader='no new reader inputs',evaluator='reconstructed scalar Bayes factors and complete future-coordinate discrepancies;source probabilities,unsupported mass and supported error remain separate;true source identities supplied in both posterior conditions'))
    return dict(passed=True,controls=checks,rows=len(all_rows),sources=total_sources,report_queries=total_sources*40,strata=len(grouped),batches=len(used_raw),numerical_acceptance=False,scope='independent source identity,likelihood,future-coordinate,scalar error cancellation,summary and timing reconstruction;separate original-raw regroup and event adjudication remain required')
