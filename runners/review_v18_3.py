"""Source-bound review computation on completed packets; no new learned fits."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path
import time
import uuid
import zipfile
import numpy as np
from ghostscale.validation.soundingline.v18_3 import world as W,calibration as C
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,canonical
from ghostscale.validation.soundingline.v18_3.runtime import load,stats,fingerprint
from ghostscale.validation.soundingline.v16.runtime import local_owner
from runners.analyze_v18_3 import contrasts


def units(root):
    for block in read(root/'COMPLETE.json')['blocks']:yield from load(root,block)


def calibration(root,pulse):
    coverage=[]
    for cell in range(16):
        for draw in range(20):
            w=W.make_world(cell,draw)
            for condition in ('matched','hidden-state-access','wrong-rule'):
                coverage.append(dict(cell=cell,lineage=draw,condition=condition,**C.native_coverage(w,condition)))
        pulse()
    coverage_summary={}
    for condition in ('matched','hidden-state-access','wrong-rule'):
        coverage_summary[condition]={m:stats([float(np.mean([r[m] for r in coverage if r['condition']==condition and r['lineage']==i])) for i in range(20)],('coverage',condition,m)) for m in ('coverage','expected_set_size')}
    complete=read(root/'neural/COMPLETE.json');manifest=read(root/'data/EVALUATOR.json');rows=[]
    for condition,entry in manifest['truth'].items():
        with np.load(root/'data'/entry['name'],allow_pickle=False) as z:
            truth=z['target'];ids=z['ids'];refs={name:z[key].copy() for name,key in (('exact','exact'),('passive-summary','passive_summary'),('intervention-summary','intervention_summary'))}
        def forecasts():
            yield from refs.items()
            for name,tests in complete['predictions'].items():
                with np.load(root/'neural'/tests[condition]['file'],allow_pickle=False) as z:yield name,z['probabilities'].copy()
        for name,p in forecasts():
            method,seed=name.rsplit('-seed',1) if '-seed' in name else (name,None)
            for lineage in sorted(set(ids[:,1])):
                chosen=ids[:,1]==lineage
                rows.append(dict(condition=condition,method=method,seed=seed,lineage=int(lineage),**C.reliability(p[chosen],truth[chosen])))
            pulse()
    summary={}
    for condition in manifest['truth']:
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['condition']==condition and r['method']==method]
            summary[condition+'|'+method]={metric:stats([float(np.mean([r[metric] for r in selected if r['lineage']==i])) for i in sorted({r['lineage'] for r in selected})],('reliability',condition,method,metric)) for metric in ('expected_absolute_calibration_gap','overconfidence')}
    return dict(fixed_instrument_coverage=coverage_summary,neural_reliability=summary,numerical_ties_and_bin_boundaries=1e-10,
        scope='fixed two-query prior-predictive coverage; E per-seed, per-lineage ten-bin expected top-choice reliability with uniform numerical ties, then fit/cell aggregation; not adaptive coverage or empirical label accuracy'),dict(coverage=coverage,reliability=rows)


def neural_factorial(roots):
    output={};points=[]
    for root in roots:
        family=read(root/'SUMMARY.json')['family'];raw=json.loads(gzip.decompress((root/'neural_points.json.gz').read_bytes()))['rows']
        configurations=[]
        if family=='E':
            configurations=[(condition,method,'direct','expected_loss') for condition in ('in-support','new-combinations','new-queries','both-new') for method in ('flat','split','exact','passive-summary','intervention-summary')]
        elif family=='E-purpose':
            configurations=[(condition,method,'raw-history','expected_loss') for condition in ('in-support','new-combinations') for method in ('flat','split','direct','exact','passive-summary','intervention-summary')]
        else:configurations=[(None,method,baseline,'counterfactual_loss') for method,baseline in (('split-IIT','split-behavior'),('split-IIT','direct-pair'),('flat-IIT','direct-pair'),('split-shuffled-IIT','split-IIT'))]
        for condition,method,baseline,metric in configurations:
            selected=[r for r in raw if (condition is None or r['condition']==condition) and r['method'] in (method,baseline)]
            draws=sorted({r['lineage'] for r in selected});values=[]
            for draw in draws:
                row=[]
                for cell in range(16):
                    arms={m:[r[metric]['value'] for r in selected if r['method']==m and r['lineage']==draw and r['cell']==cell] for m in (method,baseline)}
                    if not all(arms.values()) or any(v is None for a in arms.values() for v in a):raise ValueError('incomplete finite neural contrast')
                    value=math.fsum(arms[method])/len(arms[method])-math.fsum(arms[baseline])/len(arms[baseline]);row.append(value)
                    points.append(dict(family=family,condition=condition,method=method,baseline=baseline,metric=metric,lineage=draw,cell=cell,value=value))
                values.append(row)
            values=np.array(values);key='|'.join(str(x) for x in (family,condition,method,baseline))
            output[key]=dict(pooled=stats(values.mean(1).tolist(),key),cells={str(c):stats(values[:,c].tolist(),(key,c)) for c in range(16)},
                contrasts={name:stats(v.tolist(),(key,name)) for name,v in contrasts(values).items()},
                cells_with_positive_mean=int(np.sum(values.mean(0)>1e-10)),cells_with_negative_mean=int(np.sum(values.mean(0)<-1e-10)))
    return dict(effects=output,scope='paired coefficient-draw clusters; fixed sixteen cells, no fit-seed or role/probe pseudoreplication'),points


def examples(campaign):
    examples=[];w=W.make_world(0,0);a=W.STATES.index((1,0,0,0));b=W.STATES.index((1,1,1,0))
    ordinary=W.artifact_matrix(w,W.context());changed=W.artifact_matrix(w,W.context(signal=0))
    assert np.max(abs(ordinary[a]-ordinary[b]))<1e-12
    examples.append(dict(name='Ambiguous history',selection='fixed cell zero, coefficient draw zero; states (1,0,0,0) and (1,1,1,0)',
        states=[W.STATES[a],W.STATES[b]],ordinary_max_difference=float(np.max(abs(ordinary[a]-ordinary[b]))),intervention_total_variation=float(np.sum(abs(changed[a]-changed[b]))/2),
        meaning='ordinary outcomes cannot distinguish this goal/belief pair; a supplied signal intervention can'))
    for u in units(campaign/'A-uniform-r1'):
        row=next(r for r in u['rows'] if r['purpose']=='enactment' and r['method']=='task-three-attempts')
        if row['enactment']['success'] and row['scores']['state_mass']<.5:
            examples.append(dict(name='Enactment without state identification',selection='first raw-roster case with action-three-attempt success and true-state mass below one half',packet='A-uniform-r1',cell=u['cell'],lineage=u['index'],target=u['evaluator']['target'],row=row));break
    for u in units(campaign/'B-goal'):
        by={r['method']:r for r in u['rows']};cut=u['evaluator']['change']
        losses={m:float(np.mean([x['expected_loss']['value'] for x in by[m]['trace'][cut:]])) for m in ('static','selective-transition')}
        if losses['selective-transition']<losses['static'] and by['selective-transition']['trace'][-1]['true_skill_mass']>.8:
            examples.append(dict(name='Goal change retaining skill',selection='first raw-roster goal-change case with selective lower post-change loss and final true-skill mass above .8',packet='B-goal',cell=u['cell'],lineage=u['index'],change=cut,states=u['evaluator']['states'],post_change_losses=losses,traces={m:by[m]['trace'] for m in losses}));break
    for u in units(campaign/'C-shared-error'):
        by={r['method']:r for r in u['rows']}
        if u['roots']==2 and u['copies']==6 and by['independent']['false_confidence']:
            examples.append(dict(name='Copied false corroboration',selection='first two-root/six-copy shared-error case with naive false confidence',packet='C-shared-error',lineage=u['index'],public=u['public'],truth=u['evaluator']['truth'],rows=[by[m] for m in ('independent','known-graph','cautious-mixture')]));break
    for u in units(campaign/'D-outside-menu'):
        if u['order']!='late':continue
        by={r['method']:r for r in u['rows']};losses={m:float(np.mean([x['expected_loss']['value'] for x in by[m]['forecasts']])) for m in ('fixed','revision','mixture')}
        if losses['revision']>losses['fixed']:
            examples.append(dict(name='Failed context repair',selection='first late outside-menu case whose revision loses to fixed',packet='D-outside-menu',cell=u['cell'],lineage=u['index'],losses=losses,cue=u['cue'],revision=u['revision']));break
    for u in units(campaign/'H-core'):
        by={r['method']:r for r in u['rows'] if r['cardinality']==2};flat=by['exhaustive-flat'];product=by['observation-product']
        if flat['new_loss']<=product['new_loss']+1e-8:continue
        pair=next(((a,b) for a in range(8) for b in range(a+1,8) if flat['code'][a]==flat['code'][b] and product['code'][a]!=product['code'][b] and np.max(abs(np.array(u['new_predictions'][a])-u['new_predictions'][b]))>1e-8),None)
        if pair:
            examples.append(dict(name='A memory losing a newly useful distinction',selection='first two-symbol case with flat worse new loss and a collapsed pair separated by the product code',packet='H-core',cell=u['cell'],lineage=u['index'],history_indices=pair,histories=[u['histories'][i] for i in pair],new_predictions=[u['new_predictions'][i] for i in pair],rows=list(by.values())));break
    if len(examples)!=6:raise ValueError('a declared worked-example selection was unavailable')
    return dict(examples=examples,scope='mechanical illustrations selected after aggregate exposure, not a representative sample or independent evidence')


def exact_costs(root):
    cases=json.loads(gzip.decompress((root/'data/in-support-points.json.gz').read_bytes()))[:32];rows=[]
    for case in cases:
        w=case['world'];history=case['history'];packet=W.packet(w,history);weights=W.posterior(packet);table=W.artifact_matrix(w,W.context())
        # Warm table caches for both methods; include public-packet parse in recomputation.
        repeated=[]
        for _ in range(5):
            start=time.perf_counter()
            for j in range(50):W.posterior(packet)@table
            full=(time.perf_counter()-start)/50
            start=time.perf_counter()
            for j in range(50):weights@table
            cached=(time.perf_counter()-start)/50
            repeated.append(dict(full_history_query_seconds=full,cached_query_seconds=cached))
        rows.append(dict(history_length=len(history),timings=repeated,posterior_floats=len(weights),world_floats=17))
    return dict(rows=rows,environment=fingerprint(),scope='Ghost interpreter; warm native tables, public packet parsing included in full inference; local timing, not a cross-runtime universal speed claim')


def source_uptake(campaign):
    from ghostscale.validation.soundingline.v18_3.source_uptake import apply
    points=[];grouped={}
    for root in sorted(campaign.glob('C-*')):
        if not (root/'COMPLETE.json').exists():continue
        for original in units(root):
            case=apply(original);points.append(case)
            for row in case['rows']:
                key=canonical({k:case[k] for k in ('mode','roots','copies')}|{'method':row['method']}).decode()
                group=grouped.setdefault(key,dict(valid=[],initial_success=[],corrected_success=[],practice_actions=[],execution_actions=[],counterfactual_check_work=[]))
                group['valid'].append(float(row['valid']))
                if row['valid']:
                    group['initial_success'].append(float(row['initial']['success']));group['corrected_success'].append(float(row['corrected']['success']))
                    group['practice_actions'].append(float(row['initial']['practice_actions']));group['execution_actions'].append(float(row['initial']['execution_actions']))
                    group['counterfactual_check_work'].append(float(row.get('counterfactual_check_work',0)))
    summary={key:{metric:stats(values,('source-uptake',key,metric)) if values else dict(n=0,mean=None,unavailable=True) for metric,values in group.items()} for key,group in grouped.items()}
    return dict(cells=summary,assigned_cases=len(points),scope='post-hoc reuse of C source cases; explicit recommendation-to-practice policy and stronger public-law task checker; no new independent data'),points


def compression_ties(campaign,pulse):
    from ghostscale.validation.soundingline.v18_3.numerical_audit import tied_codebooks
    rows=[]
    for name in ('H-core','F-H'):
        for u in units(campaign/name):
            for k in (1,2,4,8):
                result=tied_codebooks(np.array(u['history_probabilities']),np.array(u['old_predictions']),np.array(u['new_predictions']),k,query_entropy=np.log(3))
                selected=next(r for r in u['rows'] if r['method']=='exhaustive-flat' and r['cardinality']==k)
                if selected['old_loss']>result['old_minimum']+1e-10:raise ValueError('old code outside numerical optimum')
                rows.append(dict(packet=name,lineage=u['index'],cell=u['cell'],selected_new_loss=selected['new_loss'],**result))
            if u['index']%8==0:pulse()
    return dict(rows=rows,scope='all numerically tied old-task optima; no reselection on future outcomes and no new independent data')


def run(campaign,output):
    with local_owner(campaign/'scientific-worker-owner'):
        acceptance=read(campaign/'ACCEPTANCE.json');attempt=campaign/'attempts'/('review-'+uuid.uuid4().hex+'.json')
        previous=[read(p) for p in campaign.glob('attempts/*.json')]
        if any(p['state']=='running' for p in previous):raise ValueError('active scientific attempt')
        old=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        def pulse(state='running'):
            write(attempt,dict(packet='review-calibration-examples-costs',state=state,cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
            if state=='running' and (old+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start'])):raise TimeoutError('review computation cutoff')
        try:
            pulse();output.mkdir(parents=True,exist_ok=True)
            roots=[campaign/name for name in ('E-discovery-r1','E-purpose-scoring-r1','G-discovery-r1')]
            if not (roots[1]/'INDEPENDENT_REPLAY.json').exists():raise ValueError('Purpose accuracy tie repair and replay are required before review')
            if read(roots[1]/'SUMMARY.json').get('accuracy_ties',{}).get('absolute_tolerance')!=1e-10:raise ValueError('unrepaired purpose accuracy refused')
            for root in roots:
                proof=read(root/'INDEPENDENT_REPLAY.json')
                if not proof['passed'] or proof['summary_sha256']!=file_digest(root/'SUMMARY.json'):raise ValueError('neural review requires verified completed packet')
            plan=dict(source_sha256=file_digest(__file__),calibration_sha256=file_digest(C.__file__),factorial_sha256=file_digest(Path(__file__).with_name('analyze_v18_3.py')),
                packets={root.name:file_digest(root/'COMPLETE.json') for root in roots},calibration_design='16 cells x 20 draws x matched/hidden-access/wrong-rule, fixed queries 1 and 3; ten-bin E forecasts',example_selection='first retained-roster cases satisfying fixed explanatory predicates; post-outcome illustrations')
            from ghostscale.validation.soundingline.v18_3.runtime import source_files,REPO
            additional=[campaign/n for n in ('A-uniform-r1','B-goal','D-outside-menu','H-core','F-H')]+list(campaign.glob('C-*'))
            plan['packets'].update({p.name:file_digest(p/'COMPLETE.json') for p in additional if (p/'COMPLETE.json').exists()})
            plan['sources']={name:file_digest(REPO/name) for name in source_files()}
            with zipfile.ZipFile(output/'SOURCE.zip','x',zipfile.ZIP_DEFLATED) as archive:
                for name in plan['sources']:archive.write(REPO/name,name)
            plan['source_archive_sha256']=file_digest(output/'SOURCE.zip')
            write(output/'PLAN.json',plan)
            summary,points=calibration(roots[0],pulse);raw=output/'CALIBRATION_points.json.gz';raw.write_bytes(gzip.compress(canonical(points),mtime=0));summary['raw_sha256']=file_digest(raw);write(output/'CALIBRATION.json',summary)
            pulse();summary,points=neural_factorial(roots);raw=output/'NEURAL_FACTORIAL_points.json.gz';raw.write_bytes(gzip.compress(canonical(points),mtime=0));summary['raw_sha256']=file_digest(raw);write(output/'NEURAL_FACTORIAL.json',summary)
            pulse();write(output/'EXAMPLES.json',examples(campaign));write(output/'EXACT_COSTS.json',exact_costs(roots[0]))
            summary,points=source_uptake(campaign);raw=output/'SOURCE_UPTAKE_points.json.gz';raw.write_bytes(gzip.compress(canonical(points),mtime=0));summary['raw_sha256']=file_digest(raw);write(output/'SOURCE_UPTAKE.json',summary)
            write(output/'COMPRESSION_TIES.json',compression_ties(campaign,pulse));pulse('complete')
        except BaseException:pulse('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.campaign,a.output)
