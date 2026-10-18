"""Independent assembly and reconstruction of all D5 successor partitions."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import math
from pathlib import Path
import statistics
import time
import numpy as np
from scipy.stats import t
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime,profile_successor as study

ARMS=('baseline','own','donor');CONDITIONS=('planted','uninformative','constant')


def reconstruct_part(packet):
    plan=read(packet/'PLAN.json');complete=read(packet/'COMPLETE.json')
    study.verify_configuration(plan)
    if complete['state']!='verified' or digest(packet/'PLAN.json')!=complete['plan_sha256']:
        raise ValueError('partition completion binding differs')
    replay=runtime.compare(packet/'original',packet/'replay')
    if replay!=read(packet/'REPLAY.json') or digest(packet/'REPLAY.json')!=complete['replay_sha256']:
        raise ValueError('partition replay binding differs')
    raw=packet/'original';points=[];controls=[];targets=0;cold=0;cfg=plan['configuration']
    for world in plan['world_indices']:
        by_condition={}
        for condition in CONDITIONS:
            cell=raw/f'w{world:02d}-{condition}'
            records=read(cell/'READER.json');labels=read(cell/'LABELS.json');predictions=read(cell/'PREDICTIONS.json')
            truth=read(cell/'EVALUATOR.json');expected_count=cfg['writers']*cfg['rounds']
            if not (len(records)==len(labels)==len(predictions)==expected_count):raise ValueError('target coverage differs')
            if len({r['key'] for r in records})!=expected_count:raise ValueError('duplicate target')
            if any(w==d for w,d in enumerate(truth['donors'])):raise ValueError('donor is own writer')
            losses={arm:[] for arm in ARMS}
            for row,label,prediction in zip(records,labels,predictions):
                if row['key']!=label['key'] or row['key']!=prediction['key']:raise ValueError('target alignment differs')
                expected=[r['key'] for r in sorted(records,key=lambda r:(r['open_at'],r['key'])) if r['resolved_at']<row['open_at']]
                for arm in ARMS:
                    output=prediction['arms'][arm];p=output['probabilities']
                    if output['training_keys']!=expected or output['cold_start']!=(not expected):raise ValueError('chronology differs')
                    if len(p)!=4 or any(not math.isfinite(x) or x<0 for x in p) or abs(sum(p)-1)>1e-10:raise ValueError('invalid forecast')
                    losses[arm].append((sum(x*x for x in p)+1-2*p[label['action']])/2)
                cold+=int(prediction['arms']['own']['cold_start']);targets+=1
            means={arm:statistics.fmean(values) for arm,values in losses.items()}
            points.append(dict(world=world,condition=condition,gain_baseline=means['baseline']-means['own'],
                gain_donor=means['donor']-means['own'],losses=means))
            by_condition[condition]=predictions
        original=by_condition['planted'];future=read(raw/f'w{world:02d}-future/PREDICTIONS.json');option=read(raw/f'w{world:02d}-option/PREDICTIONS.json')
        prefix=cfg['future_prefix'];permutation=[2,0,3,1]
        future_difference=max(abs(x-y) for a,b in zip(original[:prefix],future[:prefix]) for arm in ARMS for x,y in zip(a['arms'][arm]['probabilities'],b['arms'][arm]['probabilities']))
        option_difference=max(abs(a['arms'][arm]['probabilities'][permutation[j]]-b['arms'][arm]['probabilities'][j]) for a,b in zip(original,option) for arm in ARMS for j in range(4))
        constant_difference=max(abs(x-y) for row in by_condition['constant'] for x,y in zip(row['arms']['own']['probabilities'],row['arms']['donor']['probabilities']))
        controls.append(dict(world=world,future_max_difference=future_difference,option_max_difference=option_difference,constant_max_difference=constant_difference))
    saved=read(raw/'POINTS.json');declared=read(raw/'CONTROLS.json')
    if len(saved)!=len(points) or declared!=controls:raise ValueError('saved aggregate or controls differ')
    for a,b in zip(points,saved):
        if (a['world'],a['condition'])!=(b['world'],b['condition']):raise ValueError('aggregate identity differs')
        for field in ('gain_baseline','gain_donor'):
            if abs(a[field]-b[field])>1e-12:raise ValueError('score reconstruction differs')
    return dict(packet=packet.name,worlds=plan['world_indices'],points=points,controls=controls,
        targets=targets,cold_start_targets=cold,files=len(replay['files']),complete_sha256=digest(packet/'COMPLETE.json'),
        replay_sha256=digest(packet/'REPLAY.json'),plan_sha256=digest(packet/'PLAN.json'))


def review(campaign,output):
    packets=[]
    for path in sorted((campaign/'packets').glob('v23-s15-profile-successor-part-*')):
        if (path/'PLAN.json').exists():packets.append(path)
    if not packets:raise ValueError('no successor partitions')
    parts=[reconstruct_part(path) for path in packets]
    worlds=[w for part in parts for w in part['worlds']]
    if sorted(worlds)!=list(range(40)) or len(worlds)!=len(set(worlds)):raise ValueError('world coverage differs')
    points=[row for part in parts for row in part['points']];controls=[row for part in parts for row in part['controls']]
    outcomes={};decisions=[]
    for condition in CONDITIONS:
        outcomes[condition]={}
        for comparator in ('baseline','donor'):
            sample=[r['gain_'+comparator] for r in points if r['condition']==condition]
            if len(sample)!=40:raise ValueError('condition coverage differs')
            mean=statistics.fmean(sample);radius=float(t.ppf(.975,39)*statistics.stdev(sample)/math.sqrt(40))
            lower,upper=mean-radius,mean+radius;passed=lower>.005 if condition=='planted' else upper<=.005
            outcomes[condition][comparator]=dict(mean=mean,lower=lower,upper=upper,n=40,criterion_passed=passed)
            decisions.append(passed)
    structural=all(row['future_max_difference']<=1e-7 and row['option_max_difference']<=1e-9 and row['constant_max_difference']<=1e-12 for row in controls)
    status='PASS' if all(decisions) and structural else 'FAIL'
    result=dict(status=status,original_stage15_status='FAIL_UNCHANGED',constructed_world=True,
        outcomes=outcomes,controls=controls,structural_controls_pass=structural,parts=parts,
        worlds=40,targets=sum(p['targets'] for p in parts),cold_start_targets=sum(p['cold_start_targets'] for p in parts),
        complete_world_coverage=True,full_fresh_process_replay=True,independent_raw_score_reconstruction=True,
        certified_fit_tag=study.TAG,certified_fit_tag_object=study.TAG_OBJECT,engine_commit=study.ENGINE_COMMIT,
        limitations=['Finite constructed method validation; no human utility, latent values or broad architecture claim.',
            'Profiles are supplied observations; language-model generation and native human adapter are not tested.',
            'Descriptive fixed-sample intervals, not a family discovery or confirmation; downstream module admission is separate.'])
    write(output/'INDEPENDENT_REVIEW.json',result)
    write(output/'GHOST_WITNESS.json',dict(scope='ghost',status=status,constructed_world=True,
        constructed_software_fixture=False,tested_comparison='S15-PROFILE-FEATURE-V2',configuration=study.configuration(),
        review_sha256=digest(output/'INDEPENDENT_REVIEW.json'),parts=[dict(packet=p['packet'],complete_sha256=p['complete_sha256'],replay_sha256=p['replay_sha256'],plan_sha256=p['plan_sha256']) for p in parts],
        outcomes=outcomes,controls=controls,limitations=result['limitations']))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.process_time()
    try:print(review(a.campaign,a.output))
    finally:write(a.campaign/'attempts'/('s15-successor-review-'+str(time.time_ns())+'.json'),dict(cpu_seconds=time.process_time()-start))
