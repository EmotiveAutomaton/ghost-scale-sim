"""Independent raw-score and chronology reconstruction for the frozen S15 study."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import math
from pathlib import Path
import statistics
import time
from scipy.stats import t
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime

ARMS=('baseline','own','donor')


def reconstruct(packet):
    plan=read(packet/'PLAN.json');complete=read(packet/'COMPLETE.json')
    if plan['task']!='profile-validation' or complete['state']!='verified':raise ValueError('complete research packet required')
    if digest(packet/'PLAN.json')!=complete['plan_sha256'] or digest(packet/'REPLAY.json')!=complete['replay_sha256']:
        raise ValueError('completion binding changed')
    replay=runtime.compare(packet/'original',packet/'replay')
    if replay!=read(packet/'REPLAY.json'):raise ValueError('replay receipt differs')
    raw=packet/'original';cfg=plan['configuration'];points=[];controls=[];cold=0;targets=0
    if cfg['worlds']!=20 or cfg['rounds']!=4 or cfg['writers']!=8 or cfg['margin']!=.005:
        raise ValueError('unexpected frozen design')
    for world in range(cfg['worlds']):
        by_condition={}
        for condition in ('planted','uninformative','constant'):
            cell=raw/f'w{world:02d}-{condition}'
            records=read(cell/'READER.json');labels=read(cell/'LABELS.json');predictions=read(cell/'PREDICTIONS.json')
            truth=read(cell/'EVALUATOR.json')
            if len(records)!=32 or len(labels)!=32 or len(predictions)!=32:raise ValueError('missing target')
            if len({r['key'] for r in records})!=32:raise ValueError('duplicate target')
            if any(w==d for w,d in enumerate(truth['donors'])):raise ValueError('donor is own writer')
            losses={a:[] for a in ARMS}
            for row,label,prediction in zip(records,labels,predictions):
                if row['key']!=label['key'] or row['key']!=prediction['key']:raise ValueError('target alignment')
                if set(row)!={'key','writer','prompt','open_at','resolved_at','profile_cutoff','features'}:
                    raise ValueError('unrecognized reader field')
                if row['profile_cutoff']>=row['open_at']:raise ValueError('future profile')
                expected=[r['key'] for r in sorted(records,key=lambda r:(r['open_at'],r['key'])) if r['resolved_at']<row['open_at']]
                for arm in ARMS:
                    output=prediction['arms'][arm];p=output['probabilities']
                    if output['training_keys']!=expected or output['cold_start']!=(not expected):raise ValueError('chronology changed')
                    if len(p)!=4 or any(not math.isfinite(x) or x<0 for x in p) or abs(sum(p)-1)>1e-10:raise ValueError('invalid forecast')
                    # Algebraically separate from production's one-hot difference.
                    losses[arm].append((sum(x*x for x in p)+1-2*p[label['action']])/2)
                cold+=int(prediction['arms']['own']['cold_start']);targets+=1
            means={a:statistics.fmean(losses[a]) for a in ARMS}
            points.append(dict(world=world,condition=condition,gain_baseline=means['baseline']-means['own'],gain_donor=means['donor']-means['own'],losses=means))
            by_condition[condition]=predictions
        original=by_condition['planted'];future=read(raw/f'w{world:02d}-future/PREDICTIONS.json');option=read(raw/f'w{world:02d}-option/PREDICTIONS.json')
        if len(future)!=32 or len(option)!=32:raise ValueError('control coverage')
        future_difference=max(abs(x-y) for a,b in zip(original[:12],future[:12]) for arm in ARMS for x,y in zip(a['arms'][arm]['probabilities'],b['arms'][arm]['probabilities']))
        permutation=[2,0,3,1]
        option_difference=max(abs(a['arms'][arm]['probabilities'][permutation[j]]-b['arms'][arm]['probabilities'][j]) for a,b in zip(original,option) for arm in ARMS for j in range(4))
        constant_difference=max(abs(x-y) for r in by_condition['constant'] for x,y in zip(r['arms']['own']['probabilities'],r['arms']['donor']['probabilities']))
        controls.append(dict(world=world,future_max_difference=future_difference,option_max_difference=option_difference,constant_max_difference=constant_difference))
    saved=read(raw/'POINTS.json')
    if len(saved)!=len(points):raise ValueError('aggregate coverage')
    for a,b in zip(points,saved):
        if (a['world'],a['condition'])!=(b['world'],b['condition']):raise ValueError('aggregate identity')
        for name in ('gain_baseline','gain_donor'):
            if abs(a[name]-b[name])>1e-12:raise ValueError('score reconstruction differs')
    declared=read(raw/'CONTROLS.json')
    if declared!=controls:raise ValueError('control reconstruction differs')
    outcomes={};decisions=[]
    for condition in ('planted','uninformative','constant'):
        outcomes[condition]={}
        for comparator in ('baseline','donor'):
            sample=[r['gain_'+comparator] for r in points if r['condition']==condition]
            mean=statistics.fmean(sample);radius=float(t.ppf(.975,19)*statistics.stdev(sample)/math.sqrt(20))
            lower,upper=mean-radius,mean+radius
            passed=lower>.005 if condition=='planted' else upper<=.005
            outcomes[condition][comparator]=dict(mean=mean,lower=lower,upper=upper,n=20,criterion_passed=passed)
            decisions.append(passed)
    structural=all(r['future_max_difference']<=1e-7 and r['option_max_difference']<=1e-7 and r['constant_max_difference']<=1e-12 for r in controls)
    status='PASS' if all(decisions) and structural else 'FAIL'
    summary=read(raw/'SUMMARY.json')
    if summary['status']!=status or summary['structural_controls_pass']!=structural:raise ValueError('disposition reconstruction differs')
    for condition,contrasts in outcomes.items():
        for comparator,result in contrasts.items():
            for field in ('mean','lower','upper'):
                if abs(result[field]-summary['results'][condition][comparator][field])>1e-12:raise ValueError('interval reconstruction differs')
    return dict(status=status,outcomes=outcomes,controls=controls,structural_controls_pass=structural,
        targets=targets,cold_start_targets=cold,raw_files=len(replay['files']),plan_sha256=digest(packet/'PLAN.json'),
        full_fresh_process_replay=True,independent_raw_score_reconstruction=True,
        score_formula='(sum of squared probabilities + 1 - twice probability of observed action) / 2',
        reviewer_sha256=digest(__file__))


def review(packet,output):
    packet=packet.resolve();result=reconstruct(packet);plan=read(packet/'PLAN.json')
    write(output/'INDEPENDENT_REVIEW.json',result)
    witness=dict(scope='ghost',status=result['status'],constructed_world=True,constructed_software_fixture=False,
        implementation_hash=plan['implementation_hash'],ledger_schema_hash=plan['ledger_schema_hash'],
        tested_comparison='S15-PROFILE-FEATURE-V1',sounding_line_plan_sha256=plan['sounding_line_plan_sha256'],
        domain='20 independently redrawn constructed profile/choice worlds, eight writers and four decisions per writer; four actions',
        configuration=plan['configuration'],source_archive_sha256=plan['source_archive_sha256'],
        raw=dict(path=str(packet/'original'),replay_path=str(packet/'replay')),
        aggregate=dict(path=str(packet/'original/SUMMARY.json'),sha256=digest(packet/'original/SUMMARY.json')),
        plan=dict(path=str(packet/'PLAN.json'),sha256=digest(packet/'PLAN.json')),
        runtime=dict(path=str(packet/'STATUS.json'),sha256=digest(packet/'STATUS.json')),
        independent_replay_receipt=dict(path=str(packet/'REPLAY.json'),sha256=digest(packet/'REPLAY.json')),
        independent_review=dict(path=str((output/'INDEPENDENT_REVIEW.json').resolve()),sha256=digest(output/'INDEPENDENT_REVIEW.json')),
        controls=result['controls'],outcomes=result['outcomes'],
        limitations=['Finite constructed method validation; no human utility, latent values or broad architecture claim.',
            'Eight constructed writers and 32 targets per condition do not reproduce CoAuthor marginals or its 31-writer/234-target support.',
            'Four decisions per writer do not exercise later personal-penalty calibration.',
            'Profiles are supplied observations; language-model profile generation and native human adapter are not tested.',
            'Descriptive fixed-sample intervals, not e-values or family confirmation; native adapter and exposed-holdout approvals remain Sounding Line responsibilities.'])
    write(output/'GHOST_WITNESS.json',witness)
    return result['status']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();runtime.below_normal();start=time.process_time()
    try:print(review(a.packet,a.output))
    finally:write(a.campaign/'attempts'/('s15-profile-review-'+str(time.time_ns())+'.json'),dict(cpu_seconds=time.process_time()-start,kind='independent-dependency-review'))
