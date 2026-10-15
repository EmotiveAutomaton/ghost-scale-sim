"""Independent aggregate audit; never reruns or changes the calibration choice."""
import argparse,math,time
from pathlib import Path
from statistics import NormalDist
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23.family import verify_packet


def wilson(success,trials):
    if not trials:return dict(rate=None,lower=None,upper=None,trials=0)
    z=NormalDist().inv_cdf(.975);z2=z*z
    centre=(success+z2/2)/(trials+z2)
    spread=z*math.sqrt(success*(trials-success)/trials+z2/4)/(trials+z2)
    return dict(rate=success/trials,lower=max(0,centre-spread),upper=min(1,centre+spread),trials=trials)


def check(row):
    keys=('strong','killed_strong','null','nominated_null')
    totals={k:sum(cell['counts'][k] for cell in row['cells']) for k in keys}
    if any(row['counts'][k]!=totals[k] for k in keys):raise ValueError('calibration count mismatch')
    for cell in row['cells']:
        counts=cell['counts']
        if not 0<=counts['killed_strong']<=counts['strong']<=row['repetitions_per_condition']:raise ValueError('kill count invalid')
        if not 0<=counts['nominated_null']<=counts['null']<=row['repetitions_per_condition']:raise ValueError('nomination count invalid')
    values=dict(false_kill=wilson(totals['killed_strong'],totals['strong']),
                false_nomination=wilson(totals['nominated_null'],totals['null']))
    for name,expected in values.items():
        for key,value in expected.items():
            actual=row[name][key]
            if value is None:
                if actual is not None:raise ValueError('empty denominator misreported')
            elif not math.isclose(value,actual,rel_tol=0,abs_tol=1e-12):raise ValueError('calibration interval differs')
    qualifies=values['false_kill']['trials']>0 and values['false_nomination']['trials']>0 and values['false_kill']['upper']<=.1 and values['false_nomination']['upper']<=.2
    return dict(counts=totals,qualifies=qualifies,**values)


def review(root,campaign):
    root,campaign=Path(root).resolve(),Path(campaign).resolve()
    started=time.process_time();plan,summary=verify_packet(root)
    design=read(root/'original/DESIGN.json');tuning=[]
    for index in range(9):
        row=read(root/f'original/tuning-{index}.json');checked=check(row)
        if row['repetitions_per_condition']!=design['tuning_repetitions'] or len(row['cells'])!=180:raise ValueError('calibration grid incomplete')
        if checked['qualifies']:tuning.append(row)
    accepted=False;validation=None
    if tuning:
        chosen=min(tuning,key=lambda x:(x['false_kill']['upper'],x['false_nomination']['upper']))
        if chosen!=read(root/'original/CHOSEN_BEFORE_VALIDATION.json'):raise ValueError('tuning selection changed')
        row=read(root/'original/VALIDATION.json');validation=check(row)
        if row['namespace']!='heldout-validation' or row['repetitions_per_condition']!=128:raise ValueError('held-out protocol changed')
        if row['prior']!=chosen['prior'] or row['kill_threshold']!=chosen['kill_threshold']:raise ValueError('validation policy changed')
        accepted=validation['qualifies']
    if (summary['state']=='CALIBRATION_PASSED')!=accepted:raise ValueError('summary decision differs')
    result=dict(state='passed',calibration_accepted=accepted,summary=summary,validation=validation,
        complete_sha256=digest(root/'COMPLETE.json'),original_failed_attempts_retained=True,
        scope='pooled declared grid; per-cell stress outcomes retained; no human-data or Sounding Line rule guarantee')
    write(root/'NUMERICAL_REVIEW.json',result)
    if accepted:
        write(campaign/'CALIBRATION_ACCEPTANCE.json',dict(acceptance_sha256=digest(campaign/'ACCEPTANCE.json'),
            summary=str(root/'original/SUMMARY.json'),summary_sha256=digest(root/'original/SUMMARY.json'),
            complete=str(root/'COMPLETE.json'),complete_sha256=digest(root/'COMPLETE.json'),
            numerical_review_sha256=digest(root/'NUMERICAL_REVIEW.json'),
            adaptive_source_sha256=plan['sources']['ghostscale/validation/soundingline/v23/adaptive.py']))
    write(campaign/'attempts/calibration-review.json',dict(cpu_seconds=time.process_time()-started,state='review_complete'))
    return result


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args();print(json.dumps(review(args.root,args.campaign),indent=2))
