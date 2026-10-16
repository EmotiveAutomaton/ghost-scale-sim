"""Reconstruct R2 policy/stratum/mixture diagnostics from retained trajectories.

No new worlds, inference, training, or scientific-source changes. Mixtures are
randomized policies, so expectations (not probabilities) are mixed. Empty
strata are explicitly counted and never assigned invented scores.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import csv
import itertools
import math
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v23 import family, runtime
from ghostscale.validation.soundingline.v23.common import read, write, digest, interval


def score(heads, truth):
    p = np.asarray(heads, dtype=float)
    if p.shape != (3, 2) or not np.allclose(p.sum(-1), 1, atol=1e-12, rtol=0):
        raise ValueError('invalid retained heads')
    return dict(log_loss=-np.log(p[np.arange(3), truth]),
                half_brier=((p-np.eye(2)[truth])**2).sum(-1)/2,
                accuracy=(p.argmax(-1) == truth).astype(float))


def summarize(values):
    # Fixed, paired whole-world descriptive normal intervals for the large
    # diagnostic table; primary decision intervals remain untouched.
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not len(a): return dict(worlds=0, mean=None, lower=None, upper=None)
    mean = float(a.mean()); se = float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
    return dict(worlds=len(a), mean=mean, lower=None if se is None else mean-1.96*se,
                upper=None if se is None else mean+1.96*se)


def build(campaign, output):
    started = time.process_time(); runtime.below_normal()
    campaign, output = Path(campaign), Path(output)
    result = {}; checks = 0; case_count = 0
    try:
        review = read(output/'REVIEW.json')
        for receipt in review['receipts']:
            root = campaign/'packets'/receipt['packet']; plan, summary = family.verify_packet(root)
            for entry in summary['variants']:
                identity = entry['variant']['id']; folder = root/'original'/identity
                target = result.setdefault(identity, dict(variant=entry['variant'], worlds=[], counts={}))
                stream = zip(family.rows(folder/'posterior_points.jsonl.gz'),
                             family.rows(folder/'truth_points.jsonl.gz'), strict=True)
                secondary = {}
                for world in range(plan['worlds']):
                    world_rows = {}; second_rows = {}; n = 0
                    for case, (row, hidden) in enumerate(itertools.islice(stream, plan['cases'])):
                        if (row['world'],row['case']) != (world,case) or hidden['identity'] != ['R2',identity,plan['batch'],world,case]:
                            raise ValueError('trajectory/truth identity mismatch')
                        truth = np.asarray(hidden['truth']['state'], dtype=int); detail = row['detail']
                        policies = detail['trajectories']; baseline = score(detail['heads']['no-gate'],truth)
                        for policy,path in policies.items():
                            if len(path) != entry['variant']['settings']['prefix']: raise ValueError('incomplete path')
                            scored = score(detail['heads'][policy],truth)
                            if abs(path[-1]['purpose']-detail['heads'][policy][0][1]) > 1e-12:
                                raise ValueError('trajectory/head mismatch')
                            fired = any(x['fired'] for x in path)
                            releases = sum(a['fired'] and not b['fired'] for a,b in zip(path,path[1:]))
                            metrics = {}
                            for index,label in enumerate(('purpose','value','technique')):
                                for metric,values in scored.items(): metrics[label+':'+metric] = float(values[index])
                                metrics[label+':memory_change'] = abs(path[-1]['memory'][index]-.5)
                            metrics.update(fired_any=float(fired), benign_false_alarm=float(truth[0]==0 and fired),
                                           releases=float(releases), released_any=float(releases>0),
                                           recognition_cost=float(scored['log_loss'][0]-baseline['log_loss'][0]),
                                           accuracy_cost=float(baseline['accuracy'][0]-scored['accuracy'][0]))
                            groups = ('all', 'flagged' if truth[0] else 'benign',
                                      'fired' if fired else 'not-fired',
                                      ('flagged' if truth[0] else 'benign')+('-fired' if fired else '-not-fired'))
                            for group in groups:
                                key = policy+'|'+group
                                target['counts'][key] = target['counts'].get(key,0)+1
                                for metric,value in metrics.items():world_rows.setdefault(key+'|'+metric,[]).append(value)
                            for metric,value in metrics.items():
                                if metric not in ('released_any','recognition_cost','accuracy_cost'):
                                    second_rows.setdefault(policy+':'+metric,[]).append(value)
                        n += 1
                    if n != plan['cases']:raise ValueError('incomplete world')
                    target['worlds'].append(dict(batch=plan['batch'],world=world,
                        metrics={key:float(np.mean(values)) for key,values in world_rows.items()}))
                    for key,values in second_rows.items():secondary.setdefault(key,[]).append(float(np.mean(values)))
                    case_count += n
                if next(stream,None) is not None:raise ValueError('extra rows')
                for key,values in secondary.items():
                    rebuilt = interval(values,[identity,plan['batch'],'secondary',key])
                    for field in ('mean','se','lower','upper'):
                        if not math.isclose(rebuilt[field],entry['secondary'][key][field],abs_tol=1e-12,rel_tol=0):
                            raise ValueError('secondary reconstruction mismatch '+key)
                        checks += 1
                print('diagnosed',root.name,identity,flush=True)
        tables = {}; mixtures = []
        for identity,item in result.items():
            keys=sorted({key for world in item['worlds'] for key in world['metrics']})
            tables[identity] = {key:dict(summarize([w['metrics'].get(key,np.nan) for w in item['worlds']]),
                cases=item['counts']['|'.join(key.split('|')[:2])]) for key in keys}
            policies=sorted({key.split('|')[0] for key in keys if key.startswith('withdraw-all-')})
            for policy,group,a,b in itertools.product(policies,('all','benign','flagged'),(0,.25,.5,.75,1),(0,.25,.5,.75,1)):
                def values(name,metric):return np.array([w['metrics'].get(name+'|'+group+'|'+metric,np.nan) for w in item['worlds']])
                row=dict(variant=identity,group=group,policy=policy,all_weight=a,content_weight=b)
                for metric in ('technique:accuracy','purpose:memory_change','value:memory_change','purpose:log_loss'):
                    ng=values('no-gate',metric)
                    row[metric]=summarize(a*values(policy,metric)+(1-a)*ng-b*values('withdraw-content',metric)-(1-b)*ng)
                row['matched_mean_technique']=abs(row['technique:accuracy']['mean'])<=.02
                row['identical_no_gate']=a==0 and b==0
                mixtures.append(row)
        write(output/'POLICY_WORLD_DIAGNOSTICS.json',result)
        write(output/'POLICY_STRATA.json',tables)
        write(output/'MATCHED_TECHNIQUE.json',mixtures)
        with (output/'POLICY_STRATA.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['variant','policy','stratum','metric','cases','nonempty_worlds','mean','lower_95','upper_95'])
            for identity,rows in tables.items():
                for key,row in rows.items():w.writerow([identity,*key.split('|'),row['cases'],row['worlds'],row['mean'],row['lower'],row['upper']])
        with (output/'MATCHED_TECHNIQUE.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['variant','stratum','policy','all_weight','content_weight','matched','identical_no_gate','metric','worlds','mean_all_minus_content','lower_95','upper_95'])
            for row in mixtures:
                for metric in ('technique:accuracy','purpose:memory_change','value:memory_change','purpose:log_loss'):
                    value=row[metric];w.writerow([row['variant'],row['group'],row['policy'],row['all_weight'],row['content_weight'],row['matched_mean_technique'],row['identical_no_gate'],metric,value['worlds'],value['mean'],value['lower'],value['upper']])
        write(output/'POLICY_REVIEW.json',dict(state='passed',cases=case_count,secondary_arithmetic_checks=checks,
            intervals='paired whole-world normal descriptive intervals; nonempty-stratum worlds only',
            mixture='fixed randomized policy weights 0,.25,.5,.75,1; mean technique tolerance .02; not equivalence testing',
            raw_receipts=review['receipts'],source_sha256=digest(__file__)))
    finally:
        write(campaign/'attempts'/('diagnostics-'+output.name+'.json'),dict(cpu_seconds=time.process_time()-started,
            state='complete' if (output/'POLICY_REVIEW.json').exists() else 'failed'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();build(a.campaign,a.output)
