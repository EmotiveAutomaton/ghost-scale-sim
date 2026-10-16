"""Retained-data diagnostics and curves for reviewed R4, N5 and N6 decisions.

No sampling, fitting, admission, selection changes or event acknowledgements.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import math
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v23 import family, runtime
from ghostscale.validation.soundingline.v23.common import read, write, digest, interval


def secondary_review(root, campaign):
    """Rebuild all N5/N6 secondary summaries from retained case records."""
    review = read(root/'REVIEW.json'); name = review['family']
    maximum = 0.; cases = 0; receipts = []
    for receipt in review['receipts']:
        packet = campaign/'packets'/receipt['packet']
        if digest(packet/'COMPLETE.json') != receipt['complete_sha256']:
            raise ValueError('reviewed packet changed')
        plan = read(packet/'PLAN.json'); summary = read(packet/'original/SUMMARY.json')
        for item in summary['variants']:
            v = item['variant']; folder = packet/'original'/v['id']; values = {}
            for prediction, truth, posterior in zip(family.rows(folder/'prediction_points.jsonl.gz'),
                    family.rows(folder/'truth_points.jsonl.gz'), family.rows(folder/'posterior_points.jsonl.gz'), strict=True):
                w, c = prediction['world'], prediction['case']; label = truth['truth']; detail = posterior['detail']
                if (posterior['world'], posterior['case']) != (w, c):
                    raise ValueError('secondary case alignment')
                row = {}
                if name == 'N5':
                    row['approximate:quadrature_max_difference'] = detail['quadrature_max_difference']
                    for reader, prob in prediction['probabilities'].items():
                        p = prob[0]; top = max(range(4), key=lambda k: p[k])
                        for k in range(4):
                            observed = int(label['y'][0] == k)
                            row[f'{reader}:class-{k}:bias'] = p[k]-observed
                            row[f'{reader}:class-{k}:named'] = float(top == k)
                            row[f'class-{k}:observed'] = float(observed)
                elif name == 'N6':
                    for mode in ('model', 'withdraw', 'absorb-everything', 'payoff-optimal'):
                        d = detail[mode]
                        if mode in ('model', 'withdraw'):
                            path = d['path']
                            rebuilt = dict(updates=sum(not x['stopped'] for x in path),
                                adopted_techniques=sum(x['technique_accepted'] for x in path),
                                adopted_purposes=sum(x['purpose_accepted'] or
                                    (v['settings']['coupled'] and x['technique_accepted']) for x in path))
                            rebuilt.update(unwanted_uptake=rebuilt['adopted_purposes']*(1-label['warmth']),
                                useful_technique=rebuilt['adopted_techniques']*label['competence'],
                                failed_technique=rebuilt['adopted_techniques']*(1-label['competence']))
                            rebuilt['nonprediction_payoff'] = (rebuilt['useful_technique']-rebuilt['unwanted_uptake']-
                                rebuilt['failed_technique']-.001*rebuilt['updates'])
                            if any(abs(d[k]-val)>1e-12 for k,val in rebuilt.items()):
                                raise ValueError('uptake path arithmetic changed')
                        for key in ('updates', 'adopted_purposes', 'adopted_techniques', 'unwanted_uptake',
                                'useful_technique', 'failed_technique', 'nonprediction_payoff'):
                            row[mode+':'+key] = float(d[key])
                        reader = 'model-and-gate' if mode == 'model' else mode
                        weight = v['settings']['stakes'] if label['warmth']==label['competence']==0 else 1.
                        loss = -math.log(prediction['probabilities'][reader][0][label['y'][0]])
                        row[mode+':total_payoff'] = d['nonprediction_payoff']-weight*loss
                    row['withdraw:false_withdrawal'] = float(any(x['stopped'] for x in detail['withdraw']['path'])
                        and (label['warmth'] or label['competence']))
                else:
                    raise ValueError('secondary reviewer is for N5/N6')
                if set(row) != set(item['secondary']):
                    raise ValueError('secondary field coverage')
                for key,val in row.items():
                    values.setdefault(key, {}).setdefault(w, []).append(val)
                cases += 1
            for key, worlds in values.items():
                if set(worlds) != set(range(plan['worlds'])) or any(len(x)!=plan['cases'] for x in worlds.values()):
                    raise ValueError('secondary world coverage')
                means = [math.fsum(worlds[w])/len(worlds[w]) for w in range(plan['worlds'])]
                rebuilt = interval(means, [v['id'],plan['batch'],'secondary',key])
                for field in ('mean','se','lower','upper'):
                    error = abs(rebuilt[field]-item['secondary'][key][field]); maximum = max(maximum,error)
                    if error > 1e-10:
                        raise ValueError('secondary interval differs: '+key)
        receipts.append(dict(packet=packet.name, complete_sha256=receipt['complete_sha256']))
        print('secondary verified', packet.name, flush=True)
    write(root/'SECONDARY_REVIEW.json', dict(state='passed', cases=cases, maximum_error=maximum,
        receipts=receipts, reviewer_sha256=digest(__file__),
        scope='All secondary raw-case aggregates and intervals; uptake path arithmetic. Original independent posterior/Bellman checks remain bound by full replay.'))


def describe(root):
    review = read(root/'REVIEW.json'); curve = read(root/'CURVE.json'); diagnostics = read(root/'DIAGNOSTICS.json')
    name = review['family']
    def avg(vid, key, section='secondary'):
        vals = [r['summary'][section][key]['mean'] for r in diagnostics[vid]]
        return math.fsum(vals)/len(vals)
    result = dict(family=name, decision=review['decision'], cases=review['cases'], variants=review['variants'],
        nominated=review['nominated'], confirmation=False)
    if name in ('R4','N5'):
        result['nominated_curve'] = next(r for r in curve if r['variant']['id']==review['nominated'])
        result['structural_zero_count'] = sum(r['structural_zero'] for r in curve)
    if name == 'R4':
        strong = [r for r in curve if r['variant']['settings']['cue_reliability']>=.9 and r['variant']['settings']['coverage']<=.25]
        result['strong_region'] = dict(settings=len(strong), mean_above_margin=sum(r['raw_mean']>.02 for r in strong), lower_above_margin=sum(r['lower']>.02 for r in strong))
        misleading = []; subtree=[]; full_coverage=[]
        def loss(vid,reader):
            return math.fsum(x['summary']['readers'][reader]['log_loss']['mean'] for x in diagnostics[vid])/len(diagnostics[vid])
        for r in curve:
            v=r['variant']; s=v['settings']
            if s['misleading']:
                roster=list(diagnostics[v['id']][0]['summary']['readers'])
                mixture=next(k for k in roster if 'marginal' in k and k!='law-marginal')
                fixed=loss(v['id'],'fixed-trust')-loss(v['id'],'law-marginal')
                mix=loss(v['id'],mixture)-loss(v['id'],'law-marginal')
                misleading.append(dict(variant=v['id'],fixed_excess=fixed,mixture_excess=mix))
            if s['cue_position']=='function' and s['cue_reliability']>.5:
                other=next(x['variant']['id'] for x in curve if x['variant']['settings']==dict(s,cue_position='root'))
                a=avg(v['id'],'law-marginal:function-subtree:log_loss')
                b=avg(other,'law-marginal:function-subtree:log_loss')
                subtree.append(dict(function_variant=v['id'],root_variant=other,function_loss=a,root_loss=b))
            if s['branching']==3 and s['coverage']==1 and s['cue_reliability']>=.9:
                baseline=next(x for x in curve if x['variant']['settings']==dict(s,coverage=0))
                full_coverage.append(dict(variant=v['id'],full_gain=r['raw_mean'],no_leaves_gain=baseline['raw_mean']))
        result.update(misleading=misleading,subtree=subtree,full_coverage=full_coverage)
    if name == 'N5':
        rows=[]
        for r in curve:
            v=r['variant']; vid=v['id']; s=v['settings']
            loss={reader:math.fsum(x['summary']['readers'][reader]['log_loss']['mean'] for x in diagnostics[vid])/len(diagnostics[vid])
                  for reader in diagnostics[vid][0]['summary']['readers']}
            gap=loss['projection']-loss['base-rates']; recovered=loss['projection']-loss['calibrated-projection']
            observed=[avg(vid,f'class-{k}:observed') for k in range(4)]
            named=[avg(vid,f'projection:class-{k}:named') for k in range(4)]
            rows.append(dict(variant=v,loss=loss,base_rate_gap=gap,recovery_fraction=recovered/gap if gap>0 else None,
                never_named_classes=[k for k in range(4) if observed[k]>=.2 and named[k]==0],
                projection_bias=[avg(vid,f'projection:class-{k}:bias') for k in range(4)],
                quadrature_mean_difference=avg(vid,'approximate:quadrature_max_difference')))
        result['settings']=rows
        positive=[r for r in rows if r['variant']['settings']['offset']>=1 and r['variant']['settings']['records']==0]
        result['no_history_large_offset']=dict(settings=len(positive),gap_above_margin=sum(r['base_rate_gap']>.02 for r in positive),
            has_never_named_class=sum(bool(r['never_named_classes']) for r in positive),
            recovers_75_percent=sum(r['recovery_fraction'] is not None and r['recovery_fraction']>=.75 for r in positive))
    if name == 'N6':
        unavailable=[]; rows=[]
        for vid, batches in diagnostics.items():
            item=batches[0]['summary']
            if not item['primary']['finite']:
                unavailable.append(dict(variant=item['variant'],empty_worlds=[i for i,n in enumerate(item['primary_counts']) if n==0],
                    counts=item['primary_counts'],unavailable_predictions=item['unavailable_predictions']))
            rows.append(dict(variant=item['variant'],primary=item['primary'],secondary=item['secondary'],strata=item['strata']))
        result.update(unavailable=unavailable, settings=rows,
            limitation='No finite family selection. Available cells remain diagnostic; no deletion of empty target worlds.')
    write(root/'REPORT_DATA.json',result)
    return result


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    review=read(root/'REVIEW.json'); curve=read(root/'CURVE.json'); name=review['family']
    plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    with (root/'SPECIFICATION_CURVE.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['variant','mean','lower_95','upper_95','status'])
        for i,r in enumerate(curve):
            estimate=r if 'raw_mean' in r else r['estimate']; mean=estimate.get('raw_mean',estimate.get('mean'))
            if mean is None:
                writer.writerow([r['variant']['id'],'','','','UNAVAILABLE']);continue
            lower,upper=estimate['lower'],estimate['upper']
            writer.writerow([r['variant']['id'],mean,lower,upper,'structural zero' if r.get('structural_zero') else 'finite'])
            ax.errorbar(i,mean,yerr=[[mean-lower],[upper-mean]],fmt='o',markersize=3,capsize=1,
                color='#d65f37' if r['variant']['id']==review['nominated'] else '#216b85')
    ax.axhline(.02,color='gray',ls='--',label='Declared 0.02-nat margin');ax.axhline(0,color='gray',lw=.6)
    ax.set_xlabel('Declared setting (see CSV for identity)');ax.set_ylabel('Primary log-loss gain (nats)')
    ax.set_title(f'{name}: {review["decision"]}; {review["variants"]} settings\nDescriptive intervals; no confirmation'+
        ('; two unavailable settings retained in CSV' if name=='N6' else ''))
    ax.legend();fig.savefig(root/'SPECIFICATION_CURVE.svg');fig.savefig(root/'SPECIFICATION_CURVE.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--secondary',action='store_true')
    args=parser.parse_args();runtime.below_normal();started=time.process_time()
    try:
        if args.secondary:secondary_review(args.root,args.campaign)
        describe(args.root);render(args.root)
    finally:
        write(args.campaign/'attempts'/('decision-report-'+args.root.name+'.json'),dict(cpu_seconds=time.process_time()-started,
            state='complete' if (args.root/'SPECIFICATION_CURVE.png').exists() else 'failed'))
