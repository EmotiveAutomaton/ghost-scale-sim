"""Reconstruct R4 node diagnostics and render its incomplete burn-in curve.

Consumes retained data only. No simulation, fit, admission or event ACK occurs.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v23 import family,runtime
from ghostscale.validation.soundingline.v23.common import read,write,interval


def build(root,campaign):
    started=time.process_time();runtime.below_normal()
    root,campaign=Path(root),Path(campaign)
    try:
        review=read(root/'REVIEW.json');curve=read(root/'CURVE.json')
        diagnostics={};checked=0;maximum=0.;identities={}
        for receipt in review['receipts']:
            packet=campaign/'packets'/receipt['packet'];plan,summary=family.verify_packet(packet)
            for item in summary['variants']:
                v=item['variant'];b=v['settings']['branching'];folder=packet/'original'/v['id']
                levels=np.concatenate([np.full(b**d,d) for d in range(5)])
                # Node 1 is the cued function; derive its descendants independently.
                parents=np.array([-1]+[(i-1)//b for i in range(1,len(levels))])
                subtree=np.zeros(len(levels),bool);subtree[1]=True
                for i in range(2,len(levels)):subtree[i]=subtree[parents[i]]
                values={};zero_error=0.;neutral_error=0.
                for detail,truth,pred in zip(family.rows(folder/'posterior_points.jsonl.gz'),
                        family.rows(folder/'truth_points.jsonl.gz'),family.rows(folder/'prediction_points.jsonl.gz'),strict=True):
                    world,case=detail['world'],detail['case'];y=np.asarray(truth['truth']['nodes'])
                    if truth['identity']!=['R4',v['id'],plan['batch'],world,case] or (pred['world'],pred['case'])!=(world,case):
                        raise ValueError('node identity mismatch')
                    nodes=detail['detail']['nodes']
                    if v['settings']['coverage']==0 and v['settings']['cue_position']=='root':
                        zero_error=max(zero_error,float(np.max(np.abs(np.array(nodes['top-down'])-nodes['law-marginal']))))
                    if v['settings']['cue_reliability']==.5:
                        neutral_error=max(neutral_error,float(np.max(np.abs(np.array(nodes['bottom-up'])-nodes['law-marginal']))))
                    for name,array in nodes.items():
                        p=np.asarray(array)
                        if p.shape!=(len(y),2) or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-12,rtol=0):
                            raise ValueError('invalid retained node probabilities')
                        if not np.allclose(p[0],pred['probabilities'][name][0],atol=1e-12,rtol=0):
                            raise ValueError('root and node forecasts differ')
                        with np.errstate(divide='ignore'): loss=-np.log(p[np.arange(len(y)),y])
                        metrics=dict(log_loss=loss,half_brier=((p-np.eye(2)[y])**2).sum(1)/2,
                                     accuracy=(p.argmax(1)==y))
                        for label,mask in [(f'depth-{d}',levels==d) for d in range(5)]+[('function-subtree',subtree),('outside-subtree',~subtree)]:
                            for metric,a in metrics.items():
                                values.setdefault(f'{name}:{label}:{metric}',{}).setdefault(world,[]).append(float(a[mask].mean()))
                    checked+=1
                if max(zero_error,neutral_error)>1e-12:raise ValueError('structural tree identity failed')
                rebuilt={}
                for key,worlds in values.items():
                    if set(worlds)!=set(range(plan['worlds'])) or any(len(a)!=plan['cases'] for a in worlds.values()):
                        raise ValueError('incomplete secondary world coverage')
                    means=[float(np.mean(worlds[w])) for w in range(plan['worlds'])]
                    stat=interval(means,[v['id'],plan['batch'],'secondary',key])
                    if key in item['secondary']:
                        for field in ('mean','se','lower','upper'):
                            error=abs(stat[field]-item['secondary'][key][field]);maximum=max(maximum,error)
                            if error>1e-12:raise ValueError('secondary reconstruction differs')
                    rebuilt[key]=stat
                diagnostics[v['id']]=rebuilt
                identities[v['id']]=dict(zero_leaf_root_cue_max_error=zero_error,neutral_cue_max_error=neutral_error)
        write(root/'NODE_DIAGNOSTICS.json',diagnostics)
        write(root/'NODE_REVIEW.json',dict(state='passed',cases=checked,secondary_max_error=maximum,identities=identities))
        with (root/'NODE_SCORES.csv').open('w',encoding='utf-8',newline='') as out:
            writer=csv.writer(out);writer.writerow(['variant','reader_target_metric','worlds','mean','lower_95','upper_95'])
            for vid,rows in diagnostics.items():
                for key,s in rows.items():writer.writerow([vid,key,s['worlds'],s['mean'],s['lower'],s['upper']])
        with (root/'SPECIFICATION_CURVE.csv').open('w',encoding='utf-8',newline='') as out:
            writer=csv.writer(out);writer.writerow(['variant','reliability','coverage','branching','cue_position','misleading','raw_gain','lower_95','upper_95','shrunk_gain'])
            for row in curve:
                s=row['variant']['settings'];writer.writerow([row['variant']['id'],s['cue_reliability'],s['coverage'],s['branching'],s['cue_position'],s['misleading'],row['raw_mean'],row['lower'],row['upper'],row['shrunk_mean']])
        render(root)
    finally:
        write(campaign/'attempts'/('r4-node-review-'+root.name+'.json'),dict(cpu_seconds=time.process_time()-started,
            state='complete' if (root/'SPECIFICATION_CURVE.svg').exists() else 'failed'))


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);curve=read(root/'CURVE.json')
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
    colors={.3:'#1f77b4',.5:'#ff7f0e',.7:'#2ca02c',.9:'#d62728',.99:'#9467bd'}
    for ax,(b,position) in zip(axes.flat,[(2,'root'),(2,'function'),(3,'root'),(3,'function')]):
        for rho in colors:
            rows=sorted([r for r in curve if r['variant']['settings']['branching']==b and r['variant']['settings']['cue_position']==position and r['variant']['settings']['cue_reliability']==rho],key=lambda r:r['variant']['settings']['coverage'])
            if not rows:continue
            x=[r['variant']['settings']['coverage'] for r in rows];mean=np.array([r['raw_mean'] for r in rows])
            ax.errorbar(x,mean,yerr=[mean-np.array([r['lower'] for r in rows]),np.array([r['upper'] for r in rows])-mean],marker='o',capsize=2,color=colors[rho],label=f'Cue accuracy {rho:g}')
        ax.axhline(.02,color='gray',ls='--',lw=.8);ax.set_title(f'Branching {b}; {position} cue');ax.grid(alpha=.2)
        ax.set_xlabel('Probability a leaf is observed');ax.set_ylabel('Root log-loss gain (nats)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('R4: first burn-in only; second batch resource-held\nRaw whole-world gains and descriptive 95% intervals; no nomination or confirmation',fontsize=13)
    fig.savefig(root/'SPECIFICATION_CURVE.svg');fig.savefig(root/'SPECIFICATION_CURVE.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    if args.render_only:render(args.root)
    else:build(args.root,args.campaign)
