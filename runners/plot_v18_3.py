"""Standalone review figures from retained summaries; no live result rankings."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest
from ghostscale.validation.soundingline.v16.runtime import local_owner


def run(root,campaign):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with local_owner(campaign/'scientific-worker-owner'):
        previous=[read(p) for p in campaign.glob('attempts/*.json')];acceptance=read(campaign/'ACCEPTANCE.json')
        if any(p['state']=='running' for p in previous):raise ValueError('active science prevents figure computation')
        charged=acceptance['prior_cpu_seconds']+sum(max(p['cpu_seconds'],p.get('native_cpu_seconds',0),p.get('uncertainty_cpu_seconds',0))+p.get('child_cpu_seconds',0) for p in previous)
        if charged+time.process_time()>=acceptance['cumulative_cpu_ceiling_seconds'] or datetime.now(timezone.utc)>=datetime.fromisoformat(acceptance['report_start']):raise TimeoutError('figure admission cutoff')
        attempt=campaign/'attempts'/('figures-'+uuid.uuid4().hex+'.json')
        write(attempt,dict(packet='review-figures',state='running',cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
        try:
            out=root/'review';figures=out/'figures';figures.mkdir(exist_ok=True)
            plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
            def save(fig,name):
                fig.savefig(figures/(name+'.png'),dpi=180);fig.savefig(figures/(name+'.svg'));plt.close(fig)
            e=read(root/'E-discovery-r1/SUMMARY.json');conditions=('in-support','new-combinations','new-queries','both-new');methods=('direct','flat','split','exact')
            fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained');x=np.arange(4)
            for j,method in enumerate(methods):
                values=[e['cells'][c+'|'+method]['expected_loss']['mean'] for c in conditions]
                ax.bar(x+(j-1.5)*.18,values,.18,label=method)
            ax.set_xticks(x,['Familiar questions\nfamiliar combinations','Familiar questions\nnew combinations','New questions\nfamiliar combinations','New questions\nnew combinations']);ax.set_ylabel('Expected logarithmic loss (nats; lower is better)');ax.set_title('E: memory transfer depends on what changes');ax.legend(ncol=4,loc='upper left');save(fig,'memory-transfer')
            factor=read(root/'FACTORIAL.json');labels=[];values=[]
            for text,entry in factor['effects'].items():
                tags=json.loads(text)
                if tags['family']=='A' and tags['purpose']=='enactment':
                    labels.append('A: action advantage, '+tags['mode']);values.append([100*entry['cells'][str(c)]['mean'] for c in range(16)])
                elif tags['family']=='B' and tags['condition'] in ('goal','belief','stationary','skill','opportunity'):
                    labels.append('B: selective advantage, '+tags['condition']);values.append([-entry['cells'][str(c)]['mean'] for c in range(16)])
            values=np.asarray(values);scale=np.maximum(abs(values).max(1,keepdims=True),1e-12)
            fig,ax=plt.subplots(figsize=(13,5.8),layout='constrained');ax.imshow(values/scale,cmap='RdBu',vmin=-1,vmax=1,aspect='auto')
            for i in range(len(labels)):
                for j in range(16):ax.text(j,i,f'{values[i,j]:.1f}' if labels[i].startswith('A:') else f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j]/scale[i,0])>.55 else 'black')
            ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(16),[format(i,'04b') for i in range(16)],rotation=45);ax.set_xlabel('Cell bits: decision rule / acquisition / opportunities / source dependence');ax.set_title('Core boundaries: blue favors first method; red favors rival\nA numbers are percentage points; B numbers are loss reduction. Color is normalized within each row.');save(fig,'architecture-boundaries')
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            for ax,family,title in zip(axes,('H','F'),('Core decision laws','Held-out lexicographic law')):
                cells=read(root/(family+'_ROLLUP.json'))['cells']
                for method in ('exhaustive-flat','observation-product'):
                    rows=sorted([(json.loads(k)['cardinality'],v) for k,v in cells.items() if json.loads(k)['family']=='H' and json.loads(k)['method']==method])
                    ax.plot([np.log2(k) for k,_ in rows],[v['new_loss']['mean'] for _,v in rows],'o-',label=method)
                ax.set_title(title);ax.set_xlabel('Maximum stored bits');ax.set_xticks(range(4));ax.set_ylabel('New-question logarithmic loss');ax.legend(fontsize=8)
            fig.suptitle('Old-question optimal compression can lose on a new question');save(fig,'compression-transfer')
            costs=read(out/'NEURAL_COSTS.json');x=np.logspace(0,8,100);summary={}
            fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
            for method in ('direct','flat','split'):
                rows=[r for r in costs['rows'] if r['method']==method]
                fit=float(np.median([r['capacity_search_fit_cpu_seconds'] for r in rows]));use=float(np.median([r['full_encoding']['median_cpu_seconds']+r['cached_query_decoding']['median_cpu_seconds'] for r in rows]));cached=float(np.median([r['cached_query_decoding']['median_cpu_seconds'] for r in rows]))
                summary[method]=dict(median_capacity_search_fit_cpu_seconds=fit,median_new_history_query_seconds=use,median_cached_query_seconds=cached,
                    parameters=[r['parameters'] for r in rows],state_floats=[r['state_floats'] for r in rows])
                ax.loglog(x,fit+x*use,label=method)
            for method in ('flat','split'):
                delta=summary['direct']['median_new_history_query_seconds']-summary[method]['median_new_history_query_seconds']
                extra=summary[method]['median_capacity_search_fit_cpu_seconds']-summary['direct']['median_capacity_search_fit_cpu_seconds']
                summary[method]['fit_only_break_even_new_histories']=max(0,extra/delta) if delta>0 else None
            ax.set_xlabel('Subsequent independent 32-slot history/query uses (projection)');ax.set_ylabel('Fit CPU seconds + projected use CPU seconds');ax.set_title('Measured costs, projected reuse; accuracy is not equated\nShared setup and failed prior work are additional campaign costs');ax.legend();save(fig,'cost-reuse')
            write(out/'COST_SUMMARY.json',dict(methods=summary,scope='medians over three selected fit seeds; fit-only optimistic reuse calculation, not full research-cost repayment; batch timings extrapolate neither accuracy nor support beyond declared data'))
            write(out/'FIGURES.json',dict(source_sha256=file_digest(__file__),files={p.relative_to(out).as_posix():file_digest(p) for p in figures.iterdir() if p.is_file()},scope='static exported figures from retained aggregate records'))
            write(attempt,dict(packet='review-figures',state='complete',cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False)
        except BaseException:write(attempt,dict(packet='review-figures',state='failed',cpu_seconds=time.process_time(),child_cpu_seconds=0),immutable=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();run(a.root,a.campaign)
