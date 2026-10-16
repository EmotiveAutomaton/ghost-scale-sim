"""Render R2's retained discovery curve and reader/secondary tables."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import csv
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime


def build(root,campaign):
    started=time.process_time();runtime.below_normal();root=Path(root);campaign=Path(campaign)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curve=read(root/'CURVE.json');diagnostic=read(root/'DIAGNOSTICS.json');review=read(root/'REVIEW.json')
        plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
        fig,(ax,settings)=plt.subplots(2,1,figsize=(12,7),height_ratios=[3,1.4],sharex=True,layout='constrained')
        x=np.arange(len(curve));mean=np.array([r['raw_mean'] for r in curve])
        ax.errorbar(x,mean,yerr=[mean-np.array([r['lower'] for r in curve]),np.array([r['upper'] for r in curve])-mean],
            fmt='o',color='#215f82',markersize=5,lw=1,capsize=3,label='Raw pooled cost and descriptive 95% interval')
        ax.scatter(x,[r['shrunk_mean'] for r in curve],marker='D',s=24,color='#b45519',label='Shrunk estimate',zorder=4)
        ax.axhline(.02,color='#8d3b43',ls='--',lw=1,label='Useful margin: 0.02 nats');ax.axhline(0,color='gray',lw=.7)
        ax.set_ylabel('Purpose-recognition log-loss cost (nats)');ax.set_title('R2 discovery: withdrawing all purpose evidence\nAll 16 settings, ordered by raw cost; nomination, not confirmation',loc='left',fontweight='bold')
        ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.2)
        for row,key in enumerate(('coupled','prefix','benign_share')):
            vals=[r['variant']['settings'][key] for r in curve];levels=sorted(set(vals))
            for i,value in enumerate(vals):
                level=levels.index(value)
                settings.scatter(i,row,c=[level],vmin=0,vmax=max(1,len(levels)-1),cmap='Blues',s=400,marker='s',edgecolor='white')
                label=('yes' if value else 'no') if key=='coupled' else f'{value:g}'
                settings.text(i,row,label,ha='center',va='center',fontsize=8,color='white' if level>=len(levels)/2 else '#142b3b')
        settings.set_yticks([0,1,2],['Goal-bearing','Steps observed','Benign share'])
        settings.set_xticks(x,[r['variant']['id'].split('-')[1] for r in curve],rotation=90,fontsize=8)
        settings.set_xlabel('Variant suffix (full identity: R2-xxxx)');settings.set_ylim(-.5,2.5)
        settings.spines[['top','right']].set_visible(False)
        fig.savefig(root/'SPECIFICATION_CURVE.svg');fig.savefig(root/'SPECIFICATION_CURVE.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,coupled in zip(axes,(False,True)):
            table=np.array([[next(r['raw_mean'] for r in curve if r['variant']['settings']==dict(coupled=coupled,prefix=p,benign_share=b)) for p in (4,8,16,32)] for b in (.5,.8)])
            im=ax.imshow(table,vmin=0,vmax=.18,cmap='YlGnBu',aspect='auto')
            for row in range(2):
                for col in range(4):ax.text(col,row,f'{table[row,col]:.3f}',ha='center',va='center',color='white' if table[row,col]>.1 else '#142b3b')
            ax.set_title('Goal-bearing procedures' if coupled else 'Separable procedures');ax.set_xticks(range(4),[4,8,16,32]);ax.set_yticks(range(2),[.5,.8]);ax.set_xlabel('Steps observed')
        axes[0].set_ylabel('Benign share');fig.colorbar(im,ax=axes,label='Recognition cost (nats)',shrink=.8)
        fig.savefig(root/'HEATMAP.svg');plt.close(fig)
        with (root/'READER_STRATUM_SCORES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['variant','group','comparison','worlds','mean','lower_95','upper_95'])
            for identity,row in diagnostic.items():
                for group in ('readers','strata','contrasts'):
                    for name,value in row[group].items():w.writerow([identity,group,name,value['worlds'],value['mean'],value['lower'],value['upper']])
        secondary={}
        for receipt in review['receipts']:
            packet=campaign/'packets'/receipt['packet']
            secondary[packet.name]={r['variant']['id']:r['secondary'] for r in read(packet/'original/SUMMARY.json')['variants']}
        write(root/'SECONDARY_DIAGNOSTICS.json',secondary)
        with (root/'SECONDARY_SCORES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['packet','variant','reader_target_metric','worlds','mean','lower_95','upper_95'])
            for packet,variants in secondary.items():
                for identity,row in variants.items():
                    for name,value in row.items():w.writerow([packet,identity,name,value['worlds'],value['mean'],value['lower'],value['upper']])
    finally:write(campaign/'attempts'/('render-'+root.name+'.json'),dict(cpu_seconds=time.process_time()-started,state='complete'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--campaign',required=True,type=Path)
    a=p.parse_args();build(a.root,a.campaign)
