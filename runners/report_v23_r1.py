"""Render the accepted R1 discovery curve and its complete diagnostic tables."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import csv
from pathlib import Path
import time
import numpy as np
from ghostscale.validation.soundingline.v23.common import read,write,digest
from ghostscale.validation.soundingline.v23 import runtime

FINDING = ('Does a steering signature help when both readers can represent steering? '
    'The repaired neutral-signature control gives zero gain in all nine settings. '
    'The strongest tested signature is nominated after three batches: its estimated '
    'content-loss reduction is 0.03127 nats, with a descriptive 95% interval of '
    '0.02291–0.03962. The broader prediction fails: four of eight predicted strong '
    'settings remain below the 0.02-nat margin in point estimate. This is a '
    'constructed-method discovery, not confirmation or evidence about people.')


def build(root,campaign):
    started=time.process_time();runtime.below_normal();root=Path(root);campaign=Path(campaign)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curve=read(root/'CURVE.json');diagnostic=read(root/'DIAGNOSTICS.json');review=read(root/'REVIEW.json')
        plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
        fig,(ax,settings)=plt.subplots(2,1,figsize=(14,7),height_ratios=[3,1.4],sharex=True,layout='constrained')
        x=np.arange(len(curve));mean=np.array([r['raw_mean'] for r in curve])
        ax.errorbar(x,mean,yerr=[mean-np.array([r['lower'] for r in curve]),np.array([r['upper'] for r in curve])-mean],
            fmt='o',color='#215f82',markersize=4,lw=1,capsize=2,label='Raw pooled gain and descriptive 95% interval')
        ax.scatter(x,[r['shrunk_mean'] for r in curve],marker='D',s=16,color='#b45519',label='Shrunk estimate',zorder=4)
        ax.axhline(.02,color='#8d3b43',ls='--',lw=1,label='Useful margin: 0.02 nats');ax.axhline(0,color='gray',lw=.7)
        ax.set_ylabel('Content log-loss reduction (nats)');ax.set_title('R1 discovery: matched signature information\nAll 36 settings, sorted by raw gain; no confirmation claim',loc='left',fontweight='bold')
        ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.2)
        for row,key in enumerate(('prevalence','strength','bits')):
            vals=[r['variant']['settings'][key] for r in curve];levels=sorted(set(vals))
            for i,value in enumerate(vals):
                settings.scatter(i,row,c=[levels.index(value)],vmin=0,vmax=max(1,len(levels)-1),cmap='Blues',s=190,marker='s',edgecolor='white')
                settings.text(i,row,f'{value:g}',ha='center',va='center',fontsize=7,color='white' if levels.index(value)>=len(levels)/2 else '#142b3b')
        settings.set_yticks([0,1,2],['Steering share','Signature reliability','Signature bits'])
        settings.set_xticks(x,[r['variant']['id'].split('-')[1] for r in curve],rotation=90,fontsize=7)
        settings.set_xlabel('Variant suffix (full identity: R1-xxxx)');settings.set_ylim(-.5,2.5)
        settings.spines[['top','right']].set_visible(False)
        fig.savefig(root/'SPECIFICATION_CURVE.svg');fig.savefig(root/'SPECIFICATION_CURVE.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,prev in zip(axes,(.05,.2,.5)):
            table=np.array([[next(r['raw_mean'] for r in curve if r['variant']['settings']==dict(prevalence=prev,strength=s,bits=b)) for b in (2,8,32)] for s in (.5,.6,.7,.9)])
            im=ax.imshow(table,vmin=-.005,vmax=.035,cmap='YlGnBu',aspect='auto')
            for row in range(4):
                for col in range(3):ax.text(col,row,f'{table[row,col]:.3f}',ha='center',va='center',color='white' if table[row,col]>.022 else '#142b3b')
            ax.set_title(f'Steering share {prev:g}');ax.set_xticks(range(3),[2,8,32]);ax.set_yticks(range(4),[.5,.6,.7,.9]);ax.set_xlabel('Signature bits')
        axes[0].set_ylabel('Signature reliability');fig.colorbar(im,ax=axes,label='Raw gain (nats)',shrink=.8)
        fig.savefig(root/'HEATMAP.svg');plt.close(fig)
        with (root/'READER_STRATUM_SCORES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['variant','group','comparison','worlds','mean','lower_95','upper_95'])
            for identity,row in diagnostic.items():
                for group in ('readers','strata','contrasts'):
                    for name,value in row[group].items():w.writerow([identity,group,name,value['worlds'],value['mean'],value['lower'],value['upper']])
        packets=[campaign/'packets'/r['packet'] for r in review['receipts']]
        secondary={p.name:{r['variant']['id']:r['secondary'] for r in read(p/'original/SUMMARY.json')['variants']} for p in packets}
        write(root/'SECONDARY_DIAGNOSTICS.json',secondary)
        with (root/'SECONDARY_SCORES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['packet','variant','reader_target_metric','worlds','mean','lower_95','upper_95'])
            for packet,variants in secondary.items():
                for identity,row in variants.items():
                    for name,value in row.items():w.writerow([packet,identity,name,value['worlds'],value['mean'],value['lower'],value['upper']])
        write(root/'FINDING.json',dict(text=FINDING))
        write(root/'ARTIFACT_INDEX.json',{p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    finally:write(campaign/'attempts'/('render-'+root.name+'.json'),dict(cpu_seconds=time.process_time()-started,state='complete'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--campaign',required=True,type=Path)
    a=p.parse_args();build(a.root,a.campaign)
