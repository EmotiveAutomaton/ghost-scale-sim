"""Render the retained conditional curves; no simulation or model fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
summary=json.loads((root/'SUMMARY.json').read_bytes())
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
try:
    for ax,name in zip(axes.flat,['goal-directed','action-habit','acquired-end','habitual-goal']):
        rows=[r for r in summary['variants'] if r['variant']['settings']['mechanism']==name]
        xs=[r['variant']['settings']['history'] for r in rows]
        for metric in ('all','A','B','action-only','sham'):
            cis=[r['probe_information_gain'] if metric=='all' else r['secondary'][metric]['interval'] for r in rows]
            means=np.array([c['mean'] for c in cis]);lo=np.array([c['lower'] for c in cis]);hi=np.array([c['upper'] for c in cis])
            ax.errorbar(xs,means,yerr=np.vstack([means-lo,hi-means]),label=metric,marker='o',capsize=2)
        ax.set_title(name);ax.set_xticks([0,8,32]);ax.grid(alpha=.2)
        ax.set_xlabel('Baseline observations');ax.set_ylabel('Mechanism log-score gain (nats)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=5)
    fig.suptitle('N4c: stipulated rules; descriptive intervals across 16 law blocks',y=.995,fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(root/'PROBE_CURVES.png',dpi=150,bbox_inches='tight')
finally:plt.close(fig)
