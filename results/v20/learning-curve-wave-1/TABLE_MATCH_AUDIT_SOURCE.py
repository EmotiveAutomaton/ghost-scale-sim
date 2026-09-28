"""Reconstruct the saved exact-match frequency diagnostic without fitting.
Raw arrays are retained locally with immutable COMPLETE hashes. The checker uses
the declared repository environment. --expected is TABLE_MATCH_AUDIT.json.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json
import numpy as np
from pathlib import Path
from ghostscale.validation.soundingline.v20.analysis import interval
p=argparse.ArgumentParser();p.add_argument('--packets',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);args=p.parse_args()
expected=json.loads(args.expected.read_bytes())
identities={(r['tier'],r['history'],r['train_cases']):r['packet'] for r in expected['conditions']}
audits=[]
for tier,h in [('complete',1),('artifact',4)]:
 for n in (128,512,2048,8192,32768):
  root=args.packets/identities[tier,h,n];None
  with np.load(root/'training/examples_points.npz') as z:x=z['evidence'];y=z['labels']
  counts={}
  for row,label in zip(x,y):
   k=row.tobytes();counts.setdefault(k,np.zeros(128))[label]+=1
  prior_p=np.bincount(y,minlength=128)+.5;prior_p/=prior_p.sum()
  per_world=[]
  for l in range(24,56):
   with np.load(root/'reader'/f'{l}_points.npz') as z:xx=z['evidence']
   matched=np.array([r.tobytes() in counts for r in xx])
   with np.load(root/'raw'/f'{l}-direct-table_points.npz') as z:table=z['prediction'];truth=z['truth'];tl=z['log_loss']
   with np.load(root/'raw'/f'{l}-prior_points.npz') as z:prior_forecast=z['prediction'];pl=z['log_loss']
   with np.load(root/'raw'/f'{l}-oracle_points.npz') as z:support=z['prediction']>0
   raw=np.array([counts.get(r.tobytes(),np.zeros(128))+.5*prior_p for r in xx])*support
   raw/=raw.sum(1,keepdims=True)
   assert np.allclose(raw,table,atol=2e-14,rtol=2e-14)
   assert np.allclose(table[~matched],prior_forecast[~matched],atol=2e-14,rtol=2e-14)
   delta=tl-pl
   per_world.append(dict(lineage=l,matched_fraction=float(matched.mean()),total_excess_loss=float(delta.mean()),matched_contribution=float((delta*matched).mean()),unmatched_contribution=float((delta*~matched).mean())))
  audits.append(dict(packet=root.name,tier=tier,history=h,train_cases=n,matched_fraction=interval([r['matched_fraction'] for r in per_world]),table_minus_prior=interval([r['total_excess_loss'] for r in per_world]),matched_contribution=interval([r['matched_contribution'] for r in per_world]),unmatched_contribution=interval([r['unmatched_contribution'] for r in per_world]),lineages=per_world,reconstructed_forecasts=True))

assert audits==expected['conditions']
print(json.dumps(dict(passed=True,conditions=len(audits))))
