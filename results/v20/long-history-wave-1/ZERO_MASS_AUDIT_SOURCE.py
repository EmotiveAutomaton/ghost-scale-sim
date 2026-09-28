"""Reconstruct the retained exact-zero diagnosis without fitting any model.
Run from the repository environment with --packets, --aggregates and --expected.
The raw arrays remain locally retained under their immutable completion hashes.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--packets',type=Path,required=True)
parser.add_argument('--aggregates',type=Path,required=True)
parser.add_argument('--expected',type=Path,required=True)
args=parser.parse_args()
def read(p): return json.loads(p.read_bytes())
start=time.process_time();records=[]
for arm in read(args.aggregates)['arms']:
   if arm['method']!='structured-joint' or not arm['exact_zero_mass_fraction']:continue
   root=args.packets/arm['packet']
   with np.load(root/'training/examples_points.npz') as z:x=z['evidence'];y=z['labels']
   prior=np.bincount(y,minlength=128)+.5;prior/=prior.sum()
   cond=np.ones((128,x.shape[1],3))*.5
   for j in range(x.shape[1]):np.add.at(cond[:,j,:],(y,x[:,j]+1),1)
   cond/=cond.sum(-1,keepdims=True);logs=np.log(cond)
   assert np.isfinite(logs).all() and (prior>0).all()
   del x,y,cond
   gaps=[];count=0;outside=0
   for lineage in range(24,56):
    if time.process_time()-start>180:raise TimeoutError('bounded underflow audit')
    with np.load(root/'raw'/f'{lineage}-structured-joint_points.npz') as z:
     p=z['prediction'];truth=z['truth'];zero=p[np.arange(len(p)),truth]==0;mode=p.argmax(1)
    with np.load(root/'raw'/f'{lineage}-oracle_points.npz') as z:
     outside+=int(np.sum(z['prediction'][np.arange(len(p)),truth][zero]==0))
    with np.load(root/'reader'/f'{lineage}_points.npz') as z:xx=z['evidence'][zero]
    yy=truth[zero];mm=mode[zero];delta=np.log(prior[yy])-np.log(prior[mm])
    for j in range(xx.shape[1]):delta+=logs[yy,j,xx[:,j]+1]-logs[mm,j,xx[:,j]+1]
    gaps.extend(delta.tolist());count+=int(zero.sum())
   assert outside==0 and count>0 and np.isfinite(gaps).all()
   # If even the truth-to-largest-class odds round to zero, normalized
   # probability must also round to zero. All conditionals have positive mass.
   assert (np.exp(np.asarray(gaps))==0).all()
   records.append(dict(packet=root.name,training_labels=arm['train_cases'],zero_truth_rows=count,total_rows=65536,zero_truth_outside_common_support=outside,minimum_true_minus_modal_log_weight=min(gaps),maximum_true_minus_modal_log_weight=max(gaps),finite_positive_smoothed_likelihoods=True,all_zero_rows_have_underflowing_truth_to_mode_odds=True))

assert records==read(args.expected)['records']
print(json.dumps(dict(passed=True,records=records)))
