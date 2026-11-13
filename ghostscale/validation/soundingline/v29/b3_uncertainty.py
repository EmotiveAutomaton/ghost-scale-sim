"""B3-H1 retained-data uncertainty repair: whole-block resampling with refitting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import rng,read,write,digest,audit_controls
from ..v23 import runtime


def score_batch(train,test,length,mu,reference=False):
    high=(train+.5)/(length+1)
    if mu==1:return np.zeros(len(train))
    logits=np.log(high)-np.log1p(-high);center=logits.mean(1,keepdims=True);x=logits-center
    if reference:
        cov=np.einsum('bni,bnj->bij',x,x);_,v=np.linalg.eigh(cov);basis=v[:,:,-2:];lowlogit=center+(x@basis)@basis.transpose(0,2,1)
        low=expit(lowlogit);ls=(test*np.log(low)+(length-test)*np.log1p(-low)).sum((1,2))/(length*train.shape[1]*train.shape[2]);hs=(test*np.log(high)+(length-test)*np.log1p(-high)).sum((1,2))/(length*train.shape[1]*train.shape[2])
    else:
        u,values,v=np.linalg.svd(x,full_matrices=False);lowlogit=center+(u[:,:,:2]*values[:,None,:2])@v[:,:2,:]
        ls=(test*(-np.logaddexp(0,-lowlogit))+(length-test)*(-np.logaddexp(0,lowlogit))).mean((1,2))/length
        hs=(test*(-np.logaddexp(0,-logits))+(length-test)*(-np.logaddexp(0,logits))).mean((1,2))/length
    return hs-ls


def controls():
    r=rng('B3-H1-controls');train=r.integers(17,size=(4,16,3)).astype(float);test=r.integers(17,size=(4,16,3)).astype(float)
    a=score_batch(train,test,16,0.);b=score_batch(train,test,16,0.,True)
    constant=np.full((4,16,3),8.)
    return audit_controls([dict(name='independent refitted predictive statistic',passed=bool(np.allclose(a,b,rtol=0,atol=1e-10))),dict(name='rank-one design has exactly zero extra gain',passed=bool(np.all(score_batch(train,test,16,1.)==0))),dict(name='constant known law has zero extra gain',passed=bool(np.all(score_batch(constant,constant,16,0.)==0))),dict(name='batched refits equal separate refits',passed=bool(np.allclose(a,np.array([score_batch(train[i:i+1],test[i:i+1],16,0.)[0] for i in range(4)]),atol=1e-12,rtol=0)))])


def run(original,destination,reference=False,cpu_limit=900):
    runtime.below_normal();start=time.process_time();original=Path(original);dest=Path(destination);dest.mkdir(exist_ok=False);gate=controls();write(dest/'CONTROLS.json',gate)
    if not gate['passed']:raise ArithmeticError('B3-H1 controls failed')
    summary=read(original/'SUMMARY.json');manifest=read(original/'RAW_MANIFEST.json');bootstrap={};corrections=[];maximum=0.
    for i,s in enumerate(summary['settings']):
        if s['part']!='B3b':continue
        path=original/f'cell-{i:03d}_points.npz'
        if digest(path)!=manifest[path.name]:raise ValueError('raw B3 input changed')
        with np.load(path,allow_pickle=False) as z:data=z['decisions']
        n,_,decisions=data.shape;blocksize=n//s['blocks'];matched=[row for row in summary['rows'] if row['setting']==s and row['measure']=='heldout_gain_per_decision'];first=None
        for row in matched:
            k=row['training_records']
            if reference:
                train=np.stack([np.add.reduce(data[w,:k],axis=0,dtype=np.int64) for w in range(n)]).astype(float);test=np.stack([np.add.reduce(data[w,k:2*k],axis=0,dtype=np.int64) for w in range(n)]).astype(float)
            else:train=data[:,:k].sum(1).astype(float);test=data[:,k:2*k].sum(1).astype(float)
            point=float(score_batch(train[None],test[None],k,s['mu'],reference)[0]);error=abs(point-row['estimate']);maximum=max(maximum,error)
            if error>1e-10:raise ArithmeticError('original B3 point estimate not reproduced')
            draws=rng('B3-H1',i,k).integers(s['blocks'],size=(4000,s['blocks']));sample=np.empty(4000)
            for begin in range(0,4000,100):
                index=(draws[begin:begin+100,:,None]*blocksize+np.arange(blocksize)[None,None,:]).reshape(-1,n)
                sample[begin:begin+len(index)]=score_batch(train[index],test[index],k,s['mu'],reference)
                if time.process_time()-start>cpu_limit:raise TimeoutError('B3-H1 CPU limit')
            interval=np.quantile(sample,[.025,.975]).tolist();bootstrap[f'cell_{i}_prefix_{k}']=sample
            corrections.append(dict(setting=s,training_records=k,estimate=point,original_interval95=row['interval95'],corrected_interval95=interval))
            row.update(interval95=interval,standard_error=float(sample.std(ddof=1)),interval_method='4000 whole-block bootstrap replicates, refitting both readers on every resampled training set; held-out records stay attached to their worlds',original_block_means_are_diagnostic=True)
            if first is None and interval[0]>0:first=k
            write(dest/'PROGRESS.json',dict(cell=i,training_records=k,cpu_seconds=time.process_time()-start),immutable=False)
        for row in summary['rows']:
            if row['setting']==s and row['measure']=='first_pointwise_positive_interval':row['training_records']=first;row['meaning']='First corrected whole-block-refit interval above zero; descriptive, no multiplicity correction.'
    np.savez_compressed(dest/'bootstrap_points.npz',**bootstrap)
    summary['B3b_uncertainty_correction']='B3-H1: shared fitted loadings require refitting within each whole-block bootstrap. Original primary point estimates and B3 per-agent intervals retained.'
    write(dest/'CORRECTED_SUMMARY.json',summary);write(dest/'CORRECTIONS.json',corrections)
    write(dest/'COMPLETE.json',dict(passed=True,source_summary_sha256=digest(original/'SUMMARY.json'),source_raw_manifest_sha256=digest(original/'RAW_MANIFEST.json'),maximum_original_point_discrepancy=maximum,corrected_rows=len(corrections),cpu_seconds=time.process_time()-start,summary_sha256=digest(dest/'CORRECTED_SUMMARY.json'),bootstrap_sha256=digest(dest/'bootstrap_points.npz')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True);parser.add_argument('--destination',required=True);parser.add_argument('--reference',action='store_true');args=parser.parse_args();run(args.original,args.destination,args.reference)