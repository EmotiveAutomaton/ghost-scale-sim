"""Bounded diagnostics on retained failed cells; no new worlds or acceptance changes."""
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import rankdata
from .common import read,digest,write
from . import r

def validate(s,dest,pulse):
    reports=[]
    for case in s["cases"]:
        job=Path(case["job"]);lane=case["lane"];root=job/"output"
        aroot=root/(lane+"-original");broot=root/(lane+"-reconstruct")
        for name,sha in case["bindings"].items():
            if digest(job/name)!=sha:raise ValueError("retained input changed")
        with np.load(aroot/"cell_measures.npz",allow_pickle=False) as za,np.load(broot/"cell_measures.npz",allow_pickle=False) as zb:
            a={k:za[k] for k in za.files};b={k:zb[k] for k in zb.files}
        differences={}
        for k in a:
            mask=~np.isclose(a[k],b[k],rtol=1e-10,atol=1e-8,equal_nan=True)
            differences[k]=dict(max_abs=float(np.nanmax(abs(a[k]-b[k]))),outside_tolerance=int(mask.sum()),
                first_indices=np.argwhere(mask)[:12].tolist())
        report=dict(job=job.name,differences=differences)
        if case["family"]=="L":
            with np.load(aroot/"cell_points.npz",allow_pickle=False) as z:
                y=z["event"];selected=z["observed_show"]
            estimates=[]
            def auc(labels,score):
                return float((rankdata(score)[labels].sum()-labels.sum()*(labels.sum()+1)/2)/(labels.sum()*(len(labels)-labels.sum())))
            for digits in (None,10,11,12,13,14):
                aa=[];bb=[]
                for i,m in enumerate(selected):
                    labels=y[i,m]
                    if not 0<labels.sum()<len(labels):continue
                    x=a["_event_probability"][i,m];v=b["_event_probability"][i,m]
                    if digits is not None:x=np.round(x,digits);v=np.round(v,digits)
                    aa.append(auc(labels,x));bb.append(auc(labels,v))
                estimates.append(dict(digits=digits,max_auc_difference=float(np.max(abs(np.array(aa)-bb))),mean_original=float(np.mean(aa)),mean_reference=float(np.mean(bb))))
            report["rank_resolution_diagnostic"]=estimates
        else:
            orders=list(itertools.permutations(r.CORE))
            ix=np.argwhere(~np.isclose(a["_cost"],b["_cost"],rtol=1e-10,atol=1e-8))
            report["offending_orders"]=[dict(order=orders[int(i)],reason=int(j),original=float(a["_cost"][i,j]),reference=float(b["_cost"][i,j])) for i,j in ix[:12]]
        reports.append(report);pulse(case=job.name)
    result=dict(passed=True,purpose="diagnosis only; original failures remain failed",scientific_draws=0,reports=reports)
    write(dest/"DIAGNOSIS.json",result)
    return result
