"""Bounded retained-score audit of the approved L ranking repair."""
from pathlib import Path
import numpy as np
from . import l
from .common import read,write,digest
from runners.run_v29 import compare

def validate(s,dest,pulse):
    tests=[];audits=[]
    y=np.array([True,False,True,False])
    for score,answer in (([1,0,1,0],1.),([0,1,0,1],0.),([.5]*4,.5)):
        value=l.auc(y,score)
        tests.append(dict(name="known ranking "+str(score),passed=value==answer))
    for fixture in s["fixtures"]:
        job=Path(fixture["job"]);lane=fixture["lane"];a=job/f"output/{lane}-original";b=job/f"output/{lane}-reconstruct"
        for p,h in fixture["bindings"].items():
            if digest(job/p)!=h:raise ValueError("retained source changed")
        with np.load(a/"cell_points.npz",allow_pickle=False) as z:
            labels=z["event"];mask=z["observed_show"]
        with np.load(a/"cell_measures.npz",allow_pickle=False) as z:ma={k:z[k] for k in z.files}
        with np.load(b/"cell_measures.npz",allow_pickle=False) as z:mb={k:z[k] for k in z.files}
        values=[]
        for measures in (ma,mb):
            values.append({eps:np.array([l.auc(y[m],p[m],eps) for y,p,m in zip(labels,measures["_event_probability"],mask)]) for eps in (1e-9,1e-10,1e-11)})
        primary,reference=values[0][1e-10],values[1][1e-10]
        reconstruction=float(np.nanmax(abs(primary-reference)))
        sensitivity=max(float(np.nanmax(abs(values[k][eps]-values[k][1e-10]))) for k in (0,1) for eps in (1e-9,1e-11))
        # Explicit independent positive-negative pair comparison on retained first world.
        yy=labels[0,mask[0]];pp=ma["_event_probability"][0,mask[0]]
        delta=pp[yy,None]-pp[~yy][None,:]
        brute=float(np.mean((delta>1e-10)+.5*(np.abs(delta)<=1e-10))) if delta.size else np.nan
        passed=bool(np.allclose(primary,reference,rtol=1e-10,atol=1e-8,equal_nan=True) and sensitivity<=.001 and np.isclose(primary[0],brute,atol=1e-12,equal_nan=True))
        tests.append(dict(name=job.name+"/"+lane,passed=passed))
        change=float(np.nanmax(abs(primary-ma["_auc"])))
        ma["_auc"]=primary;mb["_auc"]=reference;setting=read(a/"SUMMARY.json")["setting"]
        sa=dict(read(a/"SUMMARY.json"),rows=l.summarize(setting,ma));sb=dict(read(b/"SUMMARY.json"),rows=l.summarize(setting,mb))
        compare(sa,sb)
        write(dest/f"corrected/{job.name}/{lane}.json",sa)
        audits.append(dict(job=job.name,lane=lane,passed=passed,max_reconstruction_difference=reconstruction,
            max_resolution_change=sensitivity,max_change_from_original=change,
            original_summary_sha256=digest(a/"SUMMARY.json"),corrected_summary_sha256=digest(dest/f"corrected/{job.name}/{lane}.json")))
        pulse(job=job.name,lane=lane)
    return dict(passed=all(x["passed"] for x in tests),checks=tests,audit=audits,scientific_draws=0,
        ranking_resolution=1e-10,unchanged_likelihoods_and_separation=True)
