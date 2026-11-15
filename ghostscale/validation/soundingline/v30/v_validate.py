"""Retain and independently reproduce X3's two headline cells before V admission."""
from pathlib import Path
import numpy as np
from ..v29 import x3
from . import v
from .common import read,write,digest,run_cell
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    source=Path(s["headline_summary"])
    if digest(source)!=s["summary_sha256"]:raise ValueError("published headline changed")
    old=read(source);checks=v.controls()["checks"];headlines=[]
    for truth in ([0,20,0],[1,2,0]):
        setting=dict(truth=truth,tail=.1,loss=10,environment="plenty",regime="PROBE",steps=1024,worlds=200,blocks=8)
        # Inner generator only; never invoke a closed version's run().
        raw=x3.generate(setting,"study");a=x3.infer(raw,setting);b=x3.infer(raw,setting,True)
        np.savez_compressed(dest/("old-"+str(truth[0])+"_points.npz"),**raw)
        expected=next(row for row in old["rows"] if row["setting"]==setting and
                      row["measure"]=="entrusted_share" and row["step"]==1024)
        estimate=float(a["trust_curve"][:,-1].mean())
        checks.append(dict(name="published X3 exact headline "+str(truth),passed=abs(estimate-expected["estimate"])<1e-12))
        checks.append(dict(name="independent X3 whole record "+str(truth),passed=all(
            np.allclose(a[k],b[k],atol=1e-9,rtol=1e-10,equal_nan=True) for k in a)))
        headlines.append(dict(setting=setting,estimate=estimate,expected=expected))
        pulse(headline=truth)
    costs=[]
    import time
    for arm in ("STATE","INSIDE"):
        setting=dict(arm=arm,truth=[7,8,0],tail=.1,loss=10,environment="scarce",regime="MIXED",
                     steps=1024,worlds=8,blocks=8,particles=128)
        start=time.process_time();root=dest/arm
        for mode in ("original","replay","reconstruct"):
            run_cell(v,setting,root/mode,mode="reconstruct" if mode=="reconstruct" else "original",
                     original=root/"original",lane="excluded-V-native",pulse=pulse)
        equivalent(root/"original",root/"replay",True);equivalent(root/"original",root/"reconstruct")
        costs.append(dict(arm=arm,cpu_seconds=time.process_time()-start))
    return dict(passed=all(x["passed"] for x in checks),checks=checks,headlines=headlines,costs=costs,
        scientific_draws=0)
