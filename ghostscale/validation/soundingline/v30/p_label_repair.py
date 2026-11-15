"""One bounded repair of algebraically tied label ranks; retained originals stay intact."""
from pathlib import Path
import numpy as np
from .common import read,write,digest,run_cell
from . import p
from runners.run_v30 import equivalent

def rerank(raw,measures,s):
    result={k:v.copy() for k,v in measures.items()}
    named=np.argmax(np.round(measures["_estimates"],12),axis=-1)
    truth=np.argmax(np.round(raw["theta"],12),axis=1)
    cumulative=np.cumsum(raw["counts"],axis=1)
    for j,N in enumerate(p.LEVELS):
        if N>s["pages"]:break
        named[:,j,0]=np.argmax(cumulative[:,N-1,p.FEEL],axis=1)
        if s["corrector"] in ("PERFECT","HALF"):
            named[:,j,1]=np.where(np.ptp(measures["_estimates"][:,j,1],axis=1)>1e-12,named[:,j,0],0)
    result["_accuracy"]=(named==truth[:,None,None]).astype(float)
    return result

def validate(s,dest,pulse):
    checks=[];changes=[];records=[]
    for i,item in enumerate(s["retained"]):
        job=Path(item["path"])
        for rel,sha in item["hashes"].items():
            if digest(job/rel)!=sha:raise ValueError("retained evidence changed")
        setting=read(job/"PLAN.json")["setting"]
        for lane in ("pilot","study"):
            a=job/"output"/(lane+"-original");b=job/"output"/(lane+"-reconstruct")
            if not (b/"cell_measures.npz").exists():continue
            ss=dict(setting,worlds=40,blocks=8) if lane=="pilot" else setting
            with np.load(a/"cell_points.npz") as z:raw={k:z[k] for k in z.files}
            with np.load(a/"cell_measures.npz") as z:old={k:z[k] for k in z.files}
            with np.load(b/"cell_measures.npz") as z:reference={k:z[k] for k in z.files}
            aa=rerank(raw,old,ss);bb=rerank(raw,reference,ss)
            passed=np.array_equal(aa["_accuracy"],bb["_accuracy"])
            checks.append(dict(name=job.name+" "+lane+" exact symmetry rank",passed=bool(passed)))
            changed=int(np.count_nonzero(aa["_accuracy"]!=old["_accuracy"]))
            if changed:changes.append(dict(job=job.name,lane=lane,changed_world_level_readers=changed))
            write(dest/"retained"/job.name/(lane+"-SUMMARY.json"),dict(setting=ss,rows=p.summarize(ss,aa),
                 no_registered_claim=True,miniature="architecture untested",repair="exact count symmetry"))
            records.append(dict(job=job.name,lane=lane,changed=changed))
        pulse(retained_cell=i+1)
    # Fresh independent fixture of the failing grid cell, full replay and reconstruction.
    setting=s["failure_setting"]
    original=dest/"fixture-original";replay=dest/"fixture-replay";reference=dest/"fixture-reconstruct"
    run_cell(p,setting,original,lane="excluded-repair")
    run_cell(p,setting,replay,lane="excluded-repair");equivalent(original,replay,True)
    run_cell(p,setting,reference,mode="reconstruct",original=original,lane="excluded-repair");equivalent(original,reference)
    checks+=p.controls()["checks"]
    return dict(passed=all(x["passed"] for x in checks),checks=checks,changes=changes,records=records,
                scientific_draws=0,repair_number=1)
