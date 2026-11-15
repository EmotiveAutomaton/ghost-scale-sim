"""Evaluate only packet-defined follow-on triggers and freeze eligible validation plans."""
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v30.common import read,write,digest
from ghostscale.validation.soundingline.v30 import l,r_performer,k
from runners.run_v27 import frozen_source

def accepted(c,names):
    jobs=[c/"jobs"/name for name in names]
    if not all((j/"COMPLETE.json").exists() and read(j/"COMPLETE.json")["state"]=="VERIFIED" for j in jobs):return None
    rows=[];bindings=[]
    for j in jobs:
        w=read(j/"WORKER_COMPLETE.json");path=j/"output/study-original/SUMMARY.json"
        if digest(path)!=w["summary_sha256"]:raise ValueError("trigger source changed")
        rows.append(read(path));bindings.append(dict(job=j.name,summary_sha256=digest(path),completion_sha256=digest(j/"COMPLETE.json")))
    return rows,bindings

def triggers(c):
    out={}
    item=accepted(c,["L-CORE-GATED-0132"])
    if item:
        rows,bind=item
        x=next(r for r in rows[0]["rows"] if r["measure"]=="correlation_rho_given_g")["estimate"]
        out["L_MIDDLE"]=dict(decided=True,triggered=x>.6,estimate=x,bar=.6,inputs=bind,
            settings=[dict(l.settings()[-1],detector="middle")],module="l",cpu_cap_seconds=300,order=9000)
    item=accepted(c,[f"R_HEAT-CORE-{i:04d}" for i in range(24)])
    if item:
        rows,bind=item;comparisons=[]
        for row in rows:
            s=row["setting"]
            if s["truth"]!=3 or s["heat"]!=[.8,.1]:continue
            heat=next(x for x in row["rows"] if x["measure"]=="separation_heat")["estimate"]
            refusal=next(x for x in row["rows"] if x["measure"]=="separation_refusal")["estimate"]
            value=lambda x:float("inf") if isinstance(x,str) or x is None else float(x)
            comparisons.append(dict(prior=s["rule_prior"],heat=heat,refusal=refusal,triggered=value(heat)<value(refusal)))
        out["R_PERFORMER"]=dict(decided=True,triggered=any(x["triggered"] for x in comparisons),comparisons=comparisons,
            inputs=bind,settings=r_performer.settings(),module="r_performer",cpu_cap_seconds=120,order=9200)
    item=accepted(c,[f"K-CORE-R1-{i:04d}" for i in range(156)])
    if item:
        rows,bind=item;groups=[];comparisons=[]
        for a in (.02,.05,.15):
            for w in (.3,.1):
                for deadline in (False,True):
                    group=[r for r in rows if r["setting"]["arrival"]==a and r["setting"]["opening"]==w and r["setting"]["deadline"]==deadline and r["setting"]["policy"]=="PLAN"]
                    def score(q,cue):
                        rr=next(r for r in group if r["setting"]["q"]==q and r["setting"]["cue"]==cue)
                        return next(x for x in rr["rows"] if x["measure"]=="closed_per_1000")["estimate"]
                    wins=[]
                    for q in (.75,.5):
                        diffs=[score(q,cue)-score(1.,cue) for cue in (.9,.6,.3,0.)]
                        wins.append(dict(q=q,differences=diffs,triggered=all(d>0 for d in diffs)))
                    yes=any(x["triggered"] for x in wins)
                    comparisons.append(dict(arrival=a,opening=w,deadline=deadline,comparisons=wins,triggered=yes))
                    if yes and (a,deadline) not in groups:groups.append((a,deadline))
        settings=[dict(arrival=a,opening=w,deadline=d,policy="PLAN",q=q/8,cue=cue,
            steps=3000,worlds=200,blocks=8,work_sizes=[3])
            for a,d in groups for w in (.05,.1,.2,.3,.5,.75) for q in range(9) for cue in (.9,.6,.3,0.)]
        out["K_BEST_Q"]=dict(decided=True,triggered=bool(groups),comparisons=comparisons,inputs=bind,
            settings=settings,module="k",cpu_cap_seconds=900,order=9400)
    # P cannot be scored from a failed numerical reference. Missing is not agreement.
    bayes=c/"families/P_BAYES_CORE/ADMISSION.json"
    if not bayes.exists():
        out["P_BAYES_SCALE"]=dict(decided=False,reason="No admitted converged Bayesian primary; larger Bayesian grids are not admitted.")
    for name in ("L_MIDDLE","R_PERFORMER","K_BEST_Q"):
        if name not in out:out[name]=dict(decided=False,reason="Awaiting complete verified trigger inputs.")
    return out

def prepare(c,repo):
    start=time.process_time();decisions=triggers(c)
    for name,item in decisions.items():
        if not item["decided"]:continue
        decision=c/f"triggers/{name}.json"
        if decision.exists():
            if read(decision)!=item:raise ValueError("frozen trigger changed")
        else:write(decision,item)
        if not item["triggered"] or (c/f"prepared/{name}.json").exists():continue
        source=frozen_source(c,["runners/run_v30.py","runners/run_v29.py","runners/run_k3_redesign.py","ghostscale/validation/soundingline/v30/reader_tables.json"])
        base=dict(source_root=str(source),source_manifest_sha256=digest(source/"SOURCE_MANIFEST.json"),
            acceptance_sha256=digest(c/"ACCEPTANCE.json"),interpreter=str(repo/".venv/Scripts/python.exe"))
        protocol=digest(repo/"docs/versions/v30-the-readers-side/protocols/TEXTURE_AND_FOLLOWONS.md")
        write(c/f"prepared/{name}.json",dict(base,family=name,pool="reserve",module="ghostscale.validation.soundingline.v30."+item["module"],
            settings=item["settings"],cpu_cap_seconds=item["cpu_cap_seconds"],wall_cap_seconds=7200,
            order=item["order"],protocol_sha256=protocol,trigger_sha256=digest(decision)))
        gate=c/f"families/{name}_VALIDATION/ADMISSION.json"
        write(gate,dict(passed=True,purpose="excluded triggered follow-on validation",trigger_sha256=digest(decision),protocol_sha256=protocol))
        fixtures=[dict(module=item["module"],setting=item["settings"][i]) for i in sorted({0,len(item["settings"])-1})]
        write(c/f"jobs/{name}-VALIDATION/PLAN.json",dict(base,kind="validation",family=name+"_VALIDATION",pool="reserve",
            module="ghostscale.validation.soundingline.v30.texture_validate",setting=dict(fixtures=fixtures),
            order=-90,cpu_cap_seconds=1800,wall_cap_seconds=10800,admission_sha256=digest(gate)))
    write(c/"TRIGGER_STATUS.json",decisions,immutable=False)
    write(c/f"accounting/followon-prepare-{time.time_ns()}.json",dict(pool="reserve",cpu_seconds=time.process_time()-start,reservation=0,state="COMPLETE"))
    print({k:dict(decided=v["decided"],triggered=v.get("triggered")) for k,v in decisions.items()})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--campaign",type=Path,required=True)
    a=p.parse_args();prepare(a.campaign.resolve(),Path.cwd())
