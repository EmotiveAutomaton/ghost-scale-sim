"""Admit separately prepared finite maps only after their frozen native validation."""
import argparse,time
from pathlib import Path
from ghostscale.validation.soundingline.v30.common import read,write,digest

def admit(c):
    start=time.process_time();dispositions={}
    for path in sorted((c/"prepared").glob("*.json")):
        prepared=read(path);family=prepared["family"];job=c/f"jobs/{family}-VALIDATION"
        if not (job/"COMPLETE.json").exists():dispositions[family]="validation pending";continue
        if read(job/"COMPLETE.json")["state"]!="VERIFIED":dispositions[family]="validation failed; held";continue
        validation=job/"output/VALIDATION.json";answer=read(validation);assert answer["passed"]
        tested=read(job/"PLAN.json")
        # Freeze the production source to the source that actually passed validation.
        for key in ("source_root","source_manifest_sha256"):
            if prepared[key]!=tested[key]:raise ValueError("prepared source differs from validation")
        gate=c/f"families/{family}/ADMISSION.json"
        if not gate.exists():write(gate,dict(passed=True,validation_sha256=digest(validation),
            validation_completion_sha256=digest(job/"COMPLETE.json"),prepared_sha256=digest(path)))
        total=len(prepared["settings"])
        prior=[c/f"jobs/{family}-{i:04d}/COMPLETE.json" for i in range(total)]
        if all(p.exists() and read(p)["state"]=="WITHDRAWN_UNSTARTED" for p in prior):
            dispositions[family]="superseded before execution; preserved successor plans govern";continue
        for i,s in enumerate(prepared["settings"]):
            if family=="L_PAIR":s=dict(s,resolution=answer["chosen_resolution"])
            plan={k:prepared[k] for k in ("source_root","source_manifest_sha256","acceptance_sha256",
                "interpreter","family","pool","module","cpu_cap_seconds","wall_cap_seconds")}
            plan.update(setting=s,order=prepared["order"]+i,admission_sha256=digest(gate))
            dest=c/f"jobs/{family}-{i:04d}/PLAN.json"
            if dest.exists():
                if read(dest)!=plan:raise ValueError("existing prepared plan changed")
            else:write(dest,plan)
        dispositions[family]=f"{total} admitted finite cells"
    write(c/"PREPARED_STATUS.json",dispositions,immutable=False)
    write(c/f"accounting/prepared-admission-{time.time_ns()}.json",
        dict(pool="reserve",cpu_seconds=time.process_time()-start,reservation=0,state="COMPLETE"))
    print(dispositions)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--campaign",type=Path,required=True)
    a=p.parse_args();admit(a.campaign.resolve())
