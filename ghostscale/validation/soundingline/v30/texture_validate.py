"""Finite excluded checks for commissioned textures; no accepted scientific draws."""
import importlib,time
from .common import run_cell
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    checks=[];costs=[]
    for i,case in enumerate(s["fixtures"]):
        module=importlib.import_module("ghostscale.validation.soundingline.v30."+case["module"])
        checks.extend(module.controls()["checks"])
        ss=dict(case["setting"],worlds=8,blocks=8)
        root=dest/str(i);start=time.process_time()
        for mode in ("original","replay","reconstruct"):
            run_cell(module,ss,root/mode,mode="reconstruct" if mode=="reconstruct" else "original",
                     original=root/"original",lane="excluded-texture-validation",pulse=pulse)
        equivalent(root/"original",root/"replay",True);equivalent(root/"original",root/"reconstruct")
        checks.append(dict(name=case["module"]+" native whole-fixture replay and reconstruction",passed=True))
        costs.append(dict(module=case["module"],setting=ss,cpu_seconds=time.process_time()-start))
    return dict(passed=all(x["passed"] for x in checks),checks=checks,costs=costs,scientific_draws=0)
