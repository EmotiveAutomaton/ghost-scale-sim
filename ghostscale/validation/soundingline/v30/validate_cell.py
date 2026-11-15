"""Excluded fixture validation through the same finite native queue slots."""
import importlib,time
from pathlib import Path
from .common import run_cell,write
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    module=importlib.import_module(s["module"]);checks=module.controls(dest/"controls",pulse)["checks"]
    costs=[]
    for i,setting in enumerate(s["fixtures"]):
        start=time.process_time();root=dest/("fixture-"+str(i))
        original=root/"original";replay=root/"replay";reference=root/"reconstruct"
        run_cell(module,setting,original,lane="excluded-native-fixture",pulse=pulse)
        run_cell(module,setting,replay,lane="excluded-native-fixture",pulse=pulse)
        equivalent(original,replay,True)
        run_cell(module,setting,reference,mode="reconstruct",original=original,lane="excluded-native-fixture",pulse=pulse)
        equivalent(original,reference)
        costs.append(dict(setting=setting,cpu_seconds=time.process_time()-start))
        checks.append(dict(name="complete native fixture "+str(i),passed=True))
        pulse(fixture=i+1)
    return dict(passed=all(c["passed"] for c in checks),checks=checks,costs=costs,scientific_draws=0)
