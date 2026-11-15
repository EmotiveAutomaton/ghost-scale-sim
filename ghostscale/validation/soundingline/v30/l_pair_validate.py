"""L two-channel resolution ladder and exact observed controls."""
import numpy as np
from . import l,l_pair
from .common import write,run_cell
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    checks=l_pair.controls()["checks"];fixtures=[]
    for target in (.08,.16,.32):
        for truth in range(3):
            for ratio in (.5,2.):
                ss=dict(l_pair.settings()[0],target=target,truth=truth,ratio=ratio,worlds=1,blocks=1,
                    show=[.6,.05],detector="middle",steps=1500)
                fixtures.append((ss,l_pair.generate(ss,"excluded-pair-resolution")))
    previous=None;chosen=None;rows=[]
    for resolution in s["ladder"]:
        outputs=[]
        for ss,raw in fixtures:
            outputs.append(l_pair.filter_one(raw,l.makers(ss["target"])[ss["truth"]],ss,resolution=resolution))
            pulse(resolution=resolution)
        if previous is not None:
            delta=max(float(np.max(abs(a[0]-b[0]))) for a,b in zip(outputs,previous))
            event=max(float(np.max(abs(a[1]-b[1]))) for a,b in zip(outputs,previous))
            passed=delta<.01 and event<.005
            rows.append(dict(resolution=resolution,max_prefix_log_change=delta,max_event_change=event,passed=passed))
            write(dest/"RESOLUTION_PROGRESS.json",dict(rows=rows),immutable=False)
            if passed:chosen=resolution;break
        previous=outputs
    checks.append(dict(name="two-channel adjacent grid resolution",passed=chosen is not None))
    if chosen:
        for detector in ("middle","exact"):
            ss=dict(l_pair.settings()[0],worlds=8,blocks=8,steps=64,detector=detector,resolution=chosen)
            root=dest/detector
            for mode in ("original","replay","reconstruct"):
                run_cell(l_pair,ss,root/mode,mode="reconstruct" if mode=="reconstruct" else "original",
                    original=root/"original",lane="excluded-pair-native",pulse=pulse)
            equivalent(root/"original",root/"replay",True);equivalent(root/"original",root/"reconstruct")
            checks.append(dict(name=detector+" full pair replay and reconstruction",passed=True))
    return dict(passed=all(x["passed"] for x in checks),checks=checks,resolution=rows,chosen_resolution=chosen,scientific_draws=0)
