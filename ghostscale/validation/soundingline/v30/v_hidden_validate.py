"""Finite visibility and particle-resolution admission for V hidden readers."""
import time
from pathlib import Path
import numpy as np
from . import v,v_hidden
from .common import write,run_cell
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    checks=v_hidden.controls()["checks"];fixtures=[];rows=[]
    for arm in ("BLIND","RECORD"):
        for truth,env,reg,tail,loss in (([1,2,0],"plenty","LOW",0.,2),([7,8,0],"scarce","MIXED",.1,10)):
            setting=dict(arm=arm,truth=truth,tail=tail,loss=loss,environment=env,regime=reg,
                         steps=1024,worlds=4,blocks=4,particles=s["ladder"][0])
            raw=v_hidden.generate(setting,"excluded-hidden-resolution",pulse)
            np.savez_compressed(dest/(arm+"-"+str(truth[0])+"_points.npz"),**raw)
            fixtures.append((setting,raw))
    previous=None;chosen=None
    for population in s["ladder"]:
        start=time.process_time();current=[]
        for setting,raw in fixtures:
            pred,change,ll=v_hidden.reconstruct(raw,setting,particles=population)
            current.append(pred);pulse(population=population)
        if previous is not None:
            errors=np.concatenate([abs(a-b).ravel() for a,b in zip(current,previous)])
            decisions=[]
            for (setting,_),a,b in zip(fixtures,current,previous):
                decisions.append(((a<1/(1+setting["loss"]))!=(b<1/(1+setting["loss"]))).ravel())
            mismatch=float(np.concatenate(decisions).mean())
            passed=errors.mean()<=.01 and np.quantile(errors,.95)<=.025 and errors.max()<=.1 and mismatch<=.01
            rows.append(dict(particles=population,mean_forecast_change=float(errors.mean()),
                percentile95=float(np.quantile(errors,.95)),max_change=float(errors.max()),
                invitation_disagreement=mismatch,passed=bool(passed),cpu_seconds=time.process_time()-start))
            write(dest/"RESOLUTION_PROGRESS.json",dict(rows=rows),immutable=False)
            if passed:chosen=population;break
        previous=current
    checks.append(dict(name="adjacent-population forecast and action stability",passed=chosen is not None))
    # Full retained-path independent arithmetic and explicit information-boundary attacks.
    for setting,raw in fixtures:
        pred,change,ll=v_hidden.reconstruct(raw,setting,reference=True)
        err=float(np.max(abs(pred-raw["prediction"])))
        checks.append(dict(name=setting["arm"]+" independent hidden path",passed=err<1e-9,max_forecast_error=err))
        altered={key:value.copy() for key,value in raw.items()}
        altered["records"][:,:,0:3]=0;altered["records"][:,:,6:]=0
        altered["private_price"][:]=32;altered["draws"][:]=.999
        if setting["arm"]=="BLIND":altered["records"][:,:,3:6]=0
        pp,cc,l2=v_hidden.reconstruct(altered,setting,reference=True)
        checks.append(dict(name=setting["arm"]+" hidden fields cannot enter reader",passed=bool(
            np.array_equal(pred,pp) and np.array_equal(change,cc) and np.array_equal(ll,l2))))
        if setting["arm"]=="BLIND":checks.append(dict(name="blind public refusal has zero update",passed=bool(np.all(change==0))))
    if chosen:
        for arm in ("BLIND","RECORD"):
            setting=dict(fixtures[0][0],arm=arm,particles=chosen,steps=64,worlds=8,blocks=8)
            root=dest/("native-"+arm)
            for mode in ("original","replay","reconstruct"):
                run_cell(v,setting,root/mode,mode="reconstruct" if mode=="reconstruct" else "original",
                         original=root/"original",lane="excluded-hidden-native",pulse=pulse)
            equivalent(root/"original",root/"replay",True);equivalent(root/"original",root/"reconstruct")
            checks.append(dict(name=arm+" full native replay and reconstruction",passed=True))
    return dict(passed=all(x["passed"] for x in checks),checks=checks,chosen_particles=chosen,
        resolution=rows,scientific_draws=0,limit="finite particle accuracy checks; shared state-transition code")
