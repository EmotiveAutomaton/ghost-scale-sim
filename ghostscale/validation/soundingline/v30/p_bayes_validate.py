"""Bounded prospective posterior-resolution and known-answer checks."""
from pathlib import Path
import time,numpy as np
from . import p,p_bayes
from .common import write,rng,run_cell
from runners.run_v30 import equivalent
def validate(s,dest,pulse):
    checks=p_bayes.controls()["checks"];rows=[];chosen=None
    # Exact no-data posterior is the specified Dirichlet prior.
    alpha=np.full(12,.5);C=p.table("HUMAN")[np.r_[p.FEEL,p.NEUTRAL]]
    samples=np.stack([p_bayes.chain(np.empty((0,13),np.int16),C,.5,alpha,3.,1024,4096,2,9321+k)
                      for k in range(4)])
    rhat,ess=p_bayes.diagnostics(samples);error=float(np.max(abs(samples.mean((0,1))-alpha/alpha.sum())))
    checks.append(dict(name="no-data Dirichlet prior mean",passed=error<.025 and rhat.max()<1.1,
                       max_mean_error=error,max_rhat=float(rhat.max()),min_ess=float(ess.min())))
    previous=None
    for burn,draws in s["ladder"]:
        worst_rhat=0.;worst_ess=np.inf;means=[];start=time.process_time()
        for table in ("PERFECT","HUMAN"):
            setting=dict(km=6,kp=3,h=.5,m=60,table=table,corrector=table,neutral_shift=0.,
                worlds=4,blocks=4,pages=128,part="bayes",burn=burn,draws=draws,thin=2,chains=4)
            raw=p_bayes.generate(setting,"excluded-bayes-resolution")
            # Only prefixes 1 and 128 are needed for this resolution fixture;
            # all eight remain in production. Keep the complete finite fixture here.
            out=p_bayes.infer(raw,setting,pulse=pulse)
            np.savez_compressed(dest/f"{table}-{burn}_points.npz",**raw)
            np.savez_compressed(dest/f"{table}-{burn}_measures.npz",**out)
            worst_rhat=max(worst_rhat,float(out["_rhat"].max()));worst_ess=min(worst_ess,float(out["_ess"].min()))
            means.append(out["_estimate"])
        delta=None if previous is None else max(float(np.max(abs(a-b))) for a,b in zip(means,previous))
        passed=worst_rhat<1.1 and worst_ess>=50 and delta is not None and delta<.03
        rows.append(dict(burn=burn,draws=draws,max_split_rhat=worst_rhat,min_ess=worst_ess,
            max_mean_change=delta,passed=passed,cpu_seconds=time.process_time()-start))
        write(dest/"RESOLUTION_PROGRESS.json",dict(rows=rows),immutable=False)
        if passed:chosen=[burn,draws];break
        previous=means
    checks.append(dict(name="posterior convergence/resolution",passed=chosen is not None))
    if chosen:
        ss=dict(setting,worlds=8,blocks=8,pages=8,burn=chosen[0],draws=chosen[1])
        for mode in ("original","replay","reconstruct"):
            run_cell(p_bayes,ss,dest/mode,mode="reconstruct" if mode=="reconstruct" else "original",
                     original=dest/"original",lane="excluded-independent-target",pulse=pulse)
        equivalent(dest/"original",dest/"replay",True);equivalent(dest/"original",dest/"reconstruct")
        checks.append(dict(name="full posterior replay and independent density",passed=True))
    return dict(passed=all(x["passed"] for x in checks),checks=checks,rows=rows,chosen=chosen,
        scientific_draws=0,qualification="posterior sample approximation; never a proven ceiling")
