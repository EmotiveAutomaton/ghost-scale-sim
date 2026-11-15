"""P's declared uneven-label texture, using reconstructed corpus reader marginals."""
import numpy as np
from pathlib import Path
from . import p
from .common import audit_controls,write

def marginals(name):
    # Corpus reports: 100/writer-label except guilt and shame (50 each);
    # five human judgments or one released model prediction per report.
    count=np.full(13,100 if name=="MODEL" else 500)
    for label in ("guilt","shame"):count[p.INPUT["labels"].index(label)]//=2
    rounded=np.array(p.INPUT[name])
    unsmoothed=np.rint(rounded*(count[:,None]+6.5)-.5).astype(int)
    rebuilt=(unsmoothed+.5)/(count[:,None]+6.5)
    if (unsmoothed<0).any() or not np.array_equal(unsmoothed.sum(1),count) or np.max(abs(rebuilt-rounded))>5.00001e-5:
        raise ArithmeticError("rounded corpus table does not identify stated integer counts")
    return unsmoothed.sum(0)/unsmoothed.sum()

def settings():
    out=[]
    for source in ("HUMAN","MODEL"):
        marginal=marginals(source);h=float(marginal[p.NEUTRAL])
        for s in p.settings():
            if s["part"]!="labels" or s["h"]!=0:continue
            alpha=s["km"]*marginal[p.FEEL]/(1-h)
            out.append(dict(s,h=h,maker_prior=alpha.tolist(),part="uneven",marginal_source=source))
    return out

generate=p.generate
infer=p.infer
summarize=p.summarize
pilot_exclusion=p.pilot_exclusion

def controls(dest=None,pulse=lambda **kw:None):
    checks=p.controls()["checks"]
    for name in ("HUMAN","MODEL"):
        x=marginals(name)
        checks.append(dict(name=name+" exact integer reconstruction and normalized marginal",
            passed=bool(abs(x.sum()-1)<1e-14 and np.all(x>0)),marginal=x.tolist()))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
