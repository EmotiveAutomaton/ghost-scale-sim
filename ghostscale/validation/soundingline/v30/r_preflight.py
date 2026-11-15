"""Bounded exact and retained-path validation of the supplied R reference."""
import ast,itertools,zlib,json
from pathlib import Path
import numpy as np
from . import r
from .common import write,read,digest

def validate(setting,dest,pulse):
    source=Path(setting["analyst_root"])
    for name,sha in setting["input_sha256"].items():
        if digest(source/name)!=sha:raise ValueError("analyst input changed: "+name)
    text=(source/"r_probes.py").read_text(encoding="utf-8")
    tree=ast.parse(text);names={"keep","asked","watched"}
    definitions=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    code="\n\n".join(definitions)
    code=code.replace("done=None; t=0","done=None; t=0; PATH=[]")
    code=code.replace("a=rng.random()<keep(truth,p,n,w)","a=rng.random()<keep(truth,p,n,w); PATH.append((t,p,n,w,int(a)))")
    code=code.replace("steps.append(done)","CAPTURE.append((truth,PATH,lo,done)); steps.append(done)")
    env=dict(np=np,zlib=zlib,itertools=itertools,S=lambda x:1/(1+np.exp(-x)),C=list(r.REASONS),
        PRICES={"LOW":([0,1,2],[.5,.3,.2]),"MIXED":([0,1,2,4,8],[.25,.25,.2,.15,.15])},
        PROBES={k:r.CASES[k] for k in ("DEVALUE","AUDIENCE","PRICE")},CAPTURE=[])
    exec(compile(code,"retained_analyst_r.py","exec"),env)
    (dest/"instrumented_reference.py").write_text(code,encoding="utf-8")
    saved=read(source/"r_probes_out.json");checks=[];allrows=[];maxerror=0.;paths=[]
    controls=r.controls();checks.extend(controls["checks"])
    for q,b in ((1.,.5),(.9,.5),(.75,.5),(.75,.8),(.6,.5),(.6,.8)):
        actual=env["asked"](q,b);expected=saved["asked"][f"q={q},b={b}"]
        s=dict(q=q,b=b,rule_prior=.25,answer="binary",slope=1.5,heat=[.8,.1])
        ours=r.exact_asked(s)
        matches=all(list(actual[name])==expected[name] and
            round(ours["expected_odds"][i],2)==expected[name][0] and
            round(ours["settled_share"][i],2)==expected[name][1] for i,name in enumerate(r.REASONS))
        checks.append(dict(name=f"asked exact q={q},b={b}",passed=matches))
        allrows.append(dict(kind="asked",q=q,b=b,analyst=actual,independent=ours))
    for price in ("LOW","MIXED"):
        for flags in (True,False):
            env["CAPTURE"]=[];actual=env["watched"](price,flags)
            key=f'{price}, flags {"seen" if flags else "unseen"}';expected=saved["watched"][key]
            checks.append(dict(name=key+" supplied values",passed=all(list(actual[x])==expected[x] for x in r.REASONS)))
            local_error=0.;mismatches=0;beyond=0
            for truth,path,lo,done in env["CAPTURE"]:
                a=np.asarray(path);n=len(a);idx=r.REASONS.index(truth)
                raw=dict(price=a[None,:,1],need=a[None,:,2],watched=a[None,:,3],
                         act=a[None,:,4].astype(bool),occasion=np.ones((1,n),bool))
                s=dict(part="watched",truth=idx,flags=flags,slope=1.5,regime=price,rule_prior=.25,steps=n)
                probs=r.watched_probability(raw,s,True)[0]
                LL=np.cumsum(np.where(raw["act"][0,:,None],np.log(probs),np.log1p(-probs)),axis=0)
                odds=LL[:,idx]-np.max(np.delete(LL,idx,axis=1),axis=1)
                local_error=max(local_error,float(abs(odds[-1]-lo)))
                if odds[-1]>=3:
                    bad=np.flatnonzero(odds<3);entry=int(bad[-1]+1) if len(bad) else 0
                    independent_done=int(a[entry,0])
                else:independent_done=None
                mismatches+=independent_done!=done;beyond+=int(a[-1,0]>3000)
                paths.append(dict(regime=price,flags=flags,truth=truth,records=a.tolist(),
                                  analyst_final_odds=lo,analyst_done=done))
            maxerror=max(maxerror,local_error)
            checks.append(dict(name=key+" independent retained path",passed=local_error<1e-9 and mismatches==0,
                               max_log_odds_error=local_error,separation_mismatches=mismatches))
            allrows.append(dict(kind="watched",regime=price,flags=flags,analyst=actual,
                 paths=len(env["CAPTURE"]),paths_crossing_horizon=beyond,independent_error=local_error,
                 qualification="Analyst early-stop and geometric horizon overshoot retained; production uses all3000 fixed steps."))
            pulse(phase="R preflight",regime=price,flags=flags)
    # Exact no-price enumeration on the independently specified equations.
    for q in (1.,.9,.75):
        ours=r.exact_asked(dict(q=q,b=.5,rule_prior=.25,answer="binary",slope=1.5,heat=[.8,.1]),
                           ("DEVALUE","AUDIENCE"))
        expected=saved["asked_without_price"][f"q={q},b=0.5"]
        checks.append(dict(name=f"without price q={q}",passed=all(
            round(ours["expected_odds"][i],2)==expected[name]["expected_log_odds_vs_best_rival"]
            and round(ours["settled_share"][i],2)==expected[name]["share_settled"] for i,name in enumerate(r.REASONS))))
    write(dest/"retained_paths_points.json",paths)
    return dict(passed=all(x["passed"] for x in checks),checks=checks,comparisons=allrows,
        analyst_script_sha256=digest(source/"r_probes.py"),max_independent_log_odds_error=maxerror,
        owner_full_record_decision=True,scientific_draws=0)
