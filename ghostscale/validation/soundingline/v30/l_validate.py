"""Prospectively frozen L baseline and numerical-resolution admission."""
import ast,json,zlib
from pathlib import Path
import numpy as np
from . import l
from .common import rng,write,digest,sustained

def analyst_namespace(root):
    tree=ast.parse((root/"l_leftover.py").read_text(encoding="utf-8"))
    # Preserve original functions, avoiding the unrelated long Monte Carlo tuning.
    tree.body=[x for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
    ns={};exec(compile(tree,"retained-l-leftover","exec"),ns)
    supplied=json.loads((root/"l_leftover_out.json").read_text(encoding="utf-8"))
    ns.update(XS=np.array([.5,1,1.5]),GRID=np.linspace(0,6,121),TYPES=supplied["types"])
    return ns,supplied

def enumeration(raw,parameter,s):
    p,g,rho=parameter;sh,sf=s["show"];eh,ef=l.detector(s)
    out=[];events=[]
    for do,eo in zip(raw["observed_show"],raw["observed_event"]):
        a=np.array([0.]);mass=np.array([1.]);ll=[];pe=[];total=0.
        for show,event in zip(do,eo):
            aa=(rho*a[:,None]+g*np.array([0.,.5,1.,1.5])).ravel()
            mm=(mass[:,None]*np.array([1-p,p/3,p/3,p/3])).ravel()
            cause=np.tile(np.array([False,True,True,True]),len(a))
            if s["detector"]!="none":
                dp=np.where(cause,eh,ef);mm*=dp if event else 1-dp
            sp=sf+(sh-sf)*(-np.expm1(-aa));mm*=sp if show else 1-sp
            z=mm.sum();mass=mm/z;a=aa;total+=np.log(z)
            ll.append(total);pe.append(float(mass[cause].sum()))
        out.append(ll);events.append(pe)
    return np.array(out),np.array(events)

def validate(s,dest,pulse):
    root=Path(s["analyst_root"]);checks=[];comparisons=[];paths={}
    for name,sha in s["input_sha256"].items():
        if digest(root/name)!=sha:raise ValueError("analyst input changed")
    ns,supplied=analyst_namespace(root);names=list(supplied["types"])
    params=[tuple(supplied["types"][n][k] for k in ("p","g","rho")) for n in names]
    checks+=l.controls()["checks"]
    cases=[("shows only","unseen",1.,0.,1.,0.),
      ("shows only, through a reader (hit .6, false alarm .05)","unseen",1.,0.,.6,.05),
      ("shows and events through a detector (hit .7, false alarm .05), sizes unseen","noisy",.7,.05,1.,0.),
      ("as above, and shows through a reader (hit .6, false alarm .05)","noisy",.7,.05,.6,.05)]
    for ci,(label,mode,eh,ef,sh,sf) in enumerate(cases):
        for truth,name in enumerate(names):
            r=np.random.default_rng(zlib.crc32(f"{mode}|{name}|{eh}|{sh}".encode()))
            E=[];X=[];D=[]
            for w in range(60):
                e,x,d=ns["simulate"](supplied["types"][name],1500,r)
                eo=np.where(e,r.random(1500)<eh,r.random(1500)<ef)
                do=np.where(d,r.random(1500)<sh,r.random(1500)<sf)
                E.append(eo);X.append(x);D.append(do)
            E=np.array(E);X=np.array(X);D=np.array(D);LL=[]
            nodes=ns["GRID"];maximum_error=0.
            for candidate,param in enumerate(params):
                p,g,rho=param
                lands=np.clip(np.rint((rho*nodes[None,:]+g*np.array([0.,.5,1.,1.5])[:,None])/.05).astype(int),0,120)
                a,_=l.grid_forward(D,E,mode!="unseen",sh,sf,eh,ef,nodes,lands,lands,
                    np.zeros_like(lands,float),np.array([1-p,p/3,p/3,p/3]))
                original=ns["loglik_path"](supplied["types"][names[candidate]],E[0],X[0],D[0],mode,eh,ef,sh,sf)
                maximum_error=max(maximum_error,float(np.max(abs(original-a[0]))));LL.append(a)
            LL=np.stack(LL,axis=-1);odds=LL[...,truth]-np.max(np.delete(LL,truth,axis=-1),axis=-1)
            times=sustained(odds);got=times[times<=1500]
            gotrow=dict(separated=round(len(got)/60,3),median_steps=int(np.median(got)) if len(got) else None,
                        median_final_log_odds=round(float(np.median(odds[:,-1])),2))
            expected=supplied["results"][label][name]
            passed=all(gotrow[k]==v for k,v in expected.items())
            checks.append(dict(name=label+" "+name+" saved baseline",passed=passed))
            checks.append(dict(name=label+" "+name+" original-script arithmetic",passed=maximum_error<1e-9,max_error=maximum_error))
            comparisons.append(dict(kind="historical rounded/clipped grid",label=label,truth=name,expected=expected,actual=gotrow))
            paths[f"{ci}-{truth}-events"]=E;paths[f"{ci}-{truth}-sizes"]=X;paths[f"{ci}-{truth}-shows"]=D
            pulse(phase="historical baseline",cell=ci*3+truth)
    for truth,name in enumerate(names):
        r=np.random.default_rng(zlib.crc32(f"exact|{name}".encode()));E=[];X=[];D=[]
        for w in range(200):
            e,x,d=ns["simulate"](supplied["types"][name],1500,r);E.append(e);X.append(x);D.append(d)
        raw=dict(observed_event=np.array(E),size=np.array([.5,1,1.5])[np.array(X)],observed_show=np.array(D))
        LL=np.stack([l.exact_observed(raw,param,dict(show=[1.,0.]))[0] for param in params],axis=-1)
        ref=np.stack([l.exact_observed(raw,param,dict(show=[1.,0.]),True)[0] for param in params],axis=-1)
        odds=LL[...,truth]-np.max(np.delete(LL,truth,axis=-1),axis=-1);times=sustained(odds);got=times[times<=1500]
        actual=dict(separated=round(len(got)/200,3),median_steps=int(np.median(got)) if len(got) else None)
        expected=supplied["results"]["shows and events, exact"][name]
        checks.append(dict(name="exact events "+name,passed=actual==expected and np.max(abs(LL-ref))<1e-9))
        comparisons.append(dict(kind="exact events",truth=name,expected=expected,actual=actual))
        for key,val in raw.items():paths["exact-"+name+"-"+key]=val
    np.savez_compressed(dest/"baseline_points.npz",**paths)
    # Fixed full-record fixtures span every target, exact/noisy shows, absent/noisy events.
    fixtures=[]
    for target in (.08,.16,.32):
        for show in ([1.,0.],[.6,.05]):
            for detector in ("none","middle"):
                setting=dict(part="filter",target=target,show=show,detector=detector,
                    truth=2,mixed=target==.16,steps=1500,worlds=4,blocks=4,resolution=256)
                raw=l.generate(setting,"excluded-resolution")
                fixtures.append((setting,raw))
    ladder=s["ladder"];previous=None;chosen=None;resolution_rows=[]
    for resolution in ladder:
        current=[];max_ll=0.;max_pe=0.
        for setting,raw in fixtures:
            for param in l.makers(setting["target"],setting["mixed"]):
                a,b=l.filter_one(raw,param,setting,resolution=resolution);current.append((a,b))
                pulse(phase="resolution",resolution=resolution)
        if previous is not None:
            max_ll=max(float(np.max(abs(a-c))) for (a,b),(c,d) in zip(current,previous))
            max_pe=max(float(np.max(abs(b-d))) for (a,b),(c,d) in zip(current,previous))
            passed=max_ll<=s["max_prefix_loglik_change"] and max_pe<=s["max_event_probability_change"]
            resolution_rows.append(dict(resolution=resolution,max_prefix_loglik_change=max_ll,max_event_probability_change=max_pe,passed=passed))
            if passed:chosen=resolution;break
        previous=current
    checks.append(dict(name="full-history adjacent-resolution gate",passed=chosen is not None))
    short=[]
    if chosen is not None:
        for target in (.08,.16,.32):
            ss=dict(part="filter",target=target,show=[.6,.05],detector="middle",truth=1,
                mixed=target==.16,steps=8,worlds=4,blocks=4,resolution=chosen)
            raw=l.generate(ss,"exact-short-path")
            for param in l.makers(target,ss["mixed"]):
                a,b=l.filter_one(raw,param,ss);c,d=enumeration(raw,param,ss)
                dl=float(np.max(abs(a-c)));dp=float(np.max(abs(b-d)))
                short.append(dict(target=target,parameter=param,ll_error=dl,event_error=dp))
                checks.append(dict(name="exhaustive short hidden paths",passed=dl<.005 and dp<.001))
        np.savez_compressed(dest/"resolution_points.npz",**{
            f"fixture-{i}-{key}":value for i,(_,raw) in enumerate(fixtures) for key,value in raw.items()})
    return dict(passed=all(x["passed"] for x in checks),checks=checks,comparisons=comparisons,
        resolution=chosen,resolution_ladder=resolution_rows,exact_short=short,
        gate_scope="finite retained fixtures; grid approximation is not an exact ceiling",scientific_draws=0)
