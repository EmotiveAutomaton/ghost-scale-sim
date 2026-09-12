"""Pinned Stitch standalone consumer; parse and validate typed cell/part abstractions."""
import ctypes
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import time
from ..v16.records import canonical,digest,file_digest,read,write
from ..v16.runtime import local_owner
PIN="350804b7b35807c78bd21c313785ae5152ae2985"


def sexpr(text):
    tokens=re.findall(r"\(|\)|[^\s()]+",text)
    def node(index):
        token=tokens[index]
        if token=="(":
            values=[]
            index+=1
            while index<len(tokens) and tokens[index]!=")":
                value,index=node(index)
                values.append(value)
            if index==len(tokens): raise ValueError("unclosed expression")
            return values,index+1
        if token==")": raise ValueError("unexpected close")
        return token,index+1
    result,end=node(0)
    if end!=len(tokens): raise ValueError("trailing expression")
    return result


def expand(expression,definitions,arguments=(),world="graphic",depth=0):
    if depth>12: raise ValueError("recursive abstraction")
    if isinstance(expression,str):
        if expression in definitions:
            body,arity=definitions[expression]
            if arity: raise ValueError("missing abstraction arguments")
            return expand(body,definitions,(),world,depth+1)
        raise ValueError("program token not an expression")
    if not expression: raise ValueError("empty expression")
    op=expression[0]
    def argument(token):
        if not isinstance(token,str): raise ValueError("higher-order argument rejected")
        if token.startswith("#"):
            index=int(token[1:])
            if not 0<=index<len(arguments): raise ValueError("unbound typed argument")
            value=arguments[index]
        elif re.fullmatch("c[0-9]+",token):
            value=int(token[1:])
        else: raise ValueError("non-cell argument")
        if type(value) is not int or not 0<=value<(16 if world=="graphic" else 3):
            raise ValueError("cell or part outside domain")
        return value
    if op=="seq":
        return [a for item in expression[1:] for a in expand(item,definitions,arguments,world,depth+1)]
    offsets={"place":0,"remove":16} if world=="graphic" else {"attach":0,"remove":3,"rotate":6}
    if op in offsets and len(expression)==2:
        return [offsets[op]+argument(expression[1])]
    if op in definitions:
        body,arity=definitions[op]
        if len(expression)-1!=arity: raise ValueError("wrong learned arity")
        return expand(body,definitions,tuple(argument(t) for t in expression[1:]),world,depth+1)
    raise ValueError("unsupported typed program")


def encode(program,world="graphic"):
    def action(a):
        if world=="graphic": return f"({'place' if a<16 else 'remove'} c{a%16})"
        if a==9: return None
        return f"({('attach','remove','rotate')[a//3]} c{a%3})"
    return "(seq "+" ".join(x for a in program if (x:=action(a)) is not None)+")"


def learn(training,world="graphic"):
    exe=Path(os.environ["GS_V17_STITCH_EXE"]).resolve()
    cache=Path(os.environ["GS_V17_STITCH_CACHE"]).resolve()
    programs=[encode(t["program"],world) for t in training]
    key=digest([PIN,file_digest(exe),programs,world,"arity2-iterations3-uncurried"])
    root=cache/key
    with local_owner(root):
        if (root/"SEMANTIC.json").exists():
            result=read(root/"SEMANTIC.json")
            receipt=read(root/"RECEIPT.json")
            for name,sha in receipt["files"].items():
                if file_digest(root/name)!=sha: raise ValueError("Stitch cache corruption")
            return result
        if (root/"FAILURE.json").exists() or (root/"TIMEOUT.json").exists():
            raise RuntimeError("retained Stitch failure requires explicit repair; no silent retry")
        # Preserve an interrupted child's partial output before attempting recovery.
        retained=[root/name for name in ("output.json","stdout.log","stderr.log") if (root/name).exists()]
        if retained:
            archive=root/("interrupted-"+str(time.time_ns()))
            archive.mkdir()
            for path in retained:
                path.rename(archive/path.name)
        write(root/"programs.json",programs)
        command=[str(exe),str(root/"programs.json"),"--out",str(root/"output.json"),
                 "--max-arity","2","--iterations","3","--threads","1","--no-curried-bodies","--no-curried-metavars"]
        if os.name=="nt": ctypes.windll.kernel32.SetErrorMode(3)
        begin=time.monotonic()
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            stdout,stderr=process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()  # Only this explicitly owned isolated dependency child.
            stdout,stderr=process.communicate()
            write(root/"TIMEOUT.json",dict(seconds=120,returncode=process.returncode))
            raise
        cpu_seconds=None
        if os.name=="nt":
            times=[ctypes.c_ulonglong() for _ in range(4)]
            ok=ctypes.windll.kernel32.GetProcessTimes(ctypes.c_void_p(int(process._handle)),*(ctypes.byref(x) for x in times))
            if ok: cpu_seconds=(times[2].value+times[3].value)/1e7
        (root/"stdout.log").write_bytes(stdout)
        (root/"stderr.log").write_bytes(stderr)
        if process.returncode:
            write(root/"FAILURE.json",dict(returncode=process.returncode,wall_seconds=time.monotonic()-begin))
            raise RuntimeError("Stitch standalone failed; retained attempt")
        raw=read(root/"output.json")
        stats=re.findall(r"Stats \{([^}]+)\}",stdout.decode("utf-8",errors="replace"))
        if not stats: raise ValueError("Stitch did not report learning operation counters")
        counters={}
        for block in stats:
            for name,value in re.findall(r"(\w+): (\d+)",block):
                counters[name]=counters.get(name,0)+int(value)
        definitions={}
        accepted=[]
        rejected=[]
        for definition in raw["abstractions"]:
            try:
                body=sexpr(definition["body"])
                arity=definition["arity"]
                if arity not in (0,1,2): raise ValueError("arity outside commissioned range")
                # Exhaustive validation of typed expansion over its entire finite domain.
                for args in product(range(16 if world=="graphic" else 3),repeat=arity):
                    expand(body,definitions,args,world)
                definitions[definition["name"]]=(body,arity)
                accepted.append(dict(name=definition["name"],body=body,arity=arity,
                    storage_tokens=len(re.findall(r"[^\s()]+",definition["body"]))+1+arity))
            except (ValueError,IndexError,TypeError) as exc:
                rejected.append(dict(definition=definition,reason=str(exc)))
        result=dict(pin=PIN,binary_sha256=file_digest(exe),cache_key=key,accepted=accepted,rejected=rejected,
            original_cost=raw["original_cost"],final_cost=raw["final_cost"],compression_ratio=raw["compression_ratio"],
            learning_counters=counters,learning_operations=sum(counters.get(k,0) for k in
                ("worklist_steps","calc_final_utility","calc_unargcap","azero_calc_util","azero_calc_unargcap")),
            input_tokens=sum(len(re.findall(r"[^\s()]+",p)) for p in programs),
            rewritten=raw["rewritten"])
        write(root/"SEMANTIC.json",result)
        write(root/"RECEIPT.json",dict(files={n:file_digest(root/n) for n in
            ("programs.json","output.json","stdout.log","stderr.log","SEMANTIC.json")},
            wall_seconds=time.monotonic()-begin,child_cpu_seconds=cpu_seconds,workers=1,gpu_used=False))
        return result
