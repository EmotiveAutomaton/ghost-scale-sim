"""Public-JSON-only scientific reader worker. No live status or evaluator I/O."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import argparse
import importlib

# Trusted source imports occur before the file-access guard. Scientific functions
# are fixed local implementations, not arbitrary submitted native/plugin code.
from ghostscale.validation.soundingline.v16.records import canonical
from ghostscale.validation.soundingline.v16.inference import read_public as native
from ghostscale.validation.soundingline.v16.craft import construct_public as craft
from ghostscale.validation.soundingline.v16.reconstruction import reader as reading
from ghostscale.validation.soundingline.v16.opportunity import public_reader as opportunity
from ghostscale.validation.soundingline.v16.self_monitor import public_reader as self_monitor
from ghostscale.validation.soundingline.v16.self_trajectory import reader as trajectory
from ghostscale.validation.soundingline.v16.inquiry import choose as inquiry
from ghostscale.validation.soundingline.v16.inquiry import learn_public,construct_public,read_maker_public
from ghostscale.validation.soundingline.v16.behavior_study import critic_reader


def guard(event,args):
    if event=="open" or event in {"os.listdir","os.scandir","os.system","subprocess.Popen",
                                 "socket.connect","socket.bind","ctypes.dlopen","ctypes.dlsym"}:
        raise PermissionError("reader file/process/network access is closed after trusted imports")


def main():
    code_root=str(Path(__file__).resolve().parents[1])
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--extension",action="append",default=[])
    args=parser.parse_args()
    extensions={}
    for spec in args.extension:
        module,attribute=spec.split(":")
        if not module.startswith("ghostscale.validation.soundingline.v16.") or not attribute.isidentifier():
            raise ValueError("extension must be a declared local V16 scientific function")
        extensions[spec]=getattr(importlib.import_module(module),attribute)
    sys.addaudithook(guard)
    print(json.dumps({"ready":True,"pid":os.getpid(),"code_root":code_root,
                      "access":"public JSON frames only; filesystem guard enabled"}),flush=True)
    for line in sys.stdin.buffer:
        try:
            frame=json.loads(line)
            if frame.get("kind")=="shutdown":
                break
            if set(frame)!={"kind","public","options"}:
                raise ValueError("unexpected reader request fields")
            kind,options=frame["kind"],frame["options"]
            payload=canonical(frame["public"])
            if kind=="native":
                result=native(payload,**options)
            elif kind=="craft":
                if options:
                    raise ValueError("craft reader takes no hidden options")
                result=craft(payload)
            elif kind=="reading":
                result=reading(payload,**options)
            elif kind=="opportunity":
                result=opportunity(payload,**options)
            elif kind=="self":
                result=self_monitor(payload,**options)
            elif kind=="trajectory":
                result=trajectory(payload,**options)
            elif kind=="inquiry":
                result=inquiry(payload,**options)
            elif kind=="inquiry-learning":
                result=learn_public(payload)
            elif kind=="inquiry-construction":
                result=construct_public(payload)
            elif kind=="inquiry-reading":
                result=read_maker_public(payload)
            elif kind=="critic":
                result=critic_reader(payload)
            elif kind in extensions:
                result=extensions[kind](payload,**options)
            elif kind=="_probe_forbidden_read":
                result=Path(frame["public"]["path"]).read_text()
            else:
                raise ValueError("unknown scientific reader")
            response={"ok":True,"result":result}
        except Exception as error:
            response={"ok":False,"error":type(error).__name__+": "+str(error)}
        sys.stdout.buffer.write(canonical(response)+b"\n")
        sys.stdout.buffer.flush()


if __name__=="__main__":
    main()
