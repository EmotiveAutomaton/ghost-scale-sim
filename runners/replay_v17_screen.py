"""Bounded complete-case replay; run from the packet's extracted frozen source."""
import argparse
import importlib
import json
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import canonical, file_digest, read
from ghostscale.validation.soundingline.v17.packets import load_cases

def replay(root):
    root=Path(root)
    lock=read(root/"LOCK.json")
    design=lock["design"]
    source_root=Path(__file__).resolve().parents[1]
    for name,expected in lock["source_files"].items():
        if file_digest(source_root/name)!=expected:
            raise ValueError("replay source mismatch: "+name)
    module=importlib.import_module("ghostscale.validation.soundingline.v17."+{
        "A2":"craft_extension","B":"recipient","C":"adaptive","D":"revision","E":"observer"}[design["family"]])
    items=list(load_cases(root,read(root/"INDEX.json")))
    prepared=dict(design)
    if hasattr(module,"prepare"):
        trained=module.prepare(design)
        if canonical(trained)!=canonical(read(root/"TRAINING.json")):
            raise ValueError("controller training replay differs")
        prepared["prepared"]=trained
    count=0
    rows=0
    for c,h in ((0,0),(design["constructors"]-1,design["histories_per_constructor"]-1)):
        for r,regime in enumerate(design["regimes"]):
            index=(c*design["histories_per_constructor"]+h)*len(design["regimes"])+r
            original=items[index]
            case=module.make_case(design["namespace"],c,h,regime)
            replayed=dict(case=case,rows=module.evaluate_case(case,prepared))
            if canonical(replayed)!=canonical(original):
                raise ValueError("case replay differs: "+case["case_id"])
            count+=1
            rows+=len(replayed["rows"])
    return dict(schema="v17.bounded-replay.1",passed=True,cases_replayed=count,rows_replayed=rows,
        selection="first and last constructor/history in every regime, fixed without outcome ranking",
        completion_sha256=file_digest(root/"COMPLETION.json"),replay_source_sha256=file_digest(Path(__file__)),
        scope="whole case construction, forecast, physical execution, counted costs and scores; deterministic fields exact",
        limitation="bounded replay, not full regeneration")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=replay(args.root)
    payload=canonical(result)+b"\n"
    if args.output.exists() and args.output.read_bytes()!=payload: raise ValueError("replay receipt differs")
    if not args.output.exists(): args.output.write_bytes(payload)
    print(json.dumps(result))
if __name__=="__main__": main()
