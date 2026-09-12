"""Export an early/final V17 observer packet as distinct reader and evaluator ZIPs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import secrets
import zipfile
from ghostscale.validation.soundingline.v16.records import canonical,read,write,file_digest
from ghostscale.validation.soundingline.v17.packets import load_cases
from ghostscale.validation.soundingline.v17.observer import project,TIERS
REPO=Path(__file__).resolve().parents[1]
# Reader source is an explicit AST whitelist. No case generator, evaluator truth,
# source namespace, training constructor or disk-accessing Stitch learner is shipped.
FUNCTIONS={
"observer.py":{"hypotheses","forward","predict"},
"recipient.py":{"normalize","softmax","recipient","model_space","infer_models","mixture","opportunities","outcome_value","utilities","accepted_distribution","predict"},
"adaptive.py":{"option_prediction"},
"revision.py":{"encode","candidate_values","recipient_world","predict"},
"craft_extension.py":{"execute","acquire","proposals","distance","solve"},
"stitch_adapter.py":{"sexpr","expand","encode"},
"contracts.py":{"BudgetExhausted","Costs","forecast_scores"},
"programs.py":{"cell","expand","encode","execute"},
}
CONSTANTS={"METHODS","TIERS","COST_KINDS","SEARCH_KINDS","OPTIONS","PIN"}
V16_FUNCTIONS={
"records.py":{"canonical","digest","seed_for"},
"graphic_world.py":{"execute","learn"},
"assembly.py":{"World","step","artifact","execute","mismatch","fragments"},
}
V16_CONSTANTS={"ATTACH","REMOVE","ROTATE","STOP","ACTIONS","EMPTY"}

def subset(path,names,constants):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    kept=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            # Imports of omitted sibling generators are unnecessary and could
            # reintroduce evaluator/file access into the exported reference.
            if isinstance(node,ast.ImportFrom):
                if node.module and node.module.split(".")[-1]=="runtime": continue
                if node.module and node.module.split(".")[-1]=="records":
                    node.names=[alias for alias in node.names if alias.name in ("canonical","digest","seed_for")]
                    if not node.names: continue
                if path.name=="observer.py" and node.module=="contracts":
                    pass
                elif path.name=="graphic_world.py" and node.module not in (None,"__future__"):
                    if node.level and node.module!="records": continue
            kept.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            kept.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
            kept.append(node)
    return ast.unparse(ast.Module(body=kept,type_ignores=[])).encode("utf-8")+b"\n"

def package(source,output,limit=12):
    source,output=Path(source),Path(output)
    lock=read(source/"LOCK.json")
    if lock["design"]["family"]!="E": raise ValueError("reader export requires an observer source")
    completion=read(source/"COMPLETION.json")
    if completion["index_sha256"]!=file_digest(source/"INDEX.json"): raise ValueError("source index changed")
    items=list(load_cases(source,read(source/"INDEX.json")))
    # Deterministic source selection spans every native family; no outcome ranking.
    selected=[]
    for family in sorted({i["case"]["regime"] for i in items}):
        selected.extend([i for i in items if i["case"]["regime"]==family][:limit])
    output.mkdir(parents=True,exist_ok=True)
    selection_path=output/"PRIVATE_SELECTION.json"
    if selection_path.exists():
        aliases=read(selection_path)
    else:
        aliases={item["case"]["case_id"]:secrets.token_hex(16) for item in selected}
        write(selection_path,aliases)
    requests=[]
    answers=[]
    for item in selected:
        case=item["case"]
        for tier in TIERS:
            request=dict(project(case["public"],tier),task_id=aliases[case["case_id"]]+"-"+str(TIERS.index(tier)))
            requests.append(request)
            answers.append(dict(task_id=request["task_id"],source_case_id=case["case_id"],
                truth=case["private"]["future_choice"],private=case["private"],
                reference_rows=[r for r in item["rows"] if r["evidence_tier"]==tier]))
    # Opaque aliases determine order; source constructor/history positions are absent.
    requests.sort(key=lambda r:r["task_id"])
    public={"requests.jsonl":b"".join(canonical(r)+b"\n" for r in requests)}
    for package_path in ("ghostscale","ghostscale/validation","ghostscale/validation/soundingline",
                         "ghostscale/validation/soundingline/v16","ghostscale/validation/soundingline/v17"):
        public[package_path+"/__init__.py"]=b""
    for name,functions in FUNCTIONS.items():
        path=REPO/"ghostscale/validation/soundingline/v17"/name
        public["ghostscale/validation/soundingline/v17/"+name]=subset(path,functions,CONSTANTS)
    # Omitted imports are retained as inert empty modules only where no used code
    # reaches them (craft_extension imports craft for constants at import time).
    public["ghostscale/validation/soundingline/v17/craft.py"]=b"REGIMES=()\n"
    for name,functions in V16_FUNCTIONS.items():
        path=REPO/"ghostscale/validation/soundingline/v16"/name
        public["ghostscale/validation/soundingline/v16/"+name]=subset(path,functions,V16_CONSTANTS)
    public["consumer.py"]=b'''"""Read public JSON lines and write probability forecasts; standard library only."""
import argparse,json,sys
from ghostscale.validation.soundingline.v17.observer import predict
parser=argparse.ArgumentParser()
parser.add_argument("--method",default="inverse_maker")
parser.add_argument("--guard-probe",action="store_true")
args=parser.parse_args()
def guard(event,values):
    if event in ("open","os.system","subprocess.Popen","socket.connect","socket.bind"):
        raise PermissionError("reader evidence boundary")
sys.addaudithook(guard)
if args.guard_probe:
    try:
        open("PRIVATE-canary.txt")
    except PermissionError:
        print(json.dumps({"private_read_denied":True}))
        raise SystemExit(0)
    raise SystemExit(2)
for line in sys.stdin:
    request=json.loads(line)
    task_id=request.pop("task_id")
    result=predict(request,args.method)
    print(json.dumps({"schema":"v17.prediction.1","task_id":task_id,**result},allow_nan=False),flush=True)
'''
    public["adapter_v16.py"]=b'''"""Versioned envelope adapter; leaves v16.transfer.1 payloads unchanged."""
def adapt(request):
    if request.get("schema")=="v17.observer.1":
        return {"adapter_schema":"ghostscale.transfer.adapter.1","payload_schema":"v17.observer.1","payload":request}
    if request.get("schema_version")=="v16.transfer.1" or request.get("schema")=="v16.transfer.1":
        return {"adapter_schema":"ghostscale.transfer.adapter.1","payload_schema":"v16.transfer.1","payload":request}
    raise ValueError("unsupported transfer schema")
'''
    public["README.md"]=("""# V17 early observer challenge
Predict a constructed maker's unseen next choice after the declared intervention.
Each JSON line contains an opaque task ID, finite answer support, query cost,
permitted evidence tier, and a known finite family of possible native makers.
The actual maker policy and goal are withheld. These are new constructed cases,
not human tasks or a claim that historical reconstruction is unique.

A uses place/remove actions 0..15/16..31 on a sixteen-cell board.
B uses the same display physics, a finite Bayesian recipient, and STOP.
C renders a decoder's most likely interpretation as an executed display; the
future choice is the decoder's next interpretation after more observations.
D uses assembly attach 0..2, remove 3..5, rotate 6..8 and STOP 9. Supporting
parts cannot be rotated or removed while dependents remain attached.

An artifact is a bit mask: bit k says cell k is present. Assembly displays encode
part k with orientation v as bit 2*k+v. The context declares earlier and future
opportunities, possible goal settings and interventions; it does not identify
which hypothesis actually produced the observations. Earlier objects and process
evidence appear only in their stated tiers. No dates are supplied.

Run: python -B consumer.py < requests.jsonl > predictions.jsonl
The standard-library reference enumerates the declared maker family. Alternative
methods are surface_continuation and episodic_continuation. It reports all
probabilities, modeled operation/storage costs and compatible histories.
It closes file/process/network access after trusted imports. This is a boundary
for this fixed Python consumer, not a sandbox for hostile native code.

The evaluator ZIP is separate. Selection takes the first bounded number of
complete cases per native family before inspecting their outcomes. This early
pilot does not claim coverage of all five desired illustrative outcome types.
The versioned adapter wraps old V16 payloads without changing their schema.
""").encode("utf-8")
    def archive(path,members):
        manifest={name:hashlib.sha256(body).hexdigest() for name,body in members.items()}
        payload=dict(members)
        payload["MANIFEST.json"]=canonical(manifest)+b"\n"
        if path.exists():
            with zipfile.ZipFile(path) as z:
                if {name:z.read(name) for name in z.namelist()}!=payload: raise ValueError("retained export differs")
        else:
            with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
                for name,body in sorted(payload.items()):
                    info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
                    info.compress_type=zipfile.ZIP_DEFLATED
                    z.writestr(info,body)
        with zipfile.ZipFile(path) as z:
            for name,body in payload.items():
                if z.read(name)!=body: raise ValueError("archive member verification failed")
        return dict(sha256=file_digest(path),members=len(payload),bytes=path.stat().st_size)
    reader=archive(output/"reader.zip",public)
    evaluator=archive(output/"evaluator.zip",{"answers.jsonl":b"".join(canonical(a)+b"\n" for a in answers),
        "SOURCE.json":canonical(dict(lock_sha256=file_digest(source/"LOCK.json"),completion=completion))+b"\n"})
    report=dict(schema="v17.transfer.1",reader=reader,evaluator=evaluator,requests=len(requests),
        cases=len(selected),selection="first bounded completed cases per native family; outcome-blind early pilot",
        reader_only=True,portable_execution_verified=False)
    write(output/"EXPORT.json",report)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--per-family",type=int,default=12)
    args=parser.parse_args()
    print(json.dumps(package(args.source,args.output,args.per_family)))
if __name__=="__main__": main()
