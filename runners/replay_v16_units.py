"""Execute a separately retained bounded whole-unit replay; never owns live status."""
import argparse
from pathlib import Path
import os
import time
from ghostscale.validation.soundingline.v16.records import read, write, now, file_digest
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.replay_plan import plan
from ghostscale.validation.soundingline.v16.whole_replay import replay_item
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess
from ghostscale.validation.soundingline.v16.expansion_runner import EXTENSIONS
REPO=Path(__file__).resolve().parents[1]

def run(root,output,include_expansion=False):
    began,cpu=time.perf_counter(),time.process_time()
    plan_path=output/"PLAN.json"
    if plan_path.exists():
        selected=read(plan_path)
        if selected["includes_completed_expansion"]!=include_expansion:
            raise ValueError("cannot change frozen replay allocation")
    else:
        selected=plan(root,REPO,include_expansion=include_expansion)
        selected["replay_sources"]={path.relative_to(REPO).as_posix():file_digest(path) for path in [
            Path(__file__).resolve(),REPO/"ghostscale/validation/soundingline/v16/replay_plan.py",
            REPO/"ghostscale/validation/soundingline/v16/whole_replay.py"]}
        write(plan_path,selected)
    source_locks(root,REPO)
    for name,expected in selected["replay_sources"].items():
        if file_digest(REPO/name)!=expected:
            raise ValueError("frozen replay implementation changed")
    receipts=[]
    failures=[]
    with ReaderProcess(output/"reader",extensions=EXTENSIONS) as reader:
        for index,item in enumerate(selected["selected"]):
            destination=output/"private"/f"case-{index:03d}"
            try:
                receipt=replay_item(root,destination,item,reader)
            except ValueError as error:
                if not str(error).startswith("whole replay"):
                    raise
                failure={"source":item["source_unit"],"error":str(error),"recorded_at":now()}
                write(output/"failed-checks"/f"case-{index:03d}.json",failure)
                failures.append(failure)
                print({"failed_comparison":index+1,"source":item["source_unit"],"error":str(error)},flush=True)
                continue
            receipt["selected_source"]=item["source_unit"]
            write(output/"checks"/f"case-{index:03d}.json",receipt)
            receipts.append(receipt)
            print({"replayed":index+1,"planned":len(selected["selected"]),"source":item["source_unit"]},flush=True)
    result={"execution_state":"completed","instrument_state":"failed" if failures else "valid","recorded_at":now(),"process_id":os.getpid(),
        "plan_sha256":file_digest(plan_path),"n_units":len(receipts),"checks":receipts,
        "selected_units_wholly_replayed":not failures,"whole_unit_replay":False,"failed_comparisons":failures,
        "attempted_units":len(receipts)+len(failures),
        "scope":"Development replay across current completed sources; final campaign coverage still requires subsequent packets.",
        "pending_coverage":selected["pending_coverage"],"wall_seconds":time.perf_counter()-began,
        "cpu_seconds":time.process_time()-cpu}
    write(output/"RECEIPT.json",result)
    return result

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--include-completed-expansion",action="store_true")
    args=parser.parse_args()
    try:
        result=run(args.root,args.output,args.include_completed_expansion)
        if result["instrument_state"]!="valid":
            raise ValueError("bounded whole replay retained failed comparisons")
    except Exception as error:
        write(args.output/"FAILURE.json",{"execution_state":"failed","instrument_state":"unresolved",
            "recorded_at":now(),"error":repr(error),"whole_unit_replay":False})
        raise

if __name__=="__main__":
    main()
